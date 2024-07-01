"""
TODO: This script needs to be updated if VPP is used for neuron
"""

import os
import argparse
import json
from pathlib import Path, PurePath
from os.path import join
from glob import glob
import re
import numpy as np
import torch

def mt2hf_fix_query_key_value_ordering(param, checkpoint_version, num_splits, num_heads, hidden_size):
    # Permutes layout of param tensor to [num_splits * num_heads * hidden_size, :]
    # for compatibility with later versions of NVIDIA Megatron-LM.
    # The inverse operation is performed inside Megatron-LM to read checkpoints:
    # https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209
    # If param is the weight tensor of the self-attention block, the returned tensor
    # will have to be transposed one more time to be read by HuggingFace GPT2.
    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param

def unpack_gqa_single_qkv(qkv_weight, num_query_groups, num_attention_heads):
    """This method de-interleaves the QKV vectors to flat matrices expected by HF

    TODO: simplify, or eliminate this method if possible
    For now, we assume megatron legacy is not true

    Chunk each head weights, and then we split them along Q, K, V, and then finally concat them together

    :param qkv_weight:
    :param num_query_groups:
    :param num_attention_heads:
    :return:
    """
    head_dimension = qkv_weight.shape[0] // (num_attention_heads + num_query_groups * 2)
    print(f'Num_group_per_shard {num_query_groups}, head_dimension {head_dimension}')
    num_queries_per_group = num_attention_heads // num_query_groups

    qkv_weight_by_head = torch.chunk(qkv_weight, num_query_groups, dim=0)
    qs = []
    ks = []
    vs = []
    for qkv_weight in qkv_weight_by_head:
        q_head, k_head, v_head = torch.split(
            qkv_weight,
            [
                head_dimension * num_queries_per_group,
                head_dimension,
                head_dimension
            ],
            dim=0)
        qs.append(q_head)
        ks.append(k_head)
        vs.append(v_head)
    qs = torch.cat(qs, dim=0)
    ks = torch.cat(ks, dim=0)
    vs = torch.cat(vs, dim=0)
    return qs, ks, vs

def get_tp_pp_degree(path_to_checkpoints):

    dir_name = PurePath(path_to_checkpoints).name
    TP = 1
    PP = 1
    for folder in os.listdir(path_to_checkpoints):
        pp_search = re.search('pp_rank_[\d]*', folder)
        if pp_search:
            PP = max(PP, 1+int(pp_search[0].split('pp_rank_')[1]))
        if PP>1:
            tp_search = re.search('tp_rank_[\d]*', folder)
            if tp_search:
                TP = max(TP, 1+int(tp_search[0].split('tp_rank_')[1]))
        else:
            tp_search = re.search('mp_rank_[\d]*', folder)
            if tp_search:
                TP = max(TP, 1+int(tp_search[0].split('mp_rank_')[1]))

    return TP, PP

def _get_tp_str(tp: int):
    tp_template = '00'
    tp = str(tp)
    leading_zeros = len(tp_template) - len(tp)
    return ''.join(['0'] * leading_zeros + list(tp))

def _get_pp_str(pp: int):
    pp_template = '000'
    pp = str(pp)
    leading_zeros = len(pp_template) - len(pp)
    return ''.join(['0'] * leading_zeros + list(pp))

def get_checkpoints_for_pp(pp: int, path_to_checkpoints: str, PP: int=1, TP: int=1, is_xser: bool=False, checkpoint_step=None):
    """
    Returns all checkpoints for specified PP rank
    """
    if PP == 1 and TP == 1:
        pp_str = ""
    else:
        pp_str = f'tp_rank_*_pp_rank_{_get_pp_str(pp)}' if PP > 1 else "mp_rank_*"

    template = join(path_to_checkpoints, pp_str, '*.ckpt')

    #  if a spacific step is not specified, take largest step saved model from the available checkpoints
    if checkpoint_step == None:
        checkpoint_step = max([int(re.match(r".*-step=(\d+).*ckpt$", i).group(1))
                               for i in glob(template)])
    template = join(path_to_checkpoints, pp_str, f'*-step={checkpoint_step}*.ckpt')
    tp_paths = sorted(glob(template))
    if is_xser:
        import nemo.collections.nlp.parts.serialization as nser
        load_fn = lambda path: nser.load(path, cpu_only=True)
    else:
        load_fn = lambda path: torch.load(path, map_location='cpu')
    return {i: load_fn(p) for i, p in enumerate(tp_paths)}


def _get_nemo_key(k, nemo_key = 'model.language_model.'):
    if "final_layernorm" in k:
        nemo_key += 'encoder.'
    return k.replace(nemo_key, '')

def convert_checkpoint(config_file,
                       path_to_checkpoints,
                       output_path,
                       checkpoint_version=2.0,
                       is_xser=False,
                       checkpoint_step=None,
                       trn=False):

    with open(config_file, "r") as f:
        config = json.load(f)

    translation = {
        "embedding.word_embeddings.weight": (1, "model.embed_tokens.weight", 0, 0), # a['model']['language_model']['word_embeddings']['weight']
        "input_layernorm.weight": (0, "input_layernorm.weight", None, 0),
        "self_attention.query.weight": (1, "self_attn.query.weight", 0, 0),
        "self_attention.key_value.weight": (1, "self_attn.key_value.weight", 0, 0),
        "self_attention.query_key_value.weight": (1, "self_attn.query_key_value.weight", 0, 0),
        "self_attention.dense.weight": (1, "self_attn.o_proj.weight", 1, 0),
        "post_attention_layernorm.weight": (0, "post_attention_layernorm.weight", None, 0),
        "self_attention.core_attention.rotary_emb.inv_freq": (0, "self_attn.rotary_emb.inv_freq", None, 0),
        "mlp.dense_h_to_4h.weight": (1, "mlp.gate_proj.weight", 0, 0),
        "mlp.dense_h_to_4h_2.weight": (1, "mlp.up_proj.weight", 0, 0),
        "mlp.dense_4h_to_h.weight": (1, "mlp.down_proj.weight", 1, 0),
        "final_layernorm.weight": (0, "model.norm.weight", None, 0),
        # TODO: do based on nemo version
        # "output_layer.weight": (1, "lm_head.weight", 0, 0),  # final layer word embedding (for nemo 1.22)
        "model.word_embeddings.weight": (1, "lm_head.weight", 0, 0),  # final layer word embedding (for nemo 1.18)
    }

    nemo_key = "model.language_model."
    br_key = "model.layers."

    TP, PP = get_tp_pp_degree(path_to_checkpoints)
    print(f"TP: {TP}, PP: {PP}")

    heads = config["num_attention_heads"]
    hidden_size_per_head = config["hidden_size"] // heads

    hf_model = {}

    layer_re = re.compile("model.language_model.encoder.layers\.(\d+)\.([a-z0-9_.]+)\.([a-z_]+)")

    for pp in range(PP):
        print(f"Loading PP={pp}")
        checkpoints_for_pp = get_checkpoints_for_pp(pp, path_to_checkpoints, PP, TP, is_xser, checkpoint_step)
        tp_models = {i: v['state_dict'] for i, v in checkpoints_for_pp.items()}
        assert len(tp_models) > 0, f"was not able to get any checkpoints for pipeline rank {pp} from path {path_to_checkpoints}"
        layer_keys = tp_models[0].keys()
        for k in layer_keys:
            print(f">> {k}")
            if "position_embeddings" in k:
                nemo_key = _get_nemo_key(k)
                _, key, _, _ = translation[nemo_key]
                hf_model[key] = tp_models[0][k]
                continue

            if "word_embeddings" in k:
                nemo_key = _get_nemo_key(k)
                split, key, dim, transpose = translation[nemo_key]
                hf_model[key] = torch.concat([tp_models[i][k] for i in range(len(tp_models))], dim=0)
                continue

            if "output_layer" in k:
                nemo_key = _get_nemo_key(k)
                split, key, dim, transpose = translation[nemo_key]
                hf_model[key] = torch.concat([tp_models[i][k] for i in range(len(tp_models))], dim=dim)
                continue

            if "final_layernorm" in k:
                nemo_key = _get_nemo_key(k)
                split, key, dim, transpose = translation[nemo_key]
                hf_model[key] = tp_models[0][k]
                continue

            # on gpu checkpoints, we have a single/common rotary pos emb named 'model.language_model.rotary_pos_emb.inv_freq'
            # for all layers. Unlike in trn checkpoints, where we have a separate rotary pos emb named 'self_attention.core_attention.rotary_emb.inv_freq' for each layer
            # although, it seems they are all the same. Below condition is for GPU checkpoints
            if k == "model.language_model.rotary_pos_emb.inv_freq":
                for i in range(config["num_hidden_layers"]):
                    hf_model[f"model.layers.{i}.self_attn.rotary_emb.inv_freq"] = tp_models[0][k]
                continue

            m = layer_re.match(k)
            layer_idx = m.group(1)
            op_name = m.group(2)
            weight_or_bias = m.group(3)
            nemo_key = f"{op_name}.{weight_or_bias}"
            split, key, dim, transpose = translation[nemo_key]
            ln_idx= int(layer_idx) + pp*(config["num_hidden_layers"]//PP)
            hf_key = f"{br_key}{ln_idx}.{key}"
            if split:
                # TODO: check this based on layer keys not a passed in flag (trn)
                if "dense_h_to_4h" in k and trn:
                    split_key = key.replace("gate_proj", "up_proj")
                    split_hf_key = f"{br_key}{ln_idx}.{split_key}"
                    dense_h_to_4h_weights_concat = []
                    dense_h_to_4h_2_weights_concat = []
                    for i in range(len(tp_models)):
                        assert tp_models[i][k].size()[0] % 2 == 0
                        split_size = tp_models[i][k].size()[0] // 2
                        dense_h_to_4h_weights, dense_h_to_4h_2_weights = torch.split(tp_models[i][k], split_size)
                        dense_h_to_4h_weights_concat.append(dense_h_to_4h_weights)
                        dense_h_to_4h_2_weights_concat.append(dense_h_to_4h_2_weights)
                    hf_model[hf_key] = torch.concat(dense_h_to_4h_weights_concat, dim=dim)
                    hf_model[split_hf_key] = torch.concat(dense_h_to_4h_2_weights_concat, dim=dim)
                else:
                    hf_model[hf_key] = torch.concat([tp_models[i][k] for i in range(len(tp_models))], dim=dim)
            else:
                hf_model[hf_key] = tp_models[0][k]

            # Gpu training stack and NNM have different proj weight storage with GQA, we can normalize all here
            # if we get query_key_value and we have query groups, its GQA gpu training stack style
            # otherwise, for NNM we will have "query" and "key_value" keys
            if "self_attention.query_key_value.weight" in k:
                num_query_groups = config.get("num_query_groups", config.get("num_key_value_heads"))
                # If we are using gpu training stack, QKV are a single matrix no matter what
                # Also, if we are using the "alternative" GQA implementation for trn in Neuron-Nemo-Megatron the QKV is a single matrix
                if num_query_groups is not None and num_query_groups > 0:
                    qkv = unpack_gqa_single_qkv(hf_model.pop(hf_key), num_query_groups, heads)
                else:
                    qkv = mt2hf_fix_query_key_value_ordering(hf_model.pop(hf_key), 2.0, 3, heads, hidden_size_per_head)
                    qkv = torch.chunk(qkv, 3, dim=0)
                hf_model[f'{br_key}{ln_idx}.self_attn.q_proj.weight'] = qkv[0]
                hf_model[f'{br_key}{ln_idx}.self_attn.k_proj.weight'] = qkv[1]
                hf_model[f'{br_key}{ln_idx}.self_attn.v_proj.weight'] = qkv[2]

            if 'self_attention.query.weight' in k:
                hf_model[f"{br_key}{ln_idx}.self_attn.q_proj.weight"] = mt2hf_fix_query_key_value_ordering(hf_model[hf_key], checkpoint_version, 1, heads, hidden_size_per_head)
                hf_model.pop(hf_key)

            if 'self_attention.key_value.weight' in k:
                kv_heads = config['num_key_value_heads']
                hf_model[hf_key] = mt2hf_fix_query_key_value_ordering(
                    hf_model[hf_key], checkpoint_version, 2, kv_heads, hidden_size_per_head
                )
                hf_key_k = f"{br_key}{ln_idx}.self_attn.k_proj.weight"
                hf_key_v = f"{br_key}{ln_idx}.self_attn.v_proj.weight"
                size_per_seg = hf_model[hf_key].shape[0] // 2
                hf_model[hf_key_k], hf_model[hf_key_v] = torch.split(hf_model[hf_key], size_per_seg, dim=0)
                hf_model.pop(hf_key)

            if transpose:
                hf_model[hf_key] = torch.transpose(hf_model[hf_key], 0, 1)


    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(hf_model, str(path)+"/pytorch_model.bin")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_version", default=2.0)
    parser.add_argument(
        "--path_to_checkpoints",
        type=str,
        help="Path to the checkpoints from Nemo. Directory is parent directory of model parallel folders.",
        required=True
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="The config json file describing the pre-trained model.",
        required=True
    )
    parser.add_argument(
        "--output_path",
        default="",
        type=str,
        help="output path",
    )
    parser.add_argument(
        "--is_xser",
        action="store_true",
        help="Enable serialized loading",
    )
    parser.add_argument(
        "--step",
        type=str,
        help="Specify a checkpoint step e.g., '22500'. If not specified, the latest checkpoint will be used.",
        required=False
    )
    parser.add_argument(
        "--trn",
        action="store_true",
        help="Whether the checkpoint is a Trainium checkpoint.",
    )
    args = parser.parse_args()
    convert_checkpoint(args.config_file,
                       args.path_to_checkpoints,
                       args.output_path,
                       args.checkpoint_version,
                       args.is_xser,
                       args.step,
                       args.trn)
