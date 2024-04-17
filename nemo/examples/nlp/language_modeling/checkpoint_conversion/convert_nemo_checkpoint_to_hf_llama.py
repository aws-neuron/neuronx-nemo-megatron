import os
import argparse
import json
from pathlib import Path, PurePath
from os.path import join
from glob import glob
import re
import numpy as np
import torch

def fix_query_key_value_ordering(param, checkpoint_version, num_splits, num_heads, hidden_size):
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

def get_checkpoints_for_pp(pp: int, path_to_checkpoints: str, PP: int=1, TP: int=1, is_xser: bool=False):
    """
    Returns all checkpoints for specified PP rank
    """
    if PP == 1 and TP == 1:
        pp_str = ""
    else:
        pp_str = f'tp_rank_*_pp_rank_{_get_pp_str(pp)}' if PP > 1 else "mp_rank_*"

    template = join(path_to_checkpoints, pp_str, '*.ckpt')

    # take largest step saved model from the available checkpoints
    max_step_recorded=max(
       [int(re.match(r".*megatron_llama--step=(\d+).*ckpt$", i).group(1))
       for i in glob(template)])
    template = join(path_to_checkpoints, pp_str, f'*megatron_llama--step={max_step_recorded}*.ckpt')
    tp_paths = sorted(glob(template))
    if is_xser:
        import nemo.collections.nlp.parts.serialization as nser
        load_fn = lambda path: nser.load(path, cpu_only=True)
    else:
        load_fn = torch.load
    return {i: load_fn(p)['state_dict'] for i, p in enumerate(tp_paths)}


def get_checkpoints_for_tp(tp: int, path_to_checkpoints: str, is_xser: bool=False):
    """
    Returns all checkpoints for specified TP rank
    """
    tp_str = _get_tp_str(tp)
    template = join(path_to_checkpoints, f'tp_rank_{tp_str}_pp_rank_*', '*.ckpt')

    pp_paths = sorted(glob(template))
    if is_xser:
        import nemo.collections.nlp.parts.serialization as nser
        load_fn = lambda path: nser.load(path, cpu_only=True)
    else:
        load_fn = torch.load
    return {i: load_fn(p)['state_dict'] for i, p in enumerate(pp_paths)}

def _get_nemo_key(k, nemo_key = 'model.language_model.'):
    if "final_layernorm" in k:
        nemo_key += 'encoder.'
    return k.replace(nemo_key, '')

def convert_checkpoint(config_file,
                       path_to_checkpoints,
                       output_path,
                       checkpoint_version=2.0,
                       is_xser=False):

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
        "mlp.dense_h_to_4h.weight": (1, "mlp.gate_proj_up_proj.weight", 0, 0),
        "mlp.dense_4h_to_h.weight": (1, "mlp.down_proj.weight", 1, 0),
        "final_layernorm.weight": (0, "model.norm.weight", None, 0),
        "output_layer.weight": (1, "lm_head.weight", 0, 0),  # this is shared
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
        tp_models = get_checkpoints_for_pp(pp, path_to_checkpoints, PP, TP, is_xser)
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

            m = layer_re.match(k)
            layer_idx = m.group(1)
            op_name = m.group(2)
            weight_or_bias = m.group(3)
            nemo_key = f"{op_name}.{weight_or_bias}"
            split, key, dim, transpose = translation[nemo_key]
            ln_idx= int(layer_idx) + pp*(config["num_hidden_layers"]//PP)
            hf_key = f"{br_key}{ln_idx}.{key}"
            if split:
                hf_model[hf_key] = torch.concat([tp_models[i][k] for i in range(len(tp_models))], dim=dim)
            else:
                hf_model[hf_key] = tp_models[0][k]
            if "query_key" in k:
                hf_model[hf_key] = fix_query_key_value_ordering(hf_model[hf_key], checkpoint_version, 3, heads, hidden_size_per_head)

            if 'self_attention.query.weight' in k:
                hf_model[f"{br_key}{ln_idx}.self_attn.q_proj.weight"] = fix_query_key_value_ordering(hf_model[hf_key], checkpoint_version, 1, heads, hidden_size_per_head)
                hf_model.pop(hf_key)

            if 'self_attention.key_value.weight' in k:
                kv_heads = config['num_key_value_heads']
                hf_model[hf_key] = fix_query_key_value_ordering(
                    hf_model[hf_key], checkpoint_version, 2, kv_heads, hidden_size_per_head
                )
                hf_key_k = f"{br_key}{ln_idx}.self_attn.k_proj.weight"
                hf_key_v = f"{br_key}{ln_idx}.self_attn.v_proj.weight"
                size_per_seg = hf_model[hf_key].shape[0] // 2
                hf_model[hf_key_k], hf_model[hf_key_v] = torch.split(hf_model[hf_key], size_per_seg, dim=0)
                hf_model.pop(hf_key)

            if transpose:
                hf_model[hf_key] = torch.transpose(hf_model[hf_key], 0, 1)

            # Break Q K V into three matrices
            if "query_key" in k:
                hf_key_q = f"{br_key}{ln_idx}.self_attn.q_proj.weight"
                hf_key_k = f"{br_key}{ln_idx}.self_attn.k_proj.weight"
                hf_key_v = f"{br_key}{ln_idx}.self_attn.v_proj.weight"
                size_per_seg = hf_model[hf_key].shape[0] // 3
                hf_model[hf_key_q], hf_model[hf_key_k], hf_model[hf_key_v] = torch.split(hf_model[hf_key], size_per_seg, dim=0)
                hf_model.pop(hf_key)

            if "dense_h_to_4h" in k:
                hf_key_gate_proj = f"{br_key}{ln_idx}.mlp.gate_proj.weight"
                hf_key_up_proj = f"{br_key}{ln_idx}.mlp.up_proj.weight"
                size_per_seg = hf_model[hf_key].shape[0] // 2
                hf_model[hf_key_gate_proj], hf_model[hf_key_up_proj] = torch.split(hf_model[hf_key], size_per_seg, dim=0)
                hf_model.pop(hf_key)

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
    args = parser.parse_args()
    convert_checkpoint(args.config_file, args.path_to_checkpoints, args.output_path, args.checkpoint_version, args.is_xser)
