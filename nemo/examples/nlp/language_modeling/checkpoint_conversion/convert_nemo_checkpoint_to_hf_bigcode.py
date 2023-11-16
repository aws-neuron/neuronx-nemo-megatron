import os
import argparse
import json
from pathlib import Path, PurePath
from os.path import join
from glob import glob
import re
import numpy as np
import torch
import torch_xla.utils.serialization as xser


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
    TP = 1
    PP = 1

    for folder in glob(os.path.join(path_to_checkpoints, '**/*.ckpt')):
        tp_search = re.search('[mt]p_rank_[\d]*', folder)
        if tp_search:
            split_key = 'tp_rank_' if 'tp_rank_' in tp_search[0] else 'mp_rank_'
            TP = max(TP, 1+int(tp_search[0].split(split_key)[1]))

        pp_search = re.search('pp_rank_[\d]*', folder)
        if pp_search:
            PP = max(PP, 1+int(pp_search[0].split('pp_rank_')[1]))
            split_key = None

    return TP, PP, split_key


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


def get_step_id(path):
    # original checkpoint (useful for sanity checking)
    if 'model_optim_rng.ckpt' in path:
        return 0

    step_id = int(re.search('step\=(.*)\-', path)[1])
    return int(step_id)


def get_checkpoints_for_pp(pp: int, path_to_checkpoints: str, PP: int = 1, TP: int = 1, is_xser: bool = False, split_key=None):
    """
    Returns all checkpoints for specified PP rank
    """
    if PP == 1 and TP == 1:
        pp_str = ""
    else:
        pp_str = f'tp_rank_*_pp_rank_{_get_pp_str(pp)}' if PP > 1 else f"{split_key}*"

    template = join(path_to_checkpoints, pp_str, '*.ckpt')

    tp_paths_sorted_by_step_id = sorted(glob(template), key=get_step_id)
    last_step_id = get_step_id(tp_paths_sorted_by_step_id[-1])
    print(f'Will load model for step {last_step_id}')
    most_recent_tp_paths = [
        p for p in tp_paths_sorted_by_step_id if get_step_id(p) == last_step_id]
    most_recent_tp_paths_sorted_by_tp = sorted(most_recent_tp_paths)
    print(f'paths {most_recent_tp_paths_sorted_by_tp}')
    return {i: xser.load(p)['state_dict'] if is_xser else torch.load(p)['state_dict'] for i, p in enumerate(most_recent_tp_paths_sorted_by_tp)}


def get_checkpoints_for_tp(tp: int, path_to_checkpoints: str, is_xser: bool = False):
    """
    Returns all checkpoints for specified TP rank
    """
    tp_str = _get_tp_str(tp)
    template = join(path_to_checkpoints,
                    f'tp_rank_{tp_str}_pp_rank_*', '*.ckpt')

    pp_paths = sorted(glob(template))
    return {i: xser.load(p)['state_dict'] if is_xser else torch.load(p)['state_dict'] for i, p in enumerate(pp_paths)}


def _get_nemo_key(k, nemo_key='model.language_model.'):
    if "final_layernorm" in k:
        nemo_key += 'encoder.'
    return k.replace(nemo_key, '')


def _merge_query_and_kv(hf_model, num_layers):
    for i in range(num_layers):
        q_weight = hf_model[f"transformer.h.{i}.attn.c_attn_query.weight"]
        kv_weight = hf_model[f"transformer.h.{i}.attn.c_attn_key_value.weight"]

        q_bias = hf_model[f"transformer.h.{i}.attn.c_attn_query.bias"]
        kv_bias = hf_model[f"transformer.h.{i}.attn.c_attn_key_value.bias"]

        hf_model[f'transformer.h.{i}.attn.c_attn.weight'] = torch.concat(
            [q_weight, kv_weight], dim=0)
        hf_model[f'transformer.h.{i}.attn.c_attn.bias'] = torch.concat(
            [q_bias, kv_bias], dim=0)
        hf_model.pop(f"transformer.h.{i}.attn.c_attn_query.weight")
        hf_model.pop(f"transformer.h.{i}.attn.c_attn_query.bias")
        hf_model.pop(f"transformer.h.{i}.attn.c_attn_key_value.weight")
        hf_model.pop(f"transformer.h.{i}.attn.c_attn_key_value.bias")

    return hf_model


def convert_checkpoint(config_file,
                       path_to_checkpoints,
                       output_path,
                       checkpoint_version=2.0,
                       is_xser=False):

    with open(config_file, "r") as f:
        config = json.load(f)

    translation = {
        "embedding.word_embeddings.weight": (1, "transformer.wte.weight", 0, 0),
        "embedding.position_embeddings.weight": (0, "transformer.wpe.weight", None, 0),
        "input_layernorm.weight": (0, "ln_1.weight", None, 0),
        "input_layernorm.bias": (0, "ln_1.bias", None, 0),
        "self_attention.query.weight": (1, "attn.c_attn_query.weight", 0, 0),
        "self_attention.query.bias": (1, "attn.c_attn_query.bias", 0, 0),
        "self_attention.key_value.weight": (0, "attn.c_attn_key_value.weight", None, 0),
        "self_attention.key_value.bias": (0, "attn.c_attn_key_value.bias", None, 0),
        "self_attention.dense.weight": (1, "attn.c_proj.weight", 1, 0),
        "self_attention.dense.bias": (0, "attn.c_proj.bias", None, 0),
        "post_attention_layernorm.weight": (0, "ln_2.weight", None, 0),
        "post_attention_layernorm.bias": (0, "ln_2.bias", None, 0),
        "mlp.dense_h_to_4h.weight": (1, "mlp.c_fc.weight", 0, 0),
        "mlp.dense_h_to_4h.bias": (1, "mlp.c_fc.bias", 0, 0),
        "mlp.dense_4h_to_h.weight": (1, "mlp.c_proj.weight", 1, 0),
        "mlp.dense_4h_to_h.bias": (0, "mlp.c_proj.bias", None, 0),
        "final_layernorm.weight": (0, "transformer.ln_f.weight", None, 0),
        "final_layernorm.bias": (0, "transformer.ln_f.bias", None, 0),
        "output_layer.weight": (1, "lm_head.weight", 0, 0),
        # share_embeddings_and_output_weights
        "model.word_embeddings.weight": (1, "lm_head.weight", 0, 0),
    }

    nemo_key = "model.language_model."
    br_key = "transformer.h."

    TP, PP, split_key = get_tp_pp_degree(path_to_checkpoints)

    heads = config["n_head"]
    hidden_size_per_head = config["n_embd"] // heads

    hf_model = {}

    layer_re = re.compile(
        "model.language_model.encoder.layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")
    for pp in range(PP):
        tp_models = get_checkpoints_for_pp(
            pp, path_to_checkpoints, PP, TP, is_xser, split_key)
        layer_keys = tp_models[0].keys()
        for k in layer_keys:
            if "position_embeddings" in k:
                nemo_key = _get_nemo_key(k)
                _, key, _, _ = translation[nemo_key]
                hf_model[key] = tp_models[0][k]
                continue

            if "word_embeddings" in k:
                nemo_key = _get_nemo_key(k)
                split, key, dim, transpose = translation[nemo_key]
                hf_model[key] = torch.concat(
                    [tp_models[i][k] for i in range(len(tp_models))], dim=0)
                continue

            if "output_layer" in k:
                nemo_key = _get_nemo_key(k)
                split, key, dim, transpose = translation[nemo_key]
                hf_model[key] = torch.concat(
                    [tp_models[i][k] for i in range(len(tp_models))], dim=dim)
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
            ln_idx = int(layer_idx) + pp*(config["n_layer"]//PP)
            hf_key = f"{br_key}{ln_idx}.{key}"
            if split:
                hf_model[hf_key] = torch.concat(
                    [tp_models[i][k] for i in range(len(tp_models))], dim=dim)
            else:
                hf_model[hf_key] = tp_models[0][k]
            if "query" in k:
                hf_model[hf_key] = fix_query_key_value_ordering(
                    hf_model[hf_key], checkpoint_version, 1, heads, hidden_size_per_head)
            elif "key_value" in k:
                hf_model[hf_key] = fix_query_key_value_ordering(
                    hf_model[hf_key], checkpoint_version, 2, 1, hidden_size_per_head)
            if transpose:
                hf_model[hf_key] = torch.transpose(hf_model[hf_key], 0, 1)

    hf_model = _merge_query_and_kv(
        hf_model=hf_model, num_layers=config["n_layer"])
    config['vocab_size'] = hf_model['transformer.wte.weight'].shape[0]

    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(hf_model, str(path)+"/pytorch_model.bin")
    with open(str(path)+"/config.json", 'wt') as f:
        json.dump(config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_version", default=2.0)
    parser.add_argument(
        "--path_to_checkpoints",
        type=str,
        help="Path to the checkpoints from creating NeMo checkpoint files using `convert_hf_checkpoint_to_nemo.py`",
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
        default=False,
        type=bool
    )
    args = parser.parse_args()
    convert_checkpoint(args.config_file, args.path_to_checkpoints,
                       args.output_path, args.checkpoint_version, args.is_xser)
