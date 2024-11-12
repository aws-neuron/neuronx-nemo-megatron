import argparse
import json
from pathlib import Path
import numpy as np
import torch
import os
import itertools
import logging
import glob
from multiprocessing import Pool
from functools import partial

from examples.nlp.language_modeling.checkpoint_conversion.logger_factory import LoggerFactory

logger = LoggerFactory.create_logger(name="hf_to_nemo_llama", level=logging.INFO)


def fix_query_key_value_ordering(
        param, checkpoint_version, num_splits, num_heads, hidden_size
):
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
        saved_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


def convert_checkpoint(p, args, config):
    TP = args.tp_degree
    PP = args.pp_degree
    VPP = args.vpp_degree

    out_models = convert_state_dict_for_pp_rank(p, args.path_to_checkpoint, config, TP, PP, VPP, args.save_bf16)

    for i in range(args.tp_degree):
        output_folder = args.output_path
        if TP > 1:
            if PP > 1:
                output_folder = output_folder + f"/tp_rank_{i:02d}"
            else:
                output_folder = output_folder + f"/mp_rank_{i:02d}"
        if PP > 1:
            output_folder = output_folder + f"_pp_rank_{p:03d}"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        checkpoint_name = f"{output_folder}/llmv2_converted_checkpoint--step={args.step}.ckpt"
        if args.is_xser:
            from nemo.collections.nlp.parts.serialization import save
            save(out_models[i], checkpoint_name)
        else:
            torch.save(out_models[i], checkpoint_name)  # , (not master_only), global_master=True)
    logger.info("Done saving Megatron checkpoint")


def get_layer_map(pipeline_parallel_degree, virtual_pipeline_parallel_size, num_layers):
    """
    # https://github.com/NVIDIA/NeMo/blob/v1.22.0/nemo/collections/nlp/modules/common/megatron/transformer.py#L1147-L1159
    for PP=2, VPP=2, N=8, layer_map will be:
    [
        [ [0,1], [4,5] ],
        [ [2,3], [6,7] ],
    ]
    where each row is layers falling on PP i
    """
    layer_map = [[] for _ in range(pipeline_parallel_degree)]
    for pp_idx in range(pipeline_parallel_degree):
        layer_map[pp_idx] = [[] for _ in range(virtual_pipeline_parallel_size)]
    layers_per_stage = num_layers // pipeline_parallel_degree // virtual_pipeline_parallel_size
    pp_idx = 0
    virtual_pp_idx = 0
    num_layers_in_stage = 0
    for layer_idx in range(num_layers):
        logger.debug(f"Adding layer {layer_idx} to layer_map[pp_idx = {pp_idx}][virtual_pp_idx = {virtual_pp_idx}]")
        layer_map[pp_idx][virtual_pp_idx].append(layer_idx)
        num_layers_in_stage = (num_layers_in_stage + 1) % layers_per_stage
        if num_layers_in_stage == 0:
            pp_idx = (pp_idx + 1) % pipeline_parallel_degree
            if pp_idx == 0:
                virtual_pp_idx = (virtual_pp_idx + 1) % virtual_pipeline_parallel_size
    logger.info(f'vpp layer_map is {layer_map}')
    return layer_map


def _load_checkpoint(path):
    logger.info(f"Loading checkpoint {path}")
    x = torch.load(path, map_location='cpu')
    logger.info(f"Loaded checkpoint {path}")
    return x


def load_model_checkpoints(model_paths):
    logger.info(f"Loading checkpoints in parallel using {len(model_paths)} processes")
    
    with Pool(processes=len(model_paths)) as pool:
        checkpoint_list = pool.map(_load_checkpoint, model_paths)
        pool.close()
        pool.join()

    logger.info(f'Building llama models')
    model_llama = {}
    for ts in checkpoint_list:
        model_llama.update(ts)
    
    logger.info(f"Loaded {len(model_llama)} keys into model_llama")
    return model_llama


def convert_state_dict_for_pp_rank(p, path_to_checkpoint, config, TP, PP, VPP, save_bf16):
    br_key = "layers."  # Used to filter all transformer layers except layernorm

    translation = {
        "model.language_model.embedding.word_embeddings.weight": (1, "model.embed_tokens.weight", 0, 0),
        # a['model']['language_model']['word_embeddings']['weight']
        "input_layernorm.weight": (0, "input_layernorm.weight", None, 0),
        "self_attention.query_key_value.weight": (1, "self_attn.query_key_value.weight", 0, 0),
        "self_attention.dense.weight": (1, "self_attn.o_proj.weight", 1, 0),
        "post_attention_layernorm.weight": (0, "post_attention_layernorm.weight", None, 0),
        "self_attention.core_attention.rotary_emb.inv_freq": (0, "self_attn.rotary_emb.inv_freq", None, 0),
        "mlp.dense_h_to_4h.weight": (1, "mlp.gate_proj.weight", 0, 0),
        "mlp.dense_h_to_4h_2.weight": (1, "mlp.up_proj.weight", 0, 0),
        "mlp.dense_4h_to_h.weight": (1, "mlp.down_proj.weight", 1, 0),
        "model.language_model.encoder.final_layernorm.weight": (0, "model.norm.weight", None, 0),
        "model.language_model.output_layer.weight": (1, "lm_head.weight", 0, 0),
    }

    reverse_translation = {}
    for k, v in translation.items():
        split, br_k, dim, transpose = v
        reverse_translation[br_k] = (split, k, dim, transpose)

    model_paths = sorted(glob.glob(f'{path_to_checkpoint}/pytorch_model*.bin'))
    model_llama = load_model_checkpoints(model_paths)
    logger.info("Loaded Llama model")

    # Merge QKV
    for i in range(config['num_hidden_layers']):
        q = model_llama[f'model.layers.{i}.self_attn.q_proj.weight']
        k = model_llama[f'model.layers.{i}.self_attn.k_proj.weight']
        v = model_llama[f'model.layers.{i}.self_attn.v_proj.weight']
        model_llama[f'model.layers.{i}.self_attn.query_key_value.weight'] = torch.cat([q, k, v], dim=0)

        model_llama.pop(f'model.layers.{i}.self_attn.q_proj.weight')
        model_llama.pop(f'model.layers.{i}.self_attn.k_proj.weight')
        model_llama.pop(f'model.layers.{i}.self_attn.v_proj.weight')

    n_layers_per_pp = config['num_hidden_layers'] // PP
    if VPP is None:
        layers_on_curr_pp = list(range(p * n_layers_per_pp, (p + 1) * n_layers_per_pp))
    else:
        layer_map = get_layer_map(PP, VPP, config['num_hidden_layers'])
        layers_on_curr_pp = list(itertools.chain(*layer_map[p]))

    out_models = {}
    for i in range(TP):
        logger.info(f"=== PP {p}, TP {i} ===")
        nemo_model = {}
        for k, v in model_llama.items():
            # logger.info(f">>> {k}")
            if "attention.masked_bias" in k:
                # We don't want to copy attention mask bias, since its a constant of 1e4
                continue
            if br_key in k:
                parts = k.split(br_key)[1].split(".")
                layer_number = parts[0]

                if int(layer_number) in layers_on_curr_pp:
                    logger.info(f'layer_number in HF: {layer_number}')
                    k = ".".join(parts[1:])
                    if k == "attn.bias":
                        continue
                    split, key, dim, tranpose = reverse_translation[k]
                    layer_number = str(layers_on_curr_pp.index(int(layer_number)))
                    logger.info(f'layer_number in nemo: {layer_number}')
                    key = "model.language_model.encoder.layers." + layer_number + "." + key
                    nemo_model[key] = v
                    if tranpose:
                        nemo_model[key] = torch.transpose(
                            nemo_model[key], 0, 1
                        )

                    if "query_key_value" in key:
                        heads = config["num_attention_heads"]
                        hidden_size_per_head = config["hidden_size"] // heads
                        nemo_model[key] = fix_query_key_value_ordering(
                            nemo_model[key], 2.0, 3, heads, hidden_size_per_head
                        )

                    if split:
                        tp_last_dim_size = nemo_model[key].shape[dim] // TP
                        if dim:  # First or last dimension to shard
                            nemo_model[key] = nemo_model[key][
                                            ..., i * tp_last_dim_size: (i + 1) * tp_last_dim_size
                                            ].clone()
                        else:
                            nemo_model[key] = nemo_model[key][
                                            i * tp_last_dim_size: (i + 1) * tp_last_dim_size, ...
                                            ].clone()

                    logger.info(f'key: {key}, split: {split}, nemo_model[key].shape: {nemo_model[key].shape}, v.shape: {v.shape}')
            else:
                split, key, dim, transpose = reverse_translation[k]
                if "embed_tokens" in k and p == 0:
                    # Padding to make it divisble by TP degree
                    if v.shape[0] % TP > 0:
                        x = torch.nn.functional.pad(
                            v, (0, (TP - v.shape[0] % TP), 0, 0)
                        )
                    else:
                        x = v
                    last_dim_size = x.shape[0]
                    tp_last_dim_size = last_dim_size // TP
                    nemo_model[key] = x[
                                      i * tp_last_dim_size: (i + 1) * tp_last_dim_size, ...
                                      ].clone()
                    logger.info(f'key: {key}, split: {split}, nemo_model[key].shape: {nemo_model[key].shape}, v.shape: {v.shape}')
                elif "model.norm" in k and p == (PP - 1):
                    nemo_model[key] = v
                    logger.info(f'key: {key}, split: {split}, nemo_model[key].shape: {nemo_model[key].shape}, v.shape: {v.shape}')
                elif "lm_head" in k and p == (PP - 1):  # Not used
                    if split:
                        tp_last_dim_size = v.shape[dim] // TP
                        if dim:
                            nemo_model[key] = v[..., i * tp_last_dim_size:(i + 1) * tp_last_dim_size].clone()
                        else:
                            nemo_model[key] = v[i * tp_last_dim_size:(i + 1) * tp_last_dim_size, ...].clone()
                    logger.info(f'key: {key}, split: {split}, nemo_model[key].shape: {nemo_model[key].shape}, v.shape: {v.shape}')

        if save_bf16:
            for _k in nemo_model:
                nemo_model[_k] = nemo_model[_k].to(dtype=torch.bfloat16, device='cpu')

        # TODO: fix the hardcoded keys
        out_model = {"state_dict": nemo_model, 'pytorch-lightning_version': '1.9.5', 'epoch': 0, 'global_step': 0}

        # merge dense_h_to_4h and dense_h_to_4h_2 for Trn checkpoint
        h_to_4h_keys = [k for k in out_model['state_dict'].keys() if 'dense_h_to_4h' in k and 'dense_h_to_4h_2' not in k]
        for k in h_to_4h_keys:
            k_1 = k.replace('dense_h_to_4h', 'dense_h_to_4h_2')
            logger.info(f"merging {k} ({out_model['state_dict'][k].shape}) and {k_1} ({out_model['state_dict'][k_1].shape})")
            out_model['state_dict'][k] = torch.concat([out_model['state_dict'][k], out_model['state_dict'][k_1]], dim=0)
            out_model['state_dict'].pop(k_1)

        if VPP is not None:
            convert_checkpoint_to_vpp_format(out_model, p, PP, VPP)
        out_models[i] = out_model
    return out_models


def convert_checkpoint_to_vpp_format(checkpoint, p, PP, VPP):
    for vpp_index in range(VPP):
        checkpoint[f'model{vpp_index}'] = {'language_model': {'encoder': {}}}

    if p == 0 and 'model.language_model.embedding.word_embeddings.weight' in checkpoint['state_dict']:
        checkpoint['model0']['language_model']['embedding'] = {'word_embeddings': {'weight': checkpoint['state_dict'].pop('model.language_model.embedding.word_embeddings.weight')}}

    layer_items = []
    for key, value in list(checkpoint['state_dict'].items()):
        if 'encoder.layers' in key:
            layer_number = int(key.split('.')[4])
            layer_items.append((layer_number, key, value))
            del checkpoint['state_dict'][key]

    layer_items.sort(key=lambda x: x[0])
    assert len(layer_items) % VPP == 0
    layers_per_model = len(layer_items) // VPP

    for idx, (layer_number, key, value) in enumerate(layer_items):
        vpp_index = idx // layers_per_model
        if vpp_index >= VPP:
            vpp_index = VPP - 1
        
        new_layer_number = idx % layers_per_model
        stripped_key = key.split('encoder.', 1)[1]
        stripped_key = stripped_key.replace(f'layers.{layer_number}.', f'layers.{new_layer_number}.')
        checkpoint[f'model{vpp_index}']['language_model']['encoder'][stripped_key] = value
    del layer_items

    if p == PP - 1:
        if 'model.language_model.encoder.final_layernorm.weight' in checkpoint['state_dict']:
            checkpoint[f'model{VPP-1}']['language_model']['encoder']['final_layernorm.weight'] = checkpoint['state_dict'].pop('model.language_model.encoder.final_layernorm.weight')
        if 'model.word_embeddings.weight' in checkpoint['state_dict']: # model head
            checkpoint[f'model{VPP-1}']['word_embeddings_for_head'] = {'weight': checkpoint['state_dict'].pop('model.word_embeddings.weight')}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_version", default=2.0)
    parser.add_argument(
        "--path_to_checkpoint",
        type=str,
        help="Path to the checkpoint file (.zip archive or direct .pt file)",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="An optional config json file describing the pre-trained model.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="output path",
    )
    parser.add_argument(
        "--tp_degree",
        default=8,
        type=int,
        help="Tensor parallelism",
    )
    parser.add_argument(
        "--pp_degree",
        default=2,
        type=int,
        help="Pipeline parallelism",
    )
    parser.add_argument(
        "--vpp_degree",
        type=int,
        help="Virtual Pipeline parallelism. When set, will interleave layers through pipeline stages",
    )
    parser.add_argument(
        "--save_bf16",
        default=False,
        type=bool,
        help="Save weights in bf16.",
    )
    parser.add_argument(
        "--is_xser",
        action="store_true",
        help="Enable serialized saving",
    )
    parser.add_argument(
        "--step",
        type=int,
        help="step used for this checkpoint",
        required=True
    )

    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    PP = args.pp_degree
    f = partial(convert_checkpoint, args=args, config=config)

    # parallel processing
    with Pool(PP) as p:
        p.map(f, [i for i in range(PP)])

    # serial processing (use for debugging)
    # for p in range(PP):
    #     f(p)
