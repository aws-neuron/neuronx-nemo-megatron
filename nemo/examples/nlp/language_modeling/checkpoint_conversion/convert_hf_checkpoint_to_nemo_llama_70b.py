import argparse
import json
from pathlib import Path
import numpy as np
import torch
import os
import logging
import itertools
import glob
from multiprocessing import Pool
from functools import partial

from examples.nlp.language_modeling.checkpoint_conversion.convert_hf_checkpoint_to_nemo_llama import get_layer_map, convert_checkpoint_to_vpp_format, load_model_checkpoints
from examples.nlp.language_modeling.checkpoint_conversion.logger_factory import LoggerFactory

logger = LoggerFactory.create_logger(name="hf_to_nemo_llama_70b", level=logging.INFO)


def get_pipeline_bin_division(model_bin_file, PP, layers_on_curr_pp):
    with open(model_bin_file, "r") as json_data:
        model_info = json.load(json_data)
        json_data.close()

    pp_to_bin = {str(p): [] for p in range(PP)}
    for p in range(PP):
        for key, value in model_info["weight_map"].items():
            if "layers." in key:
                layer_number = int(key.split(".")[2])
                if int(layer_number) in layers_on_curr_pp:
                    shard_number = int(value.split("-")[1])
                    if shard_number not in pp_to_bin[str(p)]:
                        pp_to_bin[str(p)].append(shard_number)

            elif "embed_tokens" in key and p == 0:
                shard_number = int(value.split("-")[1])
                if shard_number not in pp_to_bin[str(p)]:
                    pp_to_bin[str(p)].append(shard_number)

            elif "model.norm" in key and p == (PP - 1):
                shard_number = int(value.split("-")[1])
                if shard_number not in pp_to_bin[str(p)]:
                    pp_to_bin[str(p)].append(shard_number)

            elif "lm_head" in key and p == (PP - 1):
                shard_number = int(value.split("-")[1])
                if shard_number not in pp_to_bin[str(p)]:
                    pp_to_bin[str(p)].append(shard_number)

    return pp_to_bin


def hf2mt_fix_query_key_value_ordering(
        param, checkpoint_version, num_splits, num_heads, hidden_size
):
    # Permutes layout of param tensor to [num_splits * num_heads * hidden_size, :]
    # for compatibility with later versions of NVIDIA Megatron-LM.
    # The inverse operation is performed inside Megatron-LM to read checkpoints:
    # https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209
    # If param is the weight tensor of the self-attention block, the returned tensor
    # will have to be transposed one more time to be read by HuggingFace GPT2.

    # The function will interleave the K and V in each head. i.e., convert,
    # [k_h1; k_h2, ... k_h8; v_h1; v_h2; ... v_h8] into
    # [k_h1; v_h1; k_h2; v_h2; ... k_h8; v_h8]
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

    out_models = convert_state_dict_for_pp_rank(
        p,
        args.path_to_checkpoint,
        args.model_bin_file,
        args.num_shards, config,
        TP,
        PP,
        VPP,
        args.save_bf16,
        args.gqa_qkv)

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



def convert_state_dict_for_pp_rank(
        p,
        path_to_checkpoint,
        model_bin_file,
        num_shards,
        config,
        TP,
        PP,
        VPP,
        save_bf16,
        gqa_qkv):
    logger.info(f"------- start converting parameters for pipeline pp={p}")

    br_key = "layers."  # Used to filter all transformer layers except layernorm

    translation = {
        "model.language_model.embedding.word_embeddings.weight": (1, "model.embed_tokens.weight", 0, 0),
        # a['model']['language_model']['word_embeddings']['weight']
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
        "model.language_model.encoder.final_layernorm.weight": (0, "model.norm.weight", None, 0),
        "model.word_embeddings.weight": (1, "lm_head.weight", 0, 0), # final layer word embedding
    }

    reverse_translation = {}
    for k, v in translation.items():
        split, br_k, dim, transpose = v
        reverse_translation[br_k] = (split, k, dim, transpose)

    n_layers_per_pp = config['num_hidden_layers'] // PP
    if VPP is None:
        layers_on_curr_pp = list(range(p * n_layers_per_pp, (p + 1) * n_layers_per_pp))
    else:
        layer_map = get_layer_map(PP, VPP, config['num_hidden_layers'])
        layers_on_curr_pp = list(itertools.chain(*layer_map[p]))

    # this is for sharded huggingface checkpoints
    if model_bin_file:
        pp_to_bin = get_pipeline_bin_division(model_bin_file, PP, layers_on_curr_pp)
        model_paths = [f"{path_to_checkpoint}/pytorch_model-{x:05d}-of-{num_shards:05d}.bin" for x in
                       pp_to_bin[str(p)]]
        logger.info(f"model bins for pp {p}: {model_paths}")
    else:
        model_paths = sorted(glob.glob(f'{path_to_checkpoint}/pytorch_model*.bin'))

    # model_llama = {}
    # for _path in model_paths:
    #     logger.info(f'Loading {_path}')
    #     ts = torch.load(_path, map_location='cpu')
    #     model_llama.update(ts)
    # logger.info(len(model_llama))

    model_llama = load_model_checkpoints(model_paths)
    logger.info("Loaded Llama model")

    gqa = 'num_query_groups' in config or 'num_key_value_heads' in config
    old_gqa = gqa and not gqa_qkv


    # Merge QKV for GQA models
    if old_gqa:
        # Merge QKV for (old) trainium GQA implementation
        for i in layers_on_curr_pp:
            q = model_llama[f'model.layers.{i}.self_attn.q_proj.weight']
            k = model_llama[f'model.layers.{i}.self_attn.k_proj.weight']
            v = model_llama[f'model.layers.{i}.self_attn.v_proj.weight']
            model_llama[f'model.layers.{i}.self_attn.query.weight'] = q
            model_llama[f'model.layers.{i}.self_attn.key_value.weight'] = torch.cat([k, v], dim=0)

            model_llama.pop(f'model.layers.{i}.self_attn.q_proj.weight')
            model_llama.pop(f'model.layers.{i}.self_attn.k_proj.weight')
            model_llama.pop(f'model.layers.{i}.self_attn.v_proj.weight')
    if gqa and not old_gqa:
        # new GQA
        for i in layers_on_curr_pp:
            q = model_llama[f'model.layers.{i}.self_attn.q_proj.weight']
            k = model_llama[f'model.layers.{i}.self_attn.k_proj.weight']
            v = model_llama[f'model.layers.{i}.self_attn.v_proj.weight']
            num_query_groups = config.get("num_query_groups", config.get("num_key_value_heads"))
            qkv = []
            for qc, kc, vc in zip(q.chunk(num_query_groups), k.chunk(num_query_groups), v.chunk(num_query_groups)):
                qkv.append(torch.cat([qc, kc, vc], dim=0))
            qkv = torch.cat(qkv, dim=0)
            model_llama[f'model.layers.{i}.self_attn.query_key_value.weight'] = qkv
            model_llama.pop(f'model.layers.{i}.self_attn.q_proj.weight')
            model_llama.pop(f'model.layers.{i}.self_attn.k_proj.weight')
            model_llama.pop(f'model.layers.{i}.self_attn.v_proj.weight')

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
                    split, key, dim, transpose = reverse_translation[k]
                    layer_number = str(layers_on_curr_pp.index(int(layer_number)))
                    logger.info(f'layer_number in nemo: {layer_number}')
                    key = "model.language_model.encoder.layers." + layer_number + "." + key
                    nemo_model[key] = v
                    if transpose:
                        nemo_model[key] = torch.transpose(
                            nemo_model[key], 0, 1
                        )

                    if "query" in (key) and "query_key_value" not in key: # old GQA
                        heads = config["num_attention_heads"]
                        hidden_size_per_head = config["hidden_size"] // heads
                        nemo_model[key] = hf2mt_fix_query_key_value_ordering(
                            nemo_model[key], 2.0, 1, heads, hidden_size_per_head
                        )
                    if "key_value" in (key) and "query_key_value" not in key: # old GQA
                        heads = config["num_attention_heads"]
                        hidden_size_per_head = config["hidden_size"] // heads
                        heads = config['num_key_value_heads']
                        nemo_model[key] = hf2mt_fix_query_key_value_ordering(
                            nemo_model[key], 2.0, 2, heads, hidden_size_per_head
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
                            v, (0, 0, 0, (TP - v.shape[0] % TP))
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
                elif "lm_head" in k and p == (PP - 1):
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

        # TODO: fix the hardcoded keys.
        out_model = {"state_dict": nemo_model, 'pytorch-lightning_version': '1.9.5', 'epoch': 0, 'global_step': 0}

        # merge dense_h_to_4h and dense_h_to_4h_2 for Trn checkpoint
        h_to_4h_keys = [k for k in out_model['state_dict'].keys() if 'dense_h_to_4h' in k and 'dense_h_to_4h_2' not in k]
        for k in h_to_4h_keys:
            k_1 = k.replace('dense_h_to_4h', 'dense_h_to_4h_2')
            logger.info(f"merging {k} ({out_model['state_dict'][k].shape}) and {k_1} ({out_model['state_dict'][k_1].shape})")
            out_model['state_dict'][k] = torch.concat([out_model['state_dict'][k], out_model['state_dict'][k_1]], dim=0)
            logger.info(f"merged pojections is ({out_model['state_dict'][k].shape})")
            out_model['state_dict'].pop(k_1)

        if VPP is not None:
            convert_checkpoint_to_vpp_format(out_model, p, PP, VPP)
        out_models[i] = out_model
    del model_llama
    return out_models



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_version", default=2.0)
    parser.add_argument(
        "--path_to_checkpoint",
        type=str,
        help="Path to the checkpoint folder",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="Config json file describing the pre-trained model.",
    )
    parser.add_argument(
        "--model_bin_file",
        type=str,
        help="The pytorch_model.bin.index.json file describing the mapping of layers to .bin shards.",
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
        default=8,
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
        "--num_shards",
        default=29,
        type=int,
        help="Number of shards in the saved HF checkpoint",
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
    parser.add_argument(
        "--gqa_qkv",
        action="store_true",
        help="If this is for new GQA implementation, K, Q and V will be saved as one tensor",
    )

    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)
    args.num_hidden_layers = config["num_hidden_layers"]
    PP = args.pp_degree
    f = partial(convert_checkpoint, args=args, config=config)

    # parallel processing
    # with Pool(PP) as p:
    #     p.map(f, [i for i in range(PP)])

    # serial processing (use for debugging)
    for p in range(PP):
        f(p)
