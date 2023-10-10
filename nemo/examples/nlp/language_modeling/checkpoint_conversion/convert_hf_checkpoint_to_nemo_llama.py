import argparse
import json
from pathlib import Path
import numpy as np
import torch
import os
import glob


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


def convert_checkpoint(p):
    with open(args.config_file, "r") as f:
        config = json.load(f)
    print(config)

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

    TP = args.tp_degree
    PP = args.pp_degree
    model_paths = sorted(glob.glob(f'{args.path_to_checkpoint}/pytorch_model*.bin'))
    model_llama = {}
    for _path in model_paths:
        print(f'Loading {_path}')
        ts = torch.load(_path, map_location='cpu')
        model_llama.update(ts)
    print(len(model_llama))

    print("Loaded Llama model")

    # Merge QKV
    for i in range(config['num_hidden_layers']):
        q = model_llama[f'model.layers.{i}.self_attn.q_proj.weight']
        k = model_llama[f'model.layers.{i}.self_attn.k_proj.weight']
        v = model_llama[f'model.layers.{i}.self_attn.v_proj.weight']
        model_llama[f'model.layers.{i}.self_attn.query_key_value.weight'] = torch.cat([q, k, v], dim=0)

        model_llama.pop(f'model.layers.{i}.self_attn.q_proj.weight')
        model_llama.pop(f'model.layers.{i}.self_attn.k_proj.weight')
        model_llama.pop(f'model.layers.{i}.self_attn.v_proj.weight')

    for p in range(PP):
        for i in range(TP):
            print(f"=== PP {p}, TP {i} ===")
            nemo_model = {}
            for k, v in model_llama.items():
                # print(f">>> {k}")
                if "attention.masked_bias" in k:
                    # We don't want to copy attention mask bias, since its a constant of 1e4
                    continue
                if br_key in k:
                    parts = k.split(br_key)[1].split(".")
                    layer_number = parts[0]
                    if int(layer_number) >= (config["num_hidden_layers"] // PP) * (p + 1) or int(layer_number) < (
                            config["num_hidden_layers"] // PP) * p:
                        continue
                    k = ".".join(parts[1:])
                    if k == "attn.bias":
                        continue
                    split, key, dim, tranpose = reverse_translation[k]
                    layer_number = layer_number if PP == 1 else str(
                        int(layer_number) % (config["num_hidden_layers"] // PP))
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

                    print(key, split, nemo_model[key].shape, v.shape)
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
                        print(key, split, nemo_model[key].shape, v.shape)
                    elif "model.norm" in k and p == (PP - 1):
                        nemo_model[key] = v
                        print(key, split, nemo_model[key].shape, v.shape)
                    elif "lm_head" in k and p == (PP - 1):  # Not used
                        if split:
                            tp_last_dim_size = v.shape[dim] // TP
                            if dim:
                                nemo_model[key] = v[..., i * tp_last_dim_size:(i + 1) * tp_last_dim_size].clone()
                            else:
                                nemo_model[key] = v[i * tp_last_dim_size:(i + 1) * tp_last_dim_size, ...].clone()
                        print(key, split, nemo_model[key].shape, v.shape)

            if args.save_bf16:
                for _k in nemo_model:
                    nemo_model[_k] = nemo_model[_k].to(dtype=torch.bfloat16, device='cpu')
            out_model = {"state_dict": nemo_model, 'pytorch-lightning_version': '1.9.5', 'epoch': 0, 'global_step': 0}

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
            torch.save(out_model,
                       f"{output_folder}/model_optim_rng.ckpt")  # , (not master_only), global_master=True)
            print("Done saving Megatron checkpoint")


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
        "--save_bf16",
        default=False,
        type=bool,
        help="Save weights in bf16.",
    )

    args = parser.parse_args()
    convert_checkpoint(args)
