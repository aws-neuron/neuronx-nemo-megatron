import argparse
import json
from pathlib import Path
import numpy as np
import torch
import os
import glob
from transformers import AutoTokenizer


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


def pad_to_vocab_size_and_tp(param, tokenizer_vocab_size, tp, make_vocab_size_divisible_by):
    # pad vocab size to match that of tokenizer
    if tokenizer_vocab_size > param.shape[0]:
        v_padded = torch.nn.functional.pad(
            param, (0, 0, 0, (tokenizer_vocab_size - param.shape[0]))
        )
        print(
            f"Padding vocab weights to {tokenizer_vocab_size} (so it matches the tokenizer vocab size). Added tokens {(tokenizer_vocab_size - param.shape[0])}")
        print(f"Shape before {param.shape} Padded shape {v_padded.shape}")
    else:
        v_padded = param

    # Padding to make it divisble by make_vocab_size_divisible_by and TP degree
    multiple = make_vocab_size_divisible_by * tp
    if v_padded.shape[0] % multiple > 0:
        x = torch.nn.functional.pad(
            v_padded, (0, 0, 0, (multiple - v_padded.shape[0] % multiple))
        )
        print(
            f'Padding vocab to make it divisible by args.make_vocab_size_divisible_by * TP. Padded size {x.shape[0]}')
    else:
        x = param

    return x


def convert_checkpoint(p):
    tokenizer = AutoTokenizer.from_pretrained(args.path_to_tokenizer)
    tokenizer_vocab_size = len(tokenizer)

    with open(args.config_file, "r") as f:
        config = json.load(f)
    print(config)

    br_key = "h."  # Used to filter all transformer layers except layernorm

    translation = {
        "model.language_model.embedding.word_embeddings.weight": (1, "transformer.wte.weight", 0, 0),
        "model.language_model.embedding.position_embeddings.weight": (0, "transformer.wpe.weight", None, 0),
        "input_layernorm.weight": (0, "ln_1.weight", None, 0),
        "input_layernorm.bias": (0, "ln_1.bias", None, 0),
        "self_attention.dense.weight": (1, "attn.c_proj.weight", 1, 0),
        "self_attention.dense.bias": (0, "attn.c_proj.bias", None, 0),
        "post_attention_layernorm.weight": (0, "ln_2.weight", None, 0),
        "post_attention_layernorm.bias": (0, "ln_2.bias", None, 0),
        "mlp.dense_h_to_4h.weight": (1, "mlp.c_fc.weight", 0, 0),
        "mlp.dense_h_to_4h.bias": (1, "mlp.c_fc.bias", 0, 0),
        "mlp.dense_4h_to_h.weight": (1, "mlp.c_proj.weight", 1, 0),
        "mlp.dense_4h_to_h.bias": (0, "mlp.c_proj.bias", None, 0),
        "self_attention.query.weight": (1, "attn.c_attn_q.weight", 0, 0),
        "self_attention.query.bias": (1, "attn.c_attn_q.bias", 0, 0),
        "self_attention.key_value.weight": (0, "attn.c_attn_kv.weight", 0, 0),
        "self_attention.key_value.bias": (0, "attn.c_attn_kv.bias", 0, 0),
        "model.language_model.encoder.final_layernorm.weight": (0, "transformer.ln_f.weight", None, 0),
        "model.language_model.encoder.final_layernorm.bias": (0, "transformer.ln_f.bias", None, 0),
        "model.language_model.output_layer.weight": (1, "lm_head.weight", 0, 0),
    }

    reverse_translation = {}
    for k, v in translation.items():
        split, br_k, dim, transpose = v
        reverse_translation[br_k] = (split, k, dim, transpose)

    TP = args.tp_degree
    PP = args.pp_degree
    model_paths = sorted(
        glob.glob(f'{args.path_to_checkpoint}/pytorch_model*.bin'))

    model_bigcode = {}
    for _path in model_paths:
        print(f'Loading {_path}')
        ts = torch.load(_path, map_location='cpu')
        model_bigcode.update(ts)
    print(len(model_bigcode))

    print("Loaded BigCode model")

    # Split Q and KV
    for i in range(config['n_layer']):
        q_weight = model_bigcode[f'transformer.h.{i}.attn.c_attn.weight'][:config['n_embd'], :]
        kv_weight = model_bigcode[f'transformer.h.{i}.attn.c_attn.weight'][config['n_embd']:, :]

        q_bias = model_bigcode[f'transformer.h.{i}.attn.c_attn.bias'][:config['n_embd']]
        kv_bias = model_bigcode[f'transformer.h.{i}.attn.c_attn.bias'][config['n_embd']:]

        model_bigcode[f'transformer.h.{i}.attn.c_attn_q.weight'] = q_weight.clone(
        )
        model_bigcode[f'transformer.h.{i}.attn.c_attn_kv.weight'] = kv_weight.clone(
        )
        model_bigcode[f'transformer.h.{i}.attn.c_attn_q.bias'] = q_bias.clone()
        model_bigcode[f'transformer.h.{i}.attn.c_attn_kv.bias'] = kv_bias.clone(
        )
        model_bigcode.pop(f'transformer.h.{i}.attn.c_attn.weight')
        model_bigcode.pop(f'transformer.h.{i}.attn.c_attn.bias')

    for p in range(PP):
        for i in range(TP):
            print(f"=== PP {p}, TP {i} ===")
            nemo_model = {}
            for k, v in model_bigcode.items():
                print(f">>> {k}")
                if br_key in k:
                    parts = k.split(br_key)[1].split(".")
                    layer_number = parts[0]
                    if int(layer_number) >= (config["n_layer"] // PP) * (p + 1) or int(layer_number) < (
                            config["n_layer"] // PP) * p:
                        continue
                    k = ".".join(parts[1:])
                    split, key, dim, tranpose = reverse_translation[k]
                    layer_number = layer_number if PP == 1 else str(
                        int(layer_number) % (config["n_layer"] // PP))
                    key = "model.language_model.encoder.layers." + layer_number + "." + key
                    nemo_model[key] = v
                    print(nemo_model[key].shape)
                    if tranpose:
                        nemo_model[key] = torch.transpose(
                            nemo_model[key], 0, 1
                        )

                    if "query" in key:
                        heads = config["n_head"]
                        hidden_size_per_head = config["n_embd"] // heads
                        nemo_model[key] = fix_query_key_value_ordering(
                            nemo_model[key], 2.0, 1, heads, hidden_size_per_head
                        )
                    elif "key_value" in key:
                        heads = config["n_head"]
                        hidden_size_per_head = config["n_embd"] // heads
                        nemo_model[key] = fix_query_key_value_ordering(
                            nemo_model[key], 2.0, 2, 1, hidden_size_per_head
                        )

                    if split:
                        print(key, nemo_model[key].shape, dim)
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
                    if "wte" in k and p == 0:
                        assert tokenizer_vocab_size >= v.shape[0]
                        x = pad_to_vocab_size_and_tp(
                            v, tokenizer_vocab_size, TP, args.make_vocab_size_divisible_by)
                        last_dim_size = x.shape[0]
                        tp_last_dim_size = last_dim_size // TP
                        nemo_model[key] = x[
                            i * tp_last_dim_size: (i + 1) * tp_last_dim_size, ...
                        ].clone()
                        print(key, split,
                              nemo_model[key].shape, v.shape, x.shape)
                    elif "wpe" in k and p == 0:
                        nemo_model[key] = v.clone()
                        print(key, split, nemo_model[key].shape, v.shape)
                    elif "ln_f" in k and p == (PP - 1):
                        nemo_model[key] = v
                        print(key, split, nemo_model[key].shape, v.shape)
                    elif "lm_head" in k and p == (PP - 1):  # Not used
                        x = pad_to_vocab_size_and_tp(
                            v, tokenizer_vocab_size, TP, args.make_vocab_size_divisible_by)
                        if split:
                            tp_last_dim_size = x.shape[dim] // TP
                            if dim:
                                nemo_model[key] = x[..., i *
                                                    tp_last_dim_size:(i + 1) * tp_last_dim_size].clone()
                            else:
                                nemo_model[key] = x[i * tp_last_dim_size:(
                                    i + 1) * tp_last_dim_size, ...].clone()
                        print(key, split, nemo_model[key].shape, v.shape)

            out_model = {"state_dict": nemo_model,
                         'pytorch-lightning_version': '1.9.5', 'epoch': 0, 'global_step': 0}

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
        "--path_to_tokenizer",
        type=str,
        help="Path to the tokenizer",
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
        "--make_vocab_size_divisible_by",
        default=8,
        type=int,
        help="Make vocab size divisible by (this has to match the config used in your model)",
    )

    args = parser.parse_args()
    convert_checkpoint(args)
