import argparse
import json
from pathlib import Path
import numpy as np
import torch
import os
import glob
from multiprocessing import Pool
from functools import partial

def get_pipeline_bin_division(args):
    num_hidden_layers = args.num_hidden_layers
    PP = args.pp_degree

    with open(args.model_bin_file, "r") as json_data:
        model_info = json.load(json_data)
        json_data.close()
    
    pp_to_bin = {str(p):[] for p in range(PP)}
    for p in range(PP):
        for key, value in model_info["weight_map"].items():
            if "layers." in key:
                layer_number = int(key.split(".")[2])
                if (int(layer_number) < (num_hidden_layers//PP)*(p+1)) and (int(layer_number) >= (num_hidden_layers//PP)*p):
                    shard_number = int(value.split("-")[1])
                    if shard_number not in pp_to_bin[str(p)]:
                        pp_to_bin[str(p)].append(shard_number)

            elif "embed_in" in key and p==0:
                shard_number = int(value.split("-")[1])
                if shard_number not in pp_to_bin[str(p)]:
                        pp_to_bin[str(p)].append(shard_number)

            elif "final_layer_norm" in key and p == (PP-1):
                shard_number = int(value.split("-")[1])
                if shard_number not in pp_to_bin[str(p)]:
                    pp_to_bin[str(p)].append(shard_number)
                    
            elif "embed_out" in key and p == (PP-1):
                shard_number = int(value.split("-")[1])
                if shard_number not in pp_to_bin[str(p)]:
                    pp_to_bin[str(p)].append(shard_number)
    
    return pp_to_bin

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

def convert_checkpoint(p, args):
    with open(args.config_file, "r") as f:
        config = json.load(f)
        
    print(config)
    br_key = "layers." # Used to filter all transformer layers except layernorm
    use_bias = args.bias
    
    translation = {
        "model.language_model.embedding.word_embeddings.weight": (1, "gpt_neox.embed_in.weight", 0, 0), # a['model']['language_model']['word_embeddings']['weight']
        "self_attention.core_attention.rotary_emb.inv_freq": (0, "attention.rotary_emb.inv_freq", None, 0),
        "input_layernorm.weight": (0, "input_layernorm.weight", None, 0),
        "self_attention.query_key_value.weight": (1, "attention.query_key_value.weight", 0, 0),
        "self_attention.dense.weight": (1, "attention.dense.weight", 1, 0),
        "post_attention_layernorm.weight": (0, "post_attention_layernorm.weight", None, 0),
        "mlp.dense_h_to_4h.weight": (1, "mlp.dense_h_to_4h.weight", 0, 0),
        "mlp.dense_4h_to_h.weight": (1, "mlp.dense_4h_to_h.weight", 1, 0),
        "model.language_model.encoder.final_layernorm.weight": (0, "gpt_neox.final_layer_norm.weight", None, 0),
        "model.language_model.output_layer.weight": (1, "embed_out.weight", 0, 0),  # this is shared
    }
    if use_bias: # bias not split across TP ranks
        translation.update({
        "input_layernorm.bias": (0, "input_layernorm.bias", None, 0),
        "self_attention.query_key_value.bias": (1, "attention.query_key_value.bias", 0, 0),
        "self_attention.dense.bias": (0, "attention.dense.bias", None, 0),
        "post_attention_layernorm.bias": (0, "post_attention_layernorm.bias", None, 0),
        "mlp.dense_h_to_4h.bias": (1, "mlp.dense_h_to_4h.bias", 0, 0),
        "mlp.dense_4h_to_h.bias": (0, "mlp.dense_4h_to_h.bias", None, 0),
        "model.language_model.encoder.final_layernorm.bias": (0, "gpt_neox.final_layer_norm.bias", None, 0),
        }
        )
    
    reverse_translation = {}
    for k, v in translation.items():
        split, br_k, dim, transpose = v
        reverse_translation[br_k] = (split, k, dim, transpose)
    print(reverse_translation)


    TP = args.tp_degree
    PP = args.pp_degree
    model_paths = [f"{args.path_to_checkpoint}/pytorch_model-{x:05d}-of-{args.num_shards:05d}.bin" for x in args.pp_to_bin[str(p)]]
    model_neox = {}
    for _path in model_paths:
        print(f'Loading {_path}')
        ts = torch.load(_path, map_location="cpu")
        model_neox.update(ts)
    print(len(model_neox))
    
    print("Loaded GPT model")

    for i in range(TP):

        print(f"=== PP {p}, TP {i} ===")
        nemo_model = {}
        for k, v in model_neox.items():
            # print(f">>> {k}")
            if "attention.masked_bias" in k:
                # We don't want to copy attention mask bias, since its a constant of 1e4
                continue
            if "attention.bias" in k:
                # We don't want to copy attention mask bias, since its a deterministic lower tril matrix
                continue
            if br_key in k:
                parts = k.split(br_key)[1].split(".")
                layer_number = parts[0]
                if int(layer_number) >= (config["num_hidden_layers"]//PP)*(p+1) or int(layer_number) < (config["num_hidden_layers"]//PP)*p:
                    continue
                k = ".".join(parts[1:])
                if k == "attn.bias":
                    continue

                # print(k, parts)
                split, key, dim, tranpose = reverse_translation[k]
                layer_number = layer_number if PP == 1 else str(int(layer_number) % (config["num_hidden_layers"]//PP))
                key = "model.language_model.encoder.layers." + layer_number + "." + key
                nemo_model[key] = v
                if transpose:
                    nemo_model[key]= torch.transpose(
                        nemo_model[key], 0, 1
                    )

                if "query_key_value" in (key):
                     heads = config["num_attention_heads"]
                     hidden_size_per_head = config["hidden_size"] // heads
                     nemo_model[key] = fix_query_key_value_ordering(
                         nemo_model[key], 2.0, 3, heads, hidden_size_per_head
                     )
                if split:
                    if "bias" in k:
                        tp_last_dim_size = nemo_model[key].shape[dim] // TP
                        nemo_model[key] = nemo_model[key][
                                i * tp_last_dim_size : (i + 1) * tp_last_dim_size, ...
                            ].clone()
                    else:
                        tp_last_dim_size = nemo_model[key].shape[dim] // TP
                        if dim: # First or last dimension to shard
                            nemo_model[key] = nemo_model[key][
                                ..., i * tp_last_dim_size : (i + 1) * tp_last_dim_size
                            ].clone()
                        else:
                            nemo_model[key] = nemo_model[key][
                                i * tp_last_dim_size : (i + 1) * tp_last_dim_size, ...
                            ].clone()

                print(key, split, nemo_model[key].shape, v.shape)
            else:
                split, key, dim, transpose = reverse_translation[k]
                if transpose:
                    nemo_model[key]= torch.transpose(
                        nemo_model[key], 0, 1
                    )
                if "embed_in" in k and p==0:
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
                        i * tp_last_dim_size : (i + 1) * tp_last_dim_size, ...
                    ].clone()
                    print(key, split, nemo_model[key].shape, v.shape)
                elif "final_layer_norm" in k and p == (PP-1):
                    nemo_model[key] = v
                    print(key, split, nemo_model[key].shape, v.shape)
                elif "embed_out" in k and p == (PP-1): # Not used
                    if split:
                        tp_last_dim_size = v.shape[dim]//TP
                        if dim:
                            nemo_model[key] = v[..., i*tp_last_dim_size:(i+1)*tp_last_dim_size].clone()
                        else:
                            nemo_model[key] = v[i*tp_last_dim_size:(i+1)*tp_last_dim_size, ...].clone()
                    print(key, split, nemo_model[key].shape, v.shape)

        
        if args.save_bf16:
            for _k in nemo_model:
                nemo_model[_k] = nemo_model[_k].to(dtype=torch.bfloat16, device='cpu')
        out_model = {"state_dict": nemo_model, 'pytorch-lightning_version': '1.9.5', 'epoch': 0, 'global_step': 0}

        output_folder = args.output_path
        if TP > 1:
            if PP>1:
                output_folder = output_folder + f"/tp_rank_{i:02d}"
            else:
                output_folder = output_folder + f"/mp_rank_{i:02d}"
        if PP > 1:
            output_folder = output_folder + f"_pp_rank_{p:03d}"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        torch.save(out_model, f"{output_folder}/model_optim_rng.ckpt") #, (not master_only), global_master=True)        
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
        "--model_bin_file",
        type=str,
        help="An optional config json file describing the pre-trained model. ext and pp rank added in code",
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
        default=4,
        type=int,
        help="Pipeline parallelism",
    )
    parser.add_argument(
        "--num_hidden_layers",
        default=44,
        type=int,
        help="Number of hidden layers in the GPT model",
    )
    parser.add_argument(
        "--num_shards",
        default=46,
        type=int,
        help="Number of shards in the save checkpoint",
    )
    parser.add_argument(
        "--save_bf16",
        default=False,
        type=bool,
        help="Save weights in bf16.",
    )
    parser.add_argument(
        "--bias",
        default=True,
        type=bool,
        help="To use bias in the model layers",
    )
    
    args = parser.parse_args()
    pp_to_bin = get_pipeline_bin_division(args)
    PP = args.pp_degree
    args.pp_to_bin = pp_to_bin
    f = partial(convert_checkpoint, args=args)
    with Pool(PP) as p:
        p.map(f, [i for i in range(PP)])