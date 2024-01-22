import argparse
import json
from pathlib import Path
import numpy as np
import torch
import os
import glob
from transformers import AutoModelForCausalLM


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


def convert_checkpoint(args):

    with open(args.config_file, "r") as f:
        config = json.load(f)
    print(config)
    br_key = "h." # Used to filter all transformer layers except layernorm

    translation = {
        "model.language_model.embedding.word_embeddings.weight": (1, "transformer.wte.weight", 0, 0), # a['model']['language_model']['word_embeddings']['weight']
        "input_layernorm.weight": (0, "ln_1.weight", None, 0),
        "self_attention.query_key_value.weight": (1, "attn.c_attn.weight", 0, 0),
        "self_attention.query_key_value.bias": (1, "attn.c_attn.bias", 0, 0),
        "self_attention.dense.weight": (1, "attn.c_proj.weight", 1, 0),
        "post_attention_layernorm.weight": (0, "ln_2.weight", None, 0),
        "self_attention.core_attention.rotary_emb.inv_freq": (0, "rotary_emb.inv_freq", None, 0),
        "mlp.dense_h_to_4h.weight": (1, "mlp.w2.weight", 0, 0),
        "mlp.dense_h_to_4h_2.weight": (1, "mlp.w1.weight", 0, 0),
        "mlp.dense_4h_to_h.weight": (1, "mlp.c_proj.weight", 1, 0),
        "model.language_model.encoder.final_layernorm.weight": (0, "transformer.ln_f.weight", None, 0),
        "model.language_model.output_layer.weight": (1, "lm_head.weight", 0, 0),  # this is shared
    }

    reverse_translation = {}
    for k, v in translation.items():
        split, br_k, dim, transpose = v
        reverse_translation[br_k] = (split, k, dim, transpose)

    TP = args.tp_degree
    PP = args.pp_degree

    hf_model = AutoModelForCausalLM.from_pretrained(args.path_to_checkpoint, trust_remote_code=True)
    # hf_model.resize_token_embeddings(pad_to_multiple_of=128)
    model_bedrock = hf_model.state_dict()

    for i in range(config["num_hidden_layers"]):
        model_bedrock[f"transformer.h.{i}.rotary_emb.inv_freq"] = hf_model.transformer.rotary_emb.inv_freq

    print(list(model_bedrock.keys()))

    print("Loaded QWen model")
    

    for p in range(PP):
        for i in range(TP):
            print(f"=== PP {p}, TP {i} ===")
            nemo_model = {}
            for k, v in model_bedrock.items():
                # print(f">>> {k}")
                if "attention.masked_bias" in k:
                    # We don't want to copy attention mask bias, since its a constant of 1e4
                    continue
                if br_key in k:
                    parts = k.split(br_key)[1].split(".")
                    layer_number = parts[0]
                    if int(layer_number) >= (config["num_hidden_layers"]//PP)*(p+1) or int(layer_number) < (config["num_hidden_layers"]//PP)*p:
                        continue
                    k = ".".join(parts[1:])
                    split, key, dim, tranpose = reverse_translation[k]
                    layer_number = layer_number if PP == 1 else str(int(layer_number) % (config["num_hidden_layers"]//PP))
                    key = "model.language_model.encoder.layers." + layer_number + "." + key
                    nemo_model[key] = v
                    if tranpose:
                        nemo_model[key]= torch.transpose(
                            nemo_model[key], 0, 1
                        )

                    if "query_key_value" in (key):
                        heads = config["num_attention_heads"]
                        hidden_size = config["hidden_size"]
                        hidden_size_per_head = config["hidden_size"] // heads

                        def permute_rotary(w):
                            assert w.shape == (heads, hidden_size_per_head, hidden_size*3)
                            return (
                                w.view(heads, hidden_size_per_head // 2, 2, hidden_size*3)
                                .transpose(1, 2)
                                .reshape(heads, hidden_size_per_head, hidden_size*3)
                            )
                        
                        def permute(w, n_heads=heads, dim1=hidden_size, dim2=hidden_size*3):
                            return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

                        if "weight" in key:
                            nemo_model[key] = permute_rotary(
                                permute(nemo_model[key]).view(
                                    heads, hidden_size_per_head, hidden_size*3
                                )
                            )
                            nemo_model[key] = nemo_model[key].view(
                                3,
                                heads,
                                hidden_size_per_head,
                                hidden_size,
                            ).transpose(0, 1).contiguous().view(
                                heads * 3 * hidden_size_per_head,
                                hidden_size,
                            )
                            nemo_model[key] = nemo_model[key].view(
                                TP,
                                heads * 3 * hidden_size_per_head // TP,
                                hidden_size,
                            )
                        elif "bias" in key:
                            nemo_model[key] = nemo_model[key].view(
                                3,
                                heads,
                                hidden_size_per_head,
                            ).transpose(0, 1).contiguous().view(
                                heads * 3 * hidden_size_per_head
                            )
                            nemo_model[key] = nemo_model[key].view(
                                TP,
                                heads * 3 * hidden_size_per_head // TP,
                            )

                    if split:
                        if "query_key_value" in (key):
                            nemo_model[key] = nemo_model[key][i]
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
                    if "wte" in k and p==0:
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
                            i * tp_last_dim_size : (i + 1) * tp_last_dim_size, ...
                        ].clone()
                        print(key, split, nemo_model[key].shape, v.shape)
                    elif "transformer.ln_f" in k and p == (PP-1):
                        nemo_model[key] = v
                        print(key, split, nemo_model[key].shape, v.shape)
                    elif "lm_head" in k and p == (PP-1): 
                        # Padding to make it divisble by TP degree
                        if v.shape[0] % TP > 0:
                            x = torch.nn.functional.pad(
                                v, (0, 0, 0, (TP - v.shape[0] % TP)) 
                            )
                        else:
                            x = v
                        if split:
                            tp_last_dim_size = x.shape[dim]//TP
                            if dim:
                                nemo_model[key] = x[..., i*tp_last_dim_size:(i+1)*tp_last_dim_size].clone()
                            else:
                                nemo_model[key] = x[i*tp_last_dim_size:(i+1)*tp_last_dim_size, ...].clone()
                        print(key, split, nemo_model[key].shape, v.shape)
            
            if args.save_bf16:
                for _k in nemo_model:
                    nemo_model[_k] = nemo_model[_k].to(dtype=torch.bfloat16, device='cpu')
            out_model = {"state_dict": nemo_model}

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
        default="",
        type=str,
        help="An optional config json file describing the pre-trained model.",
    )
    parser.add_argument(
        "--output_path",
        default="",
        type=str,
        help="output path",
    )
    parser.add_argument(
        "--tp_degree",
        default=1,
        type=int,
        help="Tensor parallelism",
    )
    parser.add_argument(
        "--pp_degree",
        default=1,
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
