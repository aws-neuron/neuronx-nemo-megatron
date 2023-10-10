import argparse
import json
from pathlib import Path
import numpy as np
import torch


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

    nemo_key = "model.language_model."
    br_key = "h."

    translation = {
        "embedding.word_embeddings.weight": (1, "transformer.wte.weight", 0, 0),
        "embedding.position_embeddings.weight": (0, "transformer.wpe.weight", None, 0),
        "input_layernorm.weight": (0, "ln_1.weight", None, 0),
        "input_layernorm.bias": (0, "ln_1.bias", None, 0),
        "self_attention.query_key_value.weight": (1, "attn.c_attn.weight", 0, 1),
        "self_attention.query_key_value.bias": (1, "attn.c_attn.bias", 0, 0),
        "self_attention.dense.weight": (1, "attn.c_proj.weight", 1, 1),
        "self_attention.dense.bias": (0, "attn.c_proj.bias", None, 0),
        "post_attention_layernorm.weight": (0, "ln_2.weight", None, 0),
        "post_attention_layernorm.bias": (0, "ln_2.bias", None, 0),
        "mlp.dense_h_to_4h.weight": (1, "mlp.c_fc.weight", 0, 1),
        "mlp.dense_h_to_4h.bias": (1, "mlp.c_fc.bias", 0, 0),
        "mlp.dense_4h_to_h.weight": (1, "mlp.c_proj.weight", 1, 1),
        "mlp.dense_4h_to_h.bias": (0, "mlp.c_proj.bias", None, 0),
        "final_layernorm.weight": (0, "transformer.ln_f.weight", None, 0),
        "final_layernorm.bias": (0, "transformer.ln_f.bias", None, 0),
        "output_layer.weight": (1, "lm_head.weight", 0, 0),
    }

    reverse_translation = {}
    for k, v in translation.items():
        split, br_k, dim, transpose = v
        reverse_translation[br_k] = (split, k, dim, transpose)

    nemo_models = []

    TP = args.tp_degree
    PP = args.pp_degree
    model_GPT2 = torch.load(args.path_to_checkpoint)
    print("Loaded GPT2 model")

    for p in range(PP):
        for i in range(TP):
            nemo_model = {}
            for k, v in model_GPT2.items():
                if "attn.masked_bias" in k:
                    # We don't want to copy attention mask bias, since its a constant of 1e4
                    continue
                if br_key in k:
                    parts = k.split(br_key)[1].split(".")
                    layer_number = parts[0]
                    if int(layer_number) >= (config["n_layer"]//PP)*(p+1) or int(layer_number) < (config["n_layer"]//PP)*p:
                        continue
                    k = ".".join(parts[1:])
                    if k == "attn.bias":
                        continue
                    split, key, dim, tranpose = reverse_translation[k]
                    layer_number = layer_number if PP == 1 else str(int(layer_number) % (config["n_layer"]//PP))
                    key = "encoder.layers." + layer_number + "." + key
                    nemo_model[nemo_key + key] = v
                    if tranpose:
                        nemo_model[nemo_key + key] = torch.transpose(
                            nemo_model[nemo_key + key], 0, 1
                        )

                    if "query_key_value" in (nemo_key + key):
                        heads = config["n_head"]
                        hidden_size_per_head = config["n_embd"] // heads
                        nemo_model[nemo_key + key] = fix_query_key_value_ordering(
                            nemo_model[nemo_key + key], 2.0, 3, heads, hidden_size_per_head
                        )
                    if split:
                        tp_last_dim_size = nemo_model[nemo_key + key].shape[dim] // TP
                        if dim:
                            nemo_model[nemo_key + key] = nemo_model[nemo_key + key][
                                ..., i * tp_last_dim_size : (i + 1) * tp_last_dim_size
                            ].clone()
                        else:
                            nemo_model[nemo_key + key] = nemo_model[nemo_key + key][
                                i * tp_last_dim_size : (i + 1) * tp_last_dim_size, ...
                            ].clone()

                    print(nemo_key + key, split, nemo_model[nemo_key + key].shape, v.shape)
                else:
                    split, key, dim, transpose = reverse_translation[k]
                    if "wte" in k:
                        # Padding to make it divisble by TP degree
                        if v.shape[0] % TP > 0:
                            x = torch.nn.functional.pad(
                                v, (0, (TP - v.shape[0] % TP), 0, 0)
                            )
                        else:
                            x = v
                        last_dim_size = x.shape[0]
                        tp_last_dim_size = last_dim_size // TP
                        x_sliced = x[i * tp_last_dim_size : (i + 1) * tp_last_dim_size, ...].clone()
                        if p == 0:
                            nemo_model[nemo_key + key] = x_sliced
                            print(nemo_key + key, split, nemo_model[nemo_key + key].shape, v.shape)
                        elif p == PP-1 and args.share_embeddings_and_output_weights:
                            embedding_key = "model.word_embeddings.weight"
                            nemo_model[embedding_key] = x_sliced
                            print(embedding_key, split, nemo_model[embedding_key].shape, v.shape)
                    elif "wpe" in k and p==0:
                        nemo_model[nemo_key + key] = v
                        print(nemo_key + key, split, nemo_model[nemo_key + key].shape, v.shape)
                    elif "ln_f" in k and p == (PP-1):
                        nemo_model[nemo_key + "encoder." + key] = v
                        key = "encoder." + key
                        print(nemo_key + key, split, nemo_model[nemo_key + key].shape, v.shape)
                    elif "lm_head" in k and p == (PP-1) and not args.share_embeddings_and_output_weights:
                        if split:
                            tp_last_dim_size = v.shape[dim]//TP
                            if dim:
                                nemo_model[nemo_key+key] = v[..., i*tp_last_dim_size:(i+1)*tp_last_dim_size].clone()
                            else:
                                nemo_model[nemo_key+key] = v[i*tp_last_dim_size:(i+1)*tp_last_dim_size, ...].clone()
                        else:
                            nemo_model[nemo_key + key] = v
                        if tranpose:
                            nemo_model[nemo_key + key] = torch.transpose(
                                nemo_model[nemo_key + key], 0, 1
                            )
                        print(nemo_key + key, split, nemo_model[nemo_key + key].shape, v.shape)
                    # if p == 0 or p == (P-1):
                    #     print(nemo_key + key, split, nemo_model[nemo_key + key].shape, v.shape)

            checkpoint = {"state_dict": nemo_model}
            path = args.output_path
            if TP > 1:
                path = path + f"/tp_rank_{i:02d}"
            if PP > 1:
                path = path + f"_pp_rank_{p:03d}"
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            print("saving nemo checkpoint")
            torch.save(
                checkpoint,
                str(path)
                + "/megatron_gpt.ckpt",
            )
            print("Done saving nemo checkpoint")


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
        "--share_embeddings_and_output_weights",
        default=False,
        type=bool,
        help="Share embedding and output layer weights.",
    )
    args = parser.parse_args()
    convert_checkpoint(args)
