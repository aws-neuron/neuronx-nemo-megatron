import os
import argparse
import json
import itertools
from glob import glob
from multiprocessing import Pool
import torch
import logging
from functools import partial
import re
import sys

import nemo.collections.nlp.parts.serialization as xser
import torch.multiprocessing as mp
import torch_xla.distributed.xla_multiprocessing as xmp
import torch.nn.functional as F
import torch.distributed as dist
import torch_xla.core.xla_model as xm

from examples.nlp.language_modeling.checkpoint_conversion.convert_hf_checkpoint_to_nemo_llama import get_layer_map
from examples.nlp.language_modeling.checkpoint_conversion.convert_hf_checkpoint_to_nemo_llama import convert_state_dict_for_pp_rank as convert_state_dict_for_pp_rank
from examples.nlp.language_modeling.checkpoint_conversion.convert_hf_checkpoint_to_nemo_llama_70b import convert_state_dict_for_pp_rank as convert_state_dict_for_pp_rank_gqa
from examples.nlp.language_modeling.checkpoint_conversion.convert_hf_checkpoint_to_nemo_llama_70b import hf2mt_fix_query_key_value_ordering

from examples.nlp.language_modeling.checkpoint_conversion.logger_factory import LoggerFactory

logger = LoggerFactory.create_logger(name="hf_to_nemo_llama_opt_states_mpijob", level=logging.INFO)


# TODO: only works for HF to trainium nemo checkpoint

OPT_STATE_KEYS = ['exp_avg', 'exp_avg_sq', 'param']


def interleave_qkv_optimizer_states(q, k, v, num_query_groups):
    interleaved_opt_states = {}
    for opt_key in OPT_STATE_KEYS:
        if opt_key in q:
            qkv = []
            for qc, kc, vc in zip(q[opt_key].chunk(num_query_groups), k[opt_key].chunk(num_query_groups), v[opt_key].chunk(num_query_groups)):
                qkv.append(torch.cat([qc, kc, vc], dim=0))
            qkv = torch.cat(qkv, dim=0)
            interleaved_opt_states[opt_key] = qkv
    return interleaved_opt_states


def concat_optimizer_states(to_concat, dim):
    concatenated_opt_states = {}
    for opt_key in OPT_STATE_KEYS:
        if opt_key in to_concat[0]:
            tensors_to_concat = [t[opt_key] for t in to_concat]
            concatenated_opt_states[opt_key] = torch.cat(tensors_to_concat, dim=dim)
    return concatenated_opt_states


def clone_optimizer_states(params):
    cloned_opt_states = {}
    for opt_key in OPT_STATE_KEYS:
        if opt_key in params:
            cloned_opt_states[opt_key] = params[opt_key].clone()
    return cloned_opt_states


def transpose_optimizer_states(param, dim1, dim2):
    opt_states = {}
    for opt_key in OPT_STATE_KEYS:
        if opt_key in param:
            opt_states[opt_key] = torch.transpose(param[opt_key], dim1, dim2)
    return opt_states


def fix_query_key_value_ordering_optimizer_states(param, checkpoint_version, num_splits, heads, hidden_size_per_head):
    # if num_splits = 1 : param is query tensor
    # if num_splits = 2 : param is key_value tensor
    # if num_splits = 3 : param is query_key_value tensor
    opt_states = [{} for _ in range(num_splits)]
    for opt_key in OPT_STATE_KEYS:
        if opt_key in param:
            qkv = hf2mt_fix_query_key_value_ordering(param[opt_key], checkpoint_version, num_splits, heads, hidden_size_per_head)
            qkv_split = torch.chunk(qkv, num_splits, dim=0)
            for i, split in enumerate(qkv_split):
                opt_states[i][opt_key] = split
    return opt_states


def shard_optimizer_states_for_tp(param, tp_dim_size, dim, index):
    opt_states = {}
    for opt_key in OPT_STATE_KEYS:
        if opt_key in param:
            if dim:  # First or last dimension to shard
                opt_states[opt_key] = param[opt_key][
                                             ..., index * tp_dim_size : (index + 1) * tp_dim_size
                                             ].clone()
            else:
                opt_states[opt_key] = param[opt_key][
                                             index * tp_dim_size : (index + 1) * tp_dim_size, ...
                                             ].clone()
    return opt_states


def pad_token_embeddings(param, TP):
    # first dimension of the token embeddings should be divisible by TP
    opt_states = {}
    for opt_key in OPT_STATE_KEYS:
        if opt_key in param:
            opt_states[opt_key] = torch.nn.functional.pad(param[opt_key], (0, 0, 0, TP - param[opt_key].shape[0] % TP))
    return opt_states


def convert_opt_state_for_pp_rank(
        p,
        path_to_checkpoint,
        config,
        TP,
        PP,
        VPP,
        save_bf16,
        gqa_qkv  # if True will convert to a nemo checkpoint with QKV as signle matrix (trainium new GQA implementation)
    ):
    br_key = "layers."  # Used to filter all transformer layers except layernorm

    gqa = 'num_query_groups' in config or 'num_key_value_heads' in config
    if not gqa or (gqa and gqa_qkv):
        # optimizer_states keys for TRN non-GQA and TRN GQA "new" implementation (QKV matrix)
        num_params_per_layer = 6
        translation_optstate_trn_h_2_4h = {
            # k: (split, key, dim, transpose)
            0: (0, "input_layernorm.optimizer_state", None, 0),
            1: (1, "self_attn.query_key_value.optimizer_state", 0, 0),
            2: (1, "self_attn.o_proj.optimizer_state", 1, 0),
            3: (0, "post_attention_layernorm.optimizer_state", None, 0),
            4: (1, "mlp.gate_proj.optimizer_state", 0, 0), # in neuron megatron checkpoint, this should be concatinated with up proj
            -4: (1, "mlp.up_proj.optimizer_state", 0, 0), # this is a placeholder key for up_proj before it is merged (concat) with gate_proj
            5: (1, "mlp.down_proj.optimizer_state", 1, 0),
        }
    else:
        # optimizer_states keys for TRN "old" GQA implementation (Q and KV matrices)
        num_params_per_layer = 7
        translation_optstate_trn_h_2_4h = {
            # k: (split, key, dim, transpose)
            0: (0, "input_layernorm.optimizer_state", None, 0),
            1: (1, "self_attn.query.optimizer_state", 0, 0),
            2: (1, "self_attn.key_value.optimizer_state", 0, 0),
            3: (1, "self_attn.o_proj.optimizer_state", 1, 0),
            4: (0, "post_attention_layernorm.optimizer_state", None, 0),
            5: (1, "mlp.gate_proj.optimizer_state", 0, 0), # in neuron megatron checkpoint, this should be concatinated with up proj
            -5: (1, "mlp.up_proj.optimizer_state", 0, 0), # this is a placeholder key for up_proj before it is merged (concat) with gate_proj
            6: (1, "mlp.down_proj.optimizer_state", 1, 0),
        }


    reverse_translation = {}
    for key, v in translation_optstate_trn_h_2_4h.items():
        split, k, dim, transpose = v
        reverse_translation[k] = (split, key, dim, transpose)

    def translate_optstate_trn_h_2_4h(k, pp, PP, layer, num_params_per_layer, num_layers_per_pp_rank):
        """returns (split, key, dim, transpose) given k"""

        if pp == 0 and "model.embed_tokens.optimizer_state" == k: # token embeddings
            return 1, 0, 0, 0

        if pp == PP - 1 and "model.norm.optimizer_state" == k: #final layernorm
            key = num_layers_per_pp_rank * num_params_per_layer
            return 0, key, None, 0

        if pp == PP - 1 and "lm_head.optimizer_state" == k: # final output layer
            key = num_params_per_layer * num_layers_per_pp_rank + 1
            return 1, key, 0, 0

        split, ix, dim, transpose = reverse_translation[k]
        key = layer * num_params_per_layer + abs(ix)
        if pp == 0:
            key += 1
        if ix < 0: # workaround for up_proj (dense_h_to_4h_2) to be later concatenated with gate_proj (dense_h_to_4h)
            key = -key
        return (split, key, dim, transpose)

    num_layers_per_pp_rank = config["num_hidden_layers"] // PP
    if VPP is None:
        layers_on_curr_pp = list(range(p * num_layers_per_pp_rank, (p + 1) * num_layers_per_pp_rank))
    else:
        layer_map = get_layer_map(PP, VPP, config['num_hidden_layers'])
        layers_on_curr_pp = list(itertools.chain(*layer_map[p]))

    model_paths = sorted([f'{path_to_checkpoint}/{layer}/pytorch_model.bin' for layer in layers_on_curr_pp])
    model_llama = {}
    for _path in model_paths:
        logger.info(f'Loading {_path}')
        ts = torch.load(_path, map_location='cpu')
        model_llama.update(ts)
    logger.info(len(model_llama))

    static_information_path = f'{path_to_checkpoint}/static_information.bin'
    logger.info(f'Loading {static_information_path}')
    shared_information = torch.load(static_information_path, map_location='cpu')
    logger.info("Loaded Llama model")


    if gqa:
        if gqa_qkv:
            # Interleave Q,K,V within each query head, for "new" GQA implementation
            # This way we only need to split the resulting QKV matrix to get each TP rank parameter later in the TP loop
            for i in layers_on_curr_pp:
                q = model_llama[f'model.layers.{i}.self_attn.q_proj.optimizer_state']
                k = model_llama[f'model.layers.{i}.self_attn.k_proj.optimizer_state']
                v = model_llama[f'model.layers.{i}.self_attn.v_proj.optimizer_state']
                num_query_groups = config.get("num_query_groups", config.get("num_key_value_heads"))
                model_llama[f'model.layers.{i}.self_attn.query_key_value.optimizer_state'] = interleave_qkv_optimizer_states(q, k, v, num_query_groups)
                model_llama.pop(f'model.layers.{i}.self_attn.q_proj.optimizer_state')
                model_llama.pop(f'model.layers.{i}.self_attn.k_proj.optimizer_state')
                model_llama.pop(f'model.layers.{i}.self_attn.v_proj.optimizer_state')
        else:
            # Concatenate KV for "old" GQA implementation.
            for i in layers_on_curr_pp:
                q = model_llama[f'model.layers.{i}.self_attn.q_proj.optimizer_state']
                k = model_llama[f'model.layers.{i}.self_attn.k_proj.optimizer_state']
                v = model_llama[f'model.layers.{i}.self_attn.v_proj.optimizer_state']
                model_llama[f'model.layers.{i}.self_attn.query.optimizer_state'] = q
                model_llama[f'model.layers.{i}.self_attn.key_value.optimizer_state'] = concat_optimizer_states([k, v], dim=0)
                model_llama.pop(f'model.layers.{i}.self_attn.q_proj.optimizer_state')
                model_llama.pop(f'model.layers.{i}.self_attn.k_proj.optimizer_state')
                model_llama.pop(f'model.layers.{i}.self_attn.v_proj.optimizer_state')
    else:
        # Concatenate QKV for non-GQA models
        for i in range(config['num_hidden_layers']):
            q = model_llama[f'model.layers.{i}.self_attn.q_proj.optimizer_state']
            k = model_llama[f'model.layers.{i}.self_attn.k_proj.optimizer_state']
            v = model_llama[f'model.layers.{i}.self_attn.v_proj.optimizer_state']
            model_llama[f'model.layers.{i}.self_attn.query_key_value.optimizer_state'] = concat_optimizer_states([q, k, v], dim=0)
            model_llama.pop(f'model.layers.{i}.self_attn.q_proj.optimizer_state')
            model_llama.pop(f'model.layers.{i}.self_attn.k_proj.optimizer_state')
            model_llama.pop(f'model.layers.{i}.self_attn.v_proj.optimizer_state')

    out_models = {}
    for i in range(TP):

        logger.info(f"=== generating optimizer states for PP {p}, TP {i} ===")
        nemo_model = {}
        for k, params in model_llama.items():
            if "attention.masked_bias" in k:
                # We don't want to copy attention mask bias, since its a constant of 1e4
                continue
            if br_key in k:
                parts = k.split(br_key)[1].split(".")
                layer_number = int(parts[0])

                if int(layer_number) in layers_on_curr_pp:
                    logger.info(f'layer_number in HF: {layer_number}')
                    k_orig = k
                    k = ".".join(parts[1:])
                    if "attn.bias" in k:
                        continue

                    layer_number = str(layers_on_curr_pp.index(int(layer_number)))
                    logger.info(f'layer_number in nemo: {layer_number}')
                    split, key, dim, transpose = translate_optstate_trn_h_2_4h(k, p, PP, int(layer_number), num_params_per_layer, num_layers_per_pp_rank)
                    # TODO: had to do this clone here because otherwise when slicing the tensor to do a TP split, it impact the original data and breaks the operations for next "i" in the TP loop
                    nemo_model[key] = clone_optimizer_states(params)
                    if transpose:
                        nemo_model[key] = transpose_optimizer_states(nemo_model[key], 0, 1)

                    if gqa and not gqa_qkv:
                        # "old" GQA implementation
                        if "query" in k:
                            heads = config["num_attention_heads"]
                            hidden_size_per_head = config["hidden_size"] // heads
                            nemo_model[key] = fix_query_key_value_ordering_optimizer_states(nemo_model[key], 2.0, 1, heads, hidden_size_per_head)
                        if "key_value" in k:
                            heads = config["num_attention_heads"]
                            hidden_size_per_head = config["hidden_size"] // heads
                            kv_heads = config['num_key_value_heads']
                            nemo_model[key] = fix_query_key_value_ordering_optimizer_states(nemo_model[key], 2.0, 2, kv_heads, hidden_size_per_head)
                    if not gqa and "query_key_value" in k:
                        # non-GQA
                        heads = config["num_attention_heads"]
                        hidden_size_per_head = config["hidden_size"] // heads
                        nemo_model[key] = fix_query_key_value_ordering_optimizer_states(nemo_model[key], 2.0, 3, heads, hidden_size_per_head)

                    if split:
                        tp_last_dim_size = nemo_model[key]['exp_avg'].shape[dim] // TP
                        nemo_model[key] = shard_optimizer_states_for_tp(nemo_model[key], tp_dim_size=tp_last_dim_size, dim=dim, index=i)

                    logger.info(f"k_orig: {k_orig}, params['exp_avg'].shape: {params['exp_avg'].shape}, \n=> key: {key}, split: {split}, dim: {dim}, nemo_model[key]['exp_avg'].shape: {nemo_model[key]['exp_avg'].shape}")
            else:
                if "embed_tokens" in k and p == 0:
                    split, key, dim, transpose = translate_optstate_trn_h_2_4h(k, p, PP, None, num_params_per_layer, num_layers_per_pp_rank)
                    # Padding to make it divisible by TP degree
                    if params['exp_avg'].shape[0] % TP > 0:
                        x = pad_token_embeddings(params, TP)
                    else:
                        x = params
                    last_dim_size = x['exp_avg'].shape[0]
                    tp_last_dim_size = last_dim_size // TP
                    nemo_model[key] = shard_optimizer_states_for_tp(x, tp_dim_size=tp_last_dim_size, dim=dim, index=i)

                    logger.info(f"k: {k}, params['exp_avg'].shape: {params['exp_avg'].shape}, \n=> key: {key}, split: {split}, dim: {dim}, nemo_model[key]['exp_avg'].shape: {nemo_model[key]['exp_avg'].shape}")
                elif "model.norm" in k and p == (PP - 1):
                    split, key, dim, transpose = translate_optstate_trn_h_2_4h(k, p, PP, None, num_params_per_layer, num_layers_per_pp_rank)
                    nemo_model[key] = params

                    logger.info(f"k: {k}, params['exp_avg'].shape: {params['exp_avg'].shape}, \n=> key: {key}, split: {split}, dim: {dim}, nemo_model[key]['exp_avg'].shape: {nemo_model[key]['exp_avg'].shape}")
                elif "lm_head" in k and p == (PP - 1):
                    split, key, dim, transpose = translate_optstate_trn_h_2_4h(k, p, PP, None, num_params_per_layer, num_layers_per_pp_rank)
                    if split:
                        tp_last_dim_size = params['exp_avg'].shape[dim] // TP
                        nemo_model[key] = shard_optimizer_states_for_tp(params, tp_dim_size=tp_last_dim_size, dim=dim, index=i)

                    logger.info(f"k: {k}, params['exp_avg'].shape: {params['exp_avg'].shape}, \n=> key: {key}, split: {split}, dim: {dim}, nemo_model[key]['exp_avg'].shape: {nemo_model[key]['exp_avg'].shape}")

        # Merge gate proj (dense_h_to_4h) and up proj (dense_h_to_4h_2) into one, as nemo megatron combines the two
        # negative key is an up_proj matrix (dense_h_to_4h_2) which should be concatinated with a gate proj (dense_h_to_4h)
        up_proj_keys = [key for key in nemo_model.keys() if key < 0]
        for key in up_proj_keys:
            logger.info(f"merging gate proj and up proj ({nemo_model[-key]['exp_avg'].shape}) and ({nemo_model[key]['exp_avg'].shape})")
            nemo_model[-key] = concat_optimizer_states([nemo_model[-key], nemo_model[key]], dim=0)
            logger.info(f"merged pojections is ({nemo_model[-key]['exp_avg'].shape})")
            nemo_model.pop(key)

        # set the step value. It is the same for all keys:
        for key in nemo_model:
            nemo_model[key].update({'step': shared_information['optimizer_states']['step_information']})

        if save_bf16:
            for _k in nemo_model:
                for _kk in nemo_model[_k]:
                    nemo_model[_k][_kk] = nemo_model[_k][_kk].to(dtype=torch.bfloat16, device='cpu')

        nemo_model = dict(sorted(nemo_model.items()))
        out_models[i] = {
            "optimizer_states": [
                {'state': nemo_model,
                 'param_groups': [shared_information['optimizer_states']['opt_param_groups']]}],
            'global_step': shared_information['global_step'],
            'pytorch-lightning_version': shared_information['pytorch-lightning_version'],
            'token_count': shared_information['token_count'],
            'hparams_name': shared_information['hparams_name'],
            'lr_schedulers': shared_information['lr_schedulers'],
            'flex_state': shared_information['flex_state'],
            'loops': shared_information['loops']
        }
        out_models[i]['optimizer_states'][0]['param_groups'][0]['params'] = list(range(len(nemo_model)))

    return out_models


def convert_checkpoint(args, config, func_conv_param_weights, func_conv_opt_states, rotary_pos=None, ckpt_ref=None):
    TP = args.tp_degree
    PP = args.pp_degree
    VPP = args.vpp_degree
    rank = int(args.global_rank)
    world_size = int(args.world_size)

    entry_bgn = PP * rank // world_size
    entry_end = PP * (rank + 1) // world_size

    logger.info(f"rank is {rank}, world_size is {world_size}, entry begin is {entry_bgn}, entry end is {entry_end}")
    for entry_idx in range(entry_bgn, entry_end):
        p = entry_idx

        out_models_state_dict = func_conv_param_weights(
            p,
            path_to_checkpoint=args.path_to_checkpoint, model_bin_file=args.model_bin_file, num_shards=args.num_shards,
            config=config, TP=TP, PP=PP, VPP=VPP, save_bf16=args.save_bf16, gqa_qkv=args.gqa_qkv)

        out_models_optim_state = func_conv_opt_states(
            p,
            path_to_checkpoint=args.path_to_checkpoint_opt_states,
            config=config, TP=TP, PP=PP, VPP=VPP, save_bf16=args.save_bf16, gqa_qkv=args.gqa_qkv)

        assert len(out_models_state_dict) == TP
        assert len(out_models_optim_state) == TP

        for i in range(TP):
            out_model = {'state_dict': out_models_state_dict[i]['state_dict']}
            if VPP is not None:
                for vpp in range(VPP):
                    out_model[f'model{vpp}'] = out_models_state_dict[i][f'model{vpp}']
            out_model.update(out_models_optim_state[i])

            if rotary_pos is not None: # For RoPE embeddings
                out_model_updated = out_model.copy()
                out_model_updated["pytorch-lightning_version"] = ckpt_ref['pytorch-lightning_version']
                out_model_updated['hyper_parameters'] = ckpt_ref['hyper_parameters']
                out_model_updated['hyper_parameters']['padded_vocab_size'] = 125184
                out_model_updated['state_dict'] = {}
                for key, val in out_model['state_dict'].items():
                    out_model_updated['state_dict'].update({key: val})
                    if 'post_attention_layernorm.weight' in key:
                        layer = re.findall('.*layers\.(\d+)\.post_attention_layernorm\.weight', key)[0]
                        logger.info(f"adding rotary embedding for layer {layer}")
                        out_model_updated['state_dict'][f'model.language_model.encoder.layers.{layer}.self_attention.core_attention.rotary_emb.inv_freq'] = rotary_pos
                if VPP is not None:
                    for vpp in range(VPP):
                        for key, val in out_model[f'model{vpp}'].items():
                            out_model_updated[f'model{vpp}'].update({key: val})
                            if 'post_attention_layernorm.weight' in key:
                                layer = re.findall('.*layers\.(\d+)\.post_attention_layernorm\.weight', key)[0]
                                logger.info(f"adding rotary embedding for layer {layer}")
                                out_model_updated[f'model{vpp}'][f'model.language_model.encoder.layers.{layer}.self_attention.core_attention.rotary_emb.inv_freq'] = rotary_pos
            
            output_folder = args.output_path
            if TP > 1:
                if PP > 1:
                    output_folder = output_folder + f"/tp_rank_{i:02d}"
                else:
                    output_folder = output_folder + f"/mp_rank_{i:02d}"
            if PP > 1:
                if TP > 1:
                    output_folder = output_folder + f"_pp_rank_{p:03d}"
                else:
                    output_folder = output_folder + f"/tp_rank_00_pp_rank_{p:03d}"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            checkpoint_name = f"{output_folder}/{args.save_checkpoint_prefix}_converted_checkpoint--step={args.step}-consumed_samples={args.consumed_samples}.ckpt"
            if args.is_xser:
                from nemo.collections.nlp.parts.serialization import save
                save(out_model_updated, checkpoint_name)
                logger.info(f"Done saving model PP {p} for TP {i}")
            else:
                torch.save(out_model_updated, checkpoint_name)  # , (not master_only), global_master=True)
                logger.info(f"Done saving model PP {p} for TP {i}")


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_version", default=2.0)
    parser.add_argument(
        "--path_to_checkpoint",
        type=str,
        help="Path to the checkpoint folder",
    )
    parser.add_argument(
        "--path_to_checkpoint_opt_states",
        type=str,
        help="Path to the checkpoint folder for optimizer states",
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
        type=int,
        help="Number of shards in the save checkpoint",
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
        "--consumed_samples",
        help="This is needed for trn code to know what is the last consumed sample to continue from.",
        required=True
    )
    parser.add_argument(
        "--gqa_qkv",
        action="store_true",
        help="If this is for trainium new GQA implementation, K, Q and V will be saved as one tensor",
    )
    parser.add_argument(
        "--reference_trn_ckpt",
        type=str,
        help="reference model to fill in the parameters"
    )
    parser.add_argument(
        "--reference_hf_ckpt",
        type=str,
        help="reference model to fill in the parameters"
    )
    parser.add_argument(
        "--save_checkpoint_prefix",
        default="llmv2",
        type=str,
        help="The prefix to the converted checkpoint name. Default: llmv2",
    )
    parser.add_argument(
        "--global_rank",
        type=str,
        help="reference model to fill in the parameters"
    )
    parser.add_argument(
        "--world_size",
        type=str,
        help="reference model to fill in the parameters"
    )
    args = parser.parse_args(args[1:])
    logger.info(f"after parsing!")

    with open(args.config_file, "r") as f:
        config = json.load(f)
    args.num_hidden_layers = config["num_hidden_layers"]
    PP = args.pp_degree


    # read in rotary embedding
    ckpt_ref = xser.load(args.reference_trn_ckpt, cpu_only=True)
    temp_ref_hf = str(Path(args.config_file).parent / f"pytorch_model-00001-of-{args.num_shards:05d}.bin"))

    if not args.reference_hf_ckpt and temp_ref_hf is not None and temp_ref_hf.is_file():
        logger.info(f"Using reference HF checkpoint {temp_ref_hf} found within config.json parent directory")
        ckpt_ref_hf_path = temp_ref_hf
    elif "pytorch_model.bin" in args.reference_hf_ckpt or re.match(pattern, args.reference_hf_ckpt):
        logger.info(f"Using explicitly specified reference HF single checkpoint {args.reference_hf_ckpt}")
        ckpt_ref_hf_path = args.reference_hf_ckpt
    elif args.reference_hf_ckpt:
        ckpt_ref_hf_path = args.reference_hf_ckpt + f"pytorch_model-00001-of-{args.num_shards:05d}.bin"
        logger.info(f"Using explicitly specified reference HF checkpoint {ckpt_ref_hf_path}")
    else:
        ckpt_ref_hf_path = None
        logger.info(f"WARNING: No HF checkpoint therefore skipping the addition of RoPE in this step.")

    rotary_pos = None
    if ckpt_ref_hf_path is not None:
        ckpt_ref_hf = torch.load(ckpt_ref_hf_path)
        rotary_pos = ckpt_ref_hf['model.layers.0.self_attn.rotary_emb.inv_freq']

    gqa = 'num_key_value_heads' in config or 'num_query_groups' in config

    convert_checkpoint(
        args=args, 
        config=config, 
        func_conv_param_weights=convert_state_dict_for_pp_rank_gqa if gqa else convert_state_dict_for_pp_rank, 
        func_conv_opt_states=convert_opt_state_for_pp_rank,
        rotary_pos=rotary_pos,
        ckpt_ref=ckpt_ref,
        )


    logger.info(f"Done saving Megatron checkpoint as {args.output_path}/{args.save_checkpoint_prefix}_converted_checkpoint--step={args.step}-consumed_samples={args.consumed_samples}.ckpt")

if __name__ == "__main__":
    main(sys.argv)
