import argparse
import json
from pathlib import Path
import re
import torch
from multiprocessing import Pool
import os
import sys
from functools import partial
import torch.multiprocessing as mp
import torch_xla.distributed.xla_multiprocessing as xmp
import torch.nn.functional as F
import torch.distributed as dist
import torch_xla.core.xla_model as xm
from examples.nlp.language_modeling.checkpoint_conversion.convert_nemo_checkpoint_to_hf_llama import get_tp_pp_degree, \
    get_checkpoints_for_pp, mt2hf_fix_query_key_value_ordering, unpack_gqa_single_qkv

OPT_STATE_KEYS = ['exp_avg', 'exp_avg_sq', 'param']


def concat_optimizer_states(tp_models, k, dim):
    concatenated_opt_states = {}
    for opt_key in OPT_STATE_KEYS:
        if opt_key in tp_models[0][k]:
            concatenated_opt_states[opt_key] = torch.concat(
                [tp_models[i][k][opt_key] for i in range(len(tp_models))],
                dim=dim)
    return concatenated_opt_states


def get_optimizer_states(param, k):
    opt_states = {}
    for opt_key in OPT_STATE_KEYS:
        if opt_key in param[k]:
            opt_states[opt_key] = param[k][opt_key]
    return opt_states


def transpose_optimizer_states(param, dim1, dim2):
    opt_states = {}
    for opt_key in OPT_STATE_KEYS:
        if opt_key in param:
            opt_states[opt_key] = torch.transpose(param[opt_key], dim1, dim2)
    return opt_states


def split_gate_and_proj_optimizer_states(tp_models, k, dim, gate_key, up_key):
    opt_states = {gate_key: {}, up_key: {}}
    for opt_key in OPT_STATE_KEYS:
        if opt_key in tp_models[0][k]:
            dense_h_to_4h_weights_concat = []
            dense_h_to_4h_2_weights_concat = []
            for i in range(len(tp_models)):
                assert tp_models[i][k][opt_key].size()[0] % 2 == 0
                split_size = tp_models[i][k][opt_key].size()[0] // 2
                dense_h_to_4h_weights, dense_h_to_4h_2_weights = torch.split(tp_models[i][k][opt_key], split_size)
                dense_h_to_4h_weights_concat.append(dense_h_to_4h_weights)
                dense_h_to_4h_2_weights_concat.append(dense_h_to_4h_2_weights)
            opt_states[gate_key][opt_key] = torch.concat(dense_h_to_4h_weights_concat, dim=dim)
            opt_states[up_key][opt_key] = torch.concat(dense_h_to_4h_2_weights_concat, dim=dim)
    return opt_states


def unpack_gqa_single_qkv_optimizer_states(param, num_query_groups, heads):
    opt_states = [{}, {}, {}]  # [q, k, v]
    for opt_key in OPT_STATE_KEYS:
        if opt_key in param:
            q, k, v = unpack_gqa_single_qkv(param[opt_key], num_query_groups, heads)
            opt_states[0][opt_key] = q
            opt_states[1][opt_key] = k
            opt_states[2][opt_key] = v
    return opt_states


def fix_query_key_value_ordering_optimizer_states(param, checkpoint_version, num_splits, heads, hidden_size_per_head):
    # if num_splits = 1 : param is query tensor
    # if num_splits = 2 : param is key_value tensor
    # if num_splits = 3 : param is query_key_value tensor
    opt_states = [{} for _ in range(num_splits)]
    for opt_key in OPT_STATE_KEYS:
        if opt_key in param:
            qkv = mt2hf_fix_query_key_value_ordering(
                param[opt_key], checkpoint_version, num_splits, heads, hidden_size_per_head)
            qkv_split = torch.chunk(qkv, num_splits, dim=0)
            for i, split in enumerate(qkv_split):
                opt_states[i][opt_key] = split
    return opt_states


def get_layer_map(pipeline_parallel_degree, virtual_pipeline_parallel_size, num_layers):
    layer_map = {}

    layers_per_pipeline = num_layers // pipeline_parallel_degree
    layers_per_stage = num_layers // pipeline_parallel_degree // virtual_pipeline_parallel_size

    pp_idx = 0
    virtual_pp_idx = 0
    num_layers_in_stage = 0
    for layer_idx in range(num_layers):
        layer_map[pp_idx * (layers_per_pipeline) + virtual_pp_idx * (layers_per_stage) + num_layers_in_stage] = layer_idx

        num_layers_in_stage = (num_layers_in_stage + 1) % layers_per_stage

        if num_layers_in_stage == 0:
            pp_idx = (pp_idx + 1) % pipeline_parallel_degree

            if pp_idx == 0:
                virtual_pp_idx = (virtual_pp_idx + 1) % virtual_pipeline_parallel_size

    return layer_map


def convert_checkpoint(
                       config_file,
                       path_to_checkpoints,
                       output_path,
                       checkpoint_version=2.0,
                       is_xser=False,
                       checkpoint_step=None,
                       nemo_122=False,
                       trn=False,  # whether the checkpoint is a trainium checkpoint vs gpu checkpoint
                       gqa_qkv=True,  # the new TRN GQA implementation stores qkv in one matrix
                       VPP=1,
                       rank=None,
                       world_size=None,
                       ):
    with open(config_file, "r") as f:
        config = json.load(f)

    # key mappings for optimizer_states.
    # Unlike parameter wights in "state_dict", optimizer states are saved with numerical keys, instead of layer number and param names.
    # Below dictionary and the following function translates each numerical key
    translation_optstate_gqa_trn_q_kv = {
        # k: (split, key, dim, transpose)
        0: (0, "input_layernorm.optimizer_state", None, 0),
        1: (1, "self_attn.query.optimizer_state", 0, 0),
        2: (1, "self_attn.key_value.optimizer_state", 0, 0),
        3: (1, "self_attn.o_proj.optimizer_state", 1, 0),
        4: (0, "post_attention_layernorm.optimizer_state", None, 0),
        5: (1, "mlp.gate_proj.optimizer_state", 0, 0),  # in neuron megatron, this is the concatinated gate proj and up proj
        6: (1, "mlp.down_proj.optimizer_state", 1, 0),
    }

    translation_optstate_gqa_trn_qkv = {
        # k: (split, key, dim, transpose)
        0: (0, "input_layernorm.optimizer_state", None, 0),
        1: (1, "self_attn.query_key_value.optimizer_state", 0, 0),
        2: (1, "self_attn.o_proj.optimizer_state", 1, 0),
        3: (0, "post_attention_layernorm.optimizer_state", None, 0),
        4: (1, "mlp.gate_proj.optimizer_state", 0, 0),  # in neuron megatron, this is the concatinated gate proj and up proj
        5: (1, "mlp.down_proj.optimizer_state", 1, 0),
    }

    translation_optstate_gqa_gpu = {
        # k: (split, key, dim, transpose)
        0: (0, "input_layernorm.optimizer_state", None, 0),
        1: (1, "self_attn.query_key_value.optimizer_state", 0, 0),
        2: (1, "self_attn.o_proj.optimizer_state", 1, 0),
        3: (0, "post_attention_layernorm.optimizer_state", None, 0),
        4: (1, "mlp.gate_proj.optimizer_state", 0, 0),
        5: (1, "mlp.up_proj.optimizer_state", 0, 0),
        6: (1, "mlp.down_proj.optimizer_state", 1, 0),
    }

    translation_optstate_gqa_gpu_nemo122 = {
        # k: (split, key, dim, transpose)
        0: (1, "self_attn.o_proj.optimizer_state", 1, 0),
        1: (0, "input_layernorm.optimizer_state", None, 0),
        2: (1, "self_attn.query_key_value.optimizer_state", 0, 0),
        3: (0, "post_attention_layernorm.optimizer_state", None, 0),
        4: (1, "mlp.gate_proj.optimizer_state", 0, 0),  # in nemo 1.22, this is the concatinated gate proj and up proj
        5: (1, "mlp.down_proj.optimizer_state", 1, 0),
    }

    def get_optimizer_states_param_mapping():
        if trn:
            if gqa_qkv:
                return translation_optstate_gqa_trn_qkv, 6
            else:
                return translation_optstate_gqa_trn_q_kv, 7
        else:
            if nemo_122:
                return translation_optstate_gqa_gpu_nemo122, 6
            else:
                return translation_optstate_gqa_gpu, 7

    def translate_optstate(k, pp, PP, num_layers_per_pp_rank):
        """
        returns (split, key, dim, transpose) given k

        k: the component index number throughout the whole model. As most components
        are replicated knowing the number of components in a layer and number of layers
        in a PP rank. This can retrive the exact component name
        (i.e QKV opt state, MLP down proj opt state etc) given the component index number (k)
        """
        opt_param_mapping, num_params_per_layer = get_optimizer_states_param_mapping()

        if pp == 0 and k == 0:  # first layer token embeddings
            return 1, "model.embed_tokens.optimizer_state", 0, 0

        if PP == 1: # there is only 1 pp rank, so first pp rank is also terminal rank
            if k == num_params_per_layer * num_layers_per_pp_rank + 1:  # final layernorm
                return 0, "model.norm.optimizer_state", None, 0
            if k == num_params_per_layer * num_layers_per_pp_rank + 2:  # final output layer
                return 1, "lm_head.optimizer_state", 0, 0
        else:
            if pp == PP - 1 and k == num_params_per_layer * num_layers_per_pp_rank:  # final layernorm
                return 0, "model.norm.optimizer_state", None, 0
            if pp == PP - 1 and k == num_params_per_layer * num_layers_per_pp_rank + 1:  # final output layer
                return 1, "lm_head.optimizer_state", 0, 0

        param_ix = (k - 1) % num_params_per_layer if pp == 0 else k % num_params_per_layer
        return opt_param_mapping[param_ix]

    br_key = "model.layers."

    heads = config["num_attention_heads"]
    hidden_size_per_head = config["hidden_size"] // heads

    TP, PP = get_tp_pp_degree(path_to_checkpoints)
    print(f"hidden size: {config['hidden_size']}" +
          f"\nattention heads: {heads}" +
          # f"\nquery groups: {config['num_query_groups']}" +
          f"\nTP: {TP}, PP: {PP}")

    num_layers = config["num_hidden_layers"]
    num_layers_per_pp_rank = num_layers // PP
    print(f"num_layers: {num_layers}")
    print(f"number of layers per pp rank: {num_layers_per_pp_rank}")

    # TODO: get VPP from hyperparameters saved in the checkpoint if exists, i.e.,
    # checkpoints_for_pp[0]['hyper_parameters']['cfg']['virtual_pipeline_model_parallel_size']
    if VPP > 1:
        layer_map = get_layer_map(PP, VPP, num_layers)
    def actual_layer_index(VPP, ln_idx):
        if VPP > 1:
            return layer_map[ln_idx]
        else:
            return ln_idx

    assert PP > 0

    entry_bgn = PP * rank // world_size
    entry_end = PP * (rank + 1) // world_size

    print(f"rank is {rank}, entry begin is {entry_bgn}, entry end is {entry_end}, world size is {world_size}, PP is {PP}")
    for entry_idx in range(entry_bgn, entry_end):
        pp = entry_idx

        # for pp in range(PP):
        print(f"Loading PP={pp}")
        checkpoints_for_pp = get_checkpoints_for_pp(pp, path_to_checkpoints, PP, TP, is_xser, checkpoint_step)
        tp_models = {i: v['optimizer_states'][0]['state'] for i, v in checkpoints_for_pp.items()}

        prev_ln_idx = -1
        merged_model = {}
        for k in tp_models[0].keys():  # e.g., k: 0, 1, ..., 85 for pp=0 and 0, 1, ..., 84 for pp>0
            # for gpu checkpoints the first key is the step
            if k == 'step':
                continue

            _, num_params_per_layer = get_optimizer_states_param_mapping()
            if pp == 0:
                ln_idx = int(max(0, (k - 1)) // num_params_per_layer)
            else:
                ln_idx = int(k // num_params_per_layer + pp * num_layers_per_pp_rank)
            ln_idx = min(ln_idx, num_layers - 1)

            actual_ln_indx = actual_layer_index(VPP, ln_idx)

            # save layer after all layer parameters are computed
            if actual_ln_indx != prev_ln_idx:
                if prev_ln_idx > -1:
                    path = Path(output_path).joinpath(str(prev_ln_idx))
                    path.mkdir(parents=True, exist_ok=True)
                    print(f"Saving model layer {prev_ln_idx} in path {path}")
                    torch.save(merged_model, str(path) + "/pytorch_model.bin")
                    print(f"Done saving model layer {prev_ln_idx} in path {path}")
                prev_ln_idx = actual_ln_indx
                merged_model = {}

            split, key, dim, transpose = translate_optstate(k, pp, PP, num_layers_per_pp_rank)

            print(f"--- {actual_ln_indx} (k={int(k)}): {tp_models[0][k]['exp_avg'].shape}")

            # Below are not tied to a layer, i.e., embed token or final model norm or lm head
            # TODO: write the conditions based on the original keys in the state_dict
            if "embed_tokens" in key:
                merged_model[key] = concat_optimizer_states(tp_models, k, dim)
                print(f"> {key}: {merged_model[key]['exp_avg'].shape}")
                continue

            if "lm_head" in key:
                merged_model[key] = concat_optimizer_states(tp_models, k, dim)
                print(f"> {key}: {merged_model[key]['exp_avg'].shape}")
                continue

            if "model.norm" in key:
                # layer norm is not split across TP ranks. So, we don't need to concatenate
                merged_model[key] = get_optimizer_states(tp_models[0], k)
                print(f"> {key}: {merged_model[key]['exp_avg'].shape}")
                continue

            hf_key = f"{br_key}{actual_ln_indx}.{key}"
            merged_model[hf_key] = {}
            if split:
                if "gate_proj" in key and (trn or nemo_122):
                    # Trn checkpoints are saved with gate_proj and up_proj fused as one matrix, so we split it in two
                    # Same thing for GPU checkpoints that are generated from nemo 1.22 (i.e., the 57B and 470B)
                    split_key = key.replace("gate_proj", "up_proj")
                    split_hf_key = f"{br_key}{actual_ln_indx}.{split_key}"
                    merged_model.update(
                        split_gate_and_proj_optimizer_states(tp_models, k, dim, gate_key=hf_key, up_key=split_hf_key))
                    print(f"> {hf_key}: {merged_model[hf_key]['exp_avg'].shape}")
                    print(f"> {split_hf_key}: {merged_model[split_hf_key]['exp_avg'].shape}")
                else:
                    merged_model[hf_key] = concat_optimizer_states(tp_models, k, dim)
                    print(f"> {hf_key}: {merged_model[hf_key]['exp_avg'].shape}")
            else:
                merged_model[hf_key] = get_optimizer_states(tp_models[0], k)
                print(f"> {hf_key}: {merged_model[hf_key]['exp_avg'].shape}")

            if "query_key_value" in key:
                num_query_groups = config.get("num_query_groups", config.get("num_key_value_heads"))
                num_attention_heads = config.get("num_attention_heads")
                if num_query_groups is not None and num_query_groups > 0 and num_query_groups != num_attention_heads:
                    # if we get query_key_value and we have query groups (GQA), its either,
                    # - GPU GQA or
                    # - TRN new GQA implementation in NNM
                    # either case the k, q and v are arranged per query group and concatenated to form a single qkv
                    q, k, v = unpack_gqa_single_qkv_optimizer_states(merged_model[hf_key], num_query_groups, heads)
                else:
                    q, k, v = fix_query_key_value_ordering_optimizer_states(merged_model[hf_key], checkpoint_version, 3,
                                                                            heads, hidden_size_per_head)

                merged_model.pop(hf_key)
                merged_model[f'{br_key}{actual_ln_indx}.self_attn.q_proj.optimizer_state'] = q
                merged_model[f'{br_key}{actual_ln_indx}.self_attn.k_proj.optimizer_state'] = k
                merged_model[f'{br_key}{actual_ln_indx}.self_attn.v_proj.optimizer_state'] = v

            if 'query' in key and "query_key_value" not in key:
                hf_key_q = f"{br_key}{actual_ln_indx}.self_attn.q_proj.optimizer_state"
                q_as_list = fix_query_key_value_ordering_optimizer_states(merged_model[hf_key], checkpoint_version, 1,
                                                                          heads, hidden_size_per_head)
                merged_model[hf_key_q] = q_as_list[0]
                merged_model.pop(hf_key)
                print(f"< pop {hf_key}")
                print(f"> {hf_key_q}: {merged_model[hf_key_q]['exp_avg'].shape}")

            if 'key_value' in key and "query_key_value" not in key:
                kv_heads = config['num_key_value_heads']
                k, v = fix_query_key_value_ordering_optimizer_states(merged_model[hf_key], checkpoint_version, 2,
                                                                     kv_heads, hidden_size_per_head)
                hf_key_k = f"{br_key}{actual_ln_indx}.self_attn.k_proj.optimizer_state"
                hf_key_v = f"{br_key}{actual_ln_indx}.self_attn.v_proj.optimizer_state"
                merged_model[hf_key_k] = k
                merged_model[hf_key_k] = v
                merged_model.pop(hf_key)

                print(f"< pop {hf_key}")
                print(f"> {hf_key_k}: {merged_model[hf_key_k]['exp_avg'].shape}")
                print(f"> {hf_key_v}: {merged_model[hf_key_v]['exp_avg'].shape}")

            if transpose:
                merged_model[hf_key] = transpose_optimizer_states(merged_model[hf_key], dim1=0, dim2=1)

        # save the last layer
        path = Path(output_path).joinpath(str(prev_ln_idx))
        path.mkdir(parents=True, exist_ok=True)
        print(f"Saving model layer {prev_ln_idx} in path {path}")
        torch.save(merged_model, str(path) + "/pytorch_model.bin")
        print(f"Done saving model layer {prev_ln_idx} in path {path}")

    # 'param_groups' includes information about lr, optimizer class, etc.
    # It is the same across TP ranks and PP ranks, except for the 'params' key which is the list of numbers indicating
    # the keys for opt state and differs across PP ranks. So, we get the param groups from the last pp and tp 0 but
    # drop the 'params' key
        if pp == PP-1:
            opt_param_groups = checkpoints_for_pp[0]['optimizer_states'][0]['param_groups'][0]
            opt_param_groups.pop('params')

            # 'step' information is the same across pp and tp ranks. So we get it from the last pp and tp 0
            step_information = checkpoints_for_pp[0]['optimizer_states'][0]['state'][0]['step'] if trn else checkpoints_for_pp[0]['optimizer_states'][0]['state']['step']

            # Note: We don't take the `hyper_parameters` from the checkpoint as it will be overwritten based on the destination HPs.
            # For example, TP and PP might change in destination.
            # TODO: should we still copy the `hyper_parameters` and let the override happen during resume?
            static_information = {'optimizer_states': {'opt_param_groups': opt_param_groups,
                                                    'step_information': step_information},
                                # below fields are the same across tp/pp ranks. So, we only take from the last pp and tp 0
                                'global_step': checkpoints_for_pp[0]['global_step'],
                                'pytorch-lightning_version': checkpoints_for_pp[0]['pytorch-lightning_version'],
                                'token_count': checkpoints_for_pp[0]['token_count'],
                                'hparams_name': checkpoints_for_pp[0]['hparams_name'],
                                'lr_schedulers': checkpoints_for_pp[0]['lr_schedulers'],
                                'flex_state': checkpoints_for_pp[0]['flex_state'],
                                'loops': checkpoints_for_pp[0]['loops']}

            path = Path(output_path)
            path.mkdir(parents=True, exist_ok=True)
            torch.save(static_information, str(path) + "/static_information.bin")
            print(f"Done saved HF style optimizer states in {path}")


def main(args):
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
        help="Enable serialized loading. set True for trainium checkpoints and False for gpu checkpoints",
    )
    parser.add_argument(
        "--step",
        type=str,
        help="Specify a checkpoint step e.g., '22500'. If not specified, the latest checkpoint will be used.",
        required=False
    )
    parser.add_argument(
        "--trn",
        action="store_true",
        help="Whether the checkpoint is a Trainium checkpoint.",
    )
    parser.add_argument(
        "--gqa_qkv",
        action="store_true",
        help="If this is for trainium new GQA implementation, K, Q and V will be saved as one tensor",
    )
    parser.add_argument(
        "--nemo_122",
        action="store_true",
        help="Whether the checkpoint is from nemo 1.22",
    )
    parser.add_argument(
        "--vpp",
        type=int,
        required=True
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
    convert_checkpoint(
        config_file=args.config_file,
        path_to_checkpoints=args.path_to_checkpoints,
        output_path=args.output_path,
        checkpoint_version=2.0,
        is_xser=args.is_xser,
        checkpoint_step=args.step,
        nemo_122=args.nemo_122,
        trn=args.trn,
        gqa_qkv=args.gqa_qkv,
        VPP=args.vpp,
        rank = int(args.global_rank),
        world_size = int(args.world_size),
        )

if __name__ == "__main__":
    main(sys.argv)
