#!/bin/env python

import os
import nemo.collections.nlp.parts.serialization as xser
import glob
import torch
import argparse
from functools import partial
from multiprocessing import Pool
import sys
import torch.nn.functional as F


def _load(filename):
    if os.path.exists(os.path.join(filename + ".tensors", "tensor_0.pt")):
        obj = xser.load(filename, cpu_only=True)
    else:
        obj = torch.load(filename, map_location='cpu')

    return obj

def _save(obj, filename, save_xser=True):
    if save_xser:
        xser.save(obj, filename)
    else:
        torch.save(obj, filename)


def init_full_tensor(part_tensor, dp_size, dtype=None):
    part_tensor_shape = part_tensor.shape
    full_tensor_shape = list(part_tensor_shape)
    full_tensor_shape[0] *= dp_size
    if dtype is None:
        dtype = part_tensor.dtype
    return torch.zeros(full_tensor_shape, dtype=dtype)


def copy_part_tensor_to_full_tensor(part_tensor, dp_size, dp_rank, full_tensor):
    dim0 = part_tensor.shape[0]
    if len(part_tensor.shape) == 1:
        full_tensor[dp_rank*dim0:(dp_rank + 1)*dim0] = part_tensor
    else:
        full_tensor[dp_rank*dim0:(dp_rank + 1)*dim0, :] = part_tensor

def finalize_full_tensor(full_tensor, origin_tensor_shape):
    padded_tensor_shape = list(full_tensor.shape)
    assert len(padded_tensor_shape) == len(origin_tensor_shape)
    for i in range(1, len(padded_tensor_shape)):
        assert padded_tensor_shape[i] == origin_tensor_shape[i]

    if origin_tensor_shape[0] == padded_tensor_shape[0]:
        return full_tensor

    dim0 = origin_tensor_shape[0]
    return full_tensor[0:dim0] if len(origin_tensor_shape) == 1 else full_tensor[0:dim0, :]


def merge_part_optimizer_states(full_optimizer_states, part_optimizer_states, dp_size, dp_rank):
    '''
    merge part_optimizer_states into full_optimizer_states
    there are 4 items in a part optimizer state, we processed them 1 by 1
    '''

    assert ('state' in part_optimizer_states) and (len(part_optimizer_states['state']) == 0)

    assert 'param_groups' in part_optimizer_states
    if 'param_groups' not in full_optimizer_states:
        full_optimizer_states['param_groups'] = part_optimizer_states['param_groups']

    assert 'base_state' in part_optimizer_states
    for param in part_optimizer_states['base_state'].keys():
        part_param_state = part_optimizer_states['base_state'][param]

        if not 'state' in full_optimizer_states:
            full_optimizer_states['state'] = {}

        if param not in full_optimizer_states['state']:
            assert  part_param_state['exp_avg'].shape == part_param_state['exp_avg_sq'].shape
            full_optimizer_states['state'][param] = {}
            full_optimizer_states['state'][param]['step'] = part_param_state['step']
            full_optimizer_states['state'][param]['exp_avg'] = init_full_tensor(part_param_state['exp_avg'], dp_size)
            full_optimizer_states['state'][param]['exp_avg_sq'] = init_full_tensor(part_param_state['exp_avg_sq'], dp_size)

        copy_part_tensor_to_full_tensor(part_param_state['exp_avg'], dp_size, dp_rank, full_optimizer_states['state'][param]['exp_avg'])
        copy_part_tensor_to_full_tensor(part_param_state['exp_avg_sq'], dp_size, dp_rank, full_optimizer_states['state'][param]['exp_avg_sq'])

    # shape_info contains the original shapes of full parameter before they were padded
    # after the last part state is processed, we change they to its original shape
    assert 'shape_info' in part_optimizer_states
    if dp_rank == dp_size - 1:
        shape_info = part_optimizer_states['shape_info']
        for param in shape_info:
            full_optimizer_states['state'][param]['exp_avg'] = finalize_full_tensor(
                full_optimizer_states['state'][param]['exp_avg'],
                shape_info[param]
            )
            full_optimizer_states['state'][param]['exp_avg_sq'] = finalize_full_tensor(
                full_optimizer_states['state'][param]['exp_avg_sq'],
                shape_info[param]
            )


def merge_part_master_weights(checkpoint, part_optimizer_states, dp_size, dp_rank):
    assert ('sharded_master_weights' in part_optimizer_states[0])
    assert ('untrained_buffers' in part_optimizer_states[0])

    untrained_buffers = part_optimizer_states[0]['untrained_buffers']
    part_model_weights = part_optimizer_states[0]['sharded_master_weights']
    full_model_weights = checkpoint['state_dict']
    full_tensor_index = 0
    part_tensor_index = 0
    for model_name in full_model_weights:
        if model_name in untrained_buffers:
            full_model_weights[model_name] = untrained_buffers[model_name]
        elif model_name == 'untrained_buffer_names':
            pass
        else:
            part_tensor = part_model_weights[part_tensor_index]

            # prev_full_tensor is either a low precision Tensor or TensorReference
            prev_full_tensor = full_model_weights[model_name]
            if type(prev_full_tensor) == xser.TensorReference:
                assert prev_full_tensor.tid == full_tensor_index

            if dp_rank == 0:
                full_model_weights[model_name] = init_full_tensor(part_tensor, dp_size, prev_full_tensor.dtype)

            copy_part_tensor_to_full_tensor(part_tensor, dp_size, dp_rank, full_model_weights[model_name])

            if dp_rank == dp_size - 1:
                full_model_weights[model_name] = finalize_full_tensor(
                    full_model_weights[model_name],
                    part_optimizer_states[0]['shape_info'][part_tensor_index]
                )
            part_tensor_index += 1
        full_tensor_index += 1

def merge(checkpoint, full_optimizer_states, part_optimizer_states, dp_size, dp_rank):
    if len(full_optimizer_states) == 0:
        for i in range(len(part_optimizer_states)):
            full_optimizer_states.append({})
    else:
        assert len(full_optimizer_states) == len(part_optimizer_states)

    num_optimizers = len(part_optimizer_states)

    for i in range(num_optimizers):
        merge_part_optimizer_states(full_optimizer_states[i], part_optimizer_states[i], dp_size, dp_rank)
        if 'sharded_master_weights' in part_optimizer_states[i]:
            merge_part_master_weights(checkpoint, part_optimizer_states, dp_size, dp_rank)


def set_checkpoint_wrap_with_zero(checkpoint, wrap_with_zero):
    if "hyper_parameters" not in checkpoint:
        checkpoint['hyper_parameters'] = {}

    if "cfg" not in checkpoint["hyper_parameters"]:
        checkpoint['hyper_parameters']['cfg'] = {}

    checkpoint['hyper_parameters']['cfg']['wrap_with_zero'] = wrap_with_zero
    checkpoint['hyper_parameters']['cfg']['zero_use_master_weights'] = wrap_with_zero


def merge_optimizer_states_mp(input_dir, output_dir, checkpoint_name):
    os.makedirs(output_dir, exist_ok=True)

    optim_dir = os.path.join(input_dir, "optim")
    if not os.path.exists(optim_dir):
        raise RuntimeError("Error: when merging optimizer states, the optim directory does not exist")

    checkpoint_filename = os.path.join(input_dir, checkpoint_name)
    print(f"loading checkpoint from {checkpoint_filename}")
    checkpoint = _load(checkpoint_filename)
    full_optimizer_states = []

    dp_size = len(glob.glob(optim_dir + "/dp_*"))
    for dp_rank in range(dp_size):
        assert checkpoint_name.endswith(".ckpt")
        optim_filename = checkpoint_name.replace(".ckpt", ".optimizer_states")
        optim_filename = os.path.join(optim_dir, f"dp_rank_{dp_rank:03d}", optim_filename)
        print(f"    loading optimizer states of rank {dp_rank} from {optim_filename}")
        part_optimizer_states = _load(optim_filename)
        print(f"    merging optimizer states of rank {dp_rank} into full optimizer states and checkpoint")
        merge(checkpoint, full_optimizer_states, part_optimizer_states, dp_size, dp_rank)

    print(f"put optimizer states into checkpoint")
    checkpoint["optimizer_states"] = full_optimizer_states

    print(f"set checkpoint[hyper_parameters][cfg][wrap_with_zero] to false")
    set_checkpoint_wrap_with_zero(checkpoint, False)

    checkpoint_filename = os.path.join(output_dir, checkpoint_name)
    print(f"saving checkpoint with full optimizer states to {checkpoint_filename}")
    _save(checkpoint, checkpoint_filename)


def pad_tensor_to_dp_size(tensor, dp_size):
    if tensor.size(0) % dp_size != 0:
        pad_size = dp_size - tensor.size(0) % dp_size
        tensor = F.pad(tensor, [0, 0] * (tensor.dim() - 1) + [0, pad_size])
    return tensor


def split_part_optimizer_states(full_optimizer_states, part_optimizer_states, dp_size, dp_rank):

    part_optimizer_states['state'] = {}

    assert 'param_groups' in full_optimizer_states
    part_optimizer_states['param_groups'] = full_optimizer_states['param_groups']

    assert 'state' in full_optimizer_states
    # save shape_info first, because we might pad the tensors in full state later if necessary, which will change shape
    shape_info = {}
    idx = 0
    for param_group in part_optimizer_states['param_groups']:
        for param in param_group['params']:
            shape_info[idx] = full_optimizer_states['state'][param]['exp_avg'].shape
            idx += 1
    part_optimizer_states['shape_info'] = shape_info

    for param in full_optimizer_states['state'].keys():
        full_param_state = full_optimizer_states['state'][param]

        if dp_rank == 0:
            full_param_state['padded_exp_avg'] = pad_tensor_to_dp_size(full_param_state['exp_avg'], dp_size)
            full_param_state['padded_exp_avg_sq'] = pad_tensor_to_dp_size(full_param_state['exp_avg_sq'], dp_size)

        if 'base_state' not in part_optimizer_states:
            part_optimizer_states['base_state'] = {}

        part_optimizer_states['base_state'][param] = {}
        part_optimizer_states['base_state'][param]['step'] = full_param_state['step']
        part_optimizer_states['base_state'][param]['exp_avg'] = full_param_state['padded_exp_avg'].chunk(dp_size)[dp_rank].clone()
        part_optimizer_states['base_state'][param]['exp_avg_sq'] = full_param_state['padded_exp_avg_sq'].chunk(dp_size)[dp_rank].clone()


def split_part_master_weights_from_opt_state(full_optimizer_states, part_optimizer_states, dp_size, dp_rank):
    for param in full_optimizer_states['state'].keys():
        full_param_state = full_optimizer_states['state'][param]

        if dp_rank == 0:
            full_param_state['padded_param'] = pad_tensor_to_dp_size(full_param_state['param'], dp_size)

        if 'sharded_master_weights' not in part_optimizer_states:
            part_optimizer_states['sharded_master_weights'] = {}

        part_optimizer_states['sharded_master_weights'][param] = full_param_state['padded_param'].chunk(dp_size)[dp_rank].clone()


def split_part_master_weights_from_model_state(checkpoint, part_optimizer_states, dp_size, dp_rank):
    full_model_weights = checkpoint['state_dict']

    if dp_rank == 0:
        checkpoint['padded_state_dict'] = {}

    untrained_buffer_names = full_model_weights['untrained_buffer_names']
    part_optimizer_states[0]['untrained_buffers'] = {}
    part_optimizer_states[0]['sharded_master_weights'] = {}

    part_tensor_index = 0
    full_tensor_index = 0
    for model_name in full_model_weights:
        if model_name in untrained_buffer_names:
            part_optimizer_states[0]['untrained_buffers'][model_name] = full_model_weights[model_name]
            if dp_rank == dp_size - 1:
                full_model_weights[model_name] = xser.TensorReference(
                    tid=full_tensor_index,
                    shape=torch.Size(full_model_weights[model_name].shape),
                    dtype=full_model_weights[model_name].dtype
                )
        elif model_name == 'untrained_buffer_names':
            pass
        else:
            if dp_rank == 0:
                checkpoint['padded_state_dict'][model_name] = pad_tensor_to_dp_size(full_model_weights[model_name], dp_size)

            full_tensor = checkpoint['padded_state_dict'][model_name]
            part_optimizer_states[0]['sharded_master_weights'][part_tensor_index] = full_tensor.chunk(dp_size)[dp_rank]

            if dp_rank == dp_size - 1:
                full_model_weights[model_name] = xser.TensorReference(
                    tid=full_tensor_index,
                    shape=torch.Size(full_model_weights[model_name].shape),
                    dtype=full_model_weights[model_name].dtype
                )

            part_tensor_index += 1
        full_tensor_index += 1

    if dp_rank == dp_size - 1:
        del checkpoint['padded_state_dict']


def split(checkpoint, full_optimizer_states, part_optimizer_states, dp_size, dp_rank):
    if len(part_optimizer_states) == 0:
        for i in range(len(full_optimizer_states)):
            part_optimizer_states.append({})
    else:
        assert len(full_optimizer_states) == len(part_optimizer_states)

    num_optimizers = len(part_optimizer_states)

    for i in range(num_optimizers):
        split_part_optimizer_states(full_optimizer_states[i], part_optimizer_states[i], dp_size, dp_rank)

    # If the checkpoint already has master weights saved in the optimizer states, e.g., checkpoints converted from GPU,
    # then read and split master weights from there, otherwise get them from the normal model params in state_dict.
    if 'param' in full_optimizer_states[0]['state'][list(full_optimizer_states[0]['state'].keys())[0]]:
        split_part_master_weights_from_opt_state(full_optimizer_states[0], part_optimizer_states[0], dp_size, dp_rank)
    else:
        split_part_master_weights_from_model_state(checkpoint, part_optimizer_states, dp_size, dp_rank)


def split_optimizer_states_mp(input_dir, output_dir, checkpoint_name, dp_size):
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_filename = os.path.join(input_dir, checkpoint_name)
    print(f"loading checkpoint with optimizer states from {checkpoint_filename}")
    checkpoint = _load(checkpoint_filename)
    full_optimizer_states = checkpoint["optimizer_states"]
    del checkpoint["optimizer_states"]
    for dp_rank in range(dp_size):
        optim_dir = os.path.join(output_dir, "optim", f"dp_rank_{dp_rank:03d}")
        os.makedirs(optim_dir, exist_ok=True)
        part_optimizer_states = []
        print(f"splitting optimizer states of dp rank {dp_rank} from full optimizer states and checkpoint")
        split(checkpoint, full_optimizer_states, part_optimizer_states, dp_size, dp_rank)
        optim_filename = os.path.join(optim_dir, checkpoint_name.replace(".ckpt", ".optimizer_states"))
        print(f"saving optimizer states of dp rank {dp_rank} to {optim_filename}")
        _save(part_optimizer_states, optim_filename)

    print(f"set checkpoint[hyper_parameters][cfg][wrap_with_zero] to True")
    set_checkpoint_wrap_with_zero(checkpoint, True)

    checkpoint_filename = os.path.join(output_dir, checkpoint_name)
    print(f"saving checkpoint wout optimizer states to {checkpoint_filename}")
    _save(checkpoint, checkpoint_filename)


def convert_optimizer_states(input_checkpoint_dir, output_checkpoint_dir, checkpoint_name, action, output_dp_size=None, rank=None, world_size=None):
    # for entry in os.listdir(input_checkpoint_dir):
    print(f"rank={rank} world_size={world_size} input_checkpoint_dir={input_checkpoint_dir} output_checkpoint_dir={output_checkpoint_dir} checkpoint_name={checkpoint_name} action={action}")
    entrylist = sorted(os.listdir(input_checkpoint_dir))
    rank = int(rank)
    world_size = int(world_size)
    entry_bgn = len(entrylist) * rank // world_size
    entry_end = len(entrylist) * (rank + 1) // world_size
    for entry_idx in range(entry_bgn, entry_end):
        entry = entrylist[entry_idx]
        if not os.path.isdir(os.path.join(input_checkpoint_dir, entry)):
            continue

        if entry.startswith("mp_") or entry.startswith("tp_"):
            if action == "merge":
                merge_optimizer_states_mp(
                    os.path.join(input_checkpoint_dir, entry),
                    os.path.join(output_checkpoint_dir, entry),
                    checkpoint_name,
                )
            elif action == "split":
                if type(output_dp_size) != int or output_dp_size < 0:
                    raise RuntimeError("Error: invalid output_dp_size for splitting optimizer states")

                split_optimizer_states_mp(
                    os.path.join(input_checkpoint_dir, entry),
                    os.path.join(output_checkpoint_dir, entry),
                    checkpoint_name,
                    output_dp_size
                )
            else:
                raise RuntimeError(f"Error: unknown action {action}")


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        type=str,
        help="the action to take: merge or split",
        choices=["merge", "split"],
        required=True
    )

    parser.add_argument(
        "--input_checkpoint_directory",
        type=str,
        help="path to input checkpoint directory",
        required=True
    )

    parser.add_argument(
        "--output_checkpoint_directory",
        type=str,
        help="path to output checkpoint directory",
        required=True
    )

    parser.add_argument(
        "--checkpoint_name",
        type=str,
        help="file name of the checkpoint to convert",
        required=True
    )

    parser.add_argument(
        "--output_dp_size",
        type=int,
        help="DP size for the split action",
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

    convert_optimizer_states(
        input_checkpoint_dir=args.input_checkpoint_directory,
        output_checkpoint_dir=args.output_checkpoint_directory,
        checkpoint_name=args.checkpoint_name,
        action=args.action,
        output_dp_size=args.output_dp_size,
        rank=args.global_rank,
        world_size=args.world_size,
        )

if __name__ == "__main__":
    main(sys.argv)