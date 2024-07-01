#!/bin/env python

import os
import nemo.collections.nlp.parts.serialization as xser
from nemo.collections.nlp.parts.checkpoint_storage import create_checkpoint_storage
import glob
import torch
import argparse
import torch.nn.functional as F
import torch_xla.distributed.xla_multiprocessing as xmp
import sys
import torch.distributed as dist
import torch_xla.core.xla_model as xm
if torch.__version__.startswith('2'):
    import torch_xla.experimental.pjrt_backend


class ConvertSimpleSaver:

    def __init__(self):
        pass

    def add_save_task(self, data, path):
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        checkpoint_dir = create_checkpoint_storage(dirname)

        checkpoint_dir.save_object(data, basename)

    def rank(self):
        return 0

    def world_size(self):
        return 1

    def process_group(self):
        return None
    
def resolve_optim_states_file_name(checkpoint_name, naming_schema):
    if naming_schema == 'v1':
        return checkpoint_name.replace('ckpt', 'optimizer_states')
    elif naming_schema == 'v2':
        if '-last.ckpt' in checkpoint_name:
            return checkpoint_name.replace('-last.ckpt', '-optimizer_states-last.ckpt')
        else:
            return checkpoint_name.replace('.ckpt', '-optimizer_states.ckpt')
    else:
        raise ValueError("Invalid naming schema")


def _load(filename, scope=""):
    dirname = os.path.dirname(filename)
    basename = os.path.basename(filename)
    checkpoint_dir = create_checkpoint_storage(dirname)

    if checkpoint_dir.file_exists(os.path.join(basename + ".tensors", "tensor_0.pt")):
        if scope:
            obj = _load_with_scope(filename, scope)
        else:
            obj = xser.load(filename, process_group=None, cpu_only=True)
    else:
        obj = checkpoint_dir.load_object(basename, map_location='cpu')

    return obj

def _save(obj, filename, save_xser=True):
    if save_xser:
        saver = ConvertSimpleSaver()
        xser.save(obj, filename, saver=saver)
    else:
        dirname = os.path.dirname(filename)
        basename = os.path.basename(filename)
        checkpoint_dir = create_checkpoint_storage(dirname)
        checkpoint_dir.save_object(obj, basename)


def _load_with_scope(path, scope):
    """Loads data from previously saved with the `xser.save()` API, parameterized with scope. Allowing users to save only
        master weights, optimizer states, or both.

    Args:
        path (str): The path passed to the `save()` API.
        scope (str): If only_master_weights or only_optim_states is passed in, then only those tensors will be loaded.
            The other tensors will be zeroed tensors of the same previous shape
    Returns:
        The loaded data.
    """

    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    checkpoint_dir = create_checkpoint_storage(dirname)
    ref_data = checkpoint_dir.load_object(basename)

    tids_set = set()
    if scope != 'full':
        tensor_ref_map = _map_tensor_ref(ref_data)
        if scope == 'only_master_weights':
            assert ('sharded_master_weights' in tensor_ref_map)
            assert ('untrained_buffers' in tensor_ref_map)
            tids_set.update(set(tensor_ref_map['sharded_master_weights']+tensor_ref_map['untrained_buffers']))
        elif scope == 'only_optim_states':
            assert ('base_state' in tensor_ref_map)
            tids_set.update(set(tensor_ref_map['base_state']))

    tensor_folder = xser._get_tensors_folder(basename)

    def convert_fn(tensors):
        rewritten_tensors = []
        dtype = None
        for tensor_ref in tensors:
            if hasattr(tensor_ref, "dtype"):
                # newer code save dtype in checkpoint's meta data, older code does not.
                # when checkpoint's meta data has dtype, the TensorReference "t" will have "dtype" in it
                dtype = tensor_ref.dtype
            ignore_tensor_data = False
            if scope != 'full' and tensor_ref.tid not in tids_set:
                ignore_tensor_data = True
            loaded = xser._load_tensor(checkpoint_dir, tensor_folder, tensor_ref, dtype, ignore_tensor_data, True, None)
            # All tensors have same dtype. By setting dtype here, we can apply the optimization to
            # all the remaining tensors
            dtype = loaded.dtype

            rewritten_tensors.append(loaded)

        return rewritten_tensors

    def select_fn(v):
        return type(v) == xser.TensorReference

    return xm.ToXlaTensorArena(convert_fn, select_fn).transform(ref_data)

def _map_tensor_ref(data_ref):
    """ Iterates through the nested checkpoint reference object and maps each dictionary key to all the tensor reference ids
    """

    tensor_ref_map = {}

    # Helper function to recursively search for TensorReference objects
    def recurse(obj, collector):
        if isinstance(obj, dict):
            for value in obj.values():
                recurse(value, collector)
        elif isinstance(obj, list):
            for item in obj:
                recurse(item, collector)
        elif isinstance(obj, xser.TensorReference):
            collector.append(obj.tid)

    # There should only 1 element in data ref
    for data in data_ref:
      if isinstance(data, dict):
        for key in data.keys():
            collector = []
            recurse(data[key], collector)
            if collector:  # Only add to result if there are collected tids
                tensor_ref_map[key] = collector

    return tensor_ref_map

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

def merge(checkpoint, full_optimizer_states, part_optimizer_states, dp_size, dp_rank, scope):
    rank = torch.distributed.get_rank()
    if len(full_optimizer_states) == 0:
        for i in range(len(part_optimizer_states)):
            full_optimizer_states.append({})
    else:
        assert len(full_optimizer_states) == len(part_optimizer_states)

    num_optimizers = len(part_optimizer_states)
    for i in range(num_optimizers):
        if scope != 'only_master_weights':
            merge_part_optimizer_states(full_optimizer_states[i], part_optimizer_states[i], dp_size, dp_rank)
        if 'sharded_master_weights' in part_optimizer_states[i] and scope != 'only_optim_states':
            merge_part_master_weights(checkpoint, part_optimizer_states, dp_size, dp_rank)
        else: 
            print(f"[Rank {rank}] No master weights in part optimizer state {i}")


def set_checkpoint_wrap_with_zero(checkpoint, wrap_with_zero):
    if "hyper_parameters" not in checkpoint:
        checkpoint['hyper_parameters'] = {}

    if "cfg" not in checkpoint["hyper_parameters"]:
        checkpoint['hyper_parameters']['cfg'] = {}

    checkpoint['hyper_parameters']['cfg']['wrap_with_zero'] = wrap_with_zero
    checkpoint['hyper_parameters']['cfg']['zero_use_master_weights'] = wrap_with_zero


def merge_optimizer_states_mp(input_dir, output_dir, checkpoint_name, config):

    scope = config["scope"]
    save_format = config["save_format"]
    naming_schema = config["naming_schema"]

    output_checkpoint_dir = create_checkpoint_storage(output_dir)
    output_checkpoint_dir.create_dir(".", exist_ok=True)

    input_checkpoint_dir = create_checkpoint_storage(input_dir)
    optim_dir = os.path.join(input_checkpoint_dir.dirname(), "optim")
    if not input_checkpoint_dir.dir_exists("optim"):
        raise RuntimeError("Error: when merging optimizer states, the optim directory does not exist")

    checkpoint_filename = os.path.join(input_checkpoint_dir.dirname(), checkpoint_name)
    checkpoint = _load(checkpoint_filename)
    full_optimizer_states = []
    if input_checkpoint_dir.dirname().startswith("s3://"):
        dp_size = len(_listdir_s3(input_checkpoint_dir, "optim"))
    else:
        dp_size = len(glob.glob(optim_dir + "/dp_*"))
    for dp_rank in range(dp_size):
        assert checkpoint_name.endswith(".ckpt")
        optim_filename = resolve_optim_states_file_name(checkpoint_name, naming_schema)
        optim_filename = os.path.join(optim_dir, f"dp_rank_{dp_rank:03d}", optim_filename)
        print(f"Loading and merging {optim_filename} to full checkpoint shard")
        part_optimizer_states = _load(optim_filename, scope)
        merge(checkpoint, full_optimizer_states, part_optimizer_states, dp_size, dp_rank, scope)

    if scope != "only_master_weights":
        checkpoint["optimizer_states"] = full_optimizer_states

    set_checkpoint_wrap_with_zero(checkpoint, False)

    checkpoint_filename = os.path.join(output_dir, checkpoint_name)

    print(f"saving merged checkpoint to {checkpoint_filename}")
    save_xser = True if save_format == 'xser' else False
    _save(checkpoint, checkpoint_filename, save_xser)


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


def split_optimizer_states_mp(input_dir, output_dir, checkpoint_name, dp_size, config):
    naming_schema = config["naming_schema"]
    save_format = config["save_format"]

    output_checkpoint_dir = create_checkpoint_storage(output_dir)
    output_checkpoint_dir.create_dir(".", exist_ok=True)

    checkpoint_filename = os.path.join(input_dir, checkpoint_name)
    print(f"loading checkpoint with optimizer states from {checkpoint_filename}")
    checkpoint = _load(checkpoint_filename)
    full_optimizer_states = checkpoint["optimizer_states"]
    del checkpoint["optimizer_states"]
    for dp_rank in range(dp_size):
        output_checkpoint_dir.create_shared_dir("optim", exist_ok=True)
        optim_subdir = os.path.join("optim", f"dp_rank_{dp_rank:03d}")
        output_checkpoint_dir.create_dir(optim_subdir, exist_ok=True)
        optim_dir = os.path.join(output_checkpoint_dir.dirname(), optim_subdir)
        part_optimizer_states = []
        print(f"splitting optimizer states of dp rank {dp_rank} from full optimizer states and checkpoint")
        split(checkpoint, full_optimizer_states, part_optimizer_states, dp_size, dp_rank)
        optim_filename = resolve_optim_states_file_name(checkpoint_name, naming_schema)
        optim_filename = os.path.join(optim_dir, optim_filename)
        print(f"saving optimizer states of dp rank {dp_rank} to {optim_filename}")
        _save(part_optimizer_states, optim_filename)

    print(f"set checkpoint[hyper_parameters][cfg][wrap_with_zero] to True")
    set_checkpoint_wrap_with_zero(checkpoint, True)

    checkpoint_filename = os.path.join(output_dir, checkpoint_name)
    print(f"saving checkpoint wout optimizer states to {checkpoint_filename}")
    save_xser = True if save_format == 'xser' else False
    _save(checkpoint, checkpoint_filename, save_xser)

def _listdir_s3(checkpoint_dir, prefix: str=None):
    s3 = checkpoint_dir.get_client()
    if checkpoint_dir._base_key and prefix:
        list_prefix = os.path.join(checkpoint_dir._base_key, prefix)
    else:
        list_prefix = checkpoint_dir._base_key if checkpoint_dir._base_key else prefix

    if list_prefix and list_prefix[-1] != '/':
        list_prefix += '/'    # list_object_v2 require prefix to be end with '/'

    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(
        Bucket=checkpoint_dir._bucket,
        Prefix=list_prefix,
        Delimiter="/"
    )

    results = []
    prefix_len = len(list_prefix) if list_prefix else 0
    for page in page_iterator:
        if "CommonPrefixes" in page:
            for obj in page["CommonPrefixes"]:
                results.append(obj["Prefix"][prefix_len:-1])

    return results


def convert_optimizer_states(input_checkpoint_dir, output_checkpoint_dir, checkpoint_name, action, config, output_dp_size=None):
    assert config["scope"] and config["save_format"] and config["naming_schema"]

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    s3_dir = True if input_checkpoint_dir.startswith("s3://") else False
    checkpoint_dir = create_checkpoint_storage(input_checkpoint_dir)
    paths = _listdir_s3(checkpoint_dir) if s3_dir else os.listdir(checkpoint_dir.dirname())
    if len(paths) % (world_size) != 0:
        raise ValueError("World size must be divisible by number of TP/PP sharded directories")

    start_idx = len(paths) * rank // world_size
    end_idx = len(paths) * (rank + 1) // world_size

    for path_idx in range(start_idx, end_idx):
        
        path = paths[path_idx]

        if not checkpoint_dir.dir_exists(path):
            continue

        if path.startswith("mp_") or path.startswith("tp_"):
            if action == "merge":
                merge_optimizer_states_mp(
                    os.path.join(input_checkpoint_dir, path),
                    os.path.join(output_checkpoint_dir, path),
                    checkpoint_name,
                    config,
                )
            elif action == "split":
                if type(output_dp_size) != int or output_dp_size < 0:
                    raise RuntimeError("Error: invalid output_dp_size for splitting optimizer states")

                split_optimizer_states_mp(
                    os.path.join(input_checkpoint_dir, path),
                    os.path.join(output_checkpoint_dir, path),
                    checkpoint_name,
                    output_dp_size,
                    config,
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
        "--scope",
        type=str,
        choices=['full', 'only_master_weights', 'only_optim_states'],
        required=False,
        default='full',
        help="choose the scope of merge action: full, only_master_weights, only_optim_states",
    )

    parser.add_argument(
        "--save_format",
        type=str,
        choices=['xser', 'nser'],
        required=False,
        default='xser',
        help="choose checkpoints save format: xser or nser",
    )

    parser.add_argument(
        "--naming_schema",
        type=str,
        choices=['v1', 'v2'],
        required=False,
        default='v1',
        help="choose between optim filename schema versions | v1: <file>.optimizer_states | v2: <file>-optimizer_states.ckpt",
    )

    args = parser.parse_args(args[1:])
    config = {
        "scope": args.scope,
        "save_format": args.save_format,
        "naming_schema": args.naming_schema,
    }
    
    convert_optimizer_states(
        input_checkpoint_dir=args.input_checkpoint_directory,
        output_checkpoint_dir=args.output_checkpoint_directory,
        checkpoint_name=args.checkpoint_name,
        action=args.action,
        config=config,
        output_dp_size=args.output_dp_size)

def _mp_fn(index, args):
    main(args)
    xm.rendezvous("_mp_fn finished")

if __name__ == "__main__":
    if os.environ.get("WORLD_SIZE"):
        if torch.__version__.startswith('2'):
            print("Initializing process group")
            dist.init_process_group("xla", init_method="pjrt://")
            print("XLA process group initialized")
        else:
            dist.init_process_group("xla")
        _mp_fn(0, sys.argv)
    else:
        xmp.spawn(_mp_fn, args=(sys.argv,))