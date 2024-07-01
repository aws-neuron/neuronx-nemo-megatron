from __future__ import division
from __future__ import print_function

import os
import shutil

import torch
import torch_xla
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
from .checkpoint_storage import create_checkpoint_storage

class TensorReference(object):

  def __init__(self, tid, shape, dtype):
    self.tid = tid
    self.shape = shape
    self.dtype = dtype


def _get_tensors_folder(path):
  return path + '.tensors'


def _get_tensor_file(path, tid):
  return os.path.join(path, 'tensor_{}.pt'.format(tid))

def _karmarkar_karp(unsorted_list, sorted_indices, num_partitions, sort_func):
    bins = [[] for i in range(num_partitions)]
    sizes = [0 for i in range(num_partitions)]

    def argmin(to_get):
        return to_get.index(min(to_get))

    for idx in sorted_indices:
        size = sort_func(unsorted_list[idx])
        am = argmin(sizes)
        bins[am].append(idx)
        sizes[am] += size

    return bins


class SimpleSaver:

    def __init__(self):
        pass

    def add_save_task(self, data, path):
        torch.save(data, path)

    def rank(self):
        return 0

    def world_size(self):
        return 1

    def process_group(self):
        return None

def _rewrite_data(path, data, saver, ignore_tensor_data):
  def _get_size(tensor):
      return torch.numel(tensor) * tensor.element_size()

  def convert_fn(tensors):
    torch_xla._XLAC._xla_sync_multi(
        tensors, devices=[], wait=True, sync_xla_data=True)
    rewritten_tensors = []
    sorted_indices = sorted(list(range(len(tensors))), key=lambda i: _get_size(tensors[i]), reverse=True)
    kk = _karmarkar_karp(tensors, sorted_indices, saver.world_size(), _get_size)[saver.rank()]
    for i, t in enumerate(tensors):
      if not ignore_tensor_data:
        if i in kk:
          saver.add_save_task(t.cpu(), _get_tensor_file(path, i))
      rewritten_tensors.append(TensorReference(i, t.shape, t.dtype))
    return rewritten_tensors

  def select_fn(v):
    return type(v) == torch.Tensor

  checkpoint_dir = create_checkpoint_storage(path)
  checkpoint_dir.create_shared_dir(".", exist_ok=True,
          process_group=saver.process_group())
  return xm.ToXlaTensorArena(convert_fn, select_fn).transform(data)


def save(data, path, saver=None, ignore_tensor_data=False):
  """Saves the input data into a file.

  The saved data is transferred to PyTorch CPU device before being saved, so a
  following `torch.load()` will load CPU data.
  Care must be taken when working with views. Instead of saving views it's
  recommended that you recreate them after the tensors have been loaded and
  moved to their destination device(s).

  Args:
    data: The input data to be saved. Any nested combination of Python objects
      (list, tuples, sets, dicts, ...).
    path: The destination file for the data saving operation. If `master_only`
      is ``False`` the path must point to different destinations as otherwise
      all the writes from the same host will override each other.
    saver:

    ignore_tensor_data: whether to write pt file for individual tensor.
  """
  if saver is None:
      saver = SimpleSaver()
  ref_data = _rewrite_data(_get_tensors_folder(path), data, saver, ignore_tensor_data)
  if saver.rank() == 0:
    saver.add_save_task(ref_data, path)


def _load_tensor(checkpoint_dir, tensor_folder, tensor_ref, dtype, ignore_tensor_data, cpu_only, process_group):
    assert hasattr(tensor_ref, "tid")
    assert hasattr(tensor_ref, "shape")

    device = "cpu" if cpu_only else xm.xla_device()

    if ignore_tensor_data:
        if dtype is None:
            raise RuntimeError("Error: cannot ignore tensor data when dtype was not given")
        return torch.zeros(tensor_ref.shape, dtype=dtype, device=device)

    # when dtype and process_group are both available, we use an optimization:
    # among workers in one process group, only 1 work will read tensor from disk
    # other workers will get the tensor from network broadcasting.
    # this optimization require dtype because broadcasting is implemented by
    # allreduce, and allreduce need dtype to allocate the empty tensor.
    #
    # In order to implement this optimization, we added dtype to TensorReference,
    # but previous checkpoint does not have dtype in it, therefore we
    # need to handle the case dtype is None.

    if (dtype is None) or (process_group is None):
        # when neither dtype nor process group is available,
        # there is no optimization can be done, we load tensor from disk and return.
        loaded = checkpoint_dir.load_object(_get_tensor_file(tensor_folder, tensor_ref.tid))
        if not cpu_only:
          loaded = loaded.to(xm.xla_device())

        return loaded

    group_size = process_group.size()
    my_rank_in_group = process_group.rank()

    # we used round robin to select worker who will read from disk.
    if (tensor_ref.tid  % group_size) == my_rank_in_group:
        loaded = checkpoint_dir.load_object(_get_tensor_file(tensor_folder, tensor_ref.tid))
        if not cpu_only:
            loaded = loaded.to(xm.xla_device())
    else:
        loaded = torch.zeros(tensor_ref.shape, dtype=dtype, device=device)
    # we use all_reduce to implement broadcast because xla does not have native broadcast support.
    torch.distributed.all_reduce(loaded, op=torch.distributed.ReduceOp.SUM, group=process_group)
    return loaded


def load(path, process_group=None, ignore_tensor_data=False, cpu_only=False):
  """Loads data from previously saved with the `save()` API using provided process group

  When used under distributed environment, it is possible that "path" to be different between processes.

  In that case, caller need to pass process_group, and make sure all processes in the same group
  has the same 'path'

  Args:
    path (str): The path passed to the `save()` API.
    process_group (torch.distributed.distributed_c10d.ProcessGroup): a group of processes that will load same file
    ignore_tensor_data (bool): do not load tensor file, return zero tensor with correct shape and dtype
    cpu_only (bool): when set to True, do not copy tensor to device.
  Returns:
    The loaded data.
  """
  dirname = os.path.dirname(path)
  basename = os.path.basename(path)
  checkpoint_dir = create_checkpoint_storage(dirname)

  ref_data = checkpoint_dir.load_object(basename)
  tensor_folder = _get_tensors_folder(basename)

  def convert_fn(tensors):
    rewritten_tensors = []
    if process_group is not None:
      xm.mark_step()
    dtype = None
    for tensor_ref in tensors:
      if hasattr(tensor_ref, "dtype"):
        # newer code save dtype in checkpoint's meta data, older code does not.
        # when checkpoint's meta data has dtype, the TensorReference "t" will have "dtype" in it
        dtype = tensor_ref.dtype

      loaded = _load_tensor(checkpoint_dir, tensor_folder, tensor_ref, dtype, ignore_tensor_data, cpu_only, process_group)
      # All tensors have same dtype. By setting dtype here, we can apply the optimization to
      # all the remaining tensors
      dtype = loaded.dtype

      rewritten_tensors.append(loaded)

    if process_group is not None:
      xm.mark_step()

    return rewritten_tensors

  def select_fn(v):
    return type(v) == TensorReference

  return xm.ToXlaTensorArena(convert_fn, select_fn).transform(ref_data)
