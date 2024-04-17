from __future__ import division
from __future__ import print_function

import os
import shutil

import torch
import torch_xla
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm


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

def _rewrite_data(path, data, save_tensors, partition, num_partitions, saver):
  def _get_size(tensor):
      return torch.numel(tensor) * tensor.element_size()

  def convert_fn(tensors):
    torch_xla._XLAC._xla_sync_multi(
        tensors, devices=[], wait=True, sync_xla_data=True)
    rewritten_tensors = []
    sorted_indices = sorted(list(range(len(tensors))), key=lambda i: _get_size(tensors[i]), reverse=True)
    kk = _karmarkar_karp(tensors, sorted_indices, num_partitions, _get_size)[partition]
    for i, t in enumerate(tensors):
      if i in kk:
        saver.add_save_task(t.cpu(), _get_tensor_file(path, i))
      rewritten_tensors.append(TensorReference(i, t.shape, t.dtype))
    return rewritten_tensors

  def select_fn(v):
    return type(v) == torch.Tensor

  if save_tensors:
    if os.path.isdir(path):
      shutil.rmtree(path)
    os.mkdir(path)
  xm.rendezvous('torch_xla.utils.serialization._rewrite_data')
  return xm.ToXlaTensorArena(convert_fn, select_fn).transform(data)


def save(data, path, should_save=True, partition=0, num_partitions=1, saver=None):
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
    master_only (bool, optional): Whether only the master device should save the
      data. If False, the `path` argument should be a different path for each of
      the ordinals taking part to the replication, otherwise all the replicas on
      the same host will be writing to the same location.
      Default: True
    global_master (bool, optional): When ``master_only`` is ``True`` this flag
      controls whether every host's master (if ``global_master`` is ``False``)
      saves the content, or only the global master (ordinal 0).
      Default: False
  """
  if saver is None:
      saver = SimpleSaver()
  ref_data = _rewrite_data(_get_tensors_folder(path), data, should_save, partition, num_partitions, saver)
  if should_save:
    saver.add_save_task(ref_data, path)
  xm.rendezvous('torch_xla.utils.serialization.save')


def load(path, partition=0, num_partitions=1, partition_group=None, cpu_only=False):
  """Loads data previously saved with the `save()` API.

  Args:
    path (str): The path passed to the `save()` API.
  Returns:
    The loaded data.
  """
  ref_data = torch.load(path)
  tensor_folder = _get_tensors_folder(path)

  def convert_fn(tensors):
    rewritten_tensors = []
    if partition_group is not None:
      xm.mark_step()
    dtype = None
    for t in tensors:
      if hasattr(t, "dtype"):
        # newer code save dtype in checkpoint's meta data, older code does not.
        # when checkpoint's meta data has dtype, the TensorReference "t" will have "dtype" in it
        dtype = t.dtype

      if dtype is not None:
        # when dtype is available, we use an optimization:
        # among workers in all partitions, only 1 work will read tensor from disk
        # other workers will get the tensor from network broadcasting
        # we used round robin to select which partition will read from disk to evenly
        # distribute the load tasks.
        if (t.tid  % num_partitions) == partition:
          loaded = torch.load(_get_tensor_file(tensor_folder, t.tid))
          if not cpu_only:
            loaded = loaded.to(xm.xla_device())
        else:
          loaded = torch.zeros(t.shape, dtype=dtype, device=xm.xla_device())
        if partition_group is not None:
          # we use all_reduce to implement broadcast because xla does not have native broadcast support.
          torch.distributed.all_reduce(loaded, op=torch.distributed.ReduceOp.SUM, group=partition_group)
      else:
        # when dtype is not available (this happens when checkpoint was generated using older code)
        # all workers load tensor from disk
        loaded = torch.load(_get_tensor_file(tensor_folder, t.tid))
        if not cpu_only:
          loaded = loaded.to(xm.xla_device())

        # All tensors have same dtype. By setting dtype here, we can apply the optimization to
        # all the remaining tensors
        dtype = loaded.dtype

      rewritten_tensors.append(loaded)
    if partition_group is not None:
      xm.mark_step()
    return rewritten_tensors

  def select_fn(v):
    return type(v) == TensorReference

  return xm.ToXlaTensorArena(convert_fn, select_fn).transform(ref_data)
