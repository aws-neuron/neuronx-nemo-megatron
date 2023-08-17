# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gradient clipping."""

import itertools

import torch
if torch.__version__.startswith('2'):
    from torch import inf
else:
    from torch._six import inf

from nemo.collections.nlp.modules.common.megatron.module import param_is_not_shared

try:
    # import amp_C
    from apex.multi_tensor_apply import multi_tensor_applier
    from apex.transformer import parallel_state
    from apex.transformer.tensor_parallel.layers import param_is_not_tensor_parallel_duplicate

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

try:
    import torch_xla.core.xla_model as xm
    ## NEURON ##
    def _create_tensor(t, fn_key):
        _fn_map = {
            "DoubleTensor": torch.DoubleTensor,
            "FloatTensor": torch.FloatTensor,
            "IntTensor": torch.IntTensor,
            "LongTensor": torch.LongTensor
        }
        return _fn_map[fn_key](t).to(xm.xla_device())
    
    def _create_device():
        return xm.xla_device()
    
    torch.cuda.DoubleTensor = lambda t: _create_tensor(t, "DoubleTensor")
    torch.cuda.FloatTensor = lambda t: _create_tensor(t, "FloatTensor")
    torch.cuda.IntTensor = lambda t: _create_tensor(t, "IntTensor")
    torch.cuda.LongTensor = lambda t: _create_tensor(t, "LongTensor")
    torch.cuda.current_device = _create_device
    HAVE_XLA = True
except (ImportError, ModuleNotFoundError):
    HAVE_XLA = False

HAVE_APEX_DISTRIBUTED_ADAM = False
if HAVE_APEX:
    try:
        from apex.contrib.optimizers.distributed_fused_adam import DistributedFusedAdam

        HAVE_APEX_DISTRIBUTED_ADAM = True
    except (ImportError, ModuleNotFoundError):
        pass


def clip_grad_norm_fp32(parameters, max_norm, norm_type=2):
    """Clips gradient norm of an iterable of parameters whose gradients
       are in fp32.
    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place.
    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    # import pdb; pdb.set_trace()
    xm.mark_step()

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Filter parameters based on:
    #   - grad should not be none
    #   - parameter should not be shared
    #   - should not be a replica due to tensor model parallelism
    grads = []
    grads_for_norm = []
    for param in parameters:
        grad_not_none = param.grad is not None
        is_not_shared = param_is_not_shared(param)
        is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
        if grad_not_none:
            grad = param.grad.detach()
            # Make sure the grads are in fp32
            # assert isinstance(param.grad, torch.cuda.FloatTensor)
            grads.append(grad)
        if grad_not_none and is_not_shared and is_not_tp_duplicate:
            grads_for_norm.append(grad)

    if not grads_for_norm:
        raise ValueError(f"No grads found, please disable gradient clipping {xm.get_ordinal()}")

    # Norm parameters.
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    # total_norm = 0.0
    total_norm = torch.cuda.FloatTensor([float(0.0)])

    # Calculate norm.
    if norm_type == inf:
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        # Take max across all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_tensor_model_parallel_group()
        )
        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_pipeline_model_parallel_group()
        )
        total_norm = total_norm_cuda[0].item()

    else:
        if norm_type == 2.0:
            # dummy_overflow_buf = torch.cuda.IntTensor([0])
            # # Use apex's multi-tensor applier for efficiency reasons.
            # # Multi-tensor applier takes a function and a list of list
            # # and performs the operation on that list all in one kernel.
            # grad_norm, _ = multi_tensor_applier(
            #     amp_C.multi_tensor_l2norm, dummy_overflow_buf, [grads_for_norm], False  # no per-parameter norm
            # )
            # # Since we will be summing across data parallel groups,
            # # we need the pow(norm-type).
            # total_norm = grad_norm ** norm_type
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm ** norm_type

        else:
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm ** norm_type

        # Sum across all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_tensor_model_parallel_group()
        )
        torch.distributed.all_reduce(
            total_norm, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_pipeline_model_parallel_group()
        )
        # total_norm = total_norm.item() ** (1.0 / norm_type)
        total_norm = torch.pow(total_norm, 1.0 / norm_type)

    # Scale.
    clip_coeff = max_norm / (total_norm + 1.0e-6)
    # if clip_coeff < 1.0:
    #     dummy_overflow_buf = torch.cuda.IntTensor([0])
    #     multi_tensor_applier(amp_C.multi_tensor_scale, dummy_overflow_buf, [grads, grads], clip_coeff)
    for g in grads:
        g.data.mul_(torch.where(clip_coeff < 1, clip_coeff, torch.tensor(1., device=xm.xla_device())))

    xm.mark_step()
    return total_norm


def count_zeros_fp32(parameters):

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Filter parameters based on:
    #   - grad should not be none
    #   - parameter should not be shared
    #   - should not be a replica due to tensor model parallelism
    total_num_zeros = 0.0
    for param in parameters:
        grad_not_none = param.grad is not None
        is_not_shared = param_is_not_shared(param)
        is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
        if grad_not_none and is_not_shared and is_not_tp_duplicate:
            grad = param.grad.detach()
            num_zeros = grad.numel() - torch.count_nonzero(grad)
            total_num_zeros = num_zeros + total_num_zeros

    # Sum across all model-parallel GPUs.
    torch.distributed.all_reduce(
        total_num_zeros, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_tensor_model_parallel_group()
    )
    torch.distributed.all_reduce(
        total_num_zeros, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_pipeline_model_parallel_group()
    )
    total_num_zeros = total_num_zeros.item()

    return total_num_zeros


def clip_grad_norm_distributed_optimizer(optimizer, max_norm, norm_type=2):
    """Clips gradient norm of parameters in distributed optimizer

    This is a wrapper around DistributedFusedAdam.clip_grad_norm with
    added functionality to handle model parallel parameters.

    Arguments:
        parameters (DistributedFusedAdam): distributed optimizer
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Currently
            only 2-norm is supported.

    Returns:
        Total norm of the parameters (viewed as a single vector).

    """
    assert norm_type == 2
    assert isinstance(optimizer, DistributedFusedAdam)

    # Filter parameters based on:
    #   - parameter should not be shared
    #   - should not be a replica due to tensor model parallelism
    params = itertools.chain.from_iterable(param_group['params'] for param_group in optimizer.param_groups)
    params_for_norm = []
    for param in params:
        is_not_shared = param_is_not_shared(param)
        is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
        if is_not_shared and is_not_tp_duplicate:
            params_for_norm.append(param)

    # Compute grad norm
    # Note: Compute norm of local grads and sum over all procs
    grad_norm_sq = optimizer._local_grad_norm(parameters=params_for_norm, norm_type=norm_type)
    torch.distributed.all_reduce(
        grad_norm_sq, op=torch.distributed.ReduceOp.SUM,
    )
    grad_norm = grad_norm_sq.sqrt()

    # Apply gradient clipping
    # Note: DistributedFusedAdam is only aware of the data-parallel
    # process group, so we cannot directly apply its gradient clipping
    # function. However, it caches the grad norm to avoid redundant
    # communication, so it suffices to overwrite the cache with the
    # grad norm computed over the world parallel group.
    optimizer._grad_norm = grad_norm
    return optimizer.clip_grad_norm(max_norm, norm_type=norm_type)
