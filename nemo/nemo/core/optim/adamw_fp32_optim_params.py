# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, Meta AI and PyTorch team.
# Modifications Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

# AdamW adapted from PyTorch, with high-precision optimizer states for BF16/FP32 training.
# source: https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py

import os
from collections import defaultdict, abc as container_abcs
from copy import deepcopy
from itertools import chain

import torch
from typing import Iterable, Dict, Any, Hashable, List
from torch import Tensor
from torch.optim import AdamW
from importlib.metadata import version as get_version
from packaging import version


class AdamW_FP32OptimParams(AdamW):
    """
    Implements AdamW with Fp32 Optimizer states in the presence of XLA_DOWNCAST_BF16
    It is not supported without XLA_DOWNCAST_BF16

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 1e-3):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(self, params, **args):
        self.upcast_optim_states = os.environ.get('XLA_DOWNCAST_BF16', '0') == '1'
        if not self.upcast_optim_states:
            raise RuntimeError("XLA_DOWNCAST_BF16 not set. AdamW_FP32OptimParams requires XLA_DOWNCAST_BF16 to be set.")
        super().__init__(params, **args)

    # Overriden to init optimizer states in FP64. Downcasted to FP32 by XLA
    # PyTorch 2.0 override
    def _init_group(
            self,
            group,
            params_with_grad,
            grads,
            amsgrad,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= torch.is_complex(p)
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                state["step"] = (
                    torch.zeros((), dtype=torch.int, device=p.device)
                    if group["capturable"] or group["fused"]
                    else torch.tensor(0.0, dtype=torch.int)
                )
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(
                    p, dtype=torch.double, memory_format=torch.preserve_format
                )
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(
                    p, dtype=torch.double, memory_format=torch.preserve_format
                )
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        p, dtype=torch.double, memory_format=torch.preserve_format
                    )

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])

            if group['amsgrad']:
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])
            if group['differentiable'] and state['step'].requires_grad:
                raise RuntimeError('`requires_grad` is not supported for `step` in differentiable mode')

            # Foreach without capturable does not support a tensor lr
            if group['foreach'] and isinstance(group['lr'], Tensor) and not group['capturable']:
                raise RuntimeError('lr as a Tensor is not supported for capturable=False and foreach=True')

            state_steps.append(state["step"])
        return has_complex

    # Overriden to always cast to FP64. Called by load_state_dict for every tensor.
    # Override for PyTorch 2.0+
    @staticmethod
    def _process_value_according_to_param_policy(
            param: torch.Tensor,
            value: torch.Tensor,
            param_id: int,
            param_groups: List[Dict[Any, Any]],
            key: Hashable = None,
    ) -> torch.Tensor:
        keys_to_cast = ["step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"]
        if param.is_floating_point():
            if key in keys_to_cast:
                return value.to(dtype=torch.double, device=param.device)
            else:
                return value.to(dtype=param.dtype, device=param.device)
        else:
            return value.to(device=param.device)

    # Override for PyTorch 1.13
    # Below methods would need to change if PyTorch 1.13 changes.
    if version.parse(get_version('torch')) < version.parse("2.0"):
        @torch.no_grad()
        def step(self, closure=None):
            """Performs a single optimization step.

            Args:
                closure (Callable, optional): A closure that reevaluates the model
                    and returns the loss.
            """
            from torch.optim.adamw import adamw
            self._cuda_graph_capture_health_check()

            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            for group in self.param_groups:
                params_with_grad = []
                grads = []
                exp_avgs = []
                exp_avg_sqs = []
                max_exp_avg_sqs = []
                state_steps = []
                amsgrad = group['amsgrad']
                beta1, beta2 = group['betas']

                for p in group['params']:
                    if p.grad is None:
                        continue
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('AdamW does not support sparse gradients')
                    grads.append(p.grad)

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = torch.zeros((1,), dtype=torch.double, device=p.device) \
                            if self.defaults['capturable'] else torch.tensor(0., dtype=torch.double)
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, dtype=torch.double, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.double, memory_format=torch.preserve_format)
                        if amsgrad:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, dtype=torch.double, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if amsgrad:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    state_steps.append(state['step'])

                adamw(params_with_grad,
                      grads,
                      exp_avgs,
                      exp_avg_sqs,
                      max_exp_avg_sqs,
                      state_steps,
                      amsgrad=amsgrad,
                      beta1=beta1,
                      beta2=beta2,
                      lr=group['lr'],
                      weight_decay=group['weight_decay'],
                      eps=group['eps'],
                      maximize=group['maximize'],
                      foreach=group['foreach'],
                      capturable=group['capturable'])

            return loss

        def load_state_dict(self, state_dict):
            r"""Loads the optimizer state.

            Args:
                state_dict (dict): optimizer state. Should be an object returned
                    from a call to :meth:`state_dict`.
            """
            # deepcopy, to be consistent with module API
            state_dict = deepcopy(state_dict)
            # Validate the state_dict
            groups = self.param_groups
            saved_groups = state_dict['param_groups']

            if len(groups) != len(saved_groups):
                raise ValueError("loaded state dict has a different number of "
                                 "parameter groups")
            param_lens = (len(g['params']) for g in groups)
            saved_lens = (len(g['params']) for g in saved_groups)
            if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
                raise ValueError("loaded state dict contains a parameter group "
                                 "that doesn't match the size of optimizer's group")

            # Update the state
            id_map = {old_id: p for old_id, p in
                      zip(chain.from_iterable((g['params'] for g in saved_groups)),
                          chain.from_iterable((g['params'] for g in groups)))}

            def cast(param, value, key=None):
                r"""Make a deep copy of value, casting all tensors to device of param."""
                keys_to_cast = ["step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"]
                if isinstance(value, torch.Tensor):
                    if param.is_floating_point():
                        if key in keys_to_cast:
                            return value.to(dtype=torch.double, device=param.device)
                        else:
                            return value.to(dtype=param.dtype, device=param.device)
                    else:
                        return value.to(device=param.device)
                elif isinstance(value, dict):
                    return {k: cast(param, v, key=k) for k, v in value.items()}
                elif isinstance(value, container_abcs.Iterable):
                    return type(value)(cast(param, v) for v in value)
                else:
                    return value

            # Copy state assigned to params (and cast tensors to appropriate types).
            # State that is not assigned to params is copied as is (needed for
            # backward compatibility).
            state = defaultdict(dict)
            for k, v in state_dict['state'].items():
                if k in id_map:
                    param = id_map[k]
                    state[param] = cast(param, v)
                else:
                    state[k] = v

            # Update parameter groups, setting their 'params' value
            def update_group(group, new_group):
                new_group['params'] = group['params']
                return new_group
            param_groups = [
                update_group(g, ng) for g, ng in zip(groups, saved_groups)]
            self.__setstate__({'state': state, 'param_groups': param_groups})



