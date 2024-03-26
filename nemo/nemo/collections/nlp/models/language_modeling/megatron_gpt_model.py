# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import itertools
from typing import Any, List, Optional, Union

import numpy as np
import torch
import time
import datetime
import math

from omegaconf import ListConfig
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import get_datasets_weights_and_num_samples
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import MemoryEfficientBlendableDataset
from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import build_train_valid_test_datasets
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset import GPTSFTChatDataset
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTDataset
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
    MegatronPretrainingRandomBatchSampler,
)
from nemo.collections.nlp.models.language_modeling.megatron.gpt_model import GPTModel
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.modules.common.megatron.module import Float16Module, param_is_not_shared
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_all_params_for_weight_decay_optimization,
    get_params_for_weight_decay_optimization,
)
from nemo.collections.nlp.modules.common.text_generation_utils import (
    generate,
    get_computeprob_response,
    get_default_length_params,
    get_default_sampling_params,
    megatron_gpt_generate,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import (
    LengthParam,
    OutputType,
    SamplingParam,
    TextGeneration,
)
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging
from transformers.utils import is_torch_tpu_available
import queue

try:
    from apex.transformer import parallel_state
    from apex.transformer.pipeline_parallel.schedules.common import build_model
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import (
        forward_backward_pipelining_without_interleaving,
    )
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_with_interleaving import (
        _forward_backward_pipelining_with_interleaving,
    )
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_no_pipelining import forward_backward_no_pipelining
    from apex.transformer.tensor_parallel.layers import param_is_not_tensor_parallel_duplicate
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

try:
    import transformer_engine

    HAVE_TE = True

except (ImportError, ModuleNotFoundError):
    HAVE_TE = False

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    HAVE_XLA = True
except (ImportError, ModuleNotFoundError):
    HAVE_XLA = False


class Throughput:
    def __init__(self, moving_avg_window_size):
        self.seqs_per_iteration = None #batch_size * world_size * grad_accum_usteps
        self.moving_avg_window_size = moving_avg_window_size
        self.moving_avg_window = queue.Queue()
        self.window_time = 0
        self.start_time = time.time()
        self.throughput_peak = 0
        self.throughput_sum = 0
        self.throughputs = []

    def set_seqs_per_iteration(self, batch_size, world_size, grad_accum_usteps):
        self.seqs_per_iteration = batch_size * world_size * grad_accum_usteps

    def get_throughput(self):
        step_time = time.time() - self.start_time
        self.start_time += step_time
        self.window_time += step_time
        self.moving_avg_window.put(step_time)
        window_size = self.moving_avg_window.qsize()
        if window_size > self.moving_avg_window_size:
            self.window_time -= self.moving_avg_window.get()
            window_size -= 1
        throughput = window_size * self.seqs_per_iteration / self.window_time
        self.throughputs.append(throughput)
        return throughput

class MegatronGPTModel(MegatronBaseModel, TextGeneration):
    """
    Megatron GPT pretraining
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        if not HAVE_APEX:
            raise ImportError(
                "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        # this prevents base constructor from initializing tokenizer
        self.tokenizer = None
        super().__init__(cfg, trainer=trainer, no_lm_init=True)
        
        self._validate_trainer()

        self.megatron_amp_o2 = cfg.get('megatron_amp_O2', False)

        if not self.megatron_amp_o2 and self.cfg.get('virtual_pipeline_model_parallel_size', None):
            raise ValueError('Virtual pipeline model parallel is only supported when using megatron_amp_O2')

        if self.trainer.precision == 32:
            self.autocast_dtype = torch.float
        elif self.trainer.precision == 16:
            self.autocast_dtype = torch.half
        elif self.trainer.precision == 'bf16':
            self.autocast_dtype = torch.bfloat16
        else:
            raise ValueError('precision must be in [32, 16, "bf16"]')

        self.transformer_engine = cfg.get('transformer_engine', False)

        # configuration used for inference
        self._inference_config = None

        self.wrap_with_zero = cfg.get('wrap_with_zero', False)
        self.log_parameter_norm = cfg.get('log_parameter_norm', False)
        self.log_gradient_norm = cfg.get('log_gradient_norm', False)
        self.save_logits = cfg.get('save_logits', False)
        self.save_logits_interval = cfg.get('save_logits_interval', 0)
    
        self.throughput = Throughput(10)
    def _build_model(self):
        # build_model returns a list of modules which are used for interleaved pipeline parallelism
        self.model = build_model(
            model_provider_func=self.model_provider_func,
            wrap_with_ddp=False,
            virtual_pipeline_model_parallel_size=self.cfg.get('virtual_pipeline_model_parallel_size', None),
        )
        logging.trace(f"In gpt_model._build_model() leave apex.transformer.pipeline_parallel.build_model", trace_type="recovery_time")

        # if we're not using interleaved, then self.model is a module.
        if self.cfg.get('virtual_pipeline_model_parallel_size', None) is None:
            self.model = self.model[0]

        if self.megatron_amp_o2:
            #We use XLA_USE_DOWNCAST=1 to downcast the model weights constructed in fp32
            #Not doing explicit model casting below
            return

    def set_inference_config(self, inference_config):
        self._inference_config = inference_config

    def get_inference_config(self):
        return self._inference_config

    def model_provider_func(self, pre_process, post_process):
        """Model depends on pipeline paralellism."""

        logging.trace(f"In gpt_model._build_model, enter GPTModel()", trace_type="recovery_time")
        model = GPTModel(
            vocab_size=self.padded_vocab_size,
            hidden_size=self.cfg.hidden_size,
            max_position_embeddings=self.cfg.max_position_embeddings,
            num_layers=self.cfg.num_layers,
            num_attention_heads=self.cfg.num_attention_heads,
            apply_query_key_layer_scaling=self.cfg.get('apply_query_key_layer_scaling', True),
            kv_channels=self.cfg.get('kv_channels', None),
            ffn_hidden_size=self.cfg.ffn_hidden_size,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
            init_method_std=self.cfg.get('init_method_std', 0.02),
            use_scaled_init_method=self.cfg.get('use_scaled_init_method', True),
            fp16_lm_cross_entropy=self.cfg.get('fp16_lm_cross_entropy', False),
            use_cpu_initialization=self.cfg.get('use_cpu_initialization', False),
            hidden_dropout=self.cfg.get('hidden_dropout', 0.1),
            attention_dropout=self.cfg.get('attention_dropout', 0.0),
            ffn_dropout=self.cfg.get('ffn_dropout', 0.0),
            precision=self.cfg.get('precision', 16),
            fp32_residual_connection=self.cfg.get('fp32_residual_connection', False),
            activations_checkpoint_granularity=self.cfg.get('activations_checkpoint_granularity', None),
            activations_checkpoint_method=self.cfg.get('activations_checkpoint_method', None),
            activations_checkpoint_num_layers=self.cfg.get('activations_checkpoint_num_layers', 1),
            activations_checkpoint_layers_per_pipeline=self.cfg.get(
                'activations_checkpoint_layers_per_pipeline', None
            ),
            normalization=self.cfg.get('normalization', 'layernorm'),
            layernorm_epsilon=self.cfg.get('layernorm_epsilon', 1e-5),
            onnx_safe=self.cfg.get('onnx_safe', False),
            bias_activation_fusion=self.cfg.get('bias_activation_fusion', True),
            bias_dropout_add_fusion=self.cfg.get('bias_dropout_add_fusion', True),
            share_embeddings_and_output_weights=self.cfg.get('share_embeddings_and_output_weights', True),
            position_embedding_type=self.cfg.get('position_embedding_type', 'learned_absolute'),
            rotary_percentage=self.cfg.get('rotary_percentage', 1.0),
            activation=self.cfg.get('activation', 'gelu'),
            bias=self.cfg.get('has_bias', True),
            transformer_block_type=self.cfg.get('transformer_block_type','pre_ln'),
            masked_softmax_fusion=self.cfg.get('masked_softmax_fusion', True),
            gradient_accumulation_fusion=self.cfg.get('gradient_accumulation_fusion', False),
            persist_layer_norm=self.cfg.get('persist_layer_norm', False),
            sequence_parallel=self.cfg.get('sequence_parallel', False),
            transformer_engine=self.cfg.get('transformer_engine', False),
            fp8=self.cfg.get('fp8', False),
            fp8_e4m3=self.cfg.get('fp8_e4m3', False),
            fp8_hybrid=self.cfg.get('fp8_hybrid', False),
            fp8_margin=self.cfg.get('fp8_margin', 0),
            fp8_interval=self.cfg.get('fp8_interval', 1),
            fp8_amax_history_len=self.cfg.get('fp8_amax_history_len', 1),
            fp8_amax_compute_algo=self.cfg.get('fp8_amax_compute_algo', 'most_recent'),
            use_emha=self.cfg.get('use_emha', False),
            save_logits=self.cfg.get('save_logits', False),
            position_interpolation_factor=self.cfg.get('positition_interpolation_factor', 1.0),
        )
        logging.trace(f"In gpt_model._build_model, leave GPTModel()", trace_type="recovery_time")
        return model

    def setup_optimizer_param_groups(self):
        """ModelPT override. Optimizer will get self._optimizer_param_groups"""
        if self.cfg.get('do_layer_norm_weight_decay', False):
            if isinstance(self.model, list):
                self._optimizer_param_groups = get_all_params_for_weight_decay_optimization(self.model)
            else:
                self._optimizer_param_groups = get_all_params_for_weight_decay_optimization([self.model])

        else:
            self._optimizer_param_groups = get_params_for_weight_decay_optimization(self.model)

    def configure_optimizers(self):

        if self.with_distributed_adam:

            # Disable overlapped grad sync for embedding grad when
            # pipeline parallelism is enabled
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                    if isinstance(self.model, list):
                        module = self.model[0]  # only the first virtual rank has the embeddings
                    else:
                        module = self.model
                    if module.share_token_embeddings:
                        param = module.word_embeddings_weight()
                        param._disable_greedy_grad_copy = not self.megatron_amp_o2
                        param._disable_overlap_grad_sync = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    if isinstance(self.model, list):
                        module = self.model[-1]  # only the last virtual rank has the embeddings
                    else:
                        module = self.model
                    if module.share_token_embeddings:
                        param = module.word_embeddings_weight()
                        param._disable_greedy_grad_copy = not self.megatron_amp_o2
                        param._disable_overlap_grad_sync = True

            # Disable overlapped grad sync for layer norm grads when
            # sequence parallelism is enabled
            for param in self.parameters():
                if getattr(param, 'sequence_parallel_enabled', False):
                    param._disable_greedy_grad_copy = not self.megatron_amp_o2
                    param._disable_overlap_grad_sync = True

        return super().configure_optimizers()

    def forward(self, tokens, text_position_ids, attention_mask, labels):
        output_tensor = self.model(tokens, text_position_ids, attention_mask, labels=labels)
        return output_tensor

    def _get_fwd_bwd_function(self):
        fwd_bwd_function = None
        if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
            if self.cfg.get('virtual_pipeline_model_parallel_size', None) is not None:
                fwd_bwd_function = _forward_backward_pipelining_with_interleaving
            else:
                fwd_bwd_function = forward_backward_pipelining_without_interleaving
        else:
            fwd_bwd_function = forward_backward_no_pipelining
        return fwd_bwd_function

    def training_step(self, batch, batch_idx):
        """
            Our dataloaders produce a micro-batch and then we fetch
            a number of microbatches depending on the global batch size and model parallel size
            from the dataloader to produce a list of microbatches.
            Batch should be a list of microbatches and those microbatches should on CPU.
            Microbatches are then moved to GPU during the pipeline.
            The list of microbatches is then piped through the pipeline using Apex fwd/bwd functions.
        """
        # Log start time of the training loop 
        start_time = time.time()

        # we zero grads here because we also call backward in the apex fwd/bwd functions
        self._optimizer.zero_grad()

        batch_for_pipeline = self.process_global_batch(batch)

        tensor_shape = [self.cfg.encoder_seq_length, self.cfg.micro_batch_size, self.cfg.hidden_size]

        # handle asynchronous grad reduction
        custom_sync_context_handler = None
        custom_grad_sync_func = None
        if self.with_distributed_adam:
            if self.megatron_amp_o2:
                # copy grads to main grad
                custom_sync_context_handler = lambda: self._optimizer.no_sync(greedy_grad_copy=True)
            else:
                # keep grad tensors around
                custom_sync_context_handler = lambda: self._optimizer.no_sync(greedy_grad_copy=False)
            custom_grad_sync_func = self.reduce_overlap_gradients
        else:
            #XLA doesn;t need an async context handler 
            custom_sync_context_handler = None

        # run forward and backwards passes for an entire global batch
        # we do this inside training_step to support pipeline parallelism
        fwd_bwd_function = self._get_fwd_bwd_function()
        losses_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(),
            batch=batch_for_pipeline,
            model=self.model,
            forward_only=False,
            tensor_shape=tensor_shape,
            dtype=self.autocast_dtype,
            grad_scaler=self.trainer.precision_plugin.scaler if self.cfg.precision == 16 else None,
            custom_sync_context_handler=custom_sync_context_handler,
            custom_grad_sync_func=custom_grad_sync_func,
            sequence_parallel_enabled=self.cfg.get('sequence_parallel', False),
            sync_batch_comm=self.cfg.get('sync_batch_comm', False),
            num_micro_batches_with_partial_activation_checkpoints=self.cfg.get(
                'num_micro_batches_with_partial_activation_checkpoints', None
            ),
        )

        xm.mark_step()
        # only the last stages of the pipeline return losses
        if losses_per_micro_batch:
            # average loss across micro batches
            loss_tensors_list = [average_losses_across_data_parallel_group([loss['mb_loss']]) for loss in losses_per_micro_batch]
            loss_tensor = torch.concat(loss_tensors_list)
            loss_mean = loss_tensor.mean()
        else:
            # loss_mean = torch.tensor(0.0).cuda()
            # NEURON
            loss_mean = torch.tensor(0.0, device=xm.xla_device())

        xm.mark_step()

        # when using sequence parallelism, the sequence parallel layernorm grads must be all-reduced
        if self.cfg.get('tensor_model_parallel_size', 1) > 1 and self.cfg.get('sequence_parallel', False):
            self.allreduce_sequence_parallel_gradients()

        xm.mark_step()

        if self.with_distributed_adam or self.wrap_with_zero:
            # gradients are reduced internally in distributed optimizer
            pass
        elif self.megatron_amp_o2:
            # when using pipeline parallelism grads must be all-reduced after the pipeline (not asynchronously)
            self.allreduce_gradients()
        else:
            # async grad allreduce is not currently implemented for O1/autocasting mixed precision training
            # so we all-reduce gradients after the pipeline
            self.allreduce_gradients()  # @sangkug we think this is causing memory to blow up (hurts perf)

        xm.mark_step()

        if self.cfg.get('pipeline_model_parallel_size', 1) > 1 and self.cfg.get('share_embeddings_and_output_weights', True):
            # when using pipeline parallelism the first and last stage must keep embeddings in sync
            self.allreduce_first_last_embeddings()

        xm.mark_step()
        ## logging
        # we can only log on one rank if it is rank zero so we broadcast from last rank
        # we can avoid this broadcast by updating the PTL log function to accept specific ranks
        torch.distributed.all_reduce(loss_mean, group=parallel_state.get_pipeline_model_parallel_group())

        xm.mark_step()

        if self.cfg.precision == 16:
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                def log_loss_scale(loss_scale):
                    self.log('loss_scale', loss_scale)
                xm.add_step_closure(log_loss_scale, (loss_scale,))

        lr = self._optimizer.param_groups[0]['lr']
        # TODO: make sure compute_consumed_samples works for pipeline parallelism
        consumed_samples = self.compute_consumed_samples(self.trainer.global_step - self.init_global_step)
        
        elapsed_time = time.time() - start_time
        if self.throughput.seqs_per_iteration is None:          
            self.throughput.set_seqs_per_iteration(self.cfg.get('micro_batch_size'), parallel_state.get_data_parallel_world_size(), get_num_microbatches())
        throughput = self.throughput.get_throughput()   
        throughput_peak = self.throughput.throughput_peak
        if throughput > throughput_peak:
            self.throughput.throughput_peak = throughput
        self.throughput.throughput_sum += throughput
        param_norm = None
        grad_norm = None
        if self.log_parameter_norm:
            param_norm = self.calculate_parameter_norm(self.parameters())
        if self.log_gradient_norm:
            grad_norm = self.calculate_gradient_norm(self.parameters())
        def _log_metrics(log_fn, loss_mean, lr, global_step, consumed_samples, grad_norm, param_norm, throughput, throughput_peak):
            log_fn('reduced_train_loss', loss_mean.detach().cpu(), prog_bar=True, rank_zero_only=True)
            log_fn('lr', lr, rank_zero_only=True)
            if grad_norm:
                log_fn('gradient_norm', grad_norm.detach().cpu(), prog_bar=True, rank_zero_only=True)
            if param_norm:
                log_fn('parameter_norm', param_norm.detach().cpu(), prog_bar=True, rank_zero_only=True)
            log_fn('global_step', global_step, prog_bar=True, rank_zero_only=True)
            log_fn('consumed_samples', consumed_samples, prog_bar=True, rank_zero_only=True)
            log_fn('throughput', throughput, prog_bar=True, rank_zero_only=True)
            log_fn('throughput_peak', throughput_peak, prog_bar=True, rank_zero_only=True)
        xm.add_step_closure(_log_metrics, (self.log, loss_mean, lr, float(self.trainer.global_step), float(consumed_samples), grad_norm, param_norm, float(throughput), float(throughput_peak)))

        return loss_mean

    def on_train_batch_end(self, *args, **kwargs):
        super().on_train_batch_end(*args, **kwargs)
        xm.mark_step()

    def calculate_gradient_norm(self, parameters, norm_type=2):
        """Calculate gradient norms across model parallel ranks
        Arguments:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            norm_type (float or int): type of the used p-norm. Can be ``'math.inf'`` for
                infinity norm.
        Returns:
            Total norm of the gradients (viewed as a single vector).
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        # Filter parameters based on:
        #   - grad should not be none
        #   - parameter should not be shared
        #   - should not be a replica due to tensor model parallelism
        grads_for_norm = []
        for param in parameters:
            grad_not_none = param.grad is not None
            is_not_shared = param_is_not_shared(param)
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if grad_not_none and is_not_shared and is_not_tp_duplicate:
                grads_for_norm.append(param.grad.detach())

        if not grads_for_norm:
            raise ValueError(f"No grads found on {xm.get_ordinal()}")

        # Norm parameters.
        norm_type = float(norm_type)
        total_norm = torch.tensor([float(0.0)], device=xm.xla_device())

        # Calculate norm.
        if norm_type == math.inf:
            total_norm = max(grad.abs().max() for grad in grads_for_norm)
            total_norm = torch.tensor([float(total_norm)], device=xm.xla_device())
            # Take max across all model-parallel TPUs.
            torch.distributed.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_tensor_model_parallel_group()
            )
            torch.distributed.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_pipeline_model_parallel_group()
            )
            total_norm = total_norm[0]
        else:
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm ** norm_type

            # Sum across all model-parallel TPUs.
            torch.distributed.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_tensor_model_parallel_group()
            )
            torch.distributed.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_pipeline_model_parallel_group()
            )
            total_norm = torch.pow(total_norm, 1.0 / norm_type)
        return total_norm

    def calculate_parameter_norm(self, parameters, norm_type=2):
        """Calculate parameter norms across model parallel ranks
        Arguments:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor
            norm_type (float or int): type of the used p-norm. Can be ``'math.inf'`` for
                infinity norm.
            Total norm of the parameters (viewed as a single vector).
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        # Norm parameters.
        norm_type = float(norm_type)
        total_norm = torch.tensor([float(0.0)], device=xm.xla_device())
        params_to_norm = []

        # Filter parameters based on:
        #   - parameter should not be shared
        #   - should not be a replica due to tensor model parallelism
        for param in parameters:
            is_not_shared = param_is_not_shared(param)
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if is_not_shared and is_not_tp_duplicate:
                params_to_norm.append(param)

        # Calculate norm.
        if norm_type == math.inf:
            total_norm = max(torch.abs(param) for param in params_to_norm)
            total_norm = torch.tensor([float(total_norm)], device=xm.xla_device())
            # Take max across all model-parallel TPUs.
            torch.distributed.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_tensor_model_parallel_group()
            )
            torch.distributed.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_pipeline_model_parallel_group()
            )
            total_norm = total_norm[0]
        else:
            for param in params_to_norm:
                param_norm = torch.norm(param, norm_type)
                total_norm += param_norm**norm_type
            # Sum across all model-parallel TPUs.
            torch.distributed.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_tensor_model_parallel_group()
            )
            torch.distributed.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_pipeline_model_parallel_group()
            )
            total_norm = torch.pow(total_norm, 1.0 / norm_type)
        return total_norm

    def backward(self, *args, **kwargs):
        """ LightningModule hook to do backward.
            We want this to do nothing since we run backward in the fwd/bwd functions from apex.
            No need to call it here.
        """
        return

    def optimizer_zero_grad(self, *args, **kwargs):
        """ LightningModule hook to zero grad.
            We want this to do nothing as we are zeroing grads during the training_step.
        """
        return

    def _append_sequence_parallel_module_grads(self, module, grads):
        """ Helper method for allreduce_sequence_parallel_gradients"""

        for param in module.parameters():
            if getattr(self, 'transformer_engine', False):
                sequence_parallel_param = getattr(param, 'sequence_parallel', False)
            else:
                sequence_parallel_param = getattr(param, 'sequence_parallel_enabled', False)
            if sequence_parallel_param:
                #megatron_amp_o2 also uses model gradients 
                grad = param.grad
                grads.append(grad.data)

    def allreduce_sequence_parallel_gradients(self):
        """ All-reduce layernorm parameters across model parallel nodes when sequence parallelism is used.
            Modified from megatron-lm:
            https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/blob/3f91f09bb2ab32f9904b47f46f19d2fc3f518ed8/megatron/training.py#L425
        """

        grads = []
        if isinstance(self.model, list):
            for module in self.model:
                self._append_sequence_parallel_module_grads(module, grads)
        else:
            self._append_sequence_parallel_module_grads(self.model, grads)

        coalesced = torch._utils._flatten_dense_tensors(grads)
        torch.distributed.all_reduce(coalesced, group=parallel_state.get_tensor_model_parallel_group())
        for buf, synced in zip(grads, torch._utils._unflatten_dense_tensors(coalesced, grads)):
            buf.copy_(synced)

    def allreduce_first_last_embeddings(self):

        # Modified from megatron-lm: https://github.com/NVIDIA/Megatron-LM/blob/d41696840ed0a7edb7e0499eb82a48ae112d9bb3/megatron/training.py#L407
        # All-reduce word_embeddings' grad across first and last stages to ensure
        # that word_embeddings parameters stay in sync.
        # This should only run for models that support pipelined model parallelism
        # (BERT and GPT-2).
        if parallel_state.get_pipeline_model_parallel_world_size() > 1 and (
            parallel_state.is_pipeline_first_stage(ignore_virtual=True)
            or parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        ):
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                if isinstance(self.model, list):
                    module = self.model[0]  # only the first virtual rank has the embeddings
                else:
                    module = self.model
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                if isinstance(self.model, list):
                    module = self.model[-1]  # only the last virtual rank has the embeddings
                else:
                    module = self.model
            if module.share_token_embeddings:
                word_embeddings_weight = module.word_embeddings_weight()

                #megatron_amp_o2 also uses model gradients
                grad = word_embeddings_weight.grad
                torch.distributed.all_reduce(grad, group=parallel_state.get_embedding_group())

    def get_forward_output_and_loss_func(self, validation_step=False, all_reduce_losses=False):
        def fwd_output_and_loss_func(batch, model, checkpoint_activations_all_layers=None):
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                # batch = [x.cuda(non_blocking=True) for x in batch]
                tokens, labels, loss_mask, _, position_ids = batch
            else:
                # GPT3 uses only causal mask, which doesn't need attention mask
                if parallel_state.is_pipeline_first_stage():
                    # Fist pipeline stage needs only the tokens and position_ids
                    # tokens = batch[0].cuda(non_blocking=True)
                    # position_ids = batch[4].cuda(non_blocking=True)
                    tokens = batch[0]
                    position_ids = batch[4]
                    labels, loss_mask = None, None
                elif parallel_state.is_pipeline_last_stage():
                    # Last pipeline stage needs only the labels and loss_mask
                    # labels = batch[1].cuda(non_blocking=True)
                    # loss_mask = batch[2].cuda(non_blocking=True)
                    labels = batch[1]
                    loss_mask = batch[2]
                    tokens, position_ids = None, None
                else:
                    # Intermediate pipeline stage doesn't need any inputs
                    tokens, labels, loss_mask, position_ids = None, None, None, None

            attention_mask = batch[3][0:1]
            if is_torch_tpu_available():
                attention_mask = None
            logits = None
            if parallel_state.is_pipeline_last_stage():
                # Only the last PP stage has the logits
                output_tensor, logits = model(
                    tokens,
                    position_ids,
                    attention_mask,
                    labels,
                    checkpoint_activations_all_layers=checkpoint_activations_all_layers,
                )
            else:
                output_tensor = model(
                    tokens,
                    position_ids,
                    attention_mask,
                    labels,
                    checkpoint_activations_all_layers=checkpoint_activations_all_layers,
                )
            if self.save_logits:
                def save_logits(logits):
                    # Save logits on tensor parallel rank zero, data parallel rank zero and last pipeline parallel stage
                    if parallel_state.get_tensor_model_parallel_rank() == 0 and parallel_state.is_pipeline_last_stage()\
                            and parallel_state.get_data_parallel_rank() == 0:
                        if self.trainer.global_step % self.save_logits_interval == 0:
                            np.save(f"logits-{self.trainer.global_step}.npy", logits.detach().cpu().numpy())
                xm.add_step_closure(save_logits, (logits,))

            def loss_func(output_tensor):
                loss_for_mb = self.loss_func(loss_mask, output_tensor)
                if validation_step and not self.cfg.data.get('validation_drop_last', True):
                    num_valid_samples_in_mb = int(loss_mask.sum() / loss_mask.numel() * loss_mask.shape[0])
                    loss_sum_for_mb = num_valid_samples_in_mb * loss_for_mb
                    loss_sum_and_mb_size_all_gpu = torch.cat(
                        [
                            loss_sum_for_mb.clone().detach().view(1),
                            torch.tensor([num_valid_samples_in_mb]).cuda().clone().detach(),
                        ]
                    )
                    # Could potentially reduce num_valid_samples_in_microbatch and use that to aggregate instead of len(self._validation_ds)
                    torch.distributed.all_reduce(
                        loss_sum_and_mb_size_all_gpu, group=parallel_state.get_data_parallel_group()
                    )
                    return loss_for_mb, {'loss_sum_and_mb_size': loss_sum_and_mb_size_all_gpu.detach()}
                else:
                    if all_reduce_losses:
                        reduced_loss = average_losses_across_data_parallel_group([loss_for_mb])
                        return loss_for_mb, {'avg': reduced_loss.detach()}
                    else:
                        return loss_for_mb, {'mb_loss': loss_for_mb.detach()}

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def get_forward_output_only_func(self):
        def fwd_output_only_func(batch, model):
            extra_arg = {}
            if len(batch) == 3:
                batch = [x.cuda() for x in batch]
                tokens, attention_mask, position_ids = batch
                attention_mask = attention_mask[0:1]
            else:
                (
                    tokens,
                    attention_mask,
                    position_ids,
                    set_inference_key_value_memory,
                    inference_max_sequence_len,
                ) = batch
                tokens = tokens.cuda()
                attention_mask = attention_mask.cuda()
                position_ids = position_ids.cuda()
                attention_mask = attention_mask[0:1]
                extra_arg['set_inference_key_value_memory'] = set_inference_key_value_memory[0].item()
                extra_arg['inference_max_sequence_len'] = inference_max_sequence_len[0].item()
            if is_torch_tpu_available():
                attention_mask = None
            output_tensor = model(tokens, position_ids, attention_mask, **extra_arg)

            def id_func(output_tensor):
                return output_tensor, {'logits': output_tensor}

            return output_tensor, id_func

        return fwd_output_only_func

    def validation_step(self, batch, batch_idx):
        """
            Our dataloaders produce a micro-batch and then we fetch
            a number of microbatches depending on the global batch size and model parallel size
            from the dataloader to produce a list of microbatches.
            The list of microbatches is then piped through the pipeline using Apex fwd/bwd functions.
        """

        batch_for_pipeline = self.process_global_batch(batch)
        tensor_shape = [self.cfg.encoder_seq_length, self.cfg.micro_batch_size, self.cfg.hidden_size]

        # run forward passes for an entire global batch
        # we do this inside validation_step to support pipeline parallelism
        fwd_bwd_function = self._get_fwd_bwd_function()

        losses_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(validation_step=True),
            batch=batch_for_pipeline,
            model=self.model,
            forward_only=True,
            tensor_shape=tensor_shape,
            dtype=self.autocast_dtype,
            sequence_parallel_enabled=self.cfg.get('sequence_parallel', False),
            sync_batch_comm=self.cfg.get('sync_batch_comm', False),
        )
        xm.mark_step()
        # only the last stage of the pipeline returns losses
        if losses_per_micro_batch:
            if self.cfg.data.get('validation_drop_last', True):
                # average loss across micro batches
                loss_tensors_list = [average_losses_across_data_parallel_group([loss['mb_loss']]) for loss in losses_per_micro_batch]
                return torch.concat(loss_tensors_list).mean()
            else:
                # Get the total loss since micro batches sizes are not uniform
                loss_sum_tensors_list = [
                    loss_sum['loss_sum_and_mb_size']
                    for loss_sum in losses_per_micro_batch
                    if loss_sum['loss_sum_and_mb_size'][1] > 0
                ]
                loss_sum = (
                    torch.vstack(loss_sum_tensors_list).sum(axis=0)
                    if len(loss_sum_tensors_list) > 0
                    else torch.tensor([0.0, 0.0], xm.xla_device())
                )
                return loss_sum
        else:
            # we're not on the last pipeline stage so no losses
            return []

    def validation_epoch_end(self, outputs):
        if parallel_state.is_pipeline_last_stage():
            # only the last pipeline parallel stages return loss with their batch size
            if self.cfg.data.get('validation_drop_last', True):
                averaged_loss = torch.stack(outputs).mean()
            else:
                # Compute the avg loss by total_loss across all samples / total number of samples
                total_loss_and_total_samples = torch.vstack(outputs).sum(axis=0)
                avg_loss = total_loss_and_total_samples[0] / total_loss_and_total_samples[1]
                averaged_loss = avg_loss.type(torch.float32)
        else:
            averaged_loss = torch.tensor(0.0, dtype=torch.float32, device=xm.xla_device())

        # we can only log on one rank if it is rank zero so we all_reduce from last rank
        # (effectively a broadcast since we are all_reducing with a zero tensor)
        torch.distributed.all_reduce(averaged_loss, group=parallel_state.get_pipeline_model_parallel_group())

        def _log_val_loss(log_fn, loss):
            log_fn('val_loss', loss.cpu(), prog_bar=True, on_step=True, rank_zero_only=True, batch_size=1)
        xm.add_step_closure(_log_val_loss, (self.log, averaged_loss.detach(),))

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        averaged_loss = average_losses_across_data_parallel_group(outputs)
        logging.info(f'test_loss: {averaged_loss[0]}')

    def loss_func(self, loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        # TODO: add nemo version here
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()  # sequence level nll
        return loss

    def process_global_batch(self, global_batch, global_batch_size=None):
        """ Prepares the global batch for apex fwd/bwd functions.
            Global batch is a list of micro batches.
        """
        return [
            global_batch["tokens"],
            global_batch["labels"],
            global_batch["loss_mask"],
            global_batch["attention_mask"],
            global_batch["position_ids"],
        ]

    def build_train_valid_test_datasets(self):
        logging.info('Building GPT datasets.')
        if self.trainer.limit_val_batches > 1.0 and isinstance(self.trainer.limit_val_batches, float):
            raise ValueError("limit_val_batches must be an integer or float less than or equal to 1.0.")
        global_batch_size = self.cfg.global_batch_size
        max_train_steps = self.trainer.max_steps
        eval_iters = (max_train_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        test_iters = self.trainer.limit_test_batches

        train_valid_test_num_samples = [
            max_train_steps * global_batch_size,
            eval_iters * global_batch_size,
            test_iters * global_batch_size,
        ]

        if self.trainer.limit_val_batches <= 1.0 and isinstance(self.trainer.limit_val_batches, float):
            train_valid_test_num_samples[
                1
            ] = 1  # This is to make sure we only have one epoch on every validation iteration

        self._train_ds, self._validation_ds, self._test_ds = build_train_valid_test_datasets(
            cfg=self.cfg,
            trainer=self.trainer,
            data_prefix=self.cfg.data.data_prefix,
            data_impl=self.cfg.data.data_impl,
            splits_string=self.cfg.data.splits_string,
            train_valid_test_num_samples=train_valid_test_num_samples,
            seq_length=self.cfg.data.seq_length,
            seed=self.cfg.seed,
            skip_warmup=self.cfg.data.get('skip_warmup', True),
            tokenizer=self.tokenizer,
        )
        if self._train_ds is not None:
            logging.info(f'Length of train dataset: {len(self._train_ds)}')
        if self._validation_ds is not None:
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        if self._test_ds is not None:
            logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building GPT datasets.')

        return self._train_ds, self._validation_ds, self._test_ds

    def build_pretraining_data_loader(
        self, dataset, consumed_samples, dataset_type=None, drop_last=True, pad_samples_to_global_batch_size=False
    ):
        """Buld dataloader given an input dataset."""

        logging.info(f'Building dataloader with consumed samples: {consumed_samples}')
        # Megatron sampler
        if hasattr(self.cfg.data, 'dataloader_type') and self.cfg.data.dataloader_type is not None:
            if self.cfg.data.dataloader_type == 'single':
                batch_sampler = MegatronPretrainingBatchSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    global_batch_size=self.cfg.global_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=drop_last,
                    pad_samples_to_global_batch_size=pad_samples_to_global_batch_size,
                )
            elif self.cfg.data.dataloader_type == 'cyclic':
                batch_sampler = MegatronPretrainingRandomBatchSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    global_batch_size=self.cfg.global_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=self.cfg.get('drop_last', True),
                    pad_samples_to_global_batch_size=pad_samples_to_global_batch_size,
                )
            else:
                raise ValueError('cfg.data.dataloader_type must be "single" or "cyclic"')
        else:
            raise ValueError('cfg.data.dataloader_type not found. Must be "single" or "cyclic"')
        

        return torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=self.cfg.data.num_workers, pin_memory=False, prefetch_factor=1
        )


    def setup(self, stage=None):
        """ PTL hook that is executed after DDP spawns.
            We setup datasets here as megatron datasets require DDP to instantiate.
            See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#setup for more information.
        Args:
            stage (str, optional): Can be 'fit', 'validate', 'test' or 'predict'. Defaults to None.
        """
        logging.trace(f"Enter MegatronGPTModel.setup()", trace_type="recovery_time")
        if not hasattr(self, 'server'):
            import torch_xla.debug.profiler as xp
            (dp_rank, tp_rank, pp_rank, vp_rank) = parallel_state.get_rank_info()
            if dp_rank == 0 and tp_rank == 0: # start one for each of the PP stages!
                port = 9000 + pp_rank
                self.server = xp.start_server(port)
                logging.info(f"Started profiling server for dp_rank={dp_rank}, tp_rank={tp_rank}, pp_rank={pp_rank}, vp_rank={vp_rank} on port:{port}")


        self.initialize_model_parallel_for_nemo(main_proc=False)
        logging.trace(f"In gpt_model.setup() enter _build_model()", trace_type="recovery_time")
        self._build_model()
        logging.trace(f"In gpt_model.setup() leave _build_model()", trace_type="recovery_time")

        # log number of parameters
        if isinstance(self.model, list):
            num_parameters_on_device = sum(
                [sum([p.nelement() for p in model_module.parameters()]) for model_module in self.model]
            )
            if parallel_state.get_pipeline_model_parallel_world_size() > 1 and parallel_state.is_pipeline_last_stage(
                ignore_virtual=True
            ):
                # substract the embedding weights on the last virtual stage
                num_word_embedding_parameters = sum([p.nelement() for p in self.model[-1].word_embeddings_weight()])
                num_parameters_on_device -= num_word_embedding_parameters
        else:
            self.model.to(xm.xla_device())
            num_parameters_on_device = sum([p.nelement() for p in self.model.parameters()])

            if parallel_state.get_pipeline_model_parallel_world_size() > 1 and \
               parallel_state.is_pipeline_last_stage(ignore_virtual=True) and \
               self.cfg.get('share_embeddings_and_output_weights', True):
                # substract the embedding weights on the last stage
                num_word_embedding_parameters = sum([p.nelement() for p in self.model.word_embeddings_weight()])

                num_parameters_on_device -= num_word_embedding_parameters

        # to be summed across data parallel group
        # total_num_parameters = torch.tensor(num_parameters_on_device).cuda()
        # total_num_parameters = torch.tensor(num_parameters_on_device, device=xm.xla_device())

        # torch.distributed.all_reduce(total_num_parameters, group=parallel_state.get_model_parallel_group())

        
        # #########NEURON FIX this: total_num_parameters causes compile#########
        # logging.info(
        #     f'Pipeline model parallel rank: {parallel_state.get_pipeline_model_parallel_rank()}, '
        #     f'Tensor model parallel rank: {parallel_state.get_tensor_model_parallel_rank()}, '
        #     f'Number of model parameters on device: {num_parameters_on_device:.2e}. '
        #     f'Total number of model parameters: {total_num_parameters:.2e}.'
        # )

        resume_checkpoint_path = self.trainer._checkpoint_connector.resume_from_checkpoint_fit_path
        if resume_checkpoint_path:
            init_consumed_samples = self._extract_consumed_samples_from_ckpt(resume_checkpoint_path)
        else:
            init_consumed_samples = 0
        self.init_consumed_samples = init_consumed_samples
        self.init_global_step = self.trainer.global_step

        if stage == 'predict':
            return
        elif self.cfg.data.get("fine_tuning", False):
            self.build_sft_dataets()
            self.setup_training_data(self.cfg.data)
            self.setup_validation_data(self.cfg.data)
            self.setup_test_data(self.cfg.data)
        else:
            # TODO: consider adding a ModelPT guard to check if model is being restored.
            # allowing restored models to optionally setup datasets
            self.build_train_valid_test_datasets()
            self.setup_training_data(self.cfg.data)
            self.setup_validation_data(self.cfg.data)
            self.setup_test_data(self.cfg.data)

        # when using pipeline model parallel the final stage need to initialize word embeddings
        if parallel_state.get_pipeline_model_parallel_world_size() > 1 and self.cfg.get('share_embeddings_and_output_weights', True):
            if isinstance(self.model, list):
                for i, module in enumerate(self.model):
                    parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                    module.sync_initial_word_embeddings()
                parallel_state.set_virtual_pipeline_model_parallel_rank(0)
            else:
                self.model.sync_initial_word_embeddings()
        if self.cfg.get('transformer_engine', False):
            self.setup_transformer_engine_tp_groups()

        logging.trace(f"Leave MegatronGPTModel.setup()", trace_type="recovery_time")

    def build_sft_dataets(self):
        if self.trainer.limit_val_batches > 1.0 and isinstance(self.trainer.limit_val_batches, float):
            raise ValueError("limit_val_batches must be an integer or float less than or equal to 1.0.")

        if hasattr(self.cfg.data, 'validation_ds') and self.cfg.data.validation_ds.get('file_names', None) is not None:
            logging.info('Building GPT SFT validation datasets.')
            # Wrap this in a list since the general finetuning parent class supports multi-validation.
            self._validation_ds = self._build_dataset(self.cfg.data.validation_ds, is_train=False, is_validation=True)
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')

        if hasattr(self.cfg.data, 'test_ds') and self.cfg.data.test_ds.get('file_names', None) is not None:
            logging.info('Building GPT SFT test datasets.')
            # Wrap this in a list since the general finetuning parent class supports multi-validation.
            self._test_ds = self._build_dataset(self.cfg.data.test_ds, is_train=False)
            logging.info(f'Length of test dataset: {len(self._test_ds[0])}')

        logging.info('Building GPT SFT training datasets.')
        self._train_ds = self._build_dataset(self.cfg.data.train_ds)
        logging.info(f'Length of train dataset: {len(self._train_ds)}')

    def _build_dataset(self, data_cfg, is_train=True, is_validation=False):
        datasets = []
        # Determine if we are using a single dataset or a list of datasets.
        is_list_config = isinstance(data_cfg.file_names, ListConfig)
        if not is_list_config:
            raise ValueError(f"SFT train/validation datasets must be provided as a list of individual JSONL files.")
        max_steps = 0
        if is_train:
            max_steps = self.trainer.max_steps
        elif is_validation:
            max_steps = (self.trainer.max_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches

        # Construct the data prefix list for `get_datasets_weights_and_num_samples()`
        # that is of the format [weight1,file_name1,weight2,file_name2,...]
        if data_cfg.concat_sampling_probabilities is None or not isinstance(
                data_cfg.concat_sampling_probabilities, ListConfig
        ):
            raise ValueError(
                 (
                    f"concat_sampling_probabilities must be a ListConfig with the same number of files in file_names."
                    f"Found: {data_cfg.concat_sampling_probabilities}"
                )
            )

        if len(data_cfg.get('concat_sampling_probabilities', None)) != len(data_cfg.file_names):
            raise ValueError(
                (
                    f"concat_sampling_probabilities must be of the same size as file_names.",
                       f"Provided size {len(data_cfg.concat_sampling_probabilities)}, number of datasets {len(data_cfg.file_names)}",
                )
            )

        data_prefix = []
        for weight, prefix in zip(data_cfg.concat_sampling_probabilities, data_cfg.file_names):
            data_prefix.append(weight)
            data_prefix.append(prefix)

        num_train_samples = [max_steps * self.cfg.global_batch_size]
        _, _, num_train_samples_per_dataset = get_datasets_weights_and_num_samples(data_prefix, num_train_samples)
        num_train_samples_after_blend = sum([x[0] for x in num_train_samples_per_dataset])

        # Check dataset max_seq_length and max_position_embeddings size
        if (
                self.cfg.get('position_embedding_type', None) in [None, 'learned_absolute']
                and data_cfg.max_seq_length > self.cfg.max_position_embeddings
        ):
            logging.warning(
                f"Set dataset max_seq_length to max_position_embeddings {self.cfg.max_position_embeddings} if using learned_absolute position embedding"
            )
            data_cfg.max_seq_length = self.cfg.max_position_embeddings

        for file_path, num_samples in zip(data_cfg.file_names, num_train_samples_per_dataset):
            if self.cfg.data.get("chat", False):
                dataset_cls = GPTSFTChatDataset
            else:
                dataset_cls = GPTSFTDataset

            dataset = dataset_cls(
                file_path=file_path,
                tokenizer=self.tokenizer,
                max_seq_length=data_cfg.max_seq_length,
                min_seq_length=data_cfg.min_seq_length,
                add_bos=data_cfg.get('add_bos', False),
                add_eos=data_cfg.get('add_eos', True),
                add_sep=data_cfg.get('add_sep', False),
                sep_id=None,
                max_num_samples=num_samples[0],
                seed=data_cfg.get('seed', 1234),
                label_key=data_cfg.get('label_key', 'answer'),
                answer_only_loss=self.cfg.get('answer_only_loss', True),
                truncation_field=data_cfg.get('truncation_field', 'text'),
                pad_to_max_length=data_cfg.get('pad_to_max_length', True),
                index_mapping_dir=data_cfg.get('index_mapping_dir', None),
                prompt_template=data_cfg.get('prompt_template', None),
                virtual_tokens=0,
                tokens_to_generate=data_cfg.get(
                    'tokens_to_generate', 0
                ),  # used at inference time to allocate tensor positions for tokens that will be generated by inf procedure.
                memmap_workers=data_cfg.get(
                    'memmap_workers', None
                ),  # used to set num. of workers to create the memmap index files
                hf_dataset=data_cfg.get(
                    'hf_dataset', False
                ),  # Whether to load the json file with the HuggingFace dataset. otherwise, will load the jsonl file with the JSONLMemMapDataset.
                truncation_method=data_cfg.get(
                    'truncation_method', 'right'
                ),  # used to choose truncation method. Options: ['random', 'left', 'right']
            )
            datasets.append(dataset)

        dataset = MemoryEfficientBlendableDataset(
            datasets=datasets, weights=data_cfg.concat_sampling_probabilities, size=num_train_samples_after_blend
        )
        return dataset

    def setup_training_data(self, cfg):
        if hasattr(self, '_train_ds'):
            consumed_samples = self.compute_consumed_samples(0)
            logging.info(
                f'Setting up train dataloader with len(len(self._train_ds)): {len(self._train_ds)} and consumed samples: {consumed_samples}'
            )
            if self.cfg.data.get("fine_tuning", False):
                self._train_dl = self.build_fine_tuning_data_loader(self._train_ds, cfg.train_ds)
            else:
                self._train_dl = self.build_pretraining_data_loader(self._train_ds, consumed_samples)

    def setup_validation_data(self, cfg):
        if hasattr(self, '_validation_ds'):
            consumed_samples = 0
            logging.info(
                f'Setting up validation dataloader with len(len(self._validation_ds)): {len(self._validation_ds)} and consumed samples: {consumed_samples}'
            )

            drop_last = True
            if not self.cfg.data.get('validation_drop_last', True):
                logging.info(f'Drop last in validation dataset is set to False')
                drop_last = False
            pad_samples_to_global_batch_size = False
            if self.cfg.data.get('pad_samples_to_global_batch_size', False):
                logging.info('pad_samples_to_global_batch_size set to True')
                pad_samples_to_global_batch_size = True
            if self.cfg.data.get("fine_tuning", False):
                self._validation_dl = self.build_fine_tuning_data_loader(self._validation_ds, cfg.validation_ds)
            else:
                self._validation_dl = self.build_pretraining_data_loader(
                    self._validation_ds, consumed_samples, "validation", drop_last, pad_samples_to_global_batch_size
                )

    def setup_test_data(self, cfg):
        if hasattr(self, '_test_ds'):
            consumed_samples = 0
            logging.info(
                f'Setting up test dataloader with len(len(self._test_ds)): {len(self._test_ds)} and consumed samples: {consumed_samples}'
            )
            if self.cfg.data.get("fine_tuning", False):
                self._test_dl = self.build_fine_tuning_data_loader(self._test_ds, cfg.test_ds)
            else:
                self._test_dl = self.build_pretraining_data_loader(self._test_ds, consumed_samples)

    def build_fine_tuning_data_loader(self, dataset, data_cfg, consumed_samples=0):
        """Buld fine tuning dataloader given an input dataset."""

        logging.info(f'Building fine tuning dataloader with consumed samples: {consumed_samples}')

        collate_fn = dataset.datasets[0].collate_fn

        batch_sampler = MegatronPretrainingBatchSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=self.cfg.micro_batch_size,
            global_batch_size=self.cfg.global_batch_size,
            data_parallel_rank=parallel_state.get_data_parallel_rank(),
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
            drop_last=True,
            pad_samples_to_global_batch_size=False,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=data_cfg.num_workers,
            pin_memory=False,
            prefetch_factor=1
        )

    def generate(
        self,
        inputs: Union[List[str], torch.Tensor, List[dict]],
        length_params: LengthParam,
        sampling_params: SamplingParam = None,
    ) -> OutputType:

        # check whether the DDP is initialized
        if parallel_state.is_unitialized():

            def dummy():
                return

            if self.trainer.strategy.launcher is not None:
                self.trainer.strategy.launcher.launch(dummy, trainer=self.trainer)
            self.trainer.strategy.setup_environment()

            if self.cfg.get('transformer_engine', False):
                self.setup_transformer_engine_tp_groups()

        # set the default sampling params if it is None.
        # default do greedy sampling
        if sampling_params is None:
            sampling_params = get_default_sampling_params()

        # set the default length params if it is None.
        # default do greedy sampling
        if length_params is None:
            length_params = get_default_length_params()

        return megatron_gpt_generate(self.cuda(), inputs, self.tokenizer, length_params, sampling_params)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        inference_config = self.get_inference_config()
        if inference_config is None:
            return None
        else:
            # need to overwrite some configuration, make it immutable
            inference_config = inference_config.copy()
            compute_logprob = inference_config['compute_logprob']
            if compute_logprob:
                del inference_config['compute_logprob']
                inference_config['inputs'] = batch
                inference_config['tokens_to_generate'] = 1
                inference_config['all_probs'] = True
                inference_config["add_BOS"] = False
                inference_config['greedy'] = True
                response = generate(self, **inference_config)
                compute_prob_response = get_computeprob_response(self.tokenizer, response, batch)
                return compute_prob_response
            else:
                del inference_config['compute_logprob']
                inference_config['inputs'] = batch
                return generate(self, **inference_config)

    def list_available_models(self):
        return None

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        """ PTL hook: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#transfer-batch-to-device
            When using pipeline parallelism, we need the global batch to remain on the CPU,
            since the memory overhead will be too high when using a large number of microbatches.
            Microbatches are transferred from CPU to GPU inside the pipeline.
        """
        return batch

    def _validate_trainer(self):
        """ Certain trainer configurations can break training.
            Here we try to catch them and raise an error.
        """
        if self.trainer.accumulate_grad_batches > 1:
            raise ValueError(
                f'Gradient accumulation is done within training_step. trainer.accumulate_grad_batches must equal 1'
            )

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        result = []
        result.append(
            PretrainedModelInfo(
                pretrained_model_name="megatron_gpt_345m",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/megatron_gpt_345m/versions/1/files/megatron_gpt_345m.nemo",
                description="345M parameter GPT generative Megatron model.",
            )
        )
        return result

    def _set_tp_groups(self, module):
        """ Helper method to set tp groups for transformer engine"""

        if self.cfg.get('transformer_engine', False):
            logging.info(f'Setting up transformer engine modules for tensor parallelism.')
            if self.cfg.get('megatron_amp_O2', 'False'):
                # when using O2 additional module key is added that casts the weights
                for layer in module.module.language_model.encoder.layers:
                    layer.set_tensor_parallel_group(parallel_state.get_tensor_model_parallel_group())

            else:
                for layer in module.language_model.encoder.layers:
                    layer.set_tensor_parallel_group(parallel_state.get_tensor_model_parallel_group())

    def setup_transformer_engine_tp_groups(self):
        """ This should be called after model parallel groups have been initialized
            and only needs to be called when using Transformer Engine.
        """
        if isinstance(self.model, list):
            for module in self.model:
                self._set_tp_groups(module)
        else:
            self._set_tp_groups(self.model)

    def on_save_checkpoint(self, checkpoint) -> None:
        """LightningModule hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-save-checkpoint
        """
        if isinstance(self.model, list):
            for i in range(len(self.model)):
                parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                checkpoint[f'model{i}'] = self.model[i].module.state_dict_for_save_checkpoint()
            parallel_state.set_virtual_pipeline_model_parallel_rank(0)

    def on_load_checkpoint(self, checkpoint) -> None:
        """LightningModule hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-load-checkpoint
        """
        if isinstance(self.model, list):
            for i in range(len(self.model)):
                parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                self.model[i].module.load_state_dict(checkpoint[f'model{i}'], strict=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(0)

    def parameters(self):
        if isinstance(self.model, list):
            return itertools.chain.from_iterable(module.parameters() for module in self.model)
        else:
            return self.model.parameters()
