# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import os
import re
from typing import Any, Dict, Optional, Union

import torch
from omegaconf import open_dict
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from pytorch_lightning.trainer.connectors.logger_connector.fx_validator import _FxValidator
from pytorch_lightning.trainer.trainer import Trainer
from torch_xla.distributed.zero_redundancy_optimizer import ZeroRedundancyOptimizer

from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.megatron.clip_grads import (
    clip_grad_norm_distributed_optimizer,
    clip_grad_norm_fp32,
)
from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_model_parallel_for_nemo
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.nlp.parts.nlp_overrides import GradScaler
from nemo.core.optim import MainParamsOptimizerWrapperXLA, MainParamsOptimizerWrapper, prepare_lr_scheduler
from nemo.utils import AppState, logging
from nemo.utils.get_rank import is_global_rank_zero

try:
    from apex.transformer import parallel_state
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

try:
    import torch_xla.core.xla_model as xm
    HAVE_XLA = True
except (ImportError, ModuleNotFoundError):
    HAVE_XLA = False

__all__ = ["MegatronBaseModel"]


_ALLREDUCE_BUCKET_CAP_MB = 512
########NEURON EDIT#############
def bucket_allreduce(tensor_list):
    bucket_cap = int(os.getenv('BUCKET_CAP_MB', _ALLREDUCE_BUCKET_CAP_MB))*1024*1024
    # Reverse the gradients list so that we start allreduce from the last layer
    # onwards. This allows allreduce to trigger as soon as the bucket fills up and
    # overlap with backward pass.
    gradients = reversed(tensor_list)
    total = 0
    tensor_bucket = []

    for grad in gradients:
        grad.data /= parallel_state.get_data_parallel_world_size()
        grad_bytes = grad.numel() * grad.element_size()

        # Gradient is larger than bucket_cap, don't bucketize
        if grad_bytes > bucket_cap:
            # Flush out previous buckets even if they don't fill up
            # This maintains the strict reverse ordering
            if len(tensor_bucket):
                xm.all_reduce('sum', tensor_bucket, groups = parallel_state.get_data_parallel_group()._mesh)
                total = 0
                tensor_bucket = []
            xm.all_reduce('sum', [grad], groups = parallel_state.get_data_parallel_group()._mesh)
            continue

        # Bucketize till the total spills over
        total += grad_bytes
        if total > bucket_cap:
            xm.all_reduce('sum', tensor_bucket, groups = parallel_state.get_data_parallel_group()._mesh)
            total = grad_bytes
            tensor_bucket = []
        tensor_bucket.append(grad)

    # Flush the last remaining bucket
    if len(tensor_bucket):
        xm.all_reduce('sum', tensor_bucket, groups = parallel_state.get_data_parallel_group()._mesh)


class MegatronBaseModel(NLPModel):
    """
    Megatron base class
    It does the following things:
    1. Initialize the model parallel for nemo given the model parallel parameters.
    2. Turn on all the nvidia optimizations.
    3. If `cfg.tokenizer` is available, it loads the tokenizer and pad the vocab to the correct size for tensor model parallelism.
    4. If using distributed optimizer, configure to be compatible with
       O2-level optimizations and/or model parallelism.
    5. Perform gradient clipping: `grad_clip_pl_default` triggers the
       PyTorch Lightning default implementation, `with_distributed_adam`
       triggers the distributed optimizer's implementation,
       `megatron_amp_o2` triggers gradient clipping on the main grads,
       and otherwise gradient clipping is performed on the model grads.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer, no_lm_init=True):
        # FIXME: switch to self._cfg
        if not HAVE_APEX:
            raise ImportError(
                "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        if trainer is None:
            raise ValueError(f"Trainer cannot be None for Megatron-based models. Please provide a PTL trainer object.")
        # this prevents base constructor from initializing tokenizer
        self.tokenizer = None

        super().__init__(cfg, trainer=trainer, no_lm_init=no_lm_init)

        self.with_distributed_adam = cfg.optim.get('name') == 'distributed_fused_adam'

        # used in NVIDIA NGC PyTorch containers
        self._enable_nvidia_optimizations()

        if self._cfg.get('use_cpu_initialization', False) is False:
            torch.cuda.set_device(trainer.local_rank)

        # buffer used during train_step for logging average loss over gradient accumulation steps
        self._reduced_loss_buffer = []
        if os.environ.get("TORCHELASTIC_RUN_ID") is not None:
            g_rank=int(os.environ.get("RANK"))
            l_rank=int(os.environ.get("LOCAL_RANK"))
        else:
            g_rank = trainer.global_rank
            l_rank = trainer.local_rank
        initialize_model_parallel_for_nemo(
            world_size=int(trainer.num_nodes)*int(trainer.num_devices),
            global_rank=g_rank,
            local_rank=l_rank,
            tensor_model_parallel_size=cfg.get('tensor_model_parallel_size', 1),
            pipeline_model_parallel_size=cfg.get('pipeline_model_parallel_size', 1),
            virtual_pipeline_model_parallel_size=cfg.get('virtual_pipeline_model_parallel_size', None),
            pipeline_model_parallel_split_rank=cfg.get('pipeline_model_parallel_split_rank', 0),
            micro_batch_size=cfg.get('micro_batch_size'),
            global_batch_size=cfg.get('global_batch_size'),
            seed=self.cfg.get('seed', 1234),
            apex_transformer_log_level=self.cfg.get('apex_transformer_log_level', 30),
        )

        # This must be called after initialize model parallel since it needs to know the data parallel size
        self._validate_and_override_config()

        self.grad_clip_pl_default = False  # use pytorch default for gradient clipping. Default False
        self.wrap_with_zero = self.cfg.get("wrap_with_zero", False)
        if hasattr(self._cfg, "tokenizer") or (
            hasattr(self._cfg, "encoder_tokenizer") and hasattr(self._cfg, "decoder_tokenizer")
        ):
            # build tokenizer (defaults to nemo supported tokenizers)
            self._build_tokenizer()

            # manipulate vocabulary (e.g., pad vocabulary for better efficiency)
            self._build_vocab()

        # TODO: remove this when PTL 1.7.3 is released
        _FxValidator.functions["configure_gradient_clipping"] = {
            "allowed_on_step": (False, True),
            "allowed_on_epoch": (False, True),
            "default_on_step": True,
            "default_on_epoch": False,
        }
    
    def initialize_model_parallel_for_nemo(self, main_proc=True):
        import torch_xla.core.xla_model as xm
        initialize_model_parallel_for_nemo(
            world_size=xm.xrt_world_size(),
            global_rank=xm.get_ordinal(),
            local_rank=xm.get_local_ordinal(),
            tensor_model_parallel_size=self.cfg.get('tensor_model_parallel_size', 1),
            pipeline_model_parallel_size=self.cfg.get('pipeline_model_parallel_size', 1),
            virtual_pipeline_model_parallel_size=self.cfg.get('virtual_pipeline_model_parallel_size', None),
            pipeline_model_parallel_split_rank=self.cfg.get('pipeline_model_parallel_split_rank', 0),
            micro_batch_size=self.cfg.get('micro_batch_size'),
            global_batch_size=self.cfg.get('global_batch_size'),
            seed=self.cfg.get('seed', 1234),
            apex_transformer_log_level=self.cfg.get('apex_transformer_log_level', 30),
            main_proc=main_proc
        )

        # This must be called after initialize model parallel since it needs to know the data parallel size
        self._validate_and_override_config()

    def _enable_nvidia_optimizations(self):
        "These optimizations are present in NVIDIA NGC PyTorch Containers"

        # NVIDIA container version check
        nvidia_torch_version = os.getenv('NVIDIA_PYTORCH_VERSION', None)
        if nvidia_torch_version is not None:
            try:
                NVIDIA_TORCH_MAJOR = int(nvidia_torch_version.split('.')[0])
            except Exception:
                NVIDIA_TORCH_MAJOR = 0
            try:
                NVIDIA_TORCH_MINOR = int(nvidia_torch_version.split('.')[1])
            except Exception:
                NVIDIA_TORCH_MINOR = 0

            # Apex Persistent layer norm is supported from Nvidia PyTorch container v21.11
            # This only depends on Apex version?
            if NVIDIA_TORCH_MAJOR < 21 or (NVIDIA_TORCH_MAJOR == 21 and NVIDIA_TORCH_MINOR < 11):
                self.cfg.persist_layer_norm = False

            # NVFUSER available starting with 21.11
            if NVIDIA_TORCH_MAJOR >= 21 or (NVIDIA_TORCH_MAJOR == 21 and NVIDIA_TORCH_MINOR >= 11):

                # NVFUSER
                torch._C._jit_set_profiling_executor(True)
                torch._C._jit_set_profiling_mode(True)
                torch._C._jit_override_can_fuse_on_cpu(False)
                torch._C._jit_override_can_fuse_on_gpu(False)
                torch._C._jit_set_texpr_fuser_enabled(False)
                torch._C._jit_set_nvfuser_enabled(True)
                torch._C._debug_set_autodiff_subgraph_inlining(False)
        else:
            # Not a Nvidia container. NVFUSER Dependency check is on users
            pass

    def _build_tokenizer(self):
        """
        Default tokenizer is based on available nemo tokenizers.
        Override this method to use an external tokenizer.
        All tokenizers are expected to provide compatible interface.
        Override default Encoder-decoder tokenizer to use legacy=True for sentencepiece.
        """
        if hasattr(self._cfg.tokenizer, "sentencepiece_legacy"):
            legacy = self._cfg.tokenizer.sentencepiece_legacy
        else:
            legacy = True if self._cfg.tokenizer.library == 'sentencepiece' else False
        self.tokenizer = get_nmt_tokenizer(
            library=self._cfg.tokenizer.library,
            model_name=self._cfg.tokenizer.type,
            tokenizer_model=self.register_artifact("tokenizer.model", self._cfg.tokenizer.model),
            vocab_file=self.register_artifact("tokenizer.vocab_file", self._cfg.tokenizer.vocab_file),
            merges_file=self.register_artifact("tokenizer.merge_file", self._cfg.tokenizer.merge_file),
            delimiter=self.cfg.tokenizer.get('delimiter', None),
            legacy=legacy,
            use_fast=self.cfg.tokenizer.get('use_fast', False)
        )

    def on_train_start(self) -> None:
        super().on_train_start()
        self.init_global_step = self.trainer.global_step
        if isinstance(self._optimizer, ZeroRedundancyOptimizer) and not self._optimizer.inited:
            self._optimizer.init_zero()
            xm.mark_step()
            logging.info("Zero1 optimizer inited.")

    def _build_vocab(self):
        """
        Manipulate vocabulary (e.g., pad vocabulary for increased performance)/
        """
        # TODO: add config to allow to disable it?
        self.padded_vocab_size = self._vocab_size_with_padding(
            orig_vocab_size=self.tokenizer.vocab_size,
            make_vocab_size_divisible_by=self._cfg.get('make_vocab_size_divisible_by', 128),
            tensor_model_parallel_size=self._cfg.get('tensor_model_parallel_size', 1),
        )

    def _vocab_size_with_padding(self, orig_vocab_size, make_vocab_size_divisible_by, tensor_model_parallel_size):
        """Pad vocab size so it is divisible by model parallel size and
        still having GPU friendly size."""

        after = orig_vocab_size
        multiple = make_vocab_size_divisible_by * tensor_model_parallel_size
        while (after % multiple) != 0:
            after += 1
        logging.info(
            f'Padded vocab_size: {after}, original vocab_size: {orig_vocab_size}, dummy tokens: {after - orig_vocab_size}.'
        )
        return after

    def _get_parameters(self):
        """
        private method to load all the trainable parameters from optimizer param groups
        """
        params = []
        for param_group in self._optimizer_param_groups:
            for param in param_group['params']:
                params.append(param)
        return params
    
    def get_parameters_with_grad(self):
        """
        Get all parameters with grad from optimizer param groups
        """
        params = []
        for param_group in self._optimizer_param_groups:
            for param in param_group['params']:
                if (
                    param.grad is not None
                ):  # Adapter training with pp>1 can result in params with no grads
                    params.append(param)
        return params

    def configure_gradient_clipping(self, *args, **kwargs):
        """PTL hook to configure gradients.
           We use gradient clipping implementation from megatron-lm.
        """
        clip_val = self.trainer.gradient_clip_val
        if clip_val is None or self.wrap_with_zero:
            # Zero1 optimizer handles gradient clipping for us across TP groups
            return

        clip_val = float(clip_val)
        if clip_val <= 0:
            return

        if self.grad_clip_pl_default:
            # use the default behavior
            return super().configure_gradient_clipping(*args, **kwargs)

        if self.with_distributed_adam:
            grad_norm = clip_grad_norm_distributed_optimizer(self._optimizer, clip_val)
        else:
            if self.megatron_amp_o2:
                # grep fp32 master parameters for gradient clipping
                parameters = self._optimizer.get_parameters_with_grad()
            else:
                parameters = self.get_parameters_with_grad()
            grad_norm = clip_grad_norm_fp32(parameters=parameters, max_norm=clip_val, optimizer_dtype=torch.double if self.megatron_amp_o2 else torch.float32)

    def allreduce_gradients(self):
        """Reduce gradients across data parallel ranks.
           Modified from megatron-lm: https://github.com/NVIDIA/Megatron-LM/blob/d41696840ed0a7edb7e0499eb82a48ae112d9bb3/megatron/model/distributed.py#L188
        """
        # Bucketize and all-reduce
        buckets = {}
        for param in self.parameters():
            if param.requires_grad and param.grad is not None:
                tp = param.dtype
                if tp not in buckets:
                    buckets[tp] = []
                buckets[tp].append(param)
                # param.main_grad = param.grad

        # For each bucket, all-reduce and copy all-reduced grads.
        for tp in buckets:
            bucket = buckets[tp]
            grads = [param.grad.data for param in bucket]
            bucket_allreduce(grads)
            # coalesced = torch._utils._flatten_dense_tensors(grads)
            # coalesced /= parallel_state.get_data_parallel_world_size()
            # torch.distributed.all_reduce(coalesced, group=parallel_state.get_data_parallel_group())
            # for buf, synced in zip(grads, torch._utils._unflatten_dense_tensors(coalesced, grads)):
            #     buf.copy_(synced)

    def reduce_overlap_gradients(self):
        """Reduce grads if overlapped grad sync is enabled

        Used for pipeline parallelism with the distributed Adam
        optimizer. In the first pipeline stage, the grad sync is
        overlapped with the final backward pass. In other pipeline
        stages, the grad sync is deferred until the bubble overhead.

        """
        if self.with_distributed_adam:
            self._optimizer.try_grad_sync(
                p for p in self._optimizer.parameters() if not getattr(p, '_disable_overlap_grad_sync', False)
            )

    def on_train_batch_end(self, outputs, batch, batch_idx: int, unused: Optional[int] = 0) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)

        # TODO: Replace with newer override for scheduler.step() instead of
        # search for plugins for fp16 GradScalar
        if self.trainer.precision_plugin is not None and isinstance(
            self.trainer.precision_plugin, NativeMixedPrecisionPlugin
        ):
            precision_plugin = self.trainer.precision_plugin

            if (
                hasattr(precision_plugin, 'scaler')
                and precision_plugin.scaler is not None
                and isinstance(precision_plugin.scaler, GradScaler)
            ):
                grad_scaler = precision_plugin.scaler

                # If the grad scaler skipped its optimizer step due to infs/nans,
                # decrement the step of all schedulers.
                if grad_scaler.optimizer_update_skipped is not None and grad_scaler.optimizer_update_skipped is True:
                    scheduler_cfgs = self.trainer.lr_scheduler_configs

                    if not scheduler_cfgs or not self.trainer.lightning_module.automatic_optimization:
                        return

                    for scheduler_cfg in scheduler_cfgs:
                        # Decrement the counter by 2, then perform a scheduler.step() to perform a no-up
                        # as well as update the optimizer lr in all param groups
                        scheduler_cfg.scheduler.last_epoch -= 2
                        scheduler_cfg.scheduler.step()

                    # Removing the line below because it messes up train_valid_test_num_samples calculation.
                    # self.trainer.fit_loop.max_steps = self.trainer.fit_loop.max_steps + 1

                    # Reset the optimizer update skipped to `None` - this is to prevent scheduler no-ops during
                    # accumulated gradient updates.
                    grad_scaler.optimizer_update_skipped = None

    def setup_optimization(
        self, optim_config: Optional[Union[DictConfig, Dict]] = None, optim_kwargs: Optional[Dict[str, Any]] = None,
    ):
        optim_kwargs = {} if optim_kwargs is None else optim_kwargs.copy()
        if self.with_distributed_adam:

            # Allocate grads since we are storing between microbatches
            optim_kwargs['contiguous_grad_buffer'] = True

            if self.megatron_amp_o2:
                # Match param allgather with model dtype
                if hasattr(self, 'autocast_dtype'):
                    optim_kwargs['param_sync_dtype'] = self.autocast_dtype
                    if self.autocast_dtype == torch.float:
                        optim_kwargs['store_params'] = False
                    elif self.autocast_dtype == torch.float16:
                        optim_kwargs['store_params'] = True
                    elif self.autocast_dtype == torch.bfloat16:
                        optim_kwargs['store_params'] = False
                        optim_kwargs['store_param_remainders'] = True
            else:
                # Assume FP32 params, so no need to store main params
                optim_kwargs['store_params'] = False

        return super().setup_optimization(optim_config=optim_config, optim_kwargs=optim_kwargs,
                                          wrap_with_zero=self.wrap_with_zero)

    def configure_optimizers(self):
        self.setup_optimization()

     # Wrap the baseline optimizer with the optimizer class with master parameters
        if self.megatron_amp_o2 and not self.with_distributed_adam and self._optimizer is not None:
            self._optimizer = MainParamsOptimizerWrapperXLA(
                self._optimizer,
            )

            assert self._trainer.max_steps is not None, "'max_steps' is missing in trainer config."
            if hasattr(self._cfg.optim, 'sched'):
                sched_config = self._cfg.optim.sched
                sched_config['max_steps'] = self._trainer.max_steps
                self._scheduler = prepare_lr_scheduler(
                    optimizer=self._optimizer, scheduler_config=sched_config, train_dataloader=self._train_dl
                )

        # Configure distributed optimizer
        if self.with_distributed_adam:

            # Initialize params so that main grads are available
            # Note: Consolidate grads without overlap
            self._optimizer.init_params(
                p for p in self.parameters() if getattr(p, '_disable_overlap_grad_sync', False)
            )
            self._optimizer.init_params(self.parameters())

        if self._scheduler is None:
            return self._optimizer
        else:
            return [self._optimizer], [self._scheduler]

    def compute_consumed_samples(self, steps_since_resume=0):
        app_state = AppState()
        consumed_samples = (
            self.init_consumed_samples
            + steps_since_resume * app_state.data_parallel_size * self.cfg.micro_batch_size * get_num_microbatches() ### NEURON fix num_batches
        )
        return int(consumed_samples)

    def _extract_consumed_samples_from_ckpt(self, ckpt_path):
        try:
            init_consumed_samples = int(float(re.findall(r"consumed_samples\=([0-9]+.[0-9]+)", ckpt_path)[0]))
        except (ValueError, TypeError, IndexError):
            logging.warning("Cannot parse the checkpoint file to get the consumed samples. assume it is zero.")
            init_consumed_samples = 0

        return init_consumed_samples

    def _validate_and_override_config(self):
        """ Certain configurations might be incompatible or discouraged. 
            We can check for them here and override if necessary.
        """
        app_state = AppState()

        if self.cfg.get('sequence_parallel', False) and self.cfg.get('tensor_model_parallel_size', 1) == 1:
            logging.info(
                "Sequence parallel should only be used with tensor parallel size > 1. Setting sequence parallel to False"
            )
            with open_dict(self.cfg):
                self.cfg.sequence_parallel = False

        # Gradient accumulation fusion does not work with our baseline implementaiton of
        # async grad allreduce. This should be fixed!
        # For now we must disable it whenever using the baseline implementaion.
        # The distributed adam from apex does work with gradient accumulation fusion.
        distributed_fused_adam = self.cfg.optim.get('name', 'fused_adam') == 'distributed_fused_adam'
        pipeline_model_parallel_size = self.cfg.get('pipeline_model_parallel_size', 1)
        data_parallel_size = app_state.data_parallel_size

        if self.cfg.get('gradient_accumulation_fusion', False):
            if data_parallel_size > 1 and pipeline_model_parallel_size == 1 and not distributed_fused_adam:
                logging.info(
                    "When not using pipeline model parallel, gradient accumulation fusion can only be used with distributed_fused_adam."
                )
                with open_dict(self.cfg):
                    self.cfg.gradient_accumulation_fusion = False

        if self.cfg.get('gradient_accumulation_fusion', False) and not self.cfg.get('megatron_amp_O2', False):
            logging.info("Gradient accumulation fusion can only be used with megatron amp O2 mixed precision.")
            with open_dict(self.cfg):
                self.cfg.gradient_accumulation_fusion = False

        if self.cfg.get('use_emha', False):
            raise ValueError('use_emha is not yet supported please set to False')

        if self.cfg.get('virtual_pipeline_model_parallel_size', None) is not None:
            assert (
                self.cfg.num_layers // self.cfg.pipeline_model_parallel_size
            ) % self.cfg.virtual_pipeline_model_parallel_size == 0, (
                'Make sure the number of model chunks is the same across all pipeline stages.'
            )

    def is_data_parallel_rank_zero(self):
        if is_global_rank_zero():
            return True
        else:
            try:
                data_parallel_rank = parallel_state.get_data_parallel_rank()
            except:
                data_parallel_rank = None

            if data_parallel_rank is not None and data_parallel_rank == 0:
                return True
            else:
                return False
