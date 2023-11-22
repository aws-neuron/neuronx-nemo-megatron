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
import os
import shutil
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Mapping, Optional, Sized, Union, Iterable
from datetime import timedelta
from functools import partial

import pytorch_lightning as pl
import torch
import torch.multiprocessing as mp
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch_xla.distributed.zero_redundancy_optimizer import ZeroRedundancyOptimizer
from torchmetrics import Metric
from lightning_lite.plugins import ClusterEnvironment, XLACheckpointIO
from lightning_lite.plugins.environments import XLAEnvironment
from lightning_lite.utilities.types import _PATH, Optimizable
from lightning_lite.strategies.launchers.xla import _rank_teardown
from lightning_utilities.core.apply_func import apply_to_collection, apply_to_collections
from lightning_utilities.core.imports import RequirementCache
from lightning_lite.utilities.data import _auto_add_worker_init_fn
from lightning_lite.accelerators.tpu import _XLA_AVAILABLE
from omegaconf import OmegaConf
from pytorch_lightning.profilers import Profiler
from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.plugins import PLUGIN_INPUT
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from pytorch_lightning.strategies import DDPStrategy, TPUSpawnStrategy, Strategy
from pytorch_lightning.strategies.launchers.xla import _XLALauncher
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.fetching import DataFetcher
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.auto_restart import _add_capture_metadata_collate
from pytorch_lightning.utilities.imports import _fault_tolerant_training
from pytorch_lightning.utilities.argparse import _defaults_from_env_vars
from pytorch_lightning.utilities.rank_zero import rank_zero_warn 
from pytorch_lightning.loops.utilities import _block_parallel_sync_behavior
from pytorch_lightning.trainer.connectors.accelerator_connector import AcceleratorConnector, _LITERAL_WARN
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector

from pytorch_lightning.trainer import call, setup
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks import Callback, Checkpoint
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loops import PredictionLoop, TrainingEpochLoop, TrainingBatchLoop, OptimizerLoop
from pytorch_lightning.loops.dataloader.evaluation_loop import EvaluationLoop
from pytorch_lightning.loops.fit_loop import FitLoop, _select_data_fetcher
from pytorch_lightning.trainer.states import TrainerFn, TrainerState
from pytorch_lightning.trainer.connectors.data_connector import DataConnector
from pytorch_lightning.trainer.connectors.logger_connector import LoggerConnector
from pytorch_lightning.trainer.connectors.logger_connector.result import _ResultCollection, _ResultMetric, _ResultMetricCollection
from pytorch_lightning.trainer.connectors.signal_connector import SignalConnector
from pytorch_lightning.trainer.connectors.callback_connector import CallbackConnector
from pytorch_lightning.tuner.tuning import Tuner

from torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks import noop_hook
from torch.nn.parallel import DistributedDataParallel

from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from nemo.core.optim import MainParamsOptimizerWrapper
from nemo.utils import AppState, logging
from nemo.utils.model_utils import inject_model_parallel_rank
from nemo.utils.cast_utils import cast_all

try:
    from apex.transformer import parallel_state
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches
    from apex.transformer.pipeline_parallel.utils import setup_microbatch_calculator

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

from lightning_lite.utilities.data import has_iterable_dataset as new_has_iterable_dataset

def has_len_all_ranks_patched(dataloader, strategy, model,) -> bool:
    """Checks if a given Dataloader has ``__len__`` method implemented i.e. if it is a finite dataloader or
    infinite dataloader."""
    try:
        local_length = len(dataloader)  # type: ignore [arg-type] # we are checking with duck-typing
        total_length = strategy.reduce(torch.tensor(local_length, device=strategy.root_device), reduce_op="sum")

        if total_length == 0:
            rank_zero_warn(
                f"Total length of `{dataloader.__class__.__name__}` across ranks is zero."
                " Please make sure this was your intention."
            )
        if total_length > 0 and local_length == 0:
            if model.allow_zero_length_dataloader_with_multiple_devices:
                rank_zero_warn(
                    f"Total length of `{dataloader.__class__.__name__}` across ranks is zero, but local rank has zero"
                    " length. Please be cautious of uneven batch length."
                )
                has_len = False
            else:
                raise MisconfigurationException(
                    f"`{dataloader.__class__.__name__}` within local rank has zero length."
                    " Please make sure that it returns at least 1 batch."
                )
        else:
            has_len = True

    except (TypeError, NotImplementedError):
        has_len = False

    # we are checking using lightning_lite, which doesn't know CombinedLoader
    if has_len and new_has_iterable_dataset(dataloader):  # type: ignore [arg-type]
        rank_zero_warn(
            "Your `IterableDataset` has `__len__` defined."
            " In combination with multi-process data loading (when num_workers > 1),"
            " `__len__` could be inaccurate if each worker is not configured independently"
            " to avoid having duplicate data."
        )
    if os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None):
        return True
    return has_len

class TRNPrecisionPlugin(PrecisionPlugin):
    """Precision plugin for TPU integration."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def optimizer_step(  # type: ignore[override]
        self,
        optimizer: Optimizable,
        model: "pl.LightningModule",
        optimizer_idx: int,
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        """Hook to run the optimizer step."""
        if not isinstance(optimizer, ZeroRedundancyOptimizer):
            closure = partial(self._wrap_closure, model, optimizer, optimizer_idx, closure)
        return optimizer.step(closure=closure, **kwargs)

class _NLPXLALauncher(_XLALauncher):
    def launch(self, function: Callable, *args: Any, trainer = None, **kwargs: Any) -> Any:
        """Launches processes that run the given function in parallel.
   
        The function is allowed to have a return value. However, when all processes join, only the return value
        of worker process 0 gets returned from this `launch` method in the main process.
   
        Arguments:
            function: The entry point for all launched processes.
            *args: Optional positional arguments to be passed to the given function.
            trainer: Optional reference to the :class:`~pytorch_lightning.trainer.trainer.Trainer` for which
                a selected set of attributes get restored in the main process after processes join.
            **rning AMI GPU PyTorch 1.13.1 (Ubuntu 20.04) 20230519
            kwargs: Optional keyword arguments to be passed to the given function.
        """
   
        if not self._strategy.cluster_environment.creates_processes_externally:
            context = mp.get_context(self._start_method)
            return_queue = context.SimpleQueue()
            import torch_xla.distributed.xla_multiprocessing as xmp
            xmp.spawn(
                    self._wrapping_function,
                    args=(trainer, function, args, kwargs, return_queue),
                    nprocs=self._strategy.num_processes,
                    start_method=self._start_method,
                    )
        else:
            process_idx = int(os.environ.get("LOCAL_RANK"))
            self._strategy._local_rank = process_idx
            results = function(*args, **kwargs)
            _rank_teardown(process_idx)

        return None
   
    def _wrapping_function(
        self,
        process_idx: int,
        trainer,
        function,
        args,
        kwargs,
        return_queue,
        global_states = None
    ) -> None:
        self._strategy._local_rank = process_idx
        results = function(*args, **kwargs)
   
        #### NEURON: Avoiding moving data from device to CPU
        _rank_teardown(process_idx)

class _NLPResultCollection(_ResultCollection):
    def register_key(self, key: str, meta, value) -> None:
        """Create one _ResultMetric object per value.
   
        Value can be provided as a nested collection
        """
   
        def fn(v):
            metric = _ResultMetric(meta, isinstance(v, Tensor))
            ### NEURON: Do not move metrics to device, results in unnnecessary compiles
            return metric
   
        value = apply_to_collection(value, (Tensor, Metric), fn)
        if isinstance(value, dict):
            value = _ResultMetricCollection(value)
        self[key] = value
   
    def update_metrics(self, key: str, value, batch_size: int) -> None:
        def fn(result_metric, v):
            # performance: avoid calling `__call__` to avoid the checks in `torch.nn.Module._call_impl`
            ### NEURON: Do not move metrics to device, results in unnnecessary compiles
            result_metric.forward(v, batch_size)
            result_metric.has_reset = False
   
        apply_to_collections(self[key], value, _ResultMetric, fn)

class NLPOptimizerLoop(OptimizerLoop):
    def _run_optimization(self, kwargs, optimizer):
        """Runs closure (train step + backward) together with optimization if necessary.
   
        Args:
            kwargs: the kwargs passed down to the hooks.
            optimizer: the current optimizer
        """
        opt_idx = kwargs.get("optimizer_idx", 0)
   
        # toggle model params
        self._run_optimization_start(opt_idx, optimizer)
   
        closure = self._make_closure(kwargs, optimizer)
   
        if (
            # when    if ( the strategy handles accumulation, we want to always call the optimizer step
            not self.trainer.strategy.handles_gradient_accumulation
            and self.trainer.fit_loop._should_accumulate()
        ):
            # For gradient accumulation
            # -------------------
            # calculate loss (train step + train step end)
            # -------------------
            # automatic_optimization=True: perform ddp sync only when performing optimizer_step
            with _block_parallel_sync_behavior(self.trainer.strategy, block=True):
                closure()
   
        # ------------------------------
        # BACKWARD PASS
        # ------------------------------
        # gradient update with accumulated gradients
        else:
            # the `batch_idx` is optional with inter-batch parallelism
            self._optimizer_step(optimizer, opt_idx, kwargs.get("batch_idx", 0), closure)
   
        result = closure.consume_result()
   
        if result.loss is not None:
            # if no result, user decided to skip optimization
            # otherwise update running loss + reset accumulated loss
            # TODO: find proper way to handle updating running loss
            # self.trainer.fit_loop.epoch_loop.batch_loop._update_running_loss(result.loss)
            # NEURON: Copying loss to cpu as part of step_closure
            import torch_xla.core.xla_model as xm
            def _update_loss(trainer, loss):
                trainer.fit_loop.epoch_loop.batch_loop._update_running_loss(loss.cpu())
            xm.add_step_closure(_update_loss, (self.trainer, result.loss.detach(), ))
   
        # untoggle model params
        self._run_optimization_end(opt_idx)
        return result

class NLPTrainingBatchLoop(TrainingBatchLoop):
    def __init__(self) -> None:
        super().__init__()
        self.optimizer_loop = NLPOptimizerLoop()

class NLPEvaluationLoop(EvaluationLoop):
    # We override this class to make sure we use _NLPResultCollection 
    # and avoid transferring results to device
    def __init__(self, verbose: bool = True) -> None:
        super().__init__(verbose)
        self._results = _NLPResultCollection(training=False)

    def teardown(self) -> None:
        if self._data_fetcher is not None:
            self._data_fetcher.teardown()
            self._data_fetcher = None
        self.epoch_loop.teardown()

    def _on_evaluation_start(self, *args: Any, **kwargs: Any) -> None:
        """Runs ``on_{validation/test}_start`` hooks."""
        assert self._results is not None

        hook_name = "on_test_start" if self.trainer.testing else "on_validation_start"
        self.trainer._call_callback_hooks(hook_name, *args, **kwargs)
        self.trainer._call_lightning_module_hook(hook_name, *args, **kwargs)
        self.trainer._call_strategy_hook(hook_name, *args, **kwargs)

class NLPTrainingEpochLoop(TrainingEpochLoop):
    def __init__(self, min_steps: Optional[int] = None, max_steps: int = -1) -> None:
        super().__init__(min_steps, max_steps)
        self.batch_loop = NLPTrainingBatchLoop()
        self.val_loop = NLPEvaluationLoop(verbose=True)
        self._results = _NLPResultCollection(training=True)

class NLPFitLoop(FitLoop):
    # We override this class to make sure results are on CPU on run start
    def on_run_start(self) -> None:
        """Calls the ``on_train_start`` hook."""
        # update the current_epoch in-case of checkpoint reload
        if not self._iteration_based_training():
            self.epoch_progress.current.completed = self.epoch_progress.current.processed

        self.trainer.reset_train_dataloader(self.trainer.lightning_module)
        # reload the evaluation dataloaders too for proper display in the progress bar
        if self.epoch_loop._should_check_val_epoch():
            self.epoch_loop.val_loop._reload_evaluation_dataloaders()

        data_fetcher_cls = _select_data_fetcher(self.trainer)
        self._data_fetcher = data_fetcher_cls(prefetch_batches=self.prefetch_batches)

        self._is_fresh_start_epoch = True
        self._results.cpu()

        self.trainer._call_callback_hooks("on_train_start")
        self.trainer._call_lightning_module_hook("on_train_start")
        self.trainer._call_strategy_hook("on_train_start")


class NLPCheckpointIO(XLACheckpointIO):

    def load_checkpoint(self, checkpoint_path: _PATH, load_type_xser: bool) -> Dict[str, Any]:
        """ PTL override to accomodate model parallel checkpoints """
        checkpoint_path = inject_model_parallel_rank(checkpoint_path)
        from .serialization import load

        if load_type_xser:
            loaded_checkpoint = load(checkpoint_path, parallel_state.get_data_parallel_rank(),
                                     parallel_state.get_data_parallel_group())
        else:
            loaded_checkpoint = super().load_checkpoint(checkpoint_path)

        return loaded_checkpoint

    def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: _PATH, save_type_xser: bool, storage_options: Optional[Any] = None
    ) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.
   
        Args:
            checkpoint: dict containing model and trainer state
            path: write-target path
            storage_options: not used in ``XLACheckpointIO.save_checkpoint``
   
        Raises:
            TypeError:
                If ``storage_options`` arg is passed in
        """
        if storage_options is not None:
            raise TypeError(
                "`Trainer.save_checkpoint(..., storage_options=...)` with `storage_options` arg"
                f" is not supported for `{self.__class__.__name__}`. Please implement your custom `CheckpointIO`"
                " to define how you'd like to use `storage_options`."
            )
        app_state = AppState()
        # PTL override to accomodate model parallel checkpoints
        def ensure_directory_exists(filename):
            """Build filename's path if it does not already exists."""
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

        import torch_xla.utils.serialization as xser
        import torch_xla.core.xla_model as xm
        # PTL override to accomodate model parallel checkpoints
        filepath = inject_model_parallel_rank(filepath)
        if RequirementCache("omegaconf"):
            # workaround for https://github.com/pytorch/xla/issues/2773
            from omegaconf import DictConfig, ListConfig, OmegaConf

            checkpoint = apply_to_collection(checkpoint, (DictConfig, ListConfig), OmegaConf.to_container)
        master_only = app_state.data_parallel_rank == 0
        if master_only:
            ensure_directory_exists(filepath)
        try:
            save_bf16 = self.lightning_module.cfg.save_bf16
        except:
            save_bf16 = False
        from .serialization import save
        if save_type_xser:
            save({k: v for k, v in checkpoint.items() if k != "callbacks"}, filepath, parallel_state.get_data_parallel_rank()==0, 
                    parallel_state.get_data_parallel_rank(), parallel_state.get_data_parallel_world_size())
        else:
            for tp_rank in range(0, parallel_state.get_tensor_model_parallel_world_size()):
                my_tp_rank = parallel_state.get_tensor_model_parallel_rank()
                if save_bf16:
                    checkpoint = cast_all(checkpoint, from_dtype=torch.float32, to_dtype=torch.bfloat16)
                should_write_data = True if parallel_state.get_data_parallel_rank() == 0 and my_tp_rank == tp_rank else False

                #Staggering save checkpoints
                if should_write_data:
                    cpu_data = xm._maybe_convert_to_cpu({k: v for k, v in checkpoint.items() if k != "callbacks"}, convert=True)
                    ensure_directory_exists(filepath)
                    torch.save(cpu_data, filepath)
                xm.rendezvous(f'chktp-save-{tp_rank}')

class NLPCheckpointConnector(CheckpointConnector):
    def restore_loops(self) -> None:
        """Restores the loop progress from the pre-loaded checkpoint.
   
        Calls hooks on the loops to give it a chance to restore its state from the checkpoint.
        """
        if not self._loaded_checkpoint:
            return
   
        fit_loop = self.trainer.fit_loop
        pl_module = self.trainer.lightning_module
        assert pl_module is not None
    
        global_step = self._loaded_checkpoint.get("global_step", 0)
        epoch = self._loaded_checkpoint.get("epoch", 0)
        # set the `global_step` value for checkpoints before v1.6 without the progress tracking state.
        # it will be overwritten by the loop's state if it was also saved
        batch_loop = fit_loop.epoch_loop.batch_loop
        if pl_module.automatic_optimization:
            batch_loop.optimizer_loop.optim_progress.optimizer.step.total.completed = global_step
        else:
            batch_loop.manual_loop.optim_step_progress.total.completed = global_step
    
        # set the `current_epoch` value for checkpoints before v1.6 without the progress tracking state.
        # it will be overwritten by the loop's state if it was also saved
        fit_loop.epoch_progress.current.completed = epoch
    
        assert self.trainer.state.fn is not None
        state_dict = self._loaded_checkpoint.get("loops")
        if state_dict is not None:
            if self.trainer.state.fn == TrainerFn.FITTING:
                fit_loop.load_state_dict(state_dict["fit_loop"])
            elif self.trainer.state.fn == TrainerFn.VALIDATING:
                self.trainer.validate_loop.load_state_dict(state_dict["validate_loop"])
            elif self.trainer.state.fn == TrainerFn.TESTING:
                self.trainer.test_loop.load_state_dict(state_dict["test_loop"])
            elif self.trainer.state.fn == TrainerFn.PREDICTING:
                self.trainer.predict_loop.load_state_dict(state_dict["predict_loop"])
   
        if self.trainer.state.fn != TrainerFn.FITTING:
            return
   
        # crash if max_epochs is lower then the current epoch from the checkpoint
        if (
            self.trainer.max_epochs != -1
            and self.trainer.max_epochs is not None
            and self.trainer.current_epoch > self.trainer.max_epochs
        ):
            raise MisconfigurationException(
                f"You restored a checkpoint with current_epoch={self.trainer.current_epoch},"
                f" but you have set Trainer(max_epochs={self.trainer.max_epochs})."
            )

    def restore_optimizers_and_schedulers(self) -> None:
        """Restores the optimizers and learning rate scheduler states from the pre-loaded checkpoint."""
        if not self._loaded_checkpoint:
            return
        if self.trainer.strategy.lightning_restore_optimizer:
            # validation
            if "optimizer_states" not in self._loaded_checkpoint:
                logging.warning(
                "Trying to restore optimizer state but checkpoint contains only the model."
                " This is probably due to `ModelCheckpoint.save_weights_only` being set to `True`."
                )  
                return
            self.restore_optimizers()

        if "lr_schedulers" not in self._loaded_checkpoint:
            logging.warning(
                "Trying to restore learning rate scheduler state but checkpoint contains only the model."
                " This is probably due to `ModelCheckpoint.save_weights_only` being set to `True`."
            )
            return
        self.restore_lr_schedulers()

class NLPAcceleratorConnector(AcceleratorConnector):
    def _validate_precision_choice(self) -> None:
        """Validate the combination of choices for precision, AMP type, and accelerator."""
        if self._precision_flag == 64:
            raise MisconfigurationException(
                "`Trainer(accelerator='tpu', precision=64)` is not implemented."
                " Please, open an issue in `https://github.com/Lightning-AI/lightning/issues`"
                " requesting this feature."
            )
   
    def _lazy_init_strategy(self) -> None:
        """Lazily set missing attributes on the previously instantiated strategy."""
        self.strategy.accelerator = self.accelerator
        if self.precision_plugin:
            #self.strategy.precision_plugin = self.precision_plugin
            self.strategy.precision_plugin = TRNPrecisionPlugin()

        if self.checkpoint_io:
            self.strategy.checkpoint_io = self.checkpoint_io
        if hasattr(self.strategy, "cluster_environment"):
            self.strategy.cluster_environment = self.cluster_environment
        if hasattr(self.strategy, "parallel_devices"):
            if self.strategy.parallel_devices:
                self._parallel_devices = self.strategy.parallel_devices
            else:
                self.strategy.parallel_devices = self._parallel_devices
        if hasattr(self.strategy, "num_nodes"):
            self.strategy._num_nodes = self._num_nodes_flag
        if hasattr(self.strategy, "_layer_sync"):
            self.strategy._layer_sync = self._layer_sync
        if hasattr(self.strategy, "set_world_ranks"):
            self.strategy.set_world_ranks()
        self.strategy._configure_launcher()

class NLPDataConnector(DataConnector):

    def _reset_eval_dataloader(self, mode, model):
        """Generic method to reset a dataloader for evaluation.

        Args:
            mode: The running stage of the ``Trainer``
            model: The ``LightningModule`` if calling this outside of the trainer scope.

        Returns:
            Tuple (num_batches, dataloaders)
        """
        assert mode.evaluating or mode == RunningStage.PREDICTING

        # always get the loaders first so we can count how many there are
        dataloaders = self._request_dataloader(mode)

        if self.trainer.overfit_batches > 0:
            dataloaders = self._resolve_overfit_batches(dataloaders, mode)

        if not isinstance(dataloaders, list):
            dataloaders = [dataloaders]  # type: ignore[assignment]

        if any(dl is None for dl in dataloaders):
            rank_zero_warn("One of given dataloaders is None and it will be skipped.")

        for loader in dataloaders:
            apply_to_collection(
                loader.loaders if isinstance(loader, CombinedLoader) else loader,
                DataLoader,
                self._check_eval_shuffling,
                mode=mode,
            )

        # add samplers
        dataloaders = [self._prepare_dataloader(dl, mode=mode) for dl in dataloaders if dl is not None]

        # add worker_init_fn for correct seeding in worker processes
        apply_to_collection(
            dataloaders, dtype=DataLoader, function=_auto_add_worker_init_fn, rank=self.trainer.global_rank
        )

        loader_num_batches: List[Union[int, float]] = []

        # determine number of batches
        module = model or self.trainer.lightning_module or self.datamodule
        if len(dataloaders) != 0:
            for i, dataloader in enumerate(dataloaders):
                orig_num_batches = num_batches = (
                    len(dataloader) if has_len_all_ranks_patched(dataloader, self.trainer.strategy, module) else float("inf")
                )

                if orig_num_batches == 0:
                    assert isinstance(orig_num_batches, int)
                    loader_num_batches.append(orig_num_batches)
                    continue

                self._worker_check(dataloader, f"{mode.dataloader_prefix}_dataloader {i}")

                # percent or num_steps
                limit_eval_batches = getattr(self.trainer, f"limit_{mode.dataloader_prefix}_batches")

                # limit num batches either as a percent or num steps
                if isinstance(limit_eval_batches, int):
                    num_batches = min(orig_num_batches, limit_eval_batches)
                elif isinstance(limit_eval_batches, float) and orig_num_batches != float("inf"):
                    num_batches = int(orig_num_batches * limit_eval_batches)
                elif limit_eval_batches != 1.0:
                    raise MisconfigurationException(
                        f"When using an `IterableDataset`, `Trainer(limit_{mode.dataloader_prefix}_batches)` must be"
                        f" `1.0` or an int. An int specifies `num_{mode.dataloader_prefix}_batches` to use."
                    )

                if (
                    num_batches == 0
                    and limit_eval_batches > 0.0
                    and isinstance(limit_eval_batches, float)
                    and orig_num_batches != float("inf")
                ):
                    min_percentage = 1.0 / orig_num_batches
                    raise MisconfigurationException(
                        f"You requested to check {limit_eval_batches} of the `{mode.dataloader_prefix}_dataloader` but"
                        f" {limit_eval_batches} * {orig_num_batches} < 1. Please increase the"
                        f" `limit_{mode.dataloader_prefix}_batches` argument. Try at least"
                        f" `limit_{mode.dataloader_prefix}_batches={min_percentage}`"
                    )

                loader_num_batches.append(num_batches)

        return loader_num_batches, dataloaders


class NLPTrainer(Trainer):
    @_defaults_from_env_vars
    def __init__(
        self,
        logger: Union[Logger, Iterable[Logger], bool] = True,
        enable_checkpointing: bool = True,
        callbacks: Optional[Union[List[Callback], Callback]] = None,
        default_root_dir: Optional[_PATH] = None,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
        num_nodes: int = 1,
        num_processes: Optional[int] = None,  # TODO: Remove in 2.0
        devices: Optional[Union[List[int], str, int]] = None,
        gpus: Optional[Union[List[int], str, int]] = None,  # TODO: Remove in 2.0
        auto_select_gpus: bool = False,
        tpu_cores: Optional[Union[List[int], str, int]] = None,  # TODO: Remove in 2.0
        ipus: Optional[int] = None,  # TODO: Remove in 2.0
        enable_progress_bar: bool = True,
        overfit_batches: Union[int, float] = 0.0,
        track_grad_norm: Union[int, float, str] = -1,
        check_val_every_n_epoch: Optional[int] = 1,
        fast_dev_run: Union[int, bool] = False,
        accumulate_grad_batches: Optional[Union[int, Dict[int, int]]] = None,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        max_steps: int = -1,
        min_steps: Optional[int] = None,
        max_time: Optional[Union[str, timedelta, Dict[str, int]]] = None,
        limit_train_batches: Optional[Union[int, float]] = None,
        limit_val_batches: Optional[Union[int, float]] = None,
        limit_test_batches: Optional[Union[int, float]] = None,
        limit_predict_batches: Optional[Union[int, float]] = None,
        val_check_interval: Optional[Union[int, float]] = None,
        log_every_n_steps: int = 50,
        accelerator: Optional[Union[str, Accelerator]] = None,
        strategy: Optional[Union[str, Strategy]] = None,
        sync_batchnorm: bool = False,
        precision: Union[int, str] = 32,
        enable_model_summary: bool = True,
        num_sanity_val_steps: int = 2,
        resume_from_checkpoint: Optional[Union[Path, str]] = None,
        profiler: Optional[Union[Profiler, str]] = None,
        benchmark: Optional[bool] = None,
        deterministic: Optional[Union[bool, _LITERAL_WARN]] = None,
        reload_dataloaders_every_n_epochs: int = 0,
        auto_lr_find: Union[bool, str] = False,
        replace_sampler_ddp: bool = True,
        detect_anomaly: bool = False,
        auto_scale_batch_size: Union[str, bool] = False,
        plugins: Optional[Union[PLUGIN_INPUT, List[PLUGIN_INPUT]]] = None,
        amp_backend: str = "native",
        amp_level: Optional[str] = None,
        move_metrics_to_cpu: bool = False,
        multiple_trainloader_mode: str = "max_size_cycle",
        inference_mode: bool = True,
    ) -> None:
        Trainer._log_api_event("init")
        logging.info(f"{self.__class__.__name__}: Initializing trainer with parameters: {locals()}")
        self.state = TrainerState()

        if default_root_dir is not None:
            default_root_dir = os.fspath(default_root_dir)

        # init connectors
        self._data_connector = NLPDataConnector(self, multiple_trainloader_mode)

        self._accelerator_connector = NLPAcceleratorConnector(
            num_processes=num_processes,
            devices=devices,
            tpu_cores=tpu_cores,
            ipus=ipus,
            accelerator=accelerator,
            strategy=strategy,
            gpus=gpus,
            num_nodes=num_nodes,
            sync_batchnorm=sync_batchnorm,
            benchmark=benchmark,
            replace_sampler_ddp=replace_sampler_ddp,
            deterministic=deterministic,
            auto_select_gpus=auto_select_gpus,
            precision=precision,
            amp_type=amp_backend,
            amp_level=amp_level,
            plugins=plugins,
        )
        self._logger_connector = LoggerConnector(self)
        self._callback_connector = CallbackConnector(self)
        self._resume_from_checkpoint = resume_from_checkpoint
        self._checkpoint_connector = NLPCheckpointConnector(self, resume_from_checkpoint)
        self._signal_connector = SignalConnector(self)
        self.tuner = Tuner(self)

        fit_loop = NLPFitLoop(min_epochs=min_epochs, max_epochs=max_epochs)
        training_epoch_loop = NLPTrainingEpochLoop(min_steps=min_steps, max_steps=max_steps)
        fit_loop.connect(epoch_loop=training_epoch_loop)

        # default .fit() loop
        self.fit_loop = fit_loop

        # default .validate() loop
        self.validate_loop = NLPEvaluationLoop()

        # default .test() loop
        self.test_loop = NLPEvaluationLoop()

        # default .predict() loop
        self.predict_loop = PredictionLoop()

        # set when a checkpoint is loaded via `Trainer.{fit,validate,test,predict}`.
        self._ckpt_path: Optional[str] = None

        # init callbacks
        # Declare attributes to be set in _callback_connector on_trainer_init
        self._callback_connector.on_trainer_init(
            callbacks,
            enable_checkpointing,
            enable_progress_bar,
            default_root_dir,
            enable_model_summary,
            max_time,
            accumulate_grad_batches,
        )

        # init data flags
        self.check_val_every_n_epoch: Optional[int]
        self._data_connector.on_trainer_init(
            val_check_interval,
            reload_dataloaders_every_n_epochs,
            check_val_every_n_epoch,
        )

        # gradient clipping
        if gradient_clip_val is not None and not isinstance(gradient_clip_val, (int, float)):
            raise TypeError(f"`gradient_clip_val` should be an int or a float. Got {gradient_clip_val}.")

        if gradient_clip_algorithm is not None and not GradClipAlgorithmType.supported_type(
            gradient_clip_algorithm.lower()
        ):
            raise MisconfigurationException(
                f"`gradient_clip_algorithm` {gradient_clip_algorithm} is invalid. "
                f"Allowed algorithms: {GradClipAlgorithmType.supported_types()}."
            )

        # gradient norm tracking
        if track_grad_norm != -1 and not (
            (isinstance(track_grad_norm, (int, float)) or track_grad_norm == "inf") and float(track_grad_norm) > 0
        ):
            raise MisconfigurationException(
                f"`track_grad_norm` must be a positive number or 'inf' (infinity norm). Got {track_grad_norm}."
            )

        self.gradient_clip_val: Optional[Union[int, float]] = gradient_clip_val
        self.gradient_clip_algorithm: Optional[GradClipAlgorithmType] = (
            GradClipAlgorithmType(gradient_clip_algorithm.lower()) if gradient_clip_algorithm is not None else None
        )
        self.track_grad_norm: float = float(track_grad_norm)

        self._inference_mode: bool = inference_mode

        self._detect_anomaly: bool = detect_anomaly
        self._setup_on_init()

        # configure tuner
        self.tuner.on_trainer_init(auto_lr_find, auto_scale_batch_size)

        # configure profiler
        setup._init_profiler(self, profiler)

        # init logger flags
        self._loggers: List[Logger]
        self._logger_connector.on_trainer_init(logger, log_every_n_steps, move_metrics_to_cpu)

        # init debugging flags
        self.val_check_batch: Union[int, float]
        self.val_check_interval: Union[int, float]
        self.num_sanity_val_steps: Union[int, float]
        self.limit_train_batches: Union[int, float]
        self.limit_val_batches: Union[int, float]
        self.limit_test_batches: Union[int, float]
        self.limit_predict_batches: Union[int, float]
        setup._init_debugging_flags(
            self,
            limit_train_batches,
            limit_val_batches,
            limit_test_batches,
            limit_predict_batches,
            fast_dev_run,
            overfit_batches,
            val_check_interval,
            num_sanity_val_steps,
        )

    def _restore_modules_and_callbacks(self, checkpoint_path: Optional[_PATH] = None) -> None:
        import torch_xla.core.xla_model as xm
        self._checkpoint_connector.resume_start(checkpoint_path)
        self._checkpoint_connector._restore_quantization_callbacks()
        self._checkpoint_connector.restore_model()
        self._checkpoint_connector.restore_datamodule()
        if self.state.fn == TrainerFn.FITTING:
            # restore callback states
            self._checkpoint_connector.restore_callbacks()

    def is_resuming_from_checkpoint(self):
        return self._resume_from_checkpoint is not None

    def reset_train_dataloader(self, model: Optional["pl.LightningModule"] = None) -> None:
        """Resets the train dataloader and initialises required variables (number of batches, when to validate,
        etc.).

        Args:
            model: The ``LightningModule`` if calling this outside of the trainer scope.
        """
        source = self._data_connector._train_dataloader_source
        pl_module = model or self.lightning_module
        has_step = is_overridden("training_step", pl_module)
        enable_training = self.limit_train_batches > 0
        if not (source.is_defined() and has_step and enable_training):
            return

        self.train_dataloader = self._data_connector._request_dataloader(RunningStage.TRAINING)

        if self.overfit_batches > 0:
            self.train_dataloader = self._data_connector._resolve_overfit_batches(
                self.train_dataloader, mode=RunningStage.TRAINING
            )

        # automatically add samplers
        self.train_dataloader = apply_to_collection(
            self.train_dataloader,
            (DataLoader, CombinedLoader),
            self._data_connector._prepare_dataloader,
            mode=RunningStage.TRAINING,
        )
        loaders = (
            self.train_dataloader.loaders
            if isinstance(self.train_dataloader, CombinedLoader)
            else self.train_dataloader
        )

        # check the workers recursively
        apply_to_collection(loaders, DataLoader, self._data_connector._worker_check, "train_dataloader")

        # add worker_init_fn for correct seeding in worker processes
        apply_to_collection(loaders, DataLoader, _auto_add_worker_init_fn, rank=self.global_rank)

        # add collate_fn to collect metadata for fault tolerant training
        if _fault_tolerant_training():
            apply_to_collection(loaders, DataLoader, _add_capture_metadata_collate)

        # wrap the sequence of train loaders to a CombinedLoader object for computing the num_training_batches
        if not isinstance(self.train_dataloader, CombinedLoader):
            self.train_dataloader = CombinedLoader(loaders, self._data_connector.multiple_trainloader_mode)

        module = model or self.lightning_module or self.datamodule
        orig_train_batches = self.num_training_batches = (
            len(self.train_dataloader)  # type: ignore[arg-type]
            if has_len_all_ranks_patched(self.train_dataloader, self.strategy, module)
            else float("inf")
        )
        if orig_train_batches == 0:
            return

        # store epoch of dataloader reset for reload_dataloaders_every_n_epochs
        self._last_train_dl_reload_epoch = self.current_epoch

        if isinstance(self.limit_train_batches, int):
            self.num_training_batches = min(orig_train_batches, self.limit_train_batches)
        elif self.num_training_batches != float("inf"):
            self.num_training_batches = int(orig_train_batches * self.limit_train_batches)
        elif self.limit_train_batches != 1.0:
            raise MisconfigurationException(
                "When using an `IterableDataset`, `Trainer(limit_train_batches)` must be `1.0` or an int."
                "An int specifies `num_training_batches` to use."
            )

        if isinstance(self.val_check_interval, int):
            self.val_check_batch = self.val_check_interval
            if self.val_check_batch > self.num_training_batches and self.check_val_every_n_epoch is not None:
                raise ValueError(
                    f"`val_check_interval` ({self.val_check_interval}) must be less than or equal "
                    f"to the number of the training batches ({self.num_training_batches}). "
                    "If you want to disable validation set `limit_val_batches` to 0.0 instead."
                    "If you want to validate based on the total training batches, set `check_val_every_n_epoch=None`."
                )
        else:
            if not has_len_all_ranks_patched(self.train_dataloader, self.strategy, module):
                if self.val_check_interval == 1.0:
                    self.val_check_batch = float("inf")
                else:
                    raise MisconfigurationException(
                        "When using an IterableDataset for `train_dataloader`,"
                        " `Trainer(val_check_interval)` must be `1.0` or an int. An int k specifies"
                        " checking validation every k training batches."
                    )
            else:
                self.val_check_batch = int(self.num_training_batches * self.val_check_interval)
                self.val_check_batch = max(1, self.val_check_batch)

        if self.loggers and self.num_training_batches < self.log_every_n_steps:
            rank_zero_warn(
                f"The number of training batches ({self.num_training_batches}) is smaller than the logging interval"
                f" Trainer(log_every_n_steps={self.log_every_n_steps}). Set a lower value for log_every_n_steps if"
                " you want to see logs for the training epoch.",
                category=PossibleUserWarning,
            )

        if (
            self.num_training_batches == 0
            and self.limit_train_batches > 0.0
            and isinstance(self.limit_train_batches, float)
            and orig_train_batches != float("inf")
        ):
            min_percentage = 1.0 / orig_train_batches
            raise MisconfigurationException(
                f"You requested to check {self.limit_train_batches} of the `train_dataloader` but"
                f" {self.limit_train_batches} * {orig_train_batches} < 1. Please increase the"
                f" `limit_train_batches` argument. Try at least"
                f" `limit_train_batches={min_percentage}`"
            )

class NLPDDPStrategy(TPUSpawnStrategy):
    """ DDP plugin for Pytorch Lightning. Needed to customize DDP for model parallel models.

    Args:
        no_ddp_communication_hook: Disable DDP communication hook when using AMP-O2
        with FP32 gradient accumulation.
    """

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        debug: bool = False,
        no_ddp_communication_hook: bool = False,
        megatron_amp_o2: bool=False,
        restore_path=None,
        **_: Any,
    ) -> None:
        if not _XLA_AVAILABLE:
            raise ModuleNotFoundError(str(_XLA_AVAILABLE))
        if not HAVE_APEX:
            raise ImportError(
                "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        if cluster_environment is None:
            cluster_environment=XLAEnvironment()
        super(TPUSpawnStrategy, self).__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
            start_method="fork",
        )
        self._checkpoint_io: Optional[CheckpointIO]
        self.debug = debug
        self._launched = False
        self.no_ddp_communication_hook = no_ddp_communication_hook
        self.megatron_amp_o2 = megatron_amp_o2
        self.restore_path = restore_path

    def _configure_launcher(self) -> None:
        self._launcher = _NLPXLALauncher(self)

    def setup_distributed(self, global_rank: int = None, world_size: int = None) -> None:
        import torch.distributed as dist
        if self.cluster_environment.creates_processes_externally:
            global_rank = int(os.environ.get("RANK"))
        else:
            import torch_xla.core.xla_model as xm
            global_rank = xm.get_ordinal()
        if (os.environ.get("PJRT_DEVICE", None) == "NEURON"):
            import torch_xla.experimental.pjrt_backend
            dist.init_process_group('xla', init_method="pjrt://", rank=global_rank)
        else:
            dist.init_process_group('xla', rank=global_rank)
        # call PTL init ddp
        super().setup_distributed()

        # init model parallel if needed
        if parallel_state.is_unitialized():
            app_state = AppState()

            if app_state.model_parallel_size is not None:
                self.init_model_parallel(app_state.global_rank, app_state.world_size)

    def configure_ddp(self):
        """ Override LightningModule ddp if using model parallel.
            Sets find_unused_parameters to False to use activation-checkpoint-recomputation.
        """
        if (hasattr(self.model, 'megatron_amp_o2') and self.model.megatron_amp_o2) or (
            hasattr(self.model, 'with_distributed_adam') and self.model.with_distributed_adam
        ):
            # do not use DDP if using megatron amp O2 or distributed optimizer
            self._model = LightningDistributedModule(self.model)
        else:
            app_state = AppState()

            if app_state.model_parallel_size is not None:

                logging.info(f"Configuring DDP for model parallelism.")

                # With model parallelism, multiple GPUs form a large "logical GPU"
                # this means that data parallel groups span multiple GPUs
                # and are non-trivial
                # TODO: for megatron-lm self.model is a list
                self.pre_configure_ddp()
                # device_ids = self.determine_ddp_device_ids()
                self._model = DistributedDataParallel(
                    LightningDistributedModule(self.model),
                    process_group=parallel_state.get_data_parallel_group(),
                    gradient_as_a_view=True, # FOR NEURON
                    **self._ddp_kwargs,
                )

                if self.no_ddp_communication_hook:
                    # When using custom gradient accumulation and allreduce, disable
                    # DDP communication hook that works on the gradient bucket.
                    # Instead, use the custom gradient function and communication hook,
                    # which is defined in the master optimizer wrapper.
                    self._model.require_backward_grad_sync = False
                    self._model.register_comm_hook(None, noop_hook)

            else:
                super().configure_ddp()

    def init_model_parallel(self, global_rank: int, world_size: int) -> None:
        """ Initializes Megatron-LM model parallel if using model parallelism.

        Args:
            global_rank (int): the global process index.
            world_size (int): the total number of GPUs, num_nodes * num_devices
            is_slurm_managing_tasks (bool, optional): is the cluster managed by SLURM.
        """
        app_state = AppState()

        # we initialize megatron-lm model parallel and data parallel groups
        # after initializing DDP with PTL.
        if app_state.model_parallel_size is not None:
            # destroy groups in case they have already been created
            # this happens with multiple calls to trainer.test for example
            parallel_state.destroy_model_parallel()
            # if torch.distributed.is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size_=app_state.tensor_model_parallel_size,
                pipeline_model_parallel_size_=app_state.pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank_=app_state.pipeline_model_parallel_split_rank,
                virtual_pipeline_model_parallel_size_=app_state.virtual_pipeline_model_parallel_size,
            )

            # assert that fake tp and pp rank match after model parallel init
            # assert app_state.tensor_model_parallel_rank == parallel_state.get_tensor_model_parallel_rank()
            # assert app_state.pipeline_model_parallel_rank == parallel_state.get_pipeline_model_parallel_rank()

            app_state.tensor_model_parallel_group = parallel_state.get_tensor_model_parallel_group()
            app_state.data_parallel_group = parallel_state.get_data_parallel_group()
            app_state.data_parallel_rank = parallel_state.get_data_parallel_rank()
            app_state.data_parallel_size = parallel_state.get_data_parallel_world_size()
            app_state.pipeline_model_parallel_group = parallel_state.get_pipeline_model_parallel_group()
            # app_state.global_rank = self.global_rank

            setup_microbatch_calculator(
                rank=self.global_rank,
                global_batch_size=self.model.cfg.global_batch_size,
                micro_batch_size=self.model.cfg.micro_batch_size,
                data_parallel_size=app_state.data_parallel_size,
                rampup_batch_size=None,
            )
   
    def is_save_type_xser(self):
        try:
            save_mode = self.lightning_module.cfg.save_xser
        except:
            save_mode = False
        return save_mode

    def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: _PATH, storage_options: Optional[Any] = None
    ) -> None:
        self.checkpoint_io.save_checkpoint(checkpoint, filepath, self.is_save_type_xser())

    def is_load_type_xser(self):
        try:
            load_xser_mode = self.lightning_module.cfg.load_xser
        except:
            load_xser_mode = False
        return load_xser_mode

    def load_checkpoint(self, checkpoint_path: _PATH) -> Dict[str, Any]:
        return self.checkpoint_io.load_checkpoint(checkpoint_path, self.is_load_type_xser())

    def load_model_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        # Release strict state dict matching when using Megatron AMP-O2 to skip matching
        # half-precision module wrapper module.
        # TODO: Refactor this to be more generic.
        model_key = None
        model_attr = None
        if hasattr(self.lightning_module, 'model'):
            model_key = 'model'
            model_attr = self.lightning_module.model
        elif hasattr(self.lightning_module, 'enc_dec_model'):
            model_key = 'enc_dec_model'
            model_attr = self.lightning_module.enc_dec_model
        if model_key is not None:
            if isinstance(model_attr, Float16Module):
                new_state_dict = {}
                for key in checkpoint['state_dict'].keys():
                    new_key = key.replace(f'{model_key}.', f'{model_key}.module.', 1)
                    new_state_dict[new_key] = checkpoint['state_dict'][key]
                checkpoint['state_dict'] = new_state_dict

        self.lightning_module.load_state_dict(checkpoint["state_dict"])

    def remove_checkpoint(self, filepath: _PATH) -> None:
        app_state = AppState()
        # PTL override to accomodate model parallel checkpoints
        filepath = inject_model_parallel_rank(filepath)
        if self.is_global_zero or app_state.data_parallel_rank == 0:
            if not self.restore_path or filepath != inject_model_parallel_rank(self.restore_path):
                logging.info(f'Removing checkpoint: {filepath}')
                try:
                    self.checkpoint_io.remove_checkpoint(filepath)
                except:
                    pass  # Swallow any exceptions
                if self.is_save_type_xser():
                    if os.path.exists(f"{filepath}.tensors"):
                        try:
                            shutil.rmtree(f"{filepath}.tensors")
                        except:
                            pass  # Swallow any exceptions

    @property
    def distributed_sampler_kwargs(self):
        app_state = AppState()
        if app_state.model_parallel_size is not None:
            # When using model parallel, data parallel groups are non-trivial and they
            # correspond to the logical GPUs. This means that the GPUs that form a
            # single logical GPU all need to get the same batch of data.
            distributed_sampler_kwargs = dict(
                num_replicas=app_state.data_parallel_size, rank=app_state.data_parallel_rank
            )
            return distributed_sampler_kwargs

        else:
            return super(NLPDDPStrategy, self).distributed_sampler_kwargs

    def process_dataloader(self, dataloader):
        TPUSpawnStrategy._validate_dataloader(dataloader)
        return dataloader

    def broadcast(self, obj, src: int = 0):
        return obj

    def teardown(self):
        """This method is called to teardown the training process.
   
        It is the right place to release memory and free other resources.
        """
        #### Avoid copying to CPU
        self.precision_plugin.teardown()
        assert self.accelerator is not None
        self.accelerator.teardown()
        self.checkpoint_io.teardown()


    # original implementation of this function would go over GRPC and hits message size limit
    # when number of workers is > 128
    # https://github.com/pytorch/xla/issues/1924

    def reduce(
            self, output: Union[Tensor, Any], group: Optional[Any] = None, reduce_op: Optional[Union[torch.distributed.ReduceOp, str]] = "mean",
    ) -> Tensor:
        if not isinstance(output, Tensor):
            output = torch.tensor(output, device=self.root_device)

        invalid_reduce_op = isinstance(reduce_op, torch.distributed.ReduceOp) and reduce_op != torch.distributed.ReduceOp.SUM
        invalid_reduce_op_str = isinstance(reduce_op, str) and reduce_op.lower() not in ("sum", "mean", "avg")
        if invalid_reduce_op or invalid_reduce_op_str:
            raise ValueError(
                "Currently, the TPUSpawnStrategy only supports `sum`, `mean`, `avg` for the reduce operation, got:"
                f" {reduce_op}"
            )

        import torch_xla.core.xla_model as xm
        xm.mark_step()

        torch.distributed.all_reduce(output, op=torch.distributed.ReduceOp.SUM)
        if isinstance(reduce_op, str) and reduce_op.lower() in ("avg", "mean"):
            output = output / self.world_size

        xm.mark_step()
        return output.cpu()

class NLPSaveRestoreConnector(SaveRestoreConnector):
    def __init__(self) -> None:
        if not HAVE_APEX:
            logging.warning(
                "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/apex\n"
                "Megatron-based models require Apex to function correctly."
            )
            # raise ImportError(
            #    "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            # )
        super().__init__()

    def save_to(self, model, save_path: str):
        app_state = AppState()
        if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:

            dir_name = os.path.dirname(save_path)

            # first we save the weights for each model parallel rank
            if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:
                if app_state.data_parallel_rank == 0:
                    if app_state.pipeline_model_parallel_size == 1:
                        mp_model_weights = os.path.join(
                            dir_name, f'tp_rank_{app_state.tensor_model_parallel_rank:02d}_' + self.model_weights_ckpt
                        )
                    else:
                        mp_model_weights = os.path.join(
                            dir_name,
                            f'tp_rank_{app_state.tensor_model_parallel_rank:02d}_pp_rank_{app_state.pipeline_model_parallel_rank:03d}_'
                            + self.model_weights_ckpt,
                        )

                    self._save_state_dict_to_disk(model.state_dict(), mp_model_weights)

                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

                # create nemo file from folder with all mp_ranks checkpoints
                if (
                    app_state.pipeline_model_parallel_rank == 0
                    and app_state.tensor_model_parallel_rank == 0
                    and app_state.data_parallel_rank == 0
                ):
                    with tempfile.TemporaryDirectory() as tmpdir:

                        if app_state.pipeline_model_parallel_size == 1:
                            # move weights to the tmpdir
                            for tp_rank in range(app_state.tensor_model_parallel_size):
                                os.makedirs(os.path.join(tmpdir, f'tp_rank_{tp_rank:02d}'))
                                mp_model_weights = os.path.join(
                                    dir_name, f'tp_rank_{tp_rank:02d}_' + self.model_weights_ckpt
                                )
                                shutil.move(
                                    mp_model_weights,
                                    os.path.join(tmpdir, f'tp_rank_{tp_rank:02d}', self.model_weights_ckpt),
                                )
                        else:
                            # move weights to the tmpdir
                            for tp_rank, pp_rank in itertools.product(
                                range(app_state.tensor_model_parallel_size),
                                range(app_state.pipeline_model_parallel_size),
                            ):
                                os.makedirs(os.path.join(tmpdir, f'tp_rank_{tp_rank:02d}_pp_rank_{pp_rank:03d}'))
                                mp_model_weights = os.path.join(
                                    dir_name, f'tp_rank_{tp_rank:02d}_pp_rank_{pp_rank:03d}_' + self.model_weights_ckpt
                                )
                                shutil.move(
                                    mp_model_weights,
                                    os.path.join(
                                        tmpdir, f'tp_rank_{tp_rank:02d}_pp_rank_{pp_rank:03d}', self.model_weights_ckpt
                                    ),
                                )

                        # create config and artifacts in tmpdir
                        config_yaml = os.path.join(tmpdir, self.model_config_yaml)
                        model.to_config_file(path2yaml_file=config_yaml)
                        if hasattr(model, 'artifacts') and model.artifacts is not None:
                            self._handle_artifacts(model, nemo_file_folder=tmpdir)
                            self._update_artifact_paths(model, path2yaml_file=config_yaml)

                        # create tar file
                        self._make_nemo_file_from_folder(save_path, tmpdir)

        else:
            return super().save_to(model, save_path)

    def modify_state_dict(self, conf, state_dict):
        if conf.get('megatron_legacy', False):
            new_state_dict = {}
            for key in state_dict.keys():
                new_key = key.replace('bert_model.language_model', 'bert_model.model.language_model')
                new_key = new_key.replace('transformer', 'encoder')
                new_key = new_key.replace('.attention.', '.self_attention.')
                new_state_dict[new_key] = state_dict[key]
            state_dict = new_state_dict

        if conf.get('megatron_amp_O2', False):
            new_state_dict = {}
            for key in state_dict.keys():
                new_key = key.replace('model.', 'model.module.', 1)
                new_state_dict[new_key] = state_dict[key]
            state_dict = new_state_dict

        return state_dict

    def restore_from(
        self,
        calling_cls,
        restore_path: str,
        override_config_path: Optional[Union[OmegaConf, str]] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = True,
        return_config: bool = False,
        trainer: Trainer = None,
    ):
        """
        Restores model instance (weights and configuration) into .nemo file

        Args:
            restore_path: path to .nemo file from which model should be instantiated
            override_config_path: path to a yaml config that will override the internal
                config file or an OmegaConf / DictConfig object representing the model config.
            map_location: Optional torch.device() to map the instantiated model to a device.
                By default (None), it will select a GPU if available, falling back to CPU otherwise.
            strict: Passed to load_state_dict. By default True
            return_config: If set to true, will return just the underlying config of the restored
                model as an OmegaConf DictConfig object without instantiating the model.

        Example:
            ```
            model = nemo.collections.nlp.models.TextClassification.restore_from('asr.nemo')
            assert isinstance(model, nemo.collections.nlp.models.TextClassification)
            ```

        Returns:
            An instance of type cls or its underlying config (if return_config is set).
        """
        # Get path where the command is executed - the artifacts will be "retrieved" there
        # (original .nemo behavior)
        loaded_params = super().load_config_and_state_dict(
            calling_cls, restore_path, override_config_path, map_location, strict, return_config, trainer,
        )
        if not isinstance(loaded_params, tuple):
            return loaded_params
        conf, instance, state_dict = loaded_params
        state_dict = self.modify_state_dict(conf, state_dict)
        super().load_instance_with_state_dict(instance, state_dict, strict)
        logging.info(f'Model {instance.__class__.__name__} was successfully restored from {restore_path}.')
        return instance


class PipelineMixedPrecisionPlugin(NativeMixedPrecisionPlugin):
    """ Overrides PTL autocasting to not wrap training/val/test_step.
        We do this because we have the Apex fwd/bwd functions in training_step.
        This means .backward is being called in training_step so we do not want the whole
        step wrapped in autocast.

        We instead wrap the fwd_output_and_loss_func that is passed to the Apex fwd/bwd functions.
    """

    def __init__(
        self, precision: Union[str, int], device: str, scaler: Optional[torch.cuda.amp.GradScaler] = None
    ) -> None:
        super().__init__(precision, device, scaler=scaler)

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """Have the PTL context manager do nothing."""
        yield


class GradScaler(torch.cuda.amp.GradScaler):
    """
    Gradient sclaer for model-parallel inf check. The inf in gradients are checked across tensor-parallel
    ranks in (1) executing optimizer step and (2) gradient scaler update.

    """

    def __init__(
        self,
        init_scale=2.0 ** 16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=True,
        hysteresis=1,
    ):
        super().__init__(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )
        self.optimizer_update_skipped: Optional[bool] = None
        self.hysteresis = hysteresis
        self._hysteresis_tracker = self.hysteresis

    def _unscale_grads_(self, optimizer, *args):
        if getattr(optimizer, "_custom_amp_unscale_grads", False):
            return optimizer.unscale_grads(*args)
        else:
            return super()._unscale_grads_(optimizer, *args)

    def _maybe_opt_step(self, optimizer, optimizer_state, *args, **kwargs):
        retval = None
        found_inf = torch.cuda.FloatTensor([sum(v.item() for v in optimizer_state["found_inf_per_device"].values())])

        # Update across all model parallel instances.
        torch.distributed.all_reduce(
            found_inf, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_model_parallel_group()
        )

        if found_inf.item() == 0:
            retval = optimizer.step(*args, **kwargs)
            self.optimizer_update_skipped = False
        else:
            self.optimizer_update_skipped = True
        return retval

    def update(self, new_scale=None):
        """
        Updates to native grad scaler update function.
        1. Check inf across model-parallel ranks.
        2. Update hysteresis tracker.
        3. Apply hysteresis to grad scale update.
        """
        if not self._enabled:
            return

        _scale, _growth_tracker = self._check_scale_growth_tracker("update")

        if new_scale is not None:
            # Accept a new user-defined scale.
            if isinstance(new_scale, float):
                self._scale.fill_(new_scale)  # type: ignore[union-attr]
            else:
                reason = "new_scale should be a float or a 1-element torch.cuda.FloatTensor with requires_grad=False."
                assert isinstance(new_scale, torch.cuda.FloatTensor), reason  # type: ignore[attr-defined]
                assert new_scale.numel() == 1, reason
                assert new_scale.requires_grad is False, reason
                self._scale.copy_(new_scale)  # type: ignore[union-attr]
        else:
            # Consume shared inf/nan data collected from optimizers to update the scale.
            # If all found_inf tensors are on the same device as self._scale, this operation is asynchronous.
            found_infs = [
                found_inf.to(device=_scale.device, non_blocking=True)
                for state in self._per_optimizer_states.values()
                for found_inf in state["found_inf_per_device"].values()
            ]

            assert len(found_infs) > 0, "No inf checks were recorded prior to update."

            found_inf_combined = found_infs[0]

            # Update across all model parallel instances.
            torch.distributed.all_reduce(
                found_inf_combined, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_model_parallel_group()
            )

            if len(found_infs) > 1:
                for i in range(1, len(found_infs)):
                    found_inf = found_infs[i]
                    # Update across all model parallel instances.
                    torch.distributed.all_reduce(
                        found_inf, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_model_parallel_group()
                    )
                    found_inf_combined += found_inf

            if found_inf_combined > 0:
                self._hysteresis_tracker -= 1
                if self._hysteresis_tracker <= 0:
                    # When hysteresis becomes zero, follow the native grad scale update rule.
                    # Increase scale and reset growth tracker
                    torch._amp_update_scale_(
                        _scale,
                        _growth_tracker,
                        found_inf_combined,
                        self._growth_factor,
                        self._backoff_factor,
                        self._growth_interval,
                    )
                else:
                    # Only reset the growth tracker when hysteresis is larger than zero
                    _growth_tracker.fill_(0.0)
            else:
                # When no inf found, follow the native grad scale update rule.
                # Increment growth_tracker, update scale when growth tracker reaches the interval, and
                # reset the hysteresis tracker.
                torch._amp_update_scale_(
                    _scale,
                    _growth_tracker,
                    found_inf_combined,
                    self._growth_factor,
                    self._backoff_factor,
                    self._growth_interval,
                )
                self._hysteresis_tracker = self.hysteresis

        # To prepare for next iteration, clear the data collected from optimizers this iteration.
        self._per_optimizer_states = defaultdict(torch.cuda.amp.grad_scaler._refresh_per_optimizer_state)

    def state_dict(self):
        """
        Add hysteresis_tracker to the native functions' state_dict
        """
        return (
            {
                "scale": self.get_scale(),
                "growth_factor": self._growth_factor,
                "backoff_factor": self._backoff_factor,
                "growth_interval": self._growth_interval,
                "_growth_tracker": self._get_growth_tracker(),
                "_hysteresis_tracker": self._hysteresis_tracker,
            }
            if self._enabled
            else {}
        )

    def load_state_dict(self, state_dict):
        """
        Load hysteresis_tracker in addition to the state dict of the native function
        """
        if not self._enabled:
            return

        if len(state_dict) == 0:
            raise RuntimeError(
                "The source state dict is empty, possibly because it was saved "
                "from a disabled instance of GradScaler."
            )

        self._init_scale = state_dict["scale"]
        if self._scale is not None:
            self._scale.fill_(state_dict["scale"])
        self._growth_factor = state_dict["growth_factor"]
        self._backoff_factor = state_dict["backoff_factor"]
        self._growth_interval = state_dict["growth_interval"]
        self._init_growth_tracker = state_dict["_growth_tracker"]
        if self._growth_tracker is not None:
            self._growth_tracker.fill_(state_dict["_growth_tracker"])
        if "_hysterisis_tracker" in state_dict:
            self._hysteresis_tracker = state_dict["_hysterisis_tracker"]
        else:
            self._hysteresis_tracker = 1


class MegatronHalfPrecisionPlugin(NativeMixedPrecisionPlugin):
    """
    Plugin for Half (FP16 and BF16) precision training.
    This plugin assumes the use of the optimizer with master parameters (fp32).
    This plugin uses half-precision at all operators in the model so need of input precision
    at each layer operator.

    Args:
        precision: Whether to use ``torch.float16`` (``16``) or ``torch.bfloat16`` (``'bf16'``).
        device: The device for ``torch.autocast``.
        scaler: An optional :class:`torch.cuda.amp.GradScaler` to use.
    """

    def __init__(
        self, precision: Union[str, int], device: str, scaler: Optional[torch.cuda.amp.GradScaler] = None
    ) -> None:
        super().__init__(precision, device, scaler)

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        model: Union["pl.LightningModule", torch.nn.Module],
        optimizer_idx: int,
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> None:
        assert isinstance(
            optimizer, MainParamsOptimizerWrapper
        ), "MegatronHalfPrecisionPlugin supports only the optimizer with master parameters"

        if self.scaler is None:
            assert optimizer.fp32_grad_accumulation, "BF16 uses FP32 grad accumulation"
            _ = closure()
            self._after_closure(model, optimizer, optimizer_idx)
            return optimizer.step(**kwargs)

        if isinstance(optimizer, torch.optim.LBFGS):
            raise MisconfigurationException(
                f"Native AMP and the LBFGS optimizer are not compatible (optimizer {optimizer_idx})."
            )
        assert not optimizer.fp32_grad_accumulation, "FP16 uses FP16 grad accumulation"
        closure_result = closure()

        # TODO: Add an option for merged all-reduce

        # cast fp16 grads to fp32 and copy to main grads, which are used for unscale and param update
        optimizer.copy_model_grads_to_main_grads()
        # `unscale` after the closure is executed but before the `on_before_optimizer_step` hook.
        # unscale main (fp32) gradients
        self.scaler.unscale_(optimizer)
        self._after_closure(model, optimizer, optimizer_idx)
        skipped_backward = closure_result is None
        # in manual optimization, the closure does not return a value
        if not isinstance(model, pl.LightningModule) or not model.automatic_optimization or not skipped_backward:
            # note: the scaler will skip the `optimizer.step` if nonfinite gradients are found
            self.scaler.step(optimizer, **kwargs)
            self.scaler.update()

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """ No explicit precision casting. Inputs are supposed to be manually casted """
        try:
            yield
        finally:
            pass


class GlobalBatchDataFetcher(DataFetcher):
    """ Overrides PTL DataFetcher. Used to fetch global batches."""

    def __init__(self, prefetch_batches: int = 0, store_on_device: bool = False) -> None:

        if not HAVE_APEX:
            logging.warning("Apex was not found. Using model parallel or megatron models will error out.")

        super().__init__(prefetch_batches=prefetch_batches, store_on_device=store_on_device)

    def _fetch_next_batch(self, iterator: Iterator) -> None:
        start_output = self.on_fetch_start()
        batch = [next(iterator) for _ in range(get_num_microbatches())]
        self.fetched += 1
        if not self.prefetch_batches and self._has_len:
            # when we don't prefetch but the dataloader is sized, we use the length for `done`
            dataloader = self.dataloader
            assert isinstance(dataloader, Sized)  # `_has_len` is True
            self.done = self.fetched >= len(dataloader)
        self.on_fetch_end(batch, start_output)
