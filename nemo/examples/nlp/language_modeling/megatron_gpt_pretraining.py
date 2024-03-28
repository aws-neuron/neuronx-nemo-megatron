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
import lightning_neuron_patch
import os
import datetime
from lightning_lite.plugins.environments import TorchElasticEnvironment
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector

from checkpoint_conversion.convert_nemo_checkpoint_to_hf_llama import convert_checkpoint as convert_checkpoint_llama

from checkpoint_conversion.convert_nemo_checkpoint_to_hf import convert_checkpoint as convert_checkpoint_gpt2
from checkpoint_conversion.convert_nemo_checkpoint_to_hf_neox import convert_checkpoint as convert_checkpoint_neox
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    PipelineMixedPrecisionPlugin,
    NLPTrainer,
    NLPCheckpointIO,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import StatelessTimer, exp_manager

@hydra_runner(config_path="conf", config_name="megatron_gpt_config")
def main(cfg) -> None:
    if cfg.enable_recovery_time_instrumentation:
        logging.add_allowed_trace_type("recovery_time")
        print(f"Entering main at {datetime.datetime.now()}")

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    megatron_amp_o2 = cfg.model.get('megatron_amp_O2', False)
    with_distributed_adam = cfg.model.optim.get('name') == 'distributed_fused_adam'
    plugins = []
    
    nlp_xla_checkpoint_io = NLPCheckpointIO(cfg.get("async_checkpointing", False))
    cluster_environment = None
    if os.environ.get("TORCHELASTIC_RUN_ID") is not None:
        cluster_environment=TorchElasticEnvironment()
    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,  # we don't use DDP for async grad allreduce
        gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
        find_unused_parameters=False,
        cluster_environment=cluster_environment,
        checkpoint_io=nlp_xla_checkpoint_io,
        megatron_amp_o2=megatron_amp_o2,
        restore_path=cfg.model.resume_from_checkpoint
    )
    if cfg.trainer.precision in [16, 'bf16']:
        scaler = None
        if cfg.trainer.precision == 16:
            scaler = GradScaler(
                init_scale=cfg.model.get('native_amp_init_scale', 2 ** 32),
                growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
                hysteresis=cfg.model.get('hysteresis', 2),
            )
        if megatron_amp_o2 and not with_distributed_adam:
            plugins.append(MegatronHalfPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))

    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    # update resume from checkpoint found by exp_manager
    if cfg.model.resume_from_checkpoint is not None:
        resume_from_checkpoint = cfg.model.resume_from_checkpoint
        trainer = NLPTrainer(plugins=plugins, strategy=strategy, resume_from_checkpoint=resume_from_checkpoint, **cfg.trainer)
    else:
        trainer = NLPTrainer(plugins=plugins, strategy=strategy, **cfg.trainer)


    exp_manager(trainer, cfg.exp_manager)
    # We use NLPCheckpointConnector which correctly loads global_step, epoch
    #trainer._checkpoint_connector = CheckpointConnector(trainer, resume_from_checkpoint=resume_from_checkpoint)
    # Override timer callback to a stateless one
    for idx, callback in enumerate(trainer.callbacks):
        if isinstance(callback, Timer):
            trainer.callbacks[idx] = StatelessTimer(cfg.trainer.max_time,)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    model = MegatronGPTModel(cfg.model, trainer)
    trainer.fit(model)
    # Convert checkpoint to HuggingFace
    if cfg.model.get('convert_to_hf', False) and cfg.exp_manager.create_checkpoint_callback:
        import torch_xla.core.xla_model as xm
        if xm.get_ordinal() == 0:
            if cfg.name == "megatron_llama":
                logging.info("Converting LLama checkpoints to HuggingFace format")
                checkpoint_path = os.path.join(os.getcwd(), "nemo_experiments", cfg.exp_manager.name, os.environ.get('SLURM_JOB_ID'), "checkpoints")
                convert_checkpoint_llama(cfg.model.config_path, checkpoint_path, cfg.model.output_dir, 2.0, True)
                logging.info("Finished converting Llama checkpoints")
            elif cfg.name == "megatron_neox":
                logging.info("Converting Neox checkpoints to HuggingFace format")
                checkpoint_path = os.path.join(os.getcwd(), "nemo_experiments", cfg.exp_manager.name, os.environ.get('SLURM_JOB_ID'), "checkpoints")
                convert_checkpoint_neox(cfg.model.config_path, checkpoint_path, cfg.model.output_dir, 2.0, True)
                logging.info("Finished converting Llama checkpoints")
            elif cfg.name == "megatron_gpt":
                logging.info("Converting GPT2 checkpoints to HuggingFace format")
                checkpoint_path = os.path.join(os.getcwd(), "nemo_experiments", cfg.exp_manager.name, os.environ.get('SLURM_JOB_ID'), "checkpoints")
                convert_checkpoint_gpt2(cfg.model.config_path, checkpoint_path, cfg.model.output_dir, 2.0, True)
                logging.info("Finished converting GPT2 checkpoints")

if __name__ == '__main__':
    main()
