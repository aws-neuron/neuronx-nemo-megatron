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

from lightning_lite.plugins.environments import TorchElasticEnvironment
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.models.language_modeling.megatron.llama_model import LlamaModel

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

import torch.optim.adamw as torch_adamw
from nemo.core.optim.adamw import _single_tensor_adamw_
torch_adamw._single_tensor_adamw = _single_tensor_adamw_


class PreTrainLlamaModel(MegatronGPTModel):
    """
        Pretrain LLama Model without validation
        """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)

    def model_provider_func(self, pre_process, post_process):
        """Model depends on pipeline paralellism."""
        model = LlamaModel(
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
            num_kv_heads=self.cfg.get('num_kv_heads', None),
        )

        return model


@hydra_runner(config_path="conf", config_name="megatron_llama_70b_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    megatron_amp_o2 = cfg.model.get('megatron_amp_O2', False)
    with_distributed_adam = cfg.model.optim.get('name') == 'distributed_fused_adam'
    plugins = []
    
    nlp_xla_checkpoint_io = NLPCheckpointIO()
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

    model = PreTrainLlamaModel(cfg.model, trainer)
    trainer.fit(model)

if __name__ == '__main__':
    main()
