from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from omegaconf.dictconfig import DictConfig
from pytorch_lightning import Trainer
from nemo.collections.nlp.models.language_modeling.megatron.bigcode_model import BigCodeModel


class MegatronBigCodeModel(MegatronGPTModel):
    """
        Pretrain BigCode Model without validation
        """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)

    def model_provider_func(self, pre_process, post_process):
        """Model depends on pipeline paralellism."""
        model = BigCodeModel(
            vocab_size=self.padded_vocab_size,
            hidden_size=self.cfg.hidden_size,
            max_position_embeddings=self.cfg.max_position_embeddings,
            num_layers=self.cfg.num_layers,
            num_attention_heads=self.cfg.num_attention_heads,
            apply_query_key_layer_scaling=self.cfg.get(
                'apply_query_key_layer_scaling', True),
            kv_channels=self.cfg.get('kv_channels', None),
            ffn_hidden_size=self.cfg.ffn_hidden_size,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
            init_method_std=self.cfg.get('init_method_std', 0.02),
            use_scaled_init_method=self.cfg.get(
                'use_scaled_init_method', True),
            fp16_lm_cross_entropy=self.cfg.get('fp16_lm_cross_entropy', False),
            use_cpu_initialization=self.cfg.get(
                'use_cpu_initialization', False),
            hidden_dropout=self.cfg.get('hidden_dropout', 0.1),
            attention_dropout=self.cfg.get('attention_dropout', 0.0),
            ffn_dropout=self.cfg.get('ffn_dropout', 0.0),
            precision=self.cfg.get('precision', 16),
            fp32_residual_connection=self.cfg.get(
                'fp32_residual_connection', False),
            activations_checkpoint_granularity=self.cfg.get(
                'activations_checkpoint_granularity', None),
            activations_checkpoint_method=self.cfg.get(
                'activations_checkpoint_method', None),
            activations_checkpoint_num_layers=self.cfg.get(
                'activations_checkpoint_num_layers', 1),
            activations_checkpoint_layers_per_pipeline=self.cfg.get(
                'activations_checkpoint_layers_per_pipeline', None
            ),
            normalization=self.cfg.get('normalization', 'layernorm'),
            layernorm_epsilon=self.cfg.get('layernorm_epsilon', 1e-5),
            onnx_safe=self.cfg.get('onnx_safe', False),
            bias_activation_fusion=self.cfg.get(
                'bias_activation_fusion', True),
            bias_dropout_add_fusion=self.cfg.get(
                'bias_dropout_add_fusion', True),
            share_embeddings_and_output_weights=self.cfg.get(
                'share_embeddings_and_output_weights', True),
            position_embedding_type=self.cfg.get(
                'position_embedding_type', 'learned_absolute'),
            rotary_percentage=self.cfg.get('rotary_percentage', 1.0),
            activation=self.cfg.get('activation', 'gelu'),
            bias=self.cfg.get('has_bias', True),
            transformer_block_type=self.cfg.get(
                'transformer_block_type', 'pre_ln'),
            masked_softmax_fusion=self.cfg.get('masked_softmax_fusion', True),
            gradient_accumulation_fusion=self.cfg.get(
                'gradient_accumulation_fusion', False),
            persist_layer_norm=self.cfg.get('persist_layer_norm', False),
            sequence_parallel=self.cfg.get('sequence_parallel', False),
            transformer_engine=self.cfg.get('transformer_engine', False),
            fp8=self.cfg.get('fp8', False),
            fp8_e4m3=self.cfg.get('fp8_e4m3', False),
            fp8_hybrid=self.cfg.get('fp8_hybrid', False),
            fp8_margin=self.cfg.get('fp8_margin', 0),
            fp8_interval=self.cfg.get('fp8_interval', 1),
            fp8_amax_history_len=self.cfg.get('fp8_amax_history_len', 1),
            fp8_amax_compute_algo=self.cfg.get(
                'fp8_amax_compute_algo', 'most_recent'),
            use_emha=self.cfg.get('use_emha', False),
            save_logits=self.cfg.get('save_logits', False)
        )

        return model
