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

"""Transformer based language model."""
import torch
from nemo.utils import logging
from nemo.collections.nlp.modules.common.megatron.layer_type import LayerType
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.rotary_pos_embedding import RotaryEmbedding
from nemo.collections.nlp.modules.common.megatron.transformer import ParallelTransformer
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    get_linear_layer,
    init_method_normal,
    scaled_init_method_normal,
)

try:
    from apex.transformer import parallel_state, tensor_parallel
    from apex.transformer.enums import AttnMaskType

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

    # fake missing classes with None attributes
    AttnMaskType = ApexGuardDefaults()
    LayerType = ApexGuardDefaults()


def get_language_model(
    hidden_size,
    ffn_hidden_size,
    num_layers,
    max_position_embeddings,
    num_tokentypes,
    add_pooler,
    vocab_size,
    num_attention_heads,
    encoder_attn_mask_type,
    apply_query_key_layer_scaling=True,
    kv_channels=None,
    init_method=None,
    scaled_init_method=None,
    add_decoder=False,
    decoder_attn_mask_type=AttnMaskType.causal,
    pre_process=True,
    post_process=True,
    init_method_std=0.02,
    resume_from_checkpoint=False,
    use_cpu_initialization=False,
    hidden_dropout=0.1,
    attention_dropout=0.1,
    ffn_dropout=0.0,
    precision=16,
    fp32_residual_connection=False,
    activations_checkpoint_method=None,
    activations_checkpoint_num_layers=1,
    disable_layer_norm_checkpointing=False,
    normalization='layernorm',
    layernorm_epsilon=1e-5,
    bias_activation_fusion=True,
    masked_softmax_fusion=True,
    activation='gelu',
    headscale=False,
    transformer_block_type='pre_ln',
    normalize_attention_scores=True,
    position_embedding_type='learned_absolute',
    attention_type='multihead',
    share_embeddings_and_output_weights=True,
    rotary_percentage=1.0,
    multi_query_attention=False,
    bias_dropout_add_fusion=True,
    bias=True,
    gradient_accumulation_fusion=False,
    persist_layer_norm=False,
    openai_gelu=False,
    onnx_safe=False,
    megatron_legacy=False,
    activations_checkpoint_granularity=None,
    activations_checkpoint_layers_per_pipeline=None,
    sequence_parallel=False,
    transformer_engine=False,
    fp8=False,
    fp8_e4m3=False,
    fp8_hybrid=False,
    fp8_margin=0,
    fp8_interval=1,
    fp8_amax_history_len=1,
    fp8_amax_compute_algo='most_recent',
    reduce_amax=True,
    use_emha=False,
    position_interpolation_factor=1.0,
    position_freq_base=10000,
    position_abf_factor=1,
    num_kv_heads=None,
    sliding_window=None
):
    """Build language model and return along with the key to save."""

    if kv_channels is None:
        assert (
            hidden_size % num_attention_heads == 0
        ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
        kv_channels = hidden_size // num_attention_heads
    if num_kv_heads is not None:
        assert num_attention_heads % num_kv_heads == 0, 'number of query heads should be divisible by kv heads'
    if init_method is None:
        init_method = init_method_normal(init_method_std)

    if scaled_init_method is None:
        scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)

    logging.trace(f"In get_language_model() enter TransformerLanguageModel()", trace_type="recovery_time")
    # Language model.
    language_model = TransformerLanguageModel(
        init_method=init_method,
        output_layer_init_method=scaled_init_method,
        encoder_attn_mask_type=encoder_attn_mask_type,
        num_tokentypes=num_tokentypes,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        apply_query_key_layer_scaling=apply_query_key_layer_scaling,
        kv_channels=kv_channels,
        ffn_hidden_size=ffn_hidden_size,
        add_decoder=add_decoder,
        decoder_attn_mask_type=decoder_attn_mask_type,
        add_pooler=add_pooler,
        pre_process=pre_process,
        post_process=post_process,
        resume_from_checkpoint=resume_from_checkpoint,
        use_cpu_initialization=use_cpu_initialization,
        hidden_dropout=hidden_dropout,
        attention_dropout=attention_dropout,
        ffn_dropout=ffn_dropout,
        precision=precision,
        fp32_residual_connection=fp32_residual_connection,
        activations_checkpoint_method=activations_checkpoint_method,
        activations_checkpoint_num_layers=activations_checkpoint_num_layers,
        normalization=normalization,
        layernorm_epsilon=layernorm_epsilon,
        bias_activation_fusion=bias_activation_fusion,
        bias_dropout_add_fusion=bias_dropout_add_fusion,
        bias=bias,
        rotary_percentage=rotary_percentage,
        share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        masked_softmax_fusion=masked_softmax_fusion,
        gradient_accumulation_fusion=gradient_accumulation_fusion,
        activation=activation,
        headscale=headscale,
        transformer_block_type=transformer_block_type,
        normalize_attention_scores=normalize_attention_scores,
        position_embedding_type=position_embedding_type,
        multi_query_attention=multi_query_attention,
        persist_layer_norm=persist_layer_norm,
        openai_gelu=openai_gelu,
        onnx_safe=onnx_safe,
        megatron_legacy=megatron_legacy,
        activations_checkpoint_granularity=activations_checkpoint_granularity,
        activations_checkpoint_layers_per_pipeline=activations_checkpoint_layers_per_pipeline,
        disable_layer_norm_checkpointing=disable_layer_norm_checkpointing,
        sequence_parallel=sequence_parallel,
        transformer_engine=transformer_engine,
        fp8=fp8,
        fp8_e4m3=fp8_e4m3,
        fp8_hybrid=fp8_hybrid,
        fp8_margin=fp8_margin,
        fp8_interval=fp8_interval,
        fp8_amax_history_len=fp8_amax_history_len,
        fp8_amax_compute_algo=fp8_amax_compute_algo,
        reduce_amax=reduce_amax,
        use_emha=use_emha,
        position_interpolation_factor=position_interpolation_factor,
        position_freq_base=position_freq_base,
        position_abf_factor=position_abf_factor,
        num_kv_heads=num_kv_heads,
        sliding_window=sliding_window
    )
    logging.trace(f"In get_language_model() leave TransformerLanguageModel()", trace_type="recovery_time")

    # key used for checkpoints.
    language_model_key = 'language_model'

    return language_model, language_model_key


class Pooler(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, hidden_size, init_method, sequence_parallel=False):
        super(Pooler, self).__init__()
        self.dense = get_linear_layer(hidden_size, hidden_size, init_method)
        self.sequence_parallel = sequence_parallel

    def forward(self, hidden_states, sequence_index=0):
        # hidden_states: [s, b, h] prompt_embeddings
        # sequence_index: index of the token to pool.

        # gather data along sequence dimensions
        # same pooler is run on all tensor parallel nodes
        if self.sequence_parallel:
            hidden_states = tensor_parallel.mappings.gather_from_sequence_parallel_region(hidden_states)

        pooled = hidden_states[sequence_index, :, :]
        pooled = self.dense(pooled)
        pooled = torch.tanh(pooled)
        return pooled


class Embedding(MegatronModule):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
        resume_from_checkpoint: whether the training was resuming from a checkpoint. If so, then no need to initialize the weights
        use_cpu_initialization: whether to initialize the weights in CPU
        position_embedding_type: position embedding type determines whether we instantiate a learnable position embedding table.
    """

    def __init__(
        self,
        hidden_size,
        vocab_size,
        max_sequence_length,
        embedding_dropout_prob,
        init_method,
        num_tokentypes=0,
        resume_from_checkpoint=False,
        use_cpu_initialization=False,
        fp32_residual_connection=False,
        sequence_parallel=False,
        position_embedding_type='learned_absolute',
        transpose_batch_sequence=True,
    ):
        super(Embedding, self).__init__()
        self.hidden_size = hidden_size
        self.init_method = init_method
        self.num_tokentypes = num_tokentypes
        self.position_embedding_type = position_embedding_type
        self.transpose_batch_sequence = transpose_batch_sequence

        # Word embeddings (parallel).
        self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
            vocab_size, self.hidden_size, init_method=self.init_method,
            resume_from_checkpoint=resume_from_checkpoint,
            use_cpu_initialization=use_cpu_initialization,
        )
        self._word_embeddings_key = 'word_embeddings'

        if self.position_embedding_type == 'learned_absolute':
            # Position embedding (serial).
            self.position_embeddings = torch.nn.Embedding(max_sequence_length, self.hidden_size)
            self._position_embeddings_key = 'position_embeddings'
            # Initialize the position embeddings.
            self.init_method(self.position_embeddings.weight)

        # Token type embedding.
        # Add this as an optional field that can be added through
        # method call so we can load a pretrain model without
        # token types and add them as needed.
        self._tokentype_embeddings_key = 'tokentype_embeddings'
        if self.num_tokentypes > 0:
            self.tokentype_embeddings = torch.nn.Embedding(self.num_tokentypes, self.hidden_size)
            # Initialize the token-type embeddings.
            self.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        self.fp32_residual_connection = fp32_residual_connection
        self.sequence_parallel = sequence_parallel

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

    def zero_parameters(self):
        """Zero out all parameters in embedding."""
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.shared = True
        if self.position_embedding_type == 'learned_absolute':
            self.position_embeddings.weight.data.fill_(0)
            self.position_embeddings.weight.shared = True
        if self.num_tokentypes > 0:
            self.tokentype_embeddings.weight.data.fill_(0)
            self.tokentype_embeddings.weight.shared = True

    def add_tokentype_embeddings(self, num_tokentypes):
        """Add token-type embedding. This function is provided so we can add
        token-type embeddings in case the pretrained model does not have it.
        This allows us to load the model normally and then add this embedding.
        """
        if self.tokentype_embeddings is not None:
            raise Exception('tokentype embeddings is already initialized')
        if torch.distributed.get_rank() == 0:
            print('adding embedding for {} tokentypes'.format(num_tokentypes), flush=True)
        self.num_tokentypes = num_tokentypes
        self.tokentype_embeddings = torch.nn.Embedding(num_tokentypes, self.hidden_size)
        # Initialize the token-type embeddings.
        self.init_method(self.tokentype_embeddings.weight)

    def forward(self, input_ids, position_ids=None, token_type_ids=None):
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        if self.position_embedding_type == 'learned_absolute':
            assert position_ids is not None
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = words_embeddings + position_embeddings
        else:
            embeddings = words_embeddings
        if token_type_ids is not None:
            assert self.tokentype_embeddings is not None
            embeddings = embeddings + self.tokentype_embeddings(token_type_ids)
        else:
            assert self.tokentype_embeddings is None

        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        if self.transpose_batch_sequence:
            embeddings = embeddings.transpose(0, 1).contiguous()

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()

        # Dropout.
        if self.sequence_parallel:
            embeddings = tensor_parallel.mappings.scatter_to_sequence_parallel_region(embeddings)
            with tensor_parallel.random.get_xla_rng_tracker().fork():
                embeddings = self.embedding_dropout(embeddings)
        else:
            embeddings = self.embedding_dropout(embeddings)

        return embeddings

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        state_dict_[self._word_embeddings_key] = self.word_embeddings.state_dict(destination, prefix, keep_vars)
        if self.position_embedding_type == 'learned_absolute':
            state_dict_[self._position_embeddings_key] = self.position_embeddings.state_dict(
                destination, prefix, keep_vars
            )
        if self.num_tokentypes > 0:
            state_dict_[self._tokentype_embeddings_key] = self.tokentype_embeddings.state_dict(
                destination, prefix, keep_vars
            )

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Word embedding.
        if self._word_embeddings_key in state_dict:
            state_dict_ = state_dict[self._word_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'word_embeddings' in key:
                    state_dict_[key.split('word_embeddings.')[1]] = state_dict[key]
        self.word_embeddings.load_state_dict(state_dict_, strict=strict)

        if self.position_embedding_type == 'learned_absolute':
            # Position embedding.
            if self._position_embeddings_key in state_dict:
                state_dict_ = state_dict[self._position_embeddings_key]
            else:
                # for backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if 'position_embeddings' in key:
                        state_dict_[key.split('position_embeddings.')[1]] = state_dict[key]
            self.position_embeddings.load_state_dict(state_dict_, strict=strict)

        # Tokentype embedding.
        if self.num_tokentypes > 0:
            state_dict_ = {}
            if self._tokentype_embeddings_key in state_dict:
                state_dict_ = state_dict[self._tokentype_embeddings_key]
            else:
                # for backward compatibility.
                for key in state_dict.keys():
                    if 'tokentype_embeddings' in key:
                        state_dict_[key.split('tokentype_embeddings.')[1]] = state_dict[key]
            if len(state_dict_.keys()) > 0:
                self.tokentype_embeddings.load_state_dict(state_dict_, strict=strict)
            else:
                print(
                    '***WARNING*** expected tokentype embeddings in the ' 'checkpoint but could not find it',
                    flush=True,
                )


class TransformerLanguageModel(MegatronModule):
    """Transformer language model.

    Arguments:
        transformer_hparams: transformer hyperparameters
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(
        self,
        init_method,
        output_layer_init_method,
        encoder_attn_mask_type,
        vocab_size,
        max_position_embeddings,
        hidden_size,
        ffn_hidden_size,
        num_layers,
        num_tokentypes,
        num_attention_heads,
        apply_query_key_layer_scaling=True,
        kv_channels=None,
        add_decoder=False,
        decoder_attn_mask_type=AttnMaskType.causal,
        add_pooler=False,
        pre_process=True,
        post_process=True,
        resume_from_checkpoint=False,
        use_cpu_initialization=False,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.0,
        precision=16,
        fp32_residual_connection=False,
        activations_checkpoint_method=None,
        activations_checkpoint_num_layers=1,
        normalization='layernorm',
        layernorm_epsilon=1e-5,
        bias_activation_fusion=True,
        bias_dropout_add_fusion=True,
        bias=True,
        masked_softmax_fusion=True,
        activation='gelu',
        headscale=False,
        transformer_block_type='pre_ln',
        normalize_attention_scores=True,
        position_embedding_type='learned_absolute',
        rotary_percentage=1.0,
        multi_query_attention=False,
        share_embeddings_and_output_weights=True,
        gradient_accumulation_fusion=False,
        persist_layer_norm=False,
        openai_gelu=False,
        onnx_safe=False,
        megatron_legacy=False,
        activations_checkpoint_granularity=None,
        activations_checkpoint_layers_per_pipeline=None,
        disable_layer_norm_checkpointing=False,
        sequence_parallel=False,
        transformer_engine=False,
        fp8=False,
        fp8_e4m3=False,
        fp8_hybrid=False,
        fp8_margin=0,
        fp8_interval=1,
        fp8_amax_history_len=1,
        fp8_amax_compute_algo='most_recent',
        reduce_amax=True,
        use_emha=False,
        position_interpolation_factor=1.0,
        position_freq_base=10000,
        position_abf_factor=1,
        num_kv_heads=None,
        sliding_window=None
    ):
        super(TransformerLanguageModel, self).__init__(share_token_embeddings=share_embeddings_and_output_weights)

        self.pre_process = pre_process
        self.post_process = post_process
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.num_tokentypes = num_tokentypes
        self.init_method = init_method
        self.encoder_attn_mask_type = encoder_attn_mask_type
        self.add_decoder = add_decoder
        self.decoder_attn_mask_type = decoder_attn_mask_type
        self.add_pooler = add_pooler
        self.hidden_dropout = hidden_dropout
        self.output_layer_init_method = output_layer_init_method
        self.position_embedding_type = position_embedding_type
        self.position_interpolation_factor = position_interpolation_factor
        self.position_freq_base = position_freq_base
        self.position_abf_factor = position_abf_factor
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.sequence_parallel = sequence_parallel
        self.rotary_percentage=rotary_percentage
        assert 0 < rotary_percentage <= 1

        if kv_channels is None:

            assert (
                hidden_size % num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = hidden_size // num_attention_heads

        # Embeddings.
        if self.pre_process:
            logging.trace(f"In TransformerLanguageModel() enter Embedding()", trace_type="recovery_time")
            self.embedding = Embedding(
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_position_embeddings,
                init_method=self.init_method,
                num_tokentypes=self.num_tokentypes,
                resume_from_checkpoint=resume_from_checkpoint,
                use_cpu_initialization=use_cpu_initialization,
                embedding_dropout_prob=self.hidden_dropout,
                sequence_parallel=sequence_parallel,
                position_embedding_type=position_embedding_type,
                fp32_residual_connection=fp32_residual_connection,
            )
            logging.trace(f"In TransformerLanguageModel() leave Embedding()", trace_type="recovery_time")
            self._embedding_key = 'embedding'
        # Move Rope to core attention
        # TODO: check perf penalty vs original Nemo implementation
        # if position_embedding_type == 'rope':
        #     rotary_dim = self.hidden_size // num_attention_heads if kv_channels is None else kv_channels
        #     assert 0 < rotary_percentage <= 1
        #     if rotary_percentage < 1:
        #         rotary_dim = int(rotary_dim * rotary_percentage)
        #     self.rotary_pos_emb = RotaryEmbedding(rotary_dim)
        # Transformer.
        logging.trace(f"In TransformerLanguageModel() create encoder with resume={resume_from_checkpoint} begin", trace_type="recovery_time")
        self.encoder = ParallelTransformer(
            init_method=self.init_method,
            output_layer_init_method=self.output_layer_init_method,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            num_attention_heads=num_attention_heads,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            ffn_hidden_size=ffn_hidden_size,
            self_attn_mask_type=self.encoder_attn_mask_type,
            pre_process=self.pre_process,
            post_process=self.post_process,
            precision=precision,
            fp32_residual_connection=fp32_residual_connection,
            activations_checkpoint_method=activations_checkpoint_method,
            activations_checkpoint_num_layers=activations_checkpoint_num_layers,
            normalization=normalization,
            layernorm_epsilon=layernorm_epsilon,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            resume_from_checkpoint=resume_from_checkpoint,
            use_cpu_initialization=use_cpu_initialization,
            persist_layer_norm=persist_layer_norm,
            openai_gelu=openai_gelu,
            onnx_safe=onnx_safe,
            bias=bias,
            bias_activation_fusion=bias_activation_fusion,
            bias_dropout_add_fusion=bias_dropout_add_fusion,
            masked_softmax_fusion=masked_softmax_fusion,
            activation=activation,
            headscale=headscale,
            transformer_block_type=transformer_block_type,
            normalize_attention_scores=normalize_attention_scores,
            multi_query_attention=multi_query_attention,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
            megatron_legacy=megatron_legacy,
            sequence_parallel=sequence_parallel,
            activations_checkpoint_granularity=activations_checkpoint_granularity,
            activations_checkpoint_layers_per_pipeline=activations_checkpoint_layers_per_pipeline,
            disable_layer_norm_checkpointing=disable_layer_norm_checkpointing,
            transformer_engine=transformer_engine,
            fp8=fp8,
            fp8_e4m3=fp8_e4m3,
            fp8_hybrid=fp8_hybrid,
            fp8_margin=fp8_margin,
            fp8_interval=fp8_interval,
            fp8_amax_history_len=fp8_amax_history_len,
            fp8_amax_compute_algo=fp8_amax_compute_algo,
            reduce_amax=reduce_amax,
            use_emha=use_emha,
            position_embedding_type=self.position_embedding_type,
            position_interpolation_factor=self.position_interpolation_factor,
            position_freq_base=self.position_freq_base,
            position_abf_factor=self.position_abf_factor,
            max_position_embeddings=self.max_position_embeddings,
            rotary_percentage=self.rotary_percentage,
            num_kv_heads=num_kv_heads,
            sliding_window=sliding_window
        )
        logging.trace(f"In TransformerLanguageModel() create encoder done", trace_type="recovery_time")
        self._encoder_key = 'encoder'

        # Decoder
        if self.add_decoder:
            self.decoder = ParallelTransformer(
                layer_type=LayerType.decoder,
                self_attn_mask_type=self.decoder_attn_mask_type,
                init_method=self.init_method,
                output_layer_init_method=self.output_layer_init_method,
                num_layers=self.num_layers,
                hidden_size=self.hidden_size,
                num_attention_heads=num_attention_heads,
                apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                kv_channels=kv_channels,
                ffn_hidden_size=ffn_hidden_size,
                pre_process=self.pre_process,
                post_process=self.post_process,
                precision=precision,
                fp32_residual_connection=fp32_residual_connection,
                activations_checkpoint_method=activations_checkpoint_method,
                activations_checkpoint_num_layers=activations_checkpoint_num_layers,
                normalization=normalization,
                layernorm_epsilon=layernorm_epsilon,
                hidden_dropout=hidden_dropout,
                attention_dropout=attention_dropout,
                use_cpu_initialization=use_cpu_initialization,
                bias_activation_fusion=bias_activation_fusion,
                bias_dropout_add_fusion=bias_dropout_add_fusion,
                masked_softmax_fusion=masked_softmax_fusion,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
                persist_layer_norm=persist_layer_norm,
                openai_gelu=openai_gelu,
                onnx_safe=onnx_safe,
                megatron_legacy=megatron_legacy,
                sequence_parallel=sequence_parallel,
                activations_checkpoint_granularity=activations_checkpoint_granularity,
                activations_checkpoint_layers_per_pipeline=activations_checkpoint_layers_per_pipeline,
                transformer_engine=transformer_engine,
                position_interpolation_factor=self.position_interpolation_factor,
                position_freq_base=self.position_freq_base,
                position_abf_factor=self.position_abf_factor,
                max_position_embeddings=self.max_position_embeddings,
                rotary_percentage=self.rotary_percentage,
            )
            self._decoder_key = 'decoder'

        if self.post_process:
            # Pooler.
            if self.add_pooler:
                self.pooler = Pooler(self.hidden_size, self.init_method, sequence_parallel=sequence_parallel)
                self._pooler_key = 'pooler'

            if not self.share_embeddings_and_output_weights:
                no_async_tensor_model_parallel_allreduce = (
                    parallel_state.get_tensor_model_parallel_world_size() == 1 or sequence_parallel
                )
                self.output_layer = tensor_parallel.ColumnParallelLinear(
                    self.hidden_size,
                    self.vocab_size,
                    bias=False,  # Setting bias to False always to keep it consistent with embedding tying that also does not have a bias.
                    init_method=self.init_method,
                    skip_bias_add=True,
                    use_cpu_initialization=use_cpu_initialization,
                    gather_output=False,
                    sequence_parallel_enabled=sequence_parallel,
                    no_async_tensor_model_parallel_allreduce=no_async_tensor_model_parallel_allreduce,
                    gradient_accumulation_fusion=gradient_accumulation_fusion,
                    transfer_with_static_ring=(not (activations_checkpoint_granularity=="selective")),
                )
                self._output_layer_key = 'output_layer'

    def set_input_tensor(self, input_tensor):
        """ See megatron.model.transformer.set_input_tensor()"""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        self.encoder.set_input_tensor(input_tensor[0])

    def forward(
        self,
        enc_input_ids,
        enc_position_ids,
        enc_attn_mask,
        dec_input_ids=None,
        dec_position_ids=None,
        dec_attn_mask=None,
        enc_dec_attn_mask=None,
        token_type_ids=None,
        layer_past=None,
        get_key_value=False,
        pooling_sequence_index=0,
        enc_hidden_states=None,
        output_enc_hidden_only=False,
        encoder_input=None,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
        checkpoint_activations_all_layers=None,
    ):
        # Embeddings.
        if self.pre_process and encoder_input is None:
            encoder_input = self.embedding(enc_input_ids, enc_position_ids, token_type_ids=token_type_ids)
        else:
            pass

        # encoder_input: [s, b, h]

        # enc_attn_mask: [1, 1, s, s]
        # Move rope to core attention
        # if self.position_embedding_type == 'rope':
        #     if inference_max_sequence_len is not None:
        #         rotary_pos_emb = self.rotary_pos_emb(inference_max_sequence_len)
        #     elif self.encoder.input_tensor is not None:
        #         if self.sequence_parallel:
        #             rotary_pos_emb = self.rotary_pos_emb(
        #                 self.encoder.input_tensor.size(0) * parallel_state.get_tensor_model_parallel_world_size()
        #             )
        #         else:
        #             rotary_pos_emb = self.rotary_pos_emb(self.encoder.input_tensor.size(0))
        #     else:
        #         if self.sequence_parallel:
        #             rotary_pos_emb = self.rotary_pos_emb(
        #                 encoder_input.size(0) * parallel_state.get_tensor_model_parallel_world_size()
        #             )
        #         else:
        #             rotary_pos_emb = self.rotary_pos_emb(encoder_input.size(0))
        rotary_pos_emb = None
        # encoder.
        if enc_hidden_states is None:
            encoder_output = self.encoder(
                encoder_input,
                enc_attn_mask,
                layer_past=layer_past,
                get_key_value=get_key_value,
                set_inference_key_value_memory=set_inference_key_value_memory,
                inference_max_sequence_len=inference_max_sequence_len,
                checkpoint_activations_all_layers=checkpoint_activations_all_layers,
                rotary_pos_emb=(rotary_pos_emb, None, None)
                if rotary_pos_emb is not None
                else None,  # This assumes that this being used as a GPT/BERT model only (no cross-attention)
            )
        else:
            encoder_output = enc_hidden_states.to(encoder_input.dtype)

        if self.post_process:
            if not self.share_embeddings_and_output_weights:
                encoder_output, _ = self.output_layer(encoder_output)
            if self.add_pooler:
                pooled_output = self.pooler(encoder_output, pooling_sequence_index)

        # output_enc_hidden_only refers to when we just need the encoder's
        # output. For example, it is helpful to compute
        # similarity between two sequences by average pooling
        if not self.add_decoder or output_enc_hidden_only:
            if self.add_pooler and self.post_process:
                return encoder_output, pooled_output
            else:
                return encoder_output

        # Decoder Embedding
        dec_embedding_output = self.embedding(dec_input_ids, dec_position_ids)
        # decoder
        decoder_output = self.decoder(
            dec_embedding_output,
            dec_attn_mask,
            layer_past=layer_past,
            get_key_value=get_key_value,
            encoder_output=encoder_output,
            enc_dec_attn_mask=enc_dec_attn_mask,
            set_inference_key_value_memory=set_inference_key_value_memory,
            inference_max_sequence_len=inference_max_sequence_len,
            checkpoint_activations_all_layers=checkpoint_activations_all_layers,
        )

        if self.add_pooler and self.post_process:
            return decoder_output, encoder_output, pooled_output
        else:
            return decoder_output, encoder_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        if self.pre_process:
            state_dict_[self._embedding_key] = self.embedding.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars
            )

        state_dict_[self._encoder_key] = self.encoder.state_dict_for_save_checkpoint(destination, prefix, keep_vars)
        if self.post_process:
            if self.add_pooler:
                state_dict_[self._pooler_key] = self.pooler.state_dict_for_save_checkpoint(
                    destination, prefix, keep_vars
                )
        if self.add_decoder:
            state_dict_[self._decoder_key] = self.decoder.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars
            )

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Embedding.
        if self.pre_process:
            if self._embedding_key in state_dict:
                state_dict_ = state_dict[self._embedding_key]
            else:
                # for backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if '_embeddings' in key:
                        state_dict_[key] = state_dict[key]
            self.embedding.load_state_dict(state_dict_, strict=strict)

        # Encoder.
        if self._encoder_key in state_dict:
            state_dict_ = state_dict[self._encoder_key]

        # for backward compatibility.
        elif 'transformer' in state_dict:
            state_dict_ = state_dict['transformer']
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'transformer.' in key:
                    state_dict_[key.split('transformer.')[1]] = state_dict[key]

        # for backward compatibility.
        state_dict_self_attention = {}
        for key in state_dict_.keys():
            if '.attention.' in key:
                state_dict_self_attention[key.replace(".attention.", ".self_attention.")] = state_dict_[key]
            else:
                state_dict_self_attention[key] = state_dict_[key]
        state_dict_ = state_dict_self_attention

        self.encoder.load_state_dict(state_dict_, strict=strict)

        if self.post_process:
            # pooler
            if self.add_pooler:
                assert 'pooler' in state_dict, 'could not find data for pooler in the checkpoint'
                self.pooler.load_state_dict(state_dict[self._pooler_key], strict=strict)

            if not self.share_embeddings_and_output_weights:
                self.output_layer.load_state_dict(state_dict[self._output_layer_key], strict=strict)
        # decoder
        if self.add_decoder:
            assert 'decoder' in state_dict, 'could not find data for pooler in the checkpoint'
            self.decoder.load_state_dict(state_dict[self._decoder_key], strict=strict)
