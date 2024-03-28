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

"""GPT-2 model."""

import torch
from nemo.utils import logging
from nemo.collections.nlp.modules.common.megatron.language_model import get_language_model
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    init_method_normal,
    parallel_lm_logits,
    scaled_init_method_normal,
)
import warnings
try:
    from apex.transformer import tensor_parallel, parallel_state
    from apex.transformer.enums import AttnMaskType

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False
    # fake missing classes with None attributes
    AttnMaskType = ApexGuardDefaults()

def post_language_model_processing(
    lm_output,
    labels,
    logit_weights,
    get_key_value,
    parallel_output,
    forward_method_parallel_output,
    fp16_lm_cross_entropy,
    return_logits=False,
    sequence_parallel=False,
    gradient_accumulation_fusion=False,
    share_embeddings_and_output_weights=True,
):
    if share_embeddings_and_output_weights==False:
        lm_output = lm_output.transpose(0, 1).contiguous()
        # Output.
        if labels is None:
            return lm_output
        else:
            logits = None
            if return_logits:
                logits = output.clone()
            if fp16_lm_cross_entropy:
                assert lm_output.dtype == torch.half
                loss = tensor_parallel.vocab_parallel_cross_entropy(lm_output, labels)
            else:
                loss = tensor_parallel.vocab_parallel_cross_entropy(lm_output.float(), labels)

            return loss, logits
    if get_key_value:
        lm_output, presents = lm_output

    # Output. Format is [s b h]
    if forward_method_parallel_output is not None:
        parallel_output = forward_method_parallel_output
    async_tensor_model_parallel_allreduce = (
        parallel_state.get_tensor_model_parallel_world_size() > 1 and not sequence_parallel
    )
    output = parallel_lm_logits(
        lm_output,
        logit_weights,
        parallel_output,
        sequence_parallel=sequence_parallel,
        gradient_accumulation_fusion=gradient_accumulation_fusion,
        async_tensor_model_parallel_allreduce=async_tensor_model_parallel_allreduce,
    )
    logits = None
    if return_logits:
        # Logits are [batch x seq_length x vocab_size]
        logits = output.clone()

    if get_key_value:
        output = [output, presents]

    if labels is None:
        # [s b h] -> [b s h]
        return output.transpose(0, 1).contiguous()
    else:
        # [b s] -> [s b]
        labels = labels.transpose(0, 1).contiguous()

        if fp16_lm_cross_entropy:
            assert output.dtype == torch.half
            loss = tensor_parallel.vocab_parallel_cross_entropy(output, labels)
        else:
            loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), labels)

        # [s b] -> [b, s]
        loss = loss.transpose(0, 1).contiguous()
        return loss, logits


class GPTModel(MegatronModule):
    """GPT-2 Language model."""

    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_position_embeddings,
        num_layers,
        num_attention_heads,
        ffn_hidden_size,
        apply_query_key_layer_scaling=True,
        kv_channels=None,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=True,
        post_process=True,
        init_method_std=0.02,
        use_scaled_init_method=True,
        fp16_lm_cross_entropy=False,
        resume_from_checkpoint=False,
        use_cpu_initialization=False,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.0,
        precision=16,
        fp32_residual_connection=False,
        activations_checkpoint_granularity=None,
        activations_checkpoint_method=None,
        activations_checkpoint_num_layers=1,
        activations_checkpoint_layers_per_pipeline=None,
        disable_layer_norm_checkpointing=False,
        normalization='layernorm',
        layernorm_epsilon=1e-5,
        bias=True,
        bias_activation_fusion=True,
        bias_dropout_add_fusion=True,
        masked_softmax_fusion=True,
        activation='gelu',
        headscale=False,
        transformer_block_type='pre_ln',
        normalize_attention_scores=True,
        position_embedding_type='learned_absolute',
        rotary_percentage=1.0,
        attention_type='multihead',
        share_embeddings_and_output_weights=True,
        gradient_accumulation_fusion=False,
        persist_layer_norm=False,
        openai_gelu=False,
        onnx_safe=False,
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
        multi_query_attention=False,
        save_logits=False,
        position_interpolation_factor=1.0,
        position_freq_base=10000,
        position_abf_factor=1,
        num_kv_heads=None,
        sliding_window=None,
    ):

        super(GPTModel, self).__init__(share_token_embeddings=share_embeddings_and_output_weights)
        if sliding_window:
            if sliding_window > max_position_embeddings:
                warnings.warn(f"Warning sliding window attention value:{sliding_window} is set higher than max position embeddings value:{max_position_embeddings}.")
            elif sliding_window < max_position_embeddings:
                warnings.warn(f"Warning sliding window attention value:{sliding_window} is enabled.")
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.sequence_parallel = sequence_parallel
        self.gradient_accumulation_fusion = gradient_accumulation_fusion
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.save_logits = save_logits

        if kv_channels is None:
            assert (
                hidden_size % num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = hidden_size // num_attention_heads

        scaled_init_method = (
            scaled_init_method_normal(init_method_std, num_layers)
            if use_scaled_init_method
            else init_method_normal(init_method_std)
        )

        logging.trace(f"In GPTModel._init_() enter get_language_model()", trace_type="recovery_time")

        self.language_model, self._language_model_key = get_language_model(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            num_tokentypes=num_tokentypes,
            max_position_embeddings=max_position_embeddings,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            ffn_hidden_size=ffn_hidden_size,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            init_method=init_method_normal(init_method_std),
            scaled_init_method=scaled_init_method,
            pre_process=self.pre_process,
            post_process=self.post_process,
            init_method_std=init_method_std,
            resume_from_checkpoint=resume_from_checkpoint,
            use_cpu_initialization=use_cpu_initialization,
            precision=precision,
            fp32_residual_connection=fp32_residual_connection,
            activations_checkpoint_granularity=activations_checkpoint_granularity,
            activations_checkpoint_method=activations_checkpoint_method,
            activations_checkpoint_num_layers=activations_checkpoint_num_layers,
            activations_checkpoint_layers_per_pipeline=activations_checkpoint_layers_per_pipeline,
            disable_layer_norm_checkpointing=disable_layer_norm_checkpointing,
            normalization=normalization,
            layernorm_epsilon=layernorm_epsilon,
            rotary_percentage=rotary_percentage,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            bias=bias,
            bias_activation_fusion=bias_activation_fusion,
            bias_dropout_add_fusion=bias_dropout_add_fusion,
            masked_softmax_fusion=masked_softmax_fusion,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
            activation=activation,
            headscale=headscale,
            transformer_block_type=transformer_block_type,
            normalize_attention_scores=normalize_attention_scores,
            position_embedding_type=position_embedding_type,
            attention_type=attention_type,
            persist_layer_norm=persist_layer_norm,
            openai_gelu=openai_gelu,
            onnx_safe=onnx_safe,
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
            multi_query_attention=multi_query_attention,
            position_interpolation_factor=position_interpolation_factor,
            position_freq_base=position_freq_base,
            position_abf_factor=position_abf_factor,
            num_kv_heads=num_kv_heads,
            sliding_window=sliding_window
        )

        logging.trace(f"In GPTModel._init_() leave get_language_model()", trace_type="recovery_time")

        if self.share_embeddings_and_output_weights:
            self.initialize_word_embeddings(
                init_method=init_method_normal(init_method_std), vocab_size=vocab_size, hidden_size=hidden_size
            )

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        labels=None,
        token_type_ids=None,
        layer_past=None,
        get_key_value=False,
        forward_method_parallel_output=None,
        encoder_input=None,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
        checkpoint_activations_all_layers=None,
    ):
        # input_ids: [b, s]
        # position_ids: [b, s]
        # attention_mask: [1, 1, s, s]

        lm_output = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            layer_past=layer_past,
            get_key_value=get_key_value,
            encoder_input=encoder_input,
            set_inference_key_value_memory=set_inference_key_value_memory,
            inference_max_sequence_len=inference_max_sequence_len,
            checkpoint_activations_all_layers=checkpoint_activations_all_layers,
        )
        if self.post_process:
            return post_language_model_processing(
                lm_output,
                labels,
                self.language_model.output_layer.weight
                if not self.share_embeddings_and_output_weights
                else self.word_embeddings_weight(),
                get_key_value,
                self.parallel_output,
                forward_method_parallel_output,
                self.fp16_lm_cross_entropy,
                return_logits=self.save_logits,
                sequence_parallel=self.sequence_parallel,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights
            )
        else:
            return lm_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):

        state_dict_ = {}
        state_dict_[self._language_model_key] = self.language_model.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )
        # Save word_embeddings.
        if self.post_process and not self.pre_process:
            state_dict_[self._word_embeddings_for_head_key] = self.word_embeddings.state_dict(
                destination, prefix, keep_vars
            )
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Load word_embeddings.
        if self.post_process and not self.pre_process:
            self.word_embeddings.load_state_dict(state_dict[self._word_embeddings_for_head_key], strict=strict)
        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        self.language_model.load_state_dict(state_dict, strict=strict)
