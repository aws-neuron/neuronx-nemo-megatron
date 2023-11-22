# coding=utf-8
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

"""Transformer."""
import math
from contextlib import nullcontext
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig
from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    InfusedAdapterConfig,
    MLPInfusedAdapterConfig,
    ParallelLinearAdapterConfig,
)
from nemo.collections.nlp.modules.common.megatron.fused_bias_dropout_add import (
    bias_dropout_add,
    bias_dropout_add_fused_inference,
    bias_dropout_add_fused_train,
    dropout_add,
)

from nemo.collections.nlp.modules.common.megatron.fused_bias_gelu import fused_bias_gelu
from nemo.collections.nlp.modules.common.megatron.fused_layer_norm import get_layer_norm
from nemo.collections.nlp.modules.common.megatron.layer_type import LayerType
from nemo.collections.nlp.modules.common.megatron.rotary_pos_embedding import RotaryEmbedding, apply_rotary_pos_emb
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    attention_mask_func,
    init_method_normal,
    scaled_init_method_normal
)

from nemo.core import adapter_mixins
from nemo.utils import logging
from transformers.utils import is_torch_tpu_available

try:
    from apex.transformer import parallel_state, tensor_parallel
    from apex.transformer.enums import AttnMaskType, AttnType, ModelType
    from apex.transformer.utils import divide as safe_divide
    from apex.transformer.parallel_state import get_tensor_model_parallel_world_size
    from apex.transformer.functional.fused_softmax import FusedScaleMaskSoftmax

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

    # fake missing classes with None attributes
    ModelType = AttnMaskType = AttnType = LayerType = ApexGuardDefaults()

try:
    from transformer_engine.pytorch import TransformerLayer, fp8_autocast
    from transformer_engine.common import recipe
    from transformer_engine.pytorch.distributed import checkpoint as te_checkpoint

    HAVE_TE = True

except:
    HAVE_TE = False


    # fake missing class
    class TransformerLayer(ApexGuardDefaults):
        def __init__(self):
            super().__init__()

            logging.warning(
                "Transformer Engine was not found. transformer_engine.pytorch.transformer.TransformerLayer will not work. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
"""


def _get_falcon_add(x: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor, prob: float):
    assert bias is None, 'bias should be None for falcon'
    return x + residual


def get_falcon_add():
    return _get_falcon_add


class SharedLayer(torch.nn.Module):
    """Shared layer.

    Shared laver for key value projection

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(
            self,
            input_size, output_size,
            init_method, bias=False,
            *,
            sequence_parallel=False,
            params_dtype=torch.float32
    ):
        super(SharedLayer, self).__init__()

        self.weight = torch.nn.Parameter(
            torch.empty(
                output_size,
                input_size,
                dtype=params_dtype,
            )
        )
        init_method(self.weight)
        assert bias is False, "SharedKV does not support bias yet"
        self.sequence_parallel = sequence_parallel

        # always true, since we have to reduce weights across TP groups, as they are copies
        # note: we do not set tensor_parallel attr, so when calculating gradient norm, these
        # copies will be ignored
        setattr(self.weight, "sequence_parallel_enabled", True)

    def forward(self, input_):
        # input_: [s, b, h]

        output = torch.matmul(input_, self.weight.t())
        # gather output along sequence dimensions

        if self.sequence_parallel:
            output = tensor_parallel.mappings.gather_from_sequence_parallel_region(output)

        return output


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)

    return _bias_dropout_add


def get_dropout_add(training):
    def _dropout_add(x, bias, residual, prob):
        assert bias is None
        return dropout_add(x, bias, residual, prob, training)

    return _dropout_add


class FalconParallelTransformerLayer(MegatronModule, adapter_mixins.AdapterModuleMixin):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
            self,
            init_method,
            output_layer_init_method,
            layer_number,
            hidden_size,
            ffn_hidden_size,
            num_attention_heads,
            layer_type=LayerType.encoder,
            self_attn_mask_type=AttnMaskType.padding,
            fp32_residual_connection=False,
            precision=16,
            apply_query_key_layer_scaling=True,
            kv_channels=None,
            layernorm_epsilon=1e-5,
            hidden_dropout=0.1,
            persist_layer_norm=False,
            use_cpu_initialization=False,
            bias_activation_fusion=True,
            bias_dropout_add_fusion=True,
            masked_softmax_fusion=True,
            gradient_accumulation_fusion=False,
            openai_gelu=False,
            onnx_safe=False,
            attention_dropout=0.1,
            ffn_dropout=0.0,
            activation='gelu',
            megatron_legacy=False,
            bias=True,
            chunk_size=64,
            normalization='layernorm',
            transformer_block_type='pre_ln',
            position_embedding_type='rope',
            multi_query_attention=False,
            headscale=False,
            activations_checkpoint_granularity=None,
            sequence_parallel=False,
            normalize_attention_scores=True,
            num_moe_experts=1,
            moe_frequency=1,
            moe_dropout=0.0,
    ):
        super(FalconParallelTransformerLayer, self).__init__()

        if kv_channels is None:
            assert (
                    hidden_size % num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = hidden_size // num_attention_heads

        self.layer_number = layer_number
        self.layer_type = layer_type
        self.sequence_parallel = sequence_parallel
        self.bias = bias
        self.transformer_block_type = transformer_block_type
        self.position_embedding_type = position_embedding_type
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.normalize_attention_scores = normalize_attention_scores
        projection_size = kv_channels * num_attention_heads

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = safe_divide(projection_size, num_attention_heads)
        self.num_attention_heads_per_partition = round(num_attention_heads / world_size)
        self.num_attention_heads_partition_offset = (
                self.num_attention_heads_per_partition * parallel_state.get_tensor_model_parallel_rank()
        )

        no_async_tensor_model_parallel_allreduce = (
                parallel_state.get_tensor_model_parallel_world_size() == 1 or sequence_parallel
        )
        self.multi_query_attention = multi_query_attention
        transfer_with_static_ring = True
        assert multi_query_attention
        # fused q + mlp_linear h_to_4h layer

        self.fused_in_layer = tensor_parallel.ColumnParallelLinear(
            hidden_size,
            5 * projection_size,
            gather_output=False,
            init_method=init_method,
            use_cpu_initialization=use_cpu_initialization,
            bias=bias,
            sequence_parallel_enabled=sequence_parallel,
            no_async_tensor_model_parallel_allreduce=no_async_tensor_model_parallel_allreduce,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
            transfer_with_static_ring=transfer_with_static_ring
        )
        if (parallel_state.get_tensor_model_parallel_rank() == (world_size - 1)) and num_attention_heads % world_size:
            self.num_attention_heads_per_partition = int(
                num_attention_heads - self.num_attention_heads_per_partition * (world_size - 1))

        self.hidden_size_per_partition = self.num_attention_heads_per_partition * self.hidden_size_per_attention_head

        assert normalization == 'layernorm'
        self.input_layernorm = get_layer_norm(
            hidden_size, layernorm_epsilon, persist_layer_norm, sequence_parallel=sequence_parallel
        )
        self.key_value = SharedLayer(
            hidden_size,
            2 * kv_channels,
            init_method=init_method,
            bias=bias,
            sequence_parallel=sequence_parallel,
        )
        self.fused_out_layer = tensor_parallel.RowParallelLinear(
            5 * projection_size,
            hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            use_cpu_initialization=use_cpu_initialization,
            bias=bias,
            sequence_parallel_enabled=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
            transfer_with_static_ring=transfer_with_static_ring,
        )
        self.fp16 = precision == 16
        self.bf16 = precision == 'bf16'
        self.attention_softmax_in_fp32 = False
        if apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16,
            self.bf16,
            self_attn_mask_type,
            masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff,
        )

        assert self.position_embedding_type == 'rope'
        self.rotary_emb = RotaryEmbedding(self.hidden_size_per_attention_head)
        self.activation = F.gelu

    def forward(
            self,
            hidden_states,
            attention_mask,
            encoder_output=None,
            enc_dec_attn_mask=None,
            rotary_pos_emb=None,
            layer_past=None,
            get_key_value=False,
            set_inference_key_value_memory=False,
            inference_max_sequence_len=None,
            self_attention_relative_position_bias=None,
            cross_attention_relative_position_bias=None,
            checkpoint_core_attention=False,
    ):
        # [s, b, h], here h is fused
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        self.qkv_mlp, _ = self.fused_in_layer(hidden_states)
        s, b, _ = self.qkv_mlp.shape
        mlp_out = torch.split(self.qkv_mlp, self.hidden_size_per_attention_head, dim=-1)
        h = self.num_attention_heads_per_partition
        query_layer = torch.stack(mlp_out[:h], dim=-2).contiguous()
        mixed_kv_layer = self.key_value(hidden_states)
        mixed_kv_layer = mixed_kv_layer.view(s, b, 1, 2 * self.hidden_size_per_attention_head)

        # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
        (key_layer, value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_kv_layer, 2)
        mlp_out = torch.cat(mlp_out[h:], dim=-1).contiguous()
        # print(f"q: {query_layer.shape}, mlp; {mlp_out.shape}")
        # query_layer, key_layer, value_layer = qkv[:,:,:h], qkv[:,:,[h]], qkv[:,:,[h+1]]

        b, np, sq, sk, hn = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
            query_layer.size(3),
        )
        if self.position_embedding_type == 'rope':
            # [sq, b, np, hn]
            cos, sin = self.rotary_emb(value_layer, seq_len=query_layer.shape[0])
            # print(f"cos: {cos.shape}, sin: {sin.shape}")
            # print(f"query: {query_layer.shape}, key: {key_layer.shape}")

            query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, offset=0)
        assert self.multi_query_attention
        query_layer = rearrange(query_layer, 'sq b np hn -> b (np sq) hn')
        key_layer = rearrange(key_layer, 'sk b 1 hn -> b hn sk')
        value_layer = rearrange(value_layer, 'sv b 1 hn -> b sv hn')

        matmul_input_buffer = torch.empty(
            query_layer.shape[0],
            query_layer.shape[1],
            key_layer.shape[2],
            dtype=query_layer.dtype,
            device=query_layer.device,
        )

        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer,
            key_layer,
            beta=0.0,
            alpha=(1.0 / self.norm_factor) if self.normalize_attention_scores else 1.0,
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(b, np, sq, sk)
        # Materialize attention mask right before use
        if is_torch_tpu_available():
            seq_len = hidden_states.shape[0]  # See above [b, *s*, h] shape
            if self.sequence_parallel:
                seq_len *= parallel_state.get_tensor_model_parallel_world_size()
            attention_mask = torch.triu(torch.ones(
                (1, 1, seq_len, seq_len), device='xla'), diagonal=1).bool()

        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # change view [b * np, sq, sk]
        assert self.multi_query_attention
        attention_probs = rearrange(attention_probs, 'b np sq sk -> b (np sq) sk')

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer)

        # change view [b, np, sq, hn]
        assert self.multi_query_attention
        context_layer = rearrange(context_layer, 'b (np sq) hn -> b np sq hn', np=np)
        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # MLP and dense fusion
        mlp_out = self.activation(mlp_out)
        fused_input = torch.cat((context_layer, mlp_out), dim=-1)
        fused_out, _ = self.fused_out_layer(fused_input)
        output = fused_out + residual
        return output


class ParallelTransformerLayer(FalconParallelTransformerLayer):

    def __init__(
            self,
            init_method,
            output_layer_init_method,
            layer_number,
            hidden_size,
            ffn_hidden_size,
            num_attention_heads,
            layer_type=LayerType.encoder,
            self_attn_mask_type=AttnMaskType.padding,
            fp32_residual_connection=False,
            precision=16,
            apply_query_key_layer_scaling=True,
            kv_channels=None,
            layernorm_epsilon=1e-5,
            hidden_dropout=0.1,
            bias_dropout_add_fusion=True,
            persist_layer_norm=False,
            use_cpu_initialization=False,
            bias_activation_fusion=True,
            openai_gelu=False,
            onnx_safe=False,
            masked_softmax_fusion=True,
            attention_dropout=0.1,
            ffn_dropout=0.0,
            activation='gelu',
            megatron_legacy=False,
            bias=True,
            chunk_size=64,
            normalization='layernorm',
            transformer_block_type='pre_ln',
            position_embedding_type='learned_absolute',
            multi_query_attention=False,
            headscale=False,
            activations_checkpoint_granularity=None,
            sequence_parallel=False,
            gradient_accumulation_fusion=False,
            normalize_attention_scores=True,
            num_moe_experts=1,
            moe_frequency=1,
            moe_dropout=0.0,
    ):
        super(ParallelTransformerLayer, self).__init__(
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            layer_number=layer_number,
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            num_attention_heads=num_attention_heads,
            layer_type=layer_type,
            self_attn_mask_type=self_attn_mask_type,
            fp32_residual_connection=fp32_residual_connection,
            precision=precision,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            layernorm_epsilon=layernorm_epsilon,
            hidden_dropout=hidden_dropout,
            bias_dropout_add_fusion=bias_dropout_add_fusion,
            persist_layer_norm=persist_layer_norm,
            use_cpu_initialization=use_cpu_initialization,
            bias_activation_fusion=bias_activation_fusion,
            openai_gelu=openai_gelu,
            onnx_safe=onnx_safe,
            masked_softmax_fusion=masked_softmax_fusion,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            activation=activation,
            megatron_legacy=megatron_legacy,
            bias=bias,
            chunk_size=chunk_size,
            normalization=normalization,
            transformer_block_type=transformer_block_type,
            position_embedding_type=position_embedding_type,
            headscale=headscale,
            multi_query_attention=multi_query_attention,
            activations_checkpoint_granularity=activations_checkpoint_granularity,
            sequence_parallel=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
            normalize_attention_scores=normalize_attention_scores,
            num_moe_experts=num_moe_experts,
            moe_frequency=moe_frequency,
            moe_dropout=moe_dropout,
        )

        if precision == 32:
            self.dtype = torch.float32
        elif precision == 16:
            self.dtype = torch.float16
        elif precision == 'bf16':
            self.dtype = torch.bfloat16
        else:
            raise ValueError

    def forward(
            self,
            hidden_states,
            attention_mask,
            encoder_output=None,
            enc_dec_attn_mask=None,
            rotary_pos_emb=None,
            layer_past=None,
            get_key_value=False,
            set_inference_key_value_memory=False,
            inference_max_sequence_len=None,
            self_attention_relative_position_bias=None,
            cross_attention_relative_position_bias=None,
            checkpoint_core_attention=False,
    ):
        if self.dtype == torch.float32:
            return super().forward(
                hidden_states,
                attention_mask,
                encoder_output,
                enc_dec_attn_mask,
                layer_past,
                get_key_value,
                set_inference_key_value_memory,
                inference_max_sequence_len,
                rotary_pos_emb,
                self_attention_relative_position_bias,
                cross_attention_relative_position_bias,
                checkpoint_core_attention,
            )
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            return super().forward(
                hidden_states,
                attention_mask,
                encoder_output,
                enc_dec_attn_mask,
                layer_past,
                get_key_value,
                set_inference_key_value_memory,
                inference_max_sequence_len,
                rotary_pos_emb,
                self_attention_relative_position_bias,
                cross_attention_relative_position_bias,
                checkpoint_core_attention,
            )


class AutocastTransformerLayer(TransformerLayer):
    def __init__(
            self,
            hidden_size: int,
            ffn_hidden_size: int,
            layernorm_epsilon: float,
            num_attention_heads: int,
            init_method: Callable,
            output_layer_init_method: Callable,
            hidden_dropout: float,
            attention_dropout: float,
            layer_number: Optional[int] = None,
            kv_channels: Optional[int] = None,
            self_attn_mask_type: str = "causal",
            tp_group: Optional[Any] = None,
            tp_size: int = 1,
            params_dtype: torch.dtype = torch.float32,
            get_rng_state_tracker: Optional[Callable] = None,
            fuse_wgrad_accumulation: bool = False,
            apply_query_key_layer_scaling: bool = True,
            attention_softmax_in_fp32: bool = False,
            seq_length: Optional[int] = None,
            micro_batch_size: Optional[int] = None,
            sequence_parallel: bool = False,
            apply_residual_connection_post_layernorm: bool = False,
            output_layernorm: bool = False,
            layer_type: str = "encoder",
            drop_path_rate: float = 0,
            use_emha: bool = False,
            autocast_dtype: Any = 16,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            layernorm_epsilon=layernorm_epsilon,
            num_attention_heads=num_attention_heads,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            layer_number=layer_number,
            kv_channels=kv_channels,
            self_attn_mask_type=self_attn_mask_type,
            tp_group=tp_group,
            tp_size=tp_size,
            params_dtype=params_dtype,
            get_rng_state_tracker=get_rng_state_tracker,
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            attention_softmax_in_fp32=attention_softmax_in_fp32,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            sequence_parallel=sequence_parallel,
            apply_residual_connection_post_layernorm=apply_residual_connection_post_layernorm,
            output_layernorm=output_layernorm,
            layer_type=layer_type,
            drop_path_rate=drop_path_rate,
            set_parallel_mode=tp_size > 1,
            fuse_qkv_params=True,
        )
        # use_emha=use_emha,

        if autocast_dtype == 32:
            self.dtype = torch.float32
        elif autocast_dtype == 16:
            self.dtype = torch.float16
        elif autocast_dtype == 'bf16':
            self.dtype = torch.bfloat16
        else:
            raise ValueError

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            encoder_output: Optional[torch.Tensor] = None,
            enc_dec_attn_mask: Optional[torch.Tensor] = None,
            inference_params: Optional[Any] = None,
            is_first_microbatch: Optional[bool] = None,
            checkpoint_core_attention: Optional[bool] = False,
    ) -> torch.Tensor:
        if self.dtype == torch.float32:
            return super().forward(
                hidden_states,
                attention_mask,
                encoder_output=encoder_output,
                enc_dec_attn_mask=enc_dec_attn_mask,
                inference_params=inference_params,
                is_first_microbatch=is_first_microbatch,
                checkpoint_core_attention=checkpoint_core_attention,
            )
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            return super().forward(
                hidden_states,
                attention_mask,
                encoder_output=encoder_output,
                enc_dec_attn_mask=enc_dec_attn_mask,
                inference_params=inference_params,
                is_first_microbatch=is_first_microbatch,
                checkpoint_core_attention=checkpoint_core_attention,
            )


class ParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(
            self,
            init_method,
            output_layer_init_method,
            num_layers,
            hidden_size,
            ffn_hidden_size,
            num_attention_heads,
            apply_query_key_layer_scaling=True,
            kv_channels=None,
            layer_type=LayerType.encoder,  # it can be a list of types or single type
            self_attn_mask_type=AttnMaskType.padding,
            pre_process=True,
            post_process=True,
            precision=16,
            fp32_residual_connection=False,
            activations_checkpoint_method=None,
            activations_checkpoint_num_layers=None,
            layernorm_epsilon=1e-5,
            hidden_dropout=0.1,
            attention_dropout=0.1,
            ffn_dropout=0.0,
            use_cpu_initialization=False,
            bias_activation_fusion=True,
            bias_dropout_add_fusion=True,
            masked_softmax_fusion=True,
            gradient_accumulation_fusion=False,
            persist_layer_norm=False,
            openai_gelu=False,
            onnx_safe=False,
            activation='gelu',
            model_type=ModelType.encoder_or_decoder,
            megatron_legacy=False,
            bias=True,
            chunk_size=64,
            normalization='layernorm',
            transformer_block_type='pre_ln',
            position_embedding_type='rope',
            headscale=False,
            layer_number_offset=0,  # this is use only for attention norm_factor scaling
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
            normalize_attention_scores=True,
            multi_query_attention=False,
            num_moe_experts=1,
            moe_frequency=1,
            moe_dropout=0.0,
    ):
        super(ParallelTransformer, self).__init__()

        if kv_channels is None:
            assert (
                    hidden_size % num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = hidden_size // num_attention_heads

        self.fp32_residual_connection = fp32_residual_connection
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None
        self.self_attn_mask_type = self_attn_mask_type
        self.model_type = model_type
        self.normalization = normalization
        self.transformer_block_type = transformer_block_type
        self.layer_type = layer_type
        self.position_embedding_type = position_embedding_type
        self.multi_query_attention = multi_query_attention

        self.activations_checkpoint_method = activations_checkpoint_method
        self.activations_checkpoint_num_layers = activations_checkpoint_num_layers
        self.activations_checkpoint_granularity = activations_checkpoint_granularity
        self.activations_checkpoint_layers_per_pipeline = activations_checkpoint_layers_per_pipeline

        if self.activations_checkpoint_granularity:
            if self.activations_checkpoint_granularity == 'selective':
                if self.activations_checkpoint_method == 'uniform':
                    logging.info(
                        (
                            f'Using uniform activation checkpointing with granularity selective forces all layers to use checkpointing.'
                        )
                    )
                elif self.activations_checkpoint_method == 'block':
                    logging.info(
                        (
                            f'Using block activation checkpointing requires activations_checkpoint_num_layers to be set.'
                            f'Got: {self.activations_checkpoint_num_layers}. Setting to 1 by default.'
                        )
                    )
                else:
                    raise ValueError(
                        f'activations_checkpoint_method should be "uniform" or "block" when using granularity selective.'
                    )
            elif self.activations_checkpoint_granularity == 'full':
                if self.activations_checkpoint_method in ['uniform', 'block']:
                    if not self.activations_checkpoint_num_layers:
                        logging.info(
                            (
                                f'Using uniform or block activation checkpointing requires activations_checkpoint_num_layers to be set.'
                                f'Got: {self.activations_checkpoint_num_layers}. Setting to 1 by default.'
                            )
                        )
                else:
                    raise ValueError(
                        f'activations_checkpoint_method should be "uniform" or "block" when using granularity full.'
                    )
            else:
                raise ValueError(f'activations_checkpoint_granularity should be "selective" or "full".')

        self.sequence_parallel = sequence_parallel
        self.transformer_engine = transformer_engine
        self.fp8 = fp8
        self.fp8_e4m3 = fp8_e4m3
        self.fp8_hybrid = fp8_hybrid
        self.fp8_margin = fp8_margin
        self.fp8_interval = fp8_interval
        self.fp8_amax_history_len = fp8_amax_history_len
        self.fp8_amax_compute_algo = fp8_amax_compute_algo
        self.reduce_amax = reduce_amax

        self.fp8_recipe = None

        if self.fp8:
            if self.fp8_e4m3:
                fp8_format = recipe.Format.E4M3
            elif self.fp8_hybrid:
                fp8_format = recipe.Format.HYBRID
            self.fp8_recipe = recipe.DelayedScaling(
                margin=self.fp8_margin,
                interval=self.fp8_interval,
                fp8_format=fp8_format,
                amax_history_len=self.fp8_amax_history_len,
                amax_compute_algo=self.fp8_amax_compute_algo,
                reduce_amax=reduce_amax,
            )

        self.is_first_microbatch = True
        self.microbatch_count = 0  # transformer engine forward needs to know if it is working on the first microbatch
        self.checkpoint_core_attention = (
                activations_checkpoint_granularity == 'selective'
        )  # transformer engine forward allows for more granular selective checkpointing

        if self.model_type == ModelType.encoder_or_decoder:
            assert (
                    num_layers % parallel_state.get_pipeline_model_parallel_world_size() == 0
            ), 'num_layers must be divisible by pipeline_model_parallel_size'

        assert moe_frequency <= num_layers, 'MoE frequency must be <= number of transformer layers'
        # TODO: Add similar assert for encoder-decoder.

        self.num_layers = self.get_num_layers(num_layers)

        # Transformer layers.
        def build_layer(layer_number):
            if isinstance(layer_type, list):
                lt = layer_type[layer_number - 1]
            else:
                lt = layer_type

            if self.transformer_engine:
                return AutocastTransformerLayer(
                    hidden_size=hidden_size,
                    ffn_hidden_size=ffn_hidden_size,
                    layernorm_epsilon=layernorm_epsilon,
                    num_attention_heads=num_attention_heads,
                    init_method=init_method,
                    output_layer_init_method=output_layer_init_method,
                    hidden_dropout=hidden_dropout,
                    attention_dropout=attention_dropout,
                    layer_number=layer_number + layer_number_offset,
                    kv_channels=kv_channels,
                    self_attn_mask_type=self_attn_mask_type.name,
                    tp_size=parallel_state.get_tensor_model_parallel_world_size(),
                    params_dtype=torch.float32,  # dtype params are initialized in
                    get_rng_state_tracker=tensor_parallel.random.get_xla_rng_tracker,
                    fuse_wgrad_accumulation=gradient_accumulation_fusion,
                    apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                    seq_length=None,  # used for jit warmup
                    micro_batch_size=None,  # used for jit warmup
                    sequence_parallel=sequence_parallel,
                    apply_residual_connection_post_layernorm=False,
                    autocast_dtype=precision,
                    use_emha=use_emha,
                    zero_centered_gamma=normalization == 'layernorm1p',
                )
            else:
                return ParallelTransformerLayer(
                    init_method=init_method,
                    output_layer_init_method=output_layer_init_method,
                    layer_number=layer_number + layer_number_offset,
                    hidden_size=hidden_size,
                    ffn_hidden_size=ffn_hidden_size,
                    num_attention_heads=num_attention_heads,
                    apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                    kv_channels=kv_channels,
                    layer_type=lt,
                    self_attn_mask_type=self_attn_mask_type,
                    precision=precision,
                    fp32_residual_connection=fp32_residual_connection,
                    layernorm_epsilon=layernorm_epsilon,
                    hidden_dropout=hidden_dropout,
                    attention_dropout=attention_dropout,
                    ffn_dropout=ffn_dropout,
                    use_cpu_initialization=use_cpu_initialization,
                    bias_activation_fusion=bias_activation_fusion,
                    bias_dropout_add_fusion=bias_dropout_add_fusion,
                    masked_softmax_fusion=masked_softmax_fusion,
                    gradient_accumulation_fusion=gradient_accumulation_fusion,
                    persist_layer_norm=persist_layer_norm,
                    openai_gelu=openai_gelu,
                    onnx_safe=onnx_safe,
                    activation=activation,
                    megatron_legacy=megatron_legacy,
                    bias=bias,
                    chunk_size=chunk_size,
                    normalization=normalization,
                    transformer_block_type=transformer_block_type,
                    headscale=headscale,
                    activations_checkpoint_granularity=activations_checkpoint_granularity,
                    sequence_parallel=sequence_parallel,
                    normalize_attention_scores=normalize_attention_scores,
                    num_moe_experts=num_moe_experts,
                    moe_frequency=moe_frequency,
                    moe_dropout=moe_dropout,
                    position_embedding_type=self.position_embedding_type,
                    multi_query_attention=self.multi_query_attention
                )

        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            assert num_layers % parallel_state.get_virtual_pipeline_model_parallel_world_size() == 0, (
                'num_layers_per_stage must be divisible by ' 'virtual_pipeline_model_parallel_size'
            )

            assert self.model_type.value != 2, f'virtual pipeline parallel currently only supported for GPT'

            # Number of layers in each model chunk is the number of layers in the stage,
            # divided by the number of model chunks in a stage.
            self.num_layers = self.num_layers // parallel_state.get_virtual_pipeline_model_parallel_world_size()
            # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0]  [2]  [4]  [6]
            # Stage 1: [1]  [3]  [5]  [7]
            # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]  [4, 5]
            # Stage 1: [2, 3]  [6, 7]
            offset = parallel_state.get_virtual_pipeline_model_parallel_rank() * (
                    num_layers // parallel_state.get_virtual_pipeline_model_parallel_world_size()
            ) + (parallel_state.get_pipeline_model_parallel_rank() * self.num_layers)
        else:
            # Each stage gets a contiguous set of layers.
            if (
                    self.model_type == ModelType.encoder_and_decoder
                    and parallel_state.get_pipeline_model_parallel_world_size() > 1
            ):
                pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()
                if layer_type == LayerType.encoder:
                    offset = pipeline_rank * self.num_layers
                else:
                    num_ranks_in_enc = parallel_state.get_pipeline_model_parallel_split_rank()
                    offset = (pipeline_rank - num_ranks_in_enc) * self.num_layers
            else:
                offset = parallel_state.get_pipeline_model_parallel_rank() * self.num_layers

        self.layers = torch.nn.ModuleList([build_layer(i + 1 + offset) for i in range(self.num_layers)])

        if self.post_process and self.transformer_block_type != 'post_ln':
            # Final layer norm before output.
            assert normalization == 'layernorm'
            self.final_layernorm = get_layer_norm(
                hidden_size, layernorm_epsilon, persist_layer_norm, sequence_parallel=sequence_parallel
            )

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def get_num_layers(self, num_layers):
        """Compute the number of transformer layers resident on the current rank."""
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            if self.model_type == ModelType.encoder_and_decoder:
                assert parallel_state.get_pipeline_model_parallel_split_rank() is not None
                num_ranks_in_encoder = parallel_state.get_pipeline_model_parallel_split_rank()
                num_ranks_in_decoder = parallel_state.get_pipeline_model_parallel_world_size() - num_ranks_in_encoder
                if self.layer_type == LayerType.encoder:
                    assert (
                            num_layers % num_ranks_in_encoder == 0
                    ), 'num_layers must be divisible by number of ranks given to encoder'
                elif self.layer_type == LayerType.decoder:
                    assert (
                            num_layers % num_ranks_in_decoder == 0
                    ), 'num_layers must be divisible by number of ranks given to decoder'
                else:
                    raise ValueError(f"Unknown layer type {self.layer_type}")

                if parallel_state.is_pipeline_stage_before_split():
                    num_layers = num_layers // num_ranks_in_encoder
                else:
                    num_layers = num_layers // num_ranks_in_decoder
            elif self.model_type == ModelType.encoder_or_decoder:
                assert (
                        num_layers % parallel_state.get_pipeline_model_parallel_world_size() == 0
                ), 'num_layers must be divisible by pipeline_model_parallel_size'
                num_layers = num_layers // parallel_state.get_pipeline_model_parallel_world_size()

        return num_layers

    def _checkpointed_forward(
            self,
            hidden_states,
            attention_mask,
            encoder_output,
            enc_dec_attn_mask,
            rotary_pos_emb,
            self_attention_relative_position_bias,
            cross_attention_relative_position_bias,
            checkpoint_activations_all_layers,
    ):
        """Forward method with activation checkpointing."""

        def custom(start, end):
            if self.transformer_engine:

                def custom_forward(*inputs):
                    hidden_states = inputs[0]
                    attention_mask = inputs[1]
                    encoder_output = inputs[2]
                    enc_dec_attn_mask = inputs[3]
                    for index in range(start, end):
                        layer = self._get_layer(index)
                        hidden_states = layer(
                            hidden_states,
                            attention_mask,
                            encoder_output=encoder_output,
                            enc_dec_attn_mask=enc_dec_attn_mask,
                            inference_params=None,
                            is_first_microbatch=self.is_first_microbatch,
                            checkpoint_core_attention=False,
                        )

                    return hidden_states

            else:

                def custom_forward(*inputs):
                    if len(inputs) == 9:
                        hidden_states = inputs[0]
                        attention_mask = inputs[1]
                        encoder_output = inputs[2]
                        enc_dec_attn_mask = inputs[3]
                        rotary_pos_emb = (inputs[4], inputs[5], inputs[6])
                        self_attention_relative_position_bias = inputs[7]
                        cross_attention_relative_position_bias = inputs[8]
                    elif len(inputs) == 10:
                        hidden_states = (inputs[0], inputs[1])
                        attention_mask = inputs[2]
                        encoder_output = inputs[3]
                        enc_dec_attn_mask = inputs[4]
                        rotary_pos_emb = (inputs[5], inputs[6], inputs[7])
                        self_attention_relative_position_bias = inputs[8]
                        cross_attention_relative_position_bias = inputs[9]
                    else:
                        hidden_states = inputs[0]
                        attention_mask = inputs[1]
                        encoder_output = inputs[2]
                        enc_dec_attn_mask = inputs[3]
                        rotary_pos_emb = inputs[4]
                        self_attention_relative_position_bias = inputs[5]
                        cross_attention_relative_position_bias = inputs[6]
                    for index in range(start, end):
                        layer = self._get_layer(index)
                        hidden_states = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            encoder_output=encoder_output,
                            enc_dec_attn_mask=enc_dec_attn_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            self_attention_relative_position_bias=self_attention_relative_position_bias,
                            cross_attention_relative_position_bias=cross_attention_relative_position_bias,
                        )
                        if isinstance(hidden_states, tuple):
                            pass
                        else:
                            hidden_states = hidden_states.contiguous()
                    return hidden_states

            return custom_forward

        # Make sure memory is freed.
        tensor_parallel.reset_checkpointed_activations_memory_buffer()

        if self.activations_checkpoint_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            l = 0
            while l < self.num_layers:
                if isinstance(hidden_states, tuple):
                    hidden_tuple = (hidden_states[0], hidden_states[1])
                else:
                    hidden_tuple = (hidden_states,)
                middle_tuple = (
                    attention_mask,
                    encoder_output,
                    enc_dec_attn_mask,
                )

                if rotary_pos_emb is None:
                    rot_tuple = (rotary_pos_emb,)
                else:
                    rot_tuple = (rotary_pos_emb[0], rotary_pos_emb[1], rotary_pos_emb[2])

                final_tuple = (self_attention_relative_position_bias, cross_attention_relative_position_bias)
                arg_tuple = hidden_tuple + middle_tuple + rot_tuple + final_tuple

                if self.transformer_engine:
                    hidden_states = te_checkpoint(
                        custom(l, l + self.activations_checkpoint_num_layers),
                        False,
                        tensor_parallel.random.get_xla_rng_tracker,
                        parallel_state.get_tensor_model_parallel_group(),
                        *arg_tuple,
                    )
                else:
                    hidden_states = tensor_parallel.checkpoint(
                        custom(l, l + self.activations_checkpoint_num_layers), False, *arg_tuple
                    )
                l += self.activations_checkpoint_num_layers
        elif self.activations_checkpoint_method == 'block':
            # When pipeline-parallel size > 1 and 'num_micro_batches_with_partial_activation_checkpoints' = int,
            # pipeline scheduling can force to checkpoint all layers or partial layers in a micro-batch.
            if checkpoint_activations_all_layers:
                activations_checkpoint_num_layers = self.num_layers
            else:
                activations_checkpoint_num_layers = self.activations_checkpoint_num_layers
                if (
                        parallel_state.get_pipeline_model_parallel_world_size() > 0
                        and self.activations_checkpoint_layers_per_pipeline is not None
                ):
                    # Decrease the number of layers to checkpoint at later pipeline stages
                    activations_checkpoint_num_layers -= int(
                        parallel_state.get_pipeline_model_parallel_rank()
                        * self.activations_checkpoint_layers_per_pipeline
                    )
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            for l in range(self.num_layers):
                if isinstance(hidden_states, tuple):
                    hidden_tuple = (hidden_states[0], hidden_states[1])
                else:
                    hidden_tuple = (hidden_states,)
                middle_tuple = (
                    attention_mask,
                    encoder_output,
                    enc_dec_attn_mask,
                )

                if rotary_pos_emb is None:
                    rot_tuple = (rotary_pos_emb,)
                else:
                    rot_tuple = (rotary_pos_emb[0], rotary_pos_emb[1], rotary_pos_emb[2])

                final_tuple = (self_attention_relative_position_bias, cross_attention_relative_position_bias)
                arg_tuple = hidden_tuple + middle_tuple + rot_tuple + final_tuple

                if l < activations_checkpoint_num_layers:
                    if self.transformer_engine:
                        hidden_states = te_checkpoint(
                            custom(l, l + 1),
                            False,
                            tensor_parallel.random.get_xla_rng_tracker,
                            parallel_state.get_tensor_model_parallel_group(),
                            *arg_tuple,
                        )
                    else:
                        hidden_states = tensor_parallel.checkpoint(custom(l, l + 1), False, *arg_tuple)
                else:
                    hidden_states = custom(l, l + 1)(*arg_tuple)
        else:
            raise ValueError("Invalid activation checkpoint method.")

        return hidden_states

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(
            self,
            hidden_states,
            attention_mask,
            layer_past=None,
            get_key_value=False,
            encoder_output=None,
            enc_dec_attn_mask=None,
            set_inference_key_value_memory=False,
            inference_max_sequence_len=None,
            rotary_pos_emb=None,
            # list of positional embedding tensors, first one self attention, second one and third one are for cross attention (q, k)
            retrieved_emb=None,  # tensor of retrieved embedding of shape [b, k, r, n, d]
            self_attention_relative_position_bias=None,
            cross_attention_relative_position_bias=None,
            checkpoint_activations_all_layers=None,
    ):
        # Checks.
        if inference_max_sequence_len:
            assert self.activations_checkpoint_method is None, 'inference does not work with activation checkpointing'

        if layer_past is not None:
            assert get_key_value, 'for not None values in layer_past, ' 'expected get_key_value to be set'
        if get_key_value:
            assert self.activations_checkpoint_method is None, (
                'get_key_value does not work with ' 'activation checkpointing'
            )

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # TODO: @Yi Dong, what should this be?
        if retrieved_emb is not None:
            assert len(retrieved_emb.shape) == 5
            # this is retrieval decoder, need special transpose
            encoder_output = rearrange(retrieved_emb, 'b k r n d -> k r n b d').contiguous()

        """
        is_first_microbatch is an optimization parameter for transformer engine.
        It indicates if the current step in the forward pass is the first in a gradient accumulation cycle.
        If set, FP8 weights are cached and some minor optimizations are applied to fuse_wgrad_accumulation
        """
        from apex.transformer.pipeline_parallel.utils import _GLOBAL_NUM_MICROBATCHES_CALCULATOR

        num_micro_batches = getattr(_GLOBAL_NUM_MICROBATCHES_CALCULATOR, 'num_micro_batches', 1)

        if self.sequence_parallel:
            rng_context = tensor_parallel.random.get_xla_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        with rng_context:
            # fp8_autocast will not do anything if TE or FP8 isn't used
            fp8_group = None
            if parallel_state.model_parallel_is_initialized():
                fp8_group = parallel_state.get_data_parallel_group()

            if HAVE_TE:
                # if TE is installed but fp8 is not available then this will do nothing
                fp8_context = fp8_autocast(enabled=self.fp8, fp8_recipe=self.fp8_recipe, fp8_group=fp8_group)

            else:
                fp8_context = nullcontext()

            with fp8_context:
                if self.activations_checkpoint_granularity == 'full' and self.activations_checkpoint_num_layers > 0:
                    hidden_states = self._checkpointed_forward(
                        hidden_states,
                        attention_mask,
                        encoder_output,
                        enc_dec_attn_mask,
                        rotary_pos_emb,
                        self_attention_relative_position_bias,
                        cross_attention_relative_position_bias,
                        checkpoint_activations_all_layers,
                    )
                else:
                    if get_key_value:
                        presents = []

                    for index in range(self.num_layers):
                        layer = self._get_layer(index)
                        past = None

                        if layer_past is not None:
                            past = layer_past[index]

                        if self.activations_checkpoint_granularity == 'selective':
                            # When pipeline-parallel size > 1 and 'num_micro_batches_with_partial_activation_checkpoints' = int,
                            # pipeline scheduling can force to checkpoint all layers or partial layers in a micro-batch.
                            if (
                                    checkpoint_activations_all_layers == True
                                    or self.activations_checkpoint_method == 'uniform'
                            ):
                                checkpoint_core_attention = True
                            elif self.activations_checkpoint_method == 'block':
                                activations_checkpoint_num_layers = self.activations_checkpoint_num_layers
                                # Decrease the number of layers to checkpoint at later pipeline stages
                                if self.activations_checkpoint_layers_per_pipeline is not None:
                                    activations_checkpoint_num_layers -= int(
                                        parallel_state.get_pipeline_model_parallel_rank()
                                        * self.activations_checkpoint_layers_per_pipeline
                                    )
                                checkpoint_core_attention = index < activations_checkpoint_num_layers
                        else:
                            checkpoint_core_attention = False

                        if self.transformer_engine:

                            inference_params = None

                            hidden_states = layer(
                                hidden_states,
                                attention_mask,
                                encoder_output=encoder_output,
                                enc_dec_attn_mask=enc_dec_attn_mask,
                                inference_params=inference_params,
                                is_first_microbatch=self.is_first_microbatch,
                                checkpoint_core_attention=checkpoint_core_attention,
                            )

                        else:
                            hidden_states = layer(
                                hidden_states,
                                attention_mask,
                                encoder_output=encoder_output,
                                enc_dec_attn_mask=enc_dec_attn_mask,
                                layer_past=past,
                                get_key_value=get_key_value,
                                set_inference_key_value_memory=set_inference_key_value_memory,
                                inference_max_sequence_len=inference_max_sequence_len,
                                rotary_pos_emb=rotary_pos_emb,
                                self_attention_relative_position_bias=self_attention_relative_position_bias,
                                cross_attention_relative_position_bias=cross_attention_relative_position_bias,
                                checkpoint_core_attention=checkpoint_core_attention,
                            )

        # Skip counter update for eval and activation checkpointing
        if torch.is_grad_enabled() and self.training:
            self.microbatch_count += 1
            if self.microbatch_count % num_micro_batches == 0:
                self.microbatch_count = 0
                self.is_first_microbatch = True
            else:
                self.is_first_microbatch = False

        output = hidden_states

        # Final layer norm.
        if self.post_process:
            # only apply the final_layernorm for pre-ln
            if self.transformer_block_type != 'post_ln':
                output = self.final_layernorm(hidden_states)

        if get_key_value:
            output = [output, presents]

        return output


# language mdoel


def get_falcon_language_model(
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
):
    """Build language model and return along with the key to save."""

    if kv_channels is None:
        assert (
                hidden_size % num_attention_heads == 0
        ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
        kv_channels = hidden_size // num_attention_heads

    if init_method is None:
        init_method = init_method_normal(init_method_std)

    if scaled_init_method is None:
        scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)

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
    )
    # key used for checkpoints.
    language_model_key = 'language_model'

    return language_model, language_model_key


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
            use_cpu_initialization=False,
            fp32_residual_connection=False,
            sequence_parallel=False,
            position_embedding_type='rope',
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
            vocab_size, self.hidden_size, init_method=self.init_method, use_cpu_initialization=use_cpu_initialization,
        )
        self._word_embeddings_key = 'word_embeddings'

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
            position_embedding_type='rope',
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
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.sequence_parallel = sequence_parallel

        if kv_channels is None:
            assert (
                    hidden_size % num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = hidden_size // num_attention_heads

        # Embeddings.
        if self.pre_process:
            self.embedding = Embedding(
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_position_embeddings,
                init_method=self.init_method,
                num_tokentypes=self.num_tokentypes,
                use_cpu_initialization=use_cpu_initialization,
                embedding_dropout_prob=self.hidden_dropout,
                sequence_parallel=sequence_parallel,
                position_embedding_type=position_embedding_type,
                fp32_residual_connection=fp32_residual_connection,
            )
            self._embedding_key = 'embedding'
        # Move Rope to core attention
        # TODO: check perf penalty vs original Nemo implementation

        # Transformer.
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
            position_embedding_type=self.position_embedding_type
        )
        self._encoder_key = 'encoder'

        if self.post_process:

            if not self.share_embeddings_and_output_weights:
                no_async_tensor_model_parallel_allreduce = (
                        parallel_state.get_tensor_model_parallel_world_size() == 1 or sequence_parallel
                )
                self.output_layer = tensor_parallel.ColumnParallelLinear(
                    self.hidden_size,
                    self.vocab_size,
                    bias=False,
                    # Setting bias to False always to keep it consistent with embedding tying that also does not have a bias.
                    init_method=self.init_method,
                    skip_bias_add=True,
                    use_cpu_initialization=use_cpu_initialization,
                    gather_output=False,
                    sequence_parallel_enabled=sequence_parallel,
                    no_async_tensor_model_parallel_allreduce=no_async_tensor_model_parallel_allreduce,
                    gradient_accumulation_fusion=gradient_accumulation_fusion,
                    transfer_with_static_ring=(not (activations_checkpoint_granularity == "selective")),
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

        # output_enc_hidden_only refers to when we just need the encoder's
        # output. For example, it is helpful to compute
        # similarity between two sequences by average pooling
        if not self.add_decoder or output_enc_hidden_only:
            return encoder_output

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

            if not self.share_embeddings_and_output_weights:
                self.output_layer.load_state_dict(state_dict[self._output_layer_key], strict=strict)
