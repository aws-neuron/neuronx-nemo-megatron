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
from nemo.collections.nlp.modules.common.megatron.fused_bias_geglu import fused_bias_geglu
from nemo.collections.nlp.modules.common.megatron.fused_bias_gelu import fused_bias_gelu
from nemo.collections.nlp.modules.common.megatron.fused_layer_norm import get_layer_norm
from nemo.collections.nlp.modules.common.megatron.fused_softmax import MatchedScaleMaskSoftmax
from nemo.collections.nlp.modules.common.megatron.layer_norm_1p import LayerNorm1P
from nemo.collections.nlp.modules.common.megatron.layer_type import LayerType
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults, attention_mask_func, erf_gelu
from nemo.collections.nlp.modules.common.megatron.utils import openai_gelu as openai_gelu_func
from nemo.core import adapter_mixins
from nemo.utils import logging
from transformers.utils import is_torch_tpu_available

try:
    from apex.transformer import parallel_state, tensor_parallel
    from apex.transformer.enums import AttnMaskType, AttnType, ModelType
    from apex.transformer.utils import divide as safe_divide
    from apex.transformer.parallel_state import get_tensor_model_parallel_world_size
    from apex.transformer.layers.layer_norm import FastRMSNorm as MixedFusedRMSNorm
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


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[:,None, None, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[:, None, None, :], persistent=False)
        return (
            self.cos_cached[:seq_len, :, :, ...].to(dtype=x.dtype),
            self.sin_cached[:seq_len, :, :, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos = cos[offset: q.shape[0] + offset, ...]
    sin = sin[offset: q.shape[0] + offset, ...]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


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


class ParallelMLP(MegatronModule, adapter_mixins.AdapterModuleMixin):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(
            self,
            init_method,
            output_layer_init_method,
            hidden_size,
            ffn_hidden_size,
            use_cpu_initialization=False,
            bias_activation_fusion=True,
            openai_gelu=False,
            onnx_safe=False,
            activation='gelu',
            bias=True,
            transformer_block_type='pre_ln',
            normalization='layernorm',
            layernorm_epsilon=1e-5,
            persist_layer_norm=False,
            sequence_parallel=False,
            gradient_accumulation_fusion=False,
            dropout=0.0,
            transfer_with_static_ring=True,
    ):
        super(ParallelMLP, self).__init__()
        self.activation = activation
        self.bias = bias
        self.transformer_block_type = transformer_block_type
        self.normalization = normalization
        self.layernorm_epsilon = layernorm_epsilon
        self.persist_layer_norm = persist_layer_norm
        self.activation = activation
        self.dropout = dropout
        self.set_accepted_adapter_types([MLPInfusedAdapterConfig._target_])

        if activation not in ['gelu', 'geglu', 'reglu', 'swiglu']:
            raise ValueError(f"Activation {activation} not supported. Only gelu, geglu, reglu, swiglu are supported.")

        no_async_tensor_model_parallel_allreduce = (
                parallel_state.get_tensor_model_parallel_world_size() == 1 or sequence_parallel
        )
        # Project to 4h.
        self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
            hidden_size,
            ffn_hidden_size,  # NOTE: When using geglu, divide ffn dim by 2/3 to keep overall params the same.
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            use_cpu_initialization=use_cpu_initialization,
            bias=bias,
            sequence_parallel_enabled=sequence_parallel,
            no_async_tensor_model_parallel_allreduce=no_async_tensor_model_parallel_allreduce,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
            transfer_with_static_ring=transfer_with_static_ring,
        )

        if activation in ['geglu', 'reglu', 'swiglu']:
            # Separate linear layer for *GLU activations.
            # Source: https://github.com/huggingface/transformers/blob/bee361c6f1f7704f8c688895f2f86f6e5ff84727/src/transformers/models/t5/modeling_t5.py#L292
            self.dense_h_to_4h_2 = tensor_parallel.ColumnParallelLinear(
                hidden_size,
                ffn_hidden_size,  # NOTE: When using *glu, divide ffn dim by 2/3 to keep overall params the same.
                gather_output=False,
                init_method=init_method,
                skip_bias_add=True,
                use_cpu_initialization=use_cpu_initialization,
                bias=bias,
                sequence_parallel_enabled=sequence_parallel,
                no_async_tensor_model_parallel_allreduce=no_async_tensor_model_parallel_allreduce,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
                transfer_with_static_ring=transfer_with_static_ring,
            )

        self.glu_activation_family = activation in ['geglu', 'reglu', 'swiglu']
        bias_activation_fusion_unavailable = activation in ['reglu', 'swiglu']

        if bias_activation_fusion_unavailable and bias_activation_fusion:
            raise ValueError(
                f"Cannot use bias_activation_fusion with {activation} activation. Please turn bias gelu fusion off."
            )

        if self.glu_activation_family and onnx_safe and self.bias_activation_fusion:
            raise ValueError(
                f"Cannot use onnx_safe with specificed activation function and bias_activation_fusion : {activation} Please turn onnx safe off."
            )

        if bias_activation_fusion and not bias:
            raise ValueError(
                f"Cannot use bias_activation_fusion without bias terms. Please set bias=True or bias_activation_fusion=False."
            )

        self.bias_activation_fusion = bias_activation_fusion

        # Give openai_gelu precedence over other activations if set, for HF compatibility. Normally this is off and shouldn't affect regular model training.
        if openai_gelu:
            self.activation_func = openai_gelu_func
        elif activation in ["gelu", "geglu"]:
            self.activation_func = F.gelu
        elif onnx_safe:
            self.activation_func = erf_gelu
        elif activation == "reglu":
            self.activation_func = F.relu
        elif activation == "swiglu":
            # SiLU or sigmoid linear unit is the same as swish with beta = 1 (which is what https://arxiv.org/pdf/2002.05202.pdf uses.)
            self.activation_func = F.silu

        # Project back to h.
        self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
            ffn_hidden_size,
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

    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if self.glu_activation_family:
            intermediate_parallel_2, bias_parallel_2 = self.dense_h_to_4h_2(hidden_states)

        if self.bias_activation_fusion:
            if self.activation == 'gelu':
                intermediate_parallel = fused_bias_gelu(intermediate_parallel, bias_parallel)
            elif self.activation == 'geglu':
                intermediate_parallel = fused_bias_geglu(
                    intermediate_parallel, bias_parallel, intermediate_parallel_2, bias_parallel_2
                )

        elif self.activation in ['reglu', 'swiglu'] or (
                self.glu_activation_family and not self.bias_activation_fusion
        ):
            if bias_parallel is not None:
                intermediate_parallel = self.activation_func(intermediate_parallel + bias_parallel) * (
                        intermediate_parallel_2 + bias_parallel_2
                )
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel) * intermediate_parallel_2

        else:
            if bias_parallel is not None:
                intermediate_parallel = self.activation_func(intermediate_parallel + bias_parallel)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)

        if self.dropout > 0:
            intermediate_parallel = F.dropout(intermediate_parallel, p=self.dropout, training=self.training)

        infused_adapter = self.get_from_adapter_layer(AdapterName.MLP_INFUSED)
        if infused_adapter:
            intermediate_parallel = infused_adapter(intermediate_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias



class CoreAttention(MegatronModule):
    """ Region where selective activation recomputation is applied.
        See Figure 3. in Reducing Activation Recomputation in Large Transformer Models
        https://arxiv.org/pdf/2205.05198.pdf for more details.

    """

    def __init__(
            self,
            layer_number,
            num_attention_heads,
            hidden_size,
            attention_type=AttnType.self_attn,
            attn_mask_type=AttnMaskType.padding,
            precision=16,
            apply_query_key_layer_scaling=True,
            kv_channels=None,
            masked_softmax_fusion=True,
            attention_dropout=0.1,
            sequence_parallel=False,
            normalize_attention_scores=True,
            multi_query_attention=False,
            position_embedding_type='learned_absolute',
            num_kv_heads=None,
    ):

        super(CoreAttention, self).__init__()

        self.precision = precision
        self.fp16 = precision == 16
        self.bf16 = precision == 'bf16'
        self.multi_query_attention = multi_query_attention
        self.use_gqa = (num_kv_heads is not None) and (num_kv_heads != num_attention_heads)
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = False
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.sequence_parallel = sequence_parallel
        # If True, will scale attention scores by 1 / sqrt(hidden_size_per_attention_head).
        # This arg is been provided mostly to support weight conversion of Huggingface models. (ex: T5v1.1)
        self.normalize_attention_scores = normalize_attention_scores
        self.position_embedding_type = position_embedding_type
        if kv_channels is None:
            assert (
                    hidden_size % num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = hidden_size // num_attention_heads

        projection_size = kv_channels * num_attention_heads

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = safe_divide(projection_size, world_size)
        self.hidden_size_per_attention_head = safe_divide(projection_size, num_attention_heads)
        self.num_attention_heads_per_partition = safe_divide(num_attention_heads, world_size)
        self.num_attention_heads_partition_offset = (
                self.num_attention_heads_per_partition * parallel_state.get_tensor_model_parallel_rank()
        )
        if self.use_gqa:
            self.num_query_head_per_kv_head = num_attention_heads // num_kv_heads

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16,
            self.bf16,
            self.attn_mask_type,
            masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff,
        )

        if self.position_embedding_type == 'rope':
            self.rotary_emb = RotaryEmbedding(self.hidden_size_per_attention_head)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout)

    def forward(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            layer_past=None,
            get_key_value=False,
            rotary_pos_emb=None,
            relative_position_bias=None,
            headscale_tensor=None,
    ):

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0))
        sq, b, np, hn = query_layer.shape
        sk = key_layer.shape[0]
        # TODO: figure out how to do this
        # apply relative positional encoding (rotary embedding)
        # if rotary_pos_emb is not None:
        # q_pos_emb, k_pos_emb = rotary_pos_emb

        # query_layer = apply_rotary_pos_emb(query_layer, q_pos_emb)
        # key_layer = apply_rotary_pos_emb(key_layer, k_pos_emb)
        # TODO, can apply positional embedding to value_layer so it has
        # absolute positional embedding.
        # otherwise, only relative positional embedding takes effect
        # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

        if self.position_embedding_type == 'rope':
            # [sq, b, np, hn] --> [b, np, sq, hn] TODO optimize the permute of dimension back and forth
            query_layer = query_layer.permute(1, 2, 0, 3)
            key_layer = key_layer.permute(1, 2, 0, 3)
            value_layer = value_layer.permute(1, 2, 0, 3)

            cos, sin = self.rotary_emb(value_layer, seq_len=query_layer.shape[2])
            query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, offset=0)

            # [b, np, sq, hn] --> [sq, b, np, hn] TODO optimize the permute of dimension back and forth
            query_layer = query_layer.permute(2, 0, 1, 3).contiguous()
            key_layer = key_layer.permute(2, 0, 1, 3).contiguous()
            value_layer = value_layer.permute(2, 0, 1, 3).contiguous()

        if self.multi_query_attention:
            query_layer = rearrange(query_layer, 'sq b np hn -> b (np sq) hn')
            key_layer = rearrange(key_layer, 'sk b 1 hn -> b hn sk')
            value_layer = rearrange(value_layer, 'sv b np hn -> (b np) sv hn')
        elif self.use_gqa:
            query_layer = rearrange(
                query_layer, 'sq b (nk q_head) hn -> b q_head nk sq hn', q_head=self.num_query_head_per_kv_head,
            )
            key_layer = rearrange(key_layer, 'sk b nk hn -> b 1 nk hn sk')
            value_layer = rearrange(value_layer, 'sk b nk hn -> b 1 nk sk hn')
        else:
            query_layer = rearrange(query_layer, 'sq b np hn -> (b np) sq hn')
            key_layer = rearrange(key_layer, 'sk b np hn -> (b np) hn sk')
            value_layer = rearrange(value_layer, 'sv b np hn -> (b np) sv hn')

        if self.use_gqa:
            attention_scores = torch.matmul(query_layer, key_layer, )
            if self.normalize_attention_scores:
                attention_scores *= 1.0 / self.norm_factor
            attention_scores = rearrange(attention_scores, 'b q_head nk sq sk -> b (q_head nk) sq sk')
        else:
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
            attention_scores = matmul_result.view(*output_size)

        if relative_position_bias is not None:
            attention_scores += relative_position_bias[
                                :,
                                self.num_attention_heads_partition_offset: self.num_attention_heads_partition_offset
                                                                           + self.num_attention_heads_per_partition,
                                : attention_scores.size(2),
                                : attention_scores.size(3),
                                ]

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if get_key_value:
            with torch.no_grad():
                if layer_past is not None:
                    attention_mask = attention_mask[
                                     ..., attention_scores.size(3) - 1, : attention_scores.size(3)
                                     ].unsqueeze(2)
                else:
                    attention_mask = attention_mask[..., : attention_scores.size(3), : attention_scores.size(3)]

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.sequence_parallel:
            with tensor_parallel.random.get_xla_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        if self.use_gqa:
            # GQA
            attention_probs = rearrange(
                attention_probs, 'b (q_head nk) sq sk -> b q_head nk sq sk', q_head=self.num_query_head_per_kv_head,
            )
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = rearrange(context_layer, 'b q_head nk sq hn -> b (nk q_head) sq hn')

        else:
            # change view [b * np, sq, sk]
            attention_probs = rearrange(attention_probs, 'b np sq sk -> (b np) sq sk')

            # matmul: [b * np, sq, hn]
            context_layer = torch.bmm(attention_probs, value_layer)
            # change view [b, np, sq, hn]
            context_layer = rearrange(context_layer, '(b np) sq hn -> b np sq hn', np=np)

        if headscale_tensor is not None:
            context_layer = context_layer * headscale_tensor

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class ParallelAttention(MegatronModule, adapter_mixins.AdapterModuleMixin):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
            self,
            init_method,
            output_layer_init_method,
            layer_number,
            num_attention_heads,
            hidden_size,
            attention_type=AttnType.self_attn,
            attn_mask_type=AttnMaskType.padding,
            precision=16,
            apply_query_key_layer_scaling=True,
            kv_channels=None,
            use_cpu_initialization=False,
            masked_softmax_fusion=True,
            attention_dropout=0.1,
            layer_type=None,
            megatron_legacy=False,
            bias=True,
            headscale=False,
            position_embedding_type='learned_absolute',
            multi_query_attention=False,
            activations_checkpoint_granularity=None,
            sequence_parallel=False,
            gradient_accumulation_fusion=False,
            normalize_attention_scores=True,
            transfer_with_static_ring=True,
            num_kv_heads=None,
    ):
        super(ParallelAttention, self).__init__()

        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.normalize_attention_scores = normalize_attention_scores
        self.position_embedding_type = position_embedding_type
        self.multi_query_attention = multi_query_attention
        self.num_kv_heads = num_kv_heads
        self.use_gqa = (num_kv_heads is not None) and (num_kv_heads != num_attention_heads)
        self.megatron_legacy = megatron_legacy

        self.set_accepted_adapter_types([InfusedAdapterConfig._target_])

        if kv_channels is None:
            assert (
                    hidden_size % num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = hidden_size // num_attention_heads
        projection_size = kv_channels * num_attention_heads

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = safe_divide(projection_size, num_attention_heads)
        self.num_attention_heads_per_partition = safe_divide(num_attention_heads, world_size)
        self.num_attention_heads_partition_offset = (
                self.num_attention_heads_per_partition * parallel_state.get_tensor_model_parallel_rank()
        )

        no_async_tensor_model_parallel_allreduce = (
                parallel_state.get_tensor_model_parallel_world_size() == 1 or sequence_parallel
        )
        if self.use_gqa:
            assert num_attention_heads % num_kv_heads == 0
            self.num_query_head_per_kv_head = num_attention_heads // num_kv_heads
            kv_projection_size = kv_channels * num_kv_heads
            self.num_kv_attention_heads_per_partition = safe_divide(num_kv_heads, world_size)
        else:
            self.num_kv_attention_heads_per_partition = self.num_attention_heads_per_partition
            kv_projection_size = projection_size
        self.num_kv_heads_per_partition = safe_divide(num_kv_heads, world_size)

        # Strided linear layer.
        assert attention_type in [AttnType.self_attn, AttnType.cross_attn]
        if attention_type == AttnType.self_attn and not self.use_gqa:
            self.query_key_value = tensor_parallel.ColumnParallelLinear(
                hidden_size,
                3 * projection_size,
                gather_output=False,
                init_method=init_method,
                use_cpu_initialization=use_cpu_initialization,
                bias=bias,
                sequence_parallel_enabled=sequence_parallel,
                no_async_tensor_model_parallel_allreduce=no_async_tensor_model_parallel_allreduce,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
                transfer_with_static_ring=transfer_with_static_ring,
            )
        else:
            self.query = tensor_parallel.ColumnParallelLinear(
                hidden_size,
                projection_size,
                gather_output=False,
                init_method=init_method,
                bias=bias,
                sequence_parallel_enabled=sequence_parallel,
                no_async_tensor_model_parallel_allreduce=no_async_tensor_model_parallel_allreduce,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
                transfer_with_static_ring=transfer_with_static_ring,
            )

            self.key_value = tensor_parallel.ColumnParallelLinear(
                hidden_size,
                2 * kv_projection_size,
                gather_output=False,
                init_method=init_method,
                bias=bias,
                sequence_parallel_enabled=sequence_parallel,
                no_async_tensor_model_parallel_allreduce=no_async_tensor_model_parallel_allreduce,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
                transfer_with_static_ring=transfer_with_static_ring,
            )

        self.core_attention = CoreAttention(
            layer_number=self.layer_number,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            attention_type=self.attention_type,
            attn_mask_type=self.attn_mask_type,
            precision=precision,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            masked_softmax_fusion=masked_softmax_fusion,
            attention_dropout=attention_dropout,
            multi_query_attention=multi_query_attention,
            sequence_parallel=sequence_parallel,
            normalize_attention_scores=normalize_attention_scores,
            position_embedding_type=self.position_embedding_type,
            num_kv_heads=num_kv_heads,
        )

        # Output.
        self.dense = tensor_parallel.RowParallelLinear(
            projection_size,
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

        self.headscale = headscale
        if headscale:
            self.head_scale_tensor = torch.nn.Parameter(
                torch.ones(1, self.num_attention_heads_per_partition, 1, 1), requires_grad=True
            )

        # Inference key-value memory
        self.inference_key_memory = None
        self.inference_value_memory = None
        self.inference_current_sequence_len = 0

        # relative position embedding
        self.layer_type = layer_type

    def _checkpointed_attention_forward(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            rotary_pos_emb=None,
            relative_position_bias=None,
            headscale_tensor=None,
    ):
        """Forward method with activation checkpointing."""

        def custom_forward(*inputs):
            if len(inputs) == 7:
                query_layer = inputs[0]
                key_layer = inputs[1]
                value_layer = inputs[2]
                attention_mask = inputs[3]
                rotary_pos_emb = inputs[4]
                relative_position_bias = inputs[5]
                headscale_tensor = inputs[6]
            elif len(inputs) == 8:
                query_layer = inputs[0]
                key_layer = inputs[1]
                value_layer = inputs[2]
                attention_mask = inputs[3]
                rotary_pos_emb = (inputs[4], inputs[5])
                relative_position_bias = inputs[6]
                headscale_tensor = inputs[7]
            else:
                raise ValueError('unexpected number of inputs')
            output_ = self.core_attention(
                query_layer,
                key_layer,
                value_layer,
                attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                relative_position_bias=relative_position_bias,
                headscale_tensor=headscale_tensor,
            )
            return output_

        if rotary_pos_emb is None:
            rot_tuple = (rotary_pos_emb,)
        else:
            rot_tuple = (rotary_pos_emb[0], rotary_pos_emb[1])

        hidden_states = tensor_parallel.checkpoint(
            custom_forward,
            False,
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            *rot_tuple,
            relative_position_bias,
            headscale_tensor,
        )

        return hidden_states

    def _allocate_memory(self, inference_max_sequence_len, batch_size, dtype):
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=torch.cuda.current_device(),
        )

    def _transpose_last_dim(self, mixed_layer, num_splits, num_splits_first):
        input_shape = mixed_layer.size()
        if num_splits_first:
            """[s, b, num_splits * np * hn]
            -->(view) [s, b, num_splits, np, hn]
            -->(tranpose) [s, b, np, num_splits, hn]
            -->(view) [s, b, np * num_splits * hn] """

            intermediate_shape = input_shape[:-1] + (
                num_splits,
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )

            mixed_layer = mixed_layer.view(*intermediate_shape)
            mixed_layer = mixed_layer.transpose(-2, -3).contiguous()
        else:
            """[s, b, np * hn * num_splits]
            -->(view) [s, b, np, hn, num_splits]
            -->(tranpose) [s, b, np, num_splits, hn]
            -->(view) [s, b, np * num_splits * hn] """

            intermediate_shape = input_shape[:-1] + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
                num_splits,
            )

            mixed_layer = mixed_layer.view(*intermediate_shape)
            mixed_layer = mixed_layer.transpose(-1, -2).contiguous()
        mixed_layer = mixed_layer.view(*input_shape)

        return mixed_layer

    def forward(
            self,
            hidden_states,
            attention_mask,
            layer_past=None,
            get_key_value=False,
            encoder_output=None,
            set_inference_key_value_memory=False,
            inference_max_sequence_len=None,
            rotary_pos_emb=None,  # rotary positional embedding
            relative_position_bias=None,
            checkpoint_core_attention=False,
    ):
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        if set_inference_key_value_memory:
            assert inference_max_sequence_len and inference_max_sequence_len > 0
            self.inference_key_memory = self._allocate_memory(
                inference_max_sequence_len, hidden_states.size(1), hidden_states.dtype
            )
            self.inference_value_memory = self._allocate_memory(
                inference_max_sequence_len, hidden_states.size(1), hidden_states.dtype
            )
            self.inference_current_sequence_len = 0

        # Some consistency check.
        if inference_max_sequence_len:
            assert self.inference_current_sequence_len < self.inference_key_memory.size(0)
            assert inference_max_sequence_len == self.inference_key_memory.size(0)
        # This is added for safety. In case inference_max_sequence_len
        # is not provided, make sure there is no potential memory left
        # from previous inference.
        if not inference_max_sequence_len:
            self.inference_key_memory = None
            self.inference_value_memory = None

        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == AttnType.self_attn:
            if self.use_gqa:
                mixed_kv_layer, _ = self.key_value(hidden_states)
                if self.is_adapter_available():
                    lora_kv_adapter = self.get_adapter_module(AdapterName.LORA_KV_ADAPTER)
                    if lora_kv_adapter:
                        lora_mixed_kv_layer = lora_kv_adapter(hidden_states)
                        mixed_kv_layer = mixed_kv_layer + lora_mixed_kv_layer

                # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
                new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                    self.num_kv_attention_heads_per_partition,
                    2 * self.hidden_size_per_attention_head,
                )
                if self.megatron_legacy:
                    mixed_kv_layer = self._transpose_last_dim(mixed_kv_layer, 2, True)
                mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

                # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
                (key_layer, value_layer) = tensor_parallel.split_tensor_along_last_dim(
                    mixed_kv_layer, 2, contiguous_split_chunks=True
                )
                # Attention head [sq, b, h] --> [sq, b, hp]
                query_layer, _ = self.query(hidden_states)
                if self.is_adapter_available():
                    lora_q_adapter = self.get_adapter_module(AdapterName.LORA_Q_ADAPTER)
                    if lora_q_adapter:
                        lora_q_layer = lora_q_adapter(hidden_states)
                        query_layer = query_layer + lora_q_layer
                # [sq, b, hp] --> [sq, b, np, hn]
                new_tensor_shape = query_layer.size()[:-1] + (
                    self.num_attention_heads_per_partition,
                    self.hidden_size_per_attention_head,
                )
                query_layer = query_layer.view(*new_tensor_shape)
            else:
                # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
                mixed_x_layer, _ = self.query_key_value(hidden_states)

                # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
                new_tensor_shape = mixed_x_layer.size()[:-1] + (
                    self.num_attention_heads_per_partition,
                    3 * self.hidden_size_per_attention_head,
                )
                if self.megatron_legacy:
                    mixed_x_layer = self._transpose_last_dim(mixed_x_layer, 3, True)
                mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

                # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
                (query_layer, key_layer, value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_x_layer, 3)
        else:
            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer, _ = self.key_value(encoder_output)

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                2 * self.hidden_size_per_attention_head,
            )
            if self.megatron_legacy:
                mixed_kv_layer = self._transpose_last_dim(mixed_kv_layer, 2, True)
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer, value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_kv_layer, 2)

            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer, _ = self.query(hidden_states)
            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            query_layer = query_layer.view(*new_tensor_shape)

        if self.is_adapter_available():
            key_infused_adapter = self.get_from_adapter_layer(AdapterName.KEY_INFUSED)
            value_infused_adapter = self.get_from_adapter_layer(AdapterName.VALUE_INFUSED)
            if key_infused_adapter:
                assert value_infused_adapter is not None, "Expected value_infused_adapter not found!"
                kls = key_layer.shape
                key_layer = key_infused_adapter(key_layer.reshape(kls[0], kls[1], -1)).reshape(kls)
            if value_infused_adapter:
                assert key_infused_adapter is not None, "Expected key_infused_adapter not found!"
                vls = value_layer.shape
                value_layer = value_infused_adapter(value_layer.reshape(vls[0], vls[1], -1)).reshape(vls)

        # ===================================================
        # Adjust key, value, and attention mask for inference
        # ===================================================

        # duplicate the pos_emb for self attention
        if rotary_pos_emb is not None:
            rotary_pos_emb = rotary_pos_emb if isinstance(rotary_pos_emb, tuple) else ((rotary_pos_emb,) * 2)

        if inference_max_sequence_len:
            # Adjust the range variables.
            start = self.inference_current_sequence_len
            self.inference_current_sequence_len += key_layer.size(0)
            end = self.inference_current_sequence_len
            # Copy key and values.
            self.inference_key_memory[start:end, ...] = key_layer
            self.inference_value_memory[start:end, ...] = value_layer
            key_layer = self.inference_key_memory[:end, ...]
            value_layer = self.inference_value_memory[:end, ...]
            # Adjust attention mask
            attention_mask = attention_mask[..., start:end, :end]
            # adjust the key rotary positional embedding
            if rotary_pos_emb is not None:
                q_pos_emb, k_pos_emb = rotary_pos_emb
                if not set_inference_key_value_memory:
                    # In inference, we compute one token at a time.
                    # Select the correct positional embedding.
                    q_pos_emb = q_pos_emb[end - 1: end]
                else:
                    q_pos_emb = q_pos_emb[:end, :, :, :]
                k_pos_emb = k_pos_emb[:end, :, :, :]
                rotary_pos_emb = (q_pos_emb, k_pos_emb)

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer), value_layer), dim=0)

        if get_key_value:
            present = (key_layer, value_layer)

        if checkpoint_core_attention:
            context_layer = self._checkpointed_attention_forward(
                query_layer,
                key_layer,
                value_layer,
                attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                relative_position_bias=relative_position_bias,
                headscale_tensor=self.head_scale_tensor if self.headscale else None,
            )
        else:
            context_layer = self.core_attention(
                query_layer,
                key_layer,
                value_layer,
                attention_mask,
                layer_past=layer_past,
                get_key_value=get_key_value,
                rotary_pos_emb=rotary_pos_emb,
                relative_position_bias=relative_position_bias,
                headscale_tensor=self.head_scale_tensor if self.headscale else None,
            )

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        if get_key_value:
            output = [output, present]

        return output, bias


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)

    return _bias_dropout_add


def get_dropout_add(training):
    def _dropout_add(x, bias, residual, prob):
        assert bias is None
        return dropout_add(x, bias, residual, prob, training)

    return _dropout_add


class ParallelTransformerLayer_(MegatronModule, adapter_mixins.AdapterModuleMixin):
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
            position_embedding_type='learned_absolute',
            multi_query_attention=False,
            headscale=False,
            activations_checkpoint_granularity=None,
            sequence_parallel=False,
            normalize_attention_scores=True,
            num_moe_experts=1,
            moe_frequency=1,
            moe_dropout=0.0,
            num_kv_heads=None,
    ):
        super(ParallelTransformerLayer_, self).__init__()

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
        self.set_accepted_adapter_types([LinearAdapterConfig._target_, ParallelLinearAdapterConfig._target_])
        self.num_kv_heads = num_kv_heads

        if not bias and bias_dropout_add_fusion:
            raise ValueError(
                'bias_dropout_add_fusion=True requires bias=True, found bias=False. Either set both to True or both to False.'
            )

        if normalization not in ['layernorm', 'layernorm1p', 'rmsnorm']:
            raise ValueError(f'normalization must be "layernorm", "layernorm1p" or "rmsnorm", found {normalization}')

        if transformer_block_type not in ['pre_ln', 'post_ln', 'normformer', 'gpt_j']:
            raise ValueError(
                f'transformer_block_type must be either "pre_ln" or "post_ln" or "normformer", found {transformer_block_type}'
            )

        self.fp32_residual_connection = fp32_residual_connection  # if true move residual connections to fp32
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.bias_dropout_add_fusion = bias_dropout_add_fusion  # if true, enable bias dropout fusion

        self.checkpoint_layer_norm = (
                activations_checkpoint_granularity == 'selective'
        )  # transformer engine forward allows for more granular selective checkpointing
        transfer_with_static_ring = not self.checkpoint_layer_norm  # For now do not transfer with static ring
        # when selective enabled to avoid memory pressure

        # Self attention.
        # retrieval_decoder_after_self_attn skips the self attention
        if self.layer_type != LayerType.retrieval_decoder_after_self_attn:
            # Layernorm on the input data.
            if normalization == 'layernorm':
                self.input_layernorm = get_layer_norm(
                    hidden_size, layernorm_epsilon, persist_layer_norm, sequence_parallel
                )
            elif normalization == 'layernorm1p':
                self.input_layernorm = LayerNorm1P(
                    hidden_size, layernorm_epsilon, sequence_parallel_enabled=sequence_parallel
                )
            else:
                self.input_layernorm = MixedFusedRMSNorm(hidden_size, layernorm_epsilon,
                                                         sequence_parallel_enabled=sequence_parallel)

            self.self_attention = ParallelAttention(
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
                layer_number=layer_number,
                num_attention_heads=num_attention_heads,
                hidden_size=hidden_size,
                attention_type=AttnType.self_attn,
                attn_mask_type=self_attn_mask_type,
                precision=precision,
                apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                kv_channels=kv_channels,
                use_cpu_initialization=use_cpu_initialization,
                masked_softmax_fusion=masked_softmax_fusion,
                attention_dropout=attention_dropout,
                multi_query_attention=multi_query_attention,
                layer_type=layer_type,
                megatron_legacy=megatron_legacy,
                bias=bias,
                headscale=headscale,
                activations_checkpoint_granularity=activations_checkpoint_granularity,
                position_embedding_type=position_embedding_type,
                sequence_parallel=sequence_parallel,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
                normalize_attention_scores=normalize_attention_scores,
                transfer_with_static_ring=transfer_with_static_ring,
                num_kv_heads=num_kv_heads,
            )


            if self.layer_type != LayerType.decoder_pre_mlp or self.transformer_block_type != 'post_ln':
                #  the post_attention_layernorm is used for layermorm after mlp
                # don't need it for decoder_pre_mlp and post_ln
                if normalization == 'layernorm':
                    self.post_attention_layernorm = get_layer_norm(
                        hidden_size, layernorm_epsilon, persist_layer_norm, sequence_parallel
                    )
                elif normalization == 'layernorm1p':
                    self.post_attention_layernorm = LayerNorm1P(
                        hidden_size, layernorm_epsilon, sequence_parallel_enabled=sequence_parallel
                    )
                else:
                    self.post_attention_layernorm = MixedFusedRMSNorm(hidden_size, layernorm_epsilon,
                                                                      sequence_parallel_enabled=sequence_parallel)

        if self.layer_type == LayerType.decoder_pre_mlp:
            # skip MLP and cross attention
            return

        # the post_attention_layernorm is used for layermorm after mlp
        # need it for post_ln

        # MLP
        self.mlp = ParallelMLP(
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            use_cpu_initialization=use_cpu_initialization,
            bias_activation_fusion=bias_activation_fusion,
            openai_gelu=openai_gelu,
            onnx_safe=onnx_safe,
            activation=activation,
            bias=bias,
            transformer_block_type=transformer_block_type,
            normalization=normalization,
            layernorm_epsilon=layernorm_epsilon,
            persist_layer_norm=persist_layer_norm,
            sequence_parallel=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
            dropout=ffn_dropout,
            transfer_with_static_ring=transfer_with_static_ring,
        )

    def _get_bias_droput_add_func(self, transformer_block_type='pre_ln', position_after='attention'):
        """
        Returns a function that potentially fuses the dropout and bias addition.

        This function is particularly helpful for the normformer architecture that does not the fused kernel after attention layers, but can after the MLP.
        """
        # Normformer activations at this point have no bias vector since they've gone through another normalization layer.
        if transformer_block_type == 'normformer' and position_after == 'attention':
            bias_dropout_add_func = get_dropout_add(self.training)
        # Bias dropout add fused kernel
        elif self.bias and self.bias_dropout_add_fusion:
            if self.training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        # Bias dropout add non-fused kernel
        elif self.bias and not self.bias_dropout_add_fusion:
            bias_dropout_add_func = get_bias_dropout_add(self.training)
        # Dropout add non-fused kernel for a model without bias terms.
        else:
            bias_dropout_add_func = get_dropout_add(self.training)

        return bias_dropout_add_func

    def forward(
            self,
            hidden_states,
            attention_mask,
            encoder_output=None,
            enc_dec_attn_mask=None,
            layer_past=None,
            get_key_value=False,
            set_inference_key_value_memory=False,
            inference_max_sequence_len=None,
            rotary_pos_emb=None,
            # list of positional embedding tensors, first one self attention, second one and third one are for cross attention (q, k)
            self_attention_relative_position_bias=None,
            cross_attention_relative_position_bias=None,
            checkpoint_core_attention=False,
    ):
        # Self attention.
        if rotary_pos_emb is not None:
            # self attention pos_emb is (q, q)
            self_attention_pos_emb = (rotary_pos_emb[0], rotary_pos_emb[0])
            cross_attention_pos_emb = (rotary_pos_emb[1], rotary_pos_emb[2])
        else:
            self_attention_pos_emb = None
            cross_attention_pos_emb = None

        if self.layer_type != LayerType.retrieval_decoder_after_self_attn:
            # hidden_states: [b, s, h]

            # Pre-LN: x -> LN -> MHA -> Residual -> LN -> MLP -> Residual
            # Post-LN: x -> MHA -> Residual -> LN -> MLP -> Residual -> LN
            # Normformer: x -> LN -> MHA -> LN -> Residual -> MLP (w/LN) -> Residual
            # gpt_j: https://github.com/EleutherAI/gpt-neox/blob/303d7be582ae1c969347c25c54f568cc122445fc/megatron/model/transformer.py#L804-L847
            #        x = x + MHA(Input_LN(x)) + MLP(Post_LN(x))

            residual = hidden_states
            # Layer norm at the beginning of the transformer layer.
            if self.transformer_block_type in ['pre_ln', 'normformer']:
                if self.checkpoint_layer_norm:
                    hidden_states = tensor_parallel.checkpoint(self.input_layernorm, False, hidden_states)
                else:
                    hidden_states = self.input_layernorm(hidden_states)
            elif self.transformer_block_type in ['gpt_j']:
                normalization_output = self.post_attention_layernorm(hidden_states)
                hidden_states = self.input_layernorm(hidden_states)

            # Materialize attention mask right before use
            if is_torch_tpu_available():
                seq_len = hidden_states.shape[0]  # See above [b, *s*, h] shape
                if self.sequence_parallel:
                    seq_len *= parallel_state.get_tensor_model_parallel_world_size()
                attention_mask = torch.triu(torch.ones(
                    (1, 1, seq_len, seq_len), device='xla'), diagonal=1).bool()

            attention_output, attention_bias = self.self_attention(
                hidden_states,
                attention_mask,
                layer_past=layer_past,
                get_key_value=get_key_value,
                set_inference_key_value_memory=set_inference_key_value_memory,
                inference_max_sequence_len=inference_max_sequence_len,
                rotary_pos_emb=self_attention_pos_emb,
                relative_position_bias=self_attention_relative_position_bias,
                checkpoint_core_attention=checkpoint_core_attention,
            )

            if get_key_value:
                attention_output, presents = attention_output


            # jit scripting for a nn.module (with dropout) is not
            # trigerring the fusion kernel. For now, we use two
            # different nn.functional routines to account for varying
            # dropout semantics during training and inference phases.

            bias_dropout_add_func = self._get_bias_droput_add_func(
                transformer_block_type=self.transformer_block_type, position_after='attention'
            )
            if attention_bias is not None:
                attention_bias = attention_bias.expand_as(residual)

            layernorm_input = bias_dropout_add_func(attention_output, attention_bias, residual, self.hidden_dropout)
            # print(f"Layer: {self.layer_number} Attention checksum {layernorm_input.sum()}")

            if self.is_adapter_available():
                adapter_1 = self.get_from_adapter_layer(AdapterName.PRE_ATTN_ADAPTER)
                if adapter_1:
                    strategy = adapter_1.adapter_strategy
                    layernorm_input = self.forward_single_enabled_adapter_(
                        layernorm_input,
                        adapter_1,
                        adapter_name=AdapterName.PRE_ATTN_ADAPTER,
                        adapter_strategy=strategy,
                    )

            # Post-LN normalization after residual
            if self.transformer_block_type == 'post_ln':
                normalization_output = self.input_layernorm(layernorm_input)
                layernorm_input = normalization_output
            elif self.transformer_block_type in ['pre_ln', 'normformer']:
                # Layer norm post the self attention.
                if self.checkpoint_layer_norm:
                    normalization_output = tensor_parallel.checkpoint(self.post_attention_layernorm, False,
                                                                      layernorm_input)
                else:
                    normalization_output = self.post_attention_layernorm(layernorm_input)
        else:
            layernorm_input, normalization_output = hidden_states


        # MLP.
        mlp_output, mlp_bias = self.mlp(normalization_output)

        if self.transformer_block_type in ['gpt_j']:
            bias_dropout_add_func = self._get_bias_droput_add_func(
                transformer_block_type=self.transformer_block_type, position_after='mlp'
            )
            # x = layernorm_input + MLP(Post_LN(x))
            output = bias_dropout_add_func(mlp_output, mlp_bias, layernorm_input, self.hidden_dropout)

        else:
            residual = layernorm_input

            bias_dropout_add_func = self._get_bias_droput_add_func(
                transformer_block_type=self.transformer_block_type, position_after='mlp'
            )

            output = bias_dropout_add_func(mlp_output, mlp_bias, residual, self.hidden_dropout)
            # print(f"Layer: {self.layer_number} MLP + Dropout + Residual checksum {output.sum()}")


        if get_key_value:
            output = [output, presents]

        if (
                self.is_adapter_available()
        ):  # TODO: (@adithyre) was able to move adapter_2 back to the end of the transformer after ptl 1.7 update.
            adapter_2 = self.get_from_adapter_layer(AdapterName.POST_ATTN_ADAPTER)
            if adapter_2:
                strategy = adapter_2.adapter_strategy
                output = self.forward_single_enabled_adapter_(
                    output, adapter_2, adapter_name=AdapterName.POST_ATTN_ADAPTER, adapter_strategy=strategy
                )

        return output


class ParallelTransformerLayer(ParallelTransformerLayer_):
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
            num_kv_heads=None,
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
            num_kv_heads=num_kv_heads,
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
            position_embedding_type='learned_absolute',
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
            num_kv_heads=None,
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
        self.num_kv_heads = num_kv_heads
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
                    num_kv_heads=num_kv_heads,
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
            if normalization == 'layernorm':
                self.final_layernorm = get_layer_norm(
                    hidden_size, layernorm_epsilon, persist_layer_norm, sequence_parallel=sequence_parallel
                )
            elif normalization == 'layernorm1p':
                self.final_layernorm = LayerNorm1P(
                    hidden_size, layernorm_epsilon, sequence_parallel_enabled=sequence_parallel
                )
            else:
                self.final_layernorm = MixedFusedRMSNorm(hidden_size, layernorm_epsilon,
                                                         sequence_parallel_enabled=sequence_parallel)

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


def get_llama_language_model(
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
    num_kv_heads=None,
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
        num_kv_heads=num_kv_heads
    )
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
            vocab_size, self.hidden_size, init_method=self.init_method, use_cpu_initialization=use_cpu_initialization,
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
        num_kv_heads=None,
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
        self.num_kv_heads = num_kv_heads
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
            position_embedding_type=self.position_embedding_type,
            num_kv_heads=num_kv_heads,
        )
        self._encoder_key = 'encoder'


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
            # pooler
            if self.add_pooler:
                assert 'pooler' in state_dict, 'could not find data for pooler in the checkpoint'
                self.pooler.load_state_dict(state_dict[self._pooler_key], strict=strict)

            if not self.share_embeddings_and_output_weights:
                self.output_layer.load_state_dict(state_dict[self._output_layer_key], strict=strict)

