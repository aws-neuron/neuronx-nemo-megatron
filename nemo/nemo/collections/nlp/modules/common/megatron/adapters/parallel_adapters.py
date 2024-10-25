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


import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.init as init

from nemo.collections.common.parts.adapter_modules import AbstractAdapterModule
from nemo.collections.common.parts.utils import activation_registry
from nemo.collections.nlp.modules.common.megatron.utils import init_method_const, init_method_normal
from nemo.core.classes.mixins import adapter_mixin_strategies

try:
    from apex.transformer import parallel_state
    from apex.transformer.tensor_parallel import RowParallelLinear, ColumnParallelLinear
    from apex.normalization.fused_layer_norm import MixedFusedLayerNorm
    from apex.transformer.layers.layer_norm import FastRMSNorm as MixedFusedRMSNorm
    from apex.transformer.parallel_state import get_tensor_model_parallel_world_size
    from apex.transformer.utils import divide


    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False


class AdapterName(str, enum.Enum):
    """
    Names for adapters used in NLP Adapters and IA3. Note: changing this will break backward compatibility. 
    """

    MLP_INFUSED = "mlp_infused_adapter"
    KEY_INFUSED = "key_infused_adapter"
    VALUE_INFUSED = "value_infused_adapter"
    PRE_ATTN_ADAPTER = 'adapter_1'
    POST_ATTN_ADAPTER = 'adapter_2'
    PTUNING_ADAPTER = "ptuning_adapter"
    LORA_KQV_ADAPTER = "lora_kqv_adapter"
    LORA_KV_ADAPTER = "lora_kv_adapter"
    LORA_Q_ADAPTER = "lora_q_adapter"


class InfusedAdapter(AbstractAdapterModule):
    def __init__(
        self, in_features: int, adapter_strategy: adapter_mixin_strategies.ResidualAddAdapterStrategyConfig = None,
    ) -> None:
        super().__init__()
        self.scalers = nn.Parameter(torch.ones(in_features))

        # Setup adapter strategy
        self.setup_adapter_strategy(adapter_strategy)

    def forward(self, x):
        x = x * self.scalers[None, None, :]
        return x


class MLPInfusedAdapter(InfusedAdapter):
    """
    MLPInfusedAdapter is basically a clone of InfusedAdapter. We do this to make the adapter_mixin agnostic to adapter names
    and only check adapter class types. 
    """

    pass


@dataclass
class InfusedAdapterConfig:
    in_features: int
    adapter_strategy: Optional[Any] = field(
        default_factory=lambda: adapter_mixin_strategies.ResidualAddAdapterStrategyConfig()
    )
    _target_: str = "{0}.{1}".format(InfusedAdapter.__module__, InfusedAdapter.__name__)


@dataclass
class MLPInfusedAdapterConfig(InfusedAdapterConfig):
    _target_: str = "{0}.{1}".format(MLPInfusedAdapter.__module__, MLPInfusedAdapter.__name__)


class ParallelLinearAdapter(AbstractAdapterModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dim: int,
        activation: str = 'swish',
        norm_position: Optional[str] = 'post',
        norm_type: Optional[str] = 'mixedfusedlayernorm',
        column_init_method: str = 'xavier',  # TODO: should rename this to input_init_method to be more precise.
        row_init_method: str = 'zero',  # TODO: should rename this to output_init_method to be more precise.
        gather_output: bool = True,
        dropout: float = 0.0,
        sequence_parallel: bool = False,
        adapter_strategy: adapter_mixin_strategies.ResidualAddAdapterStrategyConfig = None,
        **kwargs,
    ):
        super().__init__()
        if not HAVE_APEX:
            logging.info("Apex is required to use ParallelLinearAdapters.")
            raise RuntimeError("ParallelLinearAdapter can not run without Apex.")
        self.activation = activation_registry[activation]()
        self.norm_position = norm_position

        no_async_tensor_model_parallel_allreduce = (
            parallel_state.get_tensor_model_parallel_world_size() == 1 or sequence_parallel
        )


        # Due to the way that truncated backpropagation through time splits the input sequence when 
        # sequence_parallel is enabled, the inner dimension of the output linear layer needs to be
        # split by the world size as it an additional xm.all_gather() is performed in the forward pass
        # of LinearWithGradAccumulationAndAsyncCommunication. This increases the effective dimension 
        # by world_size and the adapter will not match the original tensor it is adapting.

        # We force the 'in' matrix to not gather the input sequence by setting sequence_parallel_enabled to False.
        self.linear_in = ColumnParallelLinear(in_features,
                                              dim,
                                              bias=False,
                                              gather_output=not sequence_parallel,
                                              init_method=self._get_init_fn(column_init_method), 
                                              no_async_tensor_model_parallel_allreduce=no_async_tensor_model_parallel_allreduce,
                                              sequence_parallel_enabled=False)

        # TODO: Check this test is actually needed and we shouldn't always split the inner dim
        if sequence_parallel:
            world_size = get_tensor_model_parallel_world_size()
            inner_dimension = divide(dim, world_size)
        else:
            inner_dimension = dim

        if gather_output:
            # TODO: This path currently deadlocks during lazy eval
            self.linear_out = RowParallelLinear(dim,
                                                out_features,
                                                bias=False,
                                                init_method=self._get_init_fn(row_init_method),
                                                no_async_tensor_model_parallel_allreduce=no_async_tensor_model_parallel_allreduce,
                                                sequence_parallel_enabled=sequence_parallel)
        else:
            # We use this option to mirror the behavior a column parallel layer with two low-rank column parallel layers
            # if the original column parallel layer uses gather_output=False, then we will use the self.liner_out layer defined below.
            self.linear_out = ColumnParallelLinear(
                inner_dimension,
                out_features,
                bias=False,
                gather_output=False,
                init_method=self._get_init_fn(row_init_method),
                no_async_tensor_model_parallel_allreduce=no_async_tensor_model_parallel_allreduce,
                sequence_parallel_enabled=sequence_parallel
            )

        if norm_type == 'mixedfusedlayernorm':
            self.layer_norm = MixedFusedLayerNorm(in_features, 1e-5, sequence_parallel_enbaled=sequence_parallel)
        elif norm_type == 'layernorm':
            self.layer_norm = nn.LayerNorm(in_features)
        elif norm_type == "rmsnorm":
            self.layer_norm = MixedFusedRMSNorm(in_features, 1e-5, sequence_parallel_enbaled=sequence_parallel)
        else:
            raise NotImplementedError(f"norm_type should be either mixedfusedlayernorm, layernorm, or rmsnorm but {norm_type} was provided")

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # Setup adapter strategy
        self.setup_adapter_strategy(adapter_strategy)

    def _get_init_fn(self, init_method: str):
        if init_method == 'xavier':
            init_fn = init.xavier_normal_
        elif init_method == 'normal':
            init_fn = init_method_normal(0.2)
        elif init_method == "zero":
            init_fn = init_method_const(0.0)
        else:
            raise NotImplementedError("out_init_method should be zero, normal or xavier")
        return init_fn

    def forward(self, x):

        if self.norm_position == 'pre':
            x = self.layer_norm(x)

        x, _ = self.linear_in(x)  # ColumnLinear returns output and bias, we are ignoring the bias term.
        x = self.activation(x)
        x, _ = self.linear_out(x)

        if self.norm_position == 'post':
            x = self.layer_norm(x)

        # Add dropout if available
        if self.dropout is not None:
            x = self.dropout(x)

        return x


@dataclass
class ParallelLinearAdapterConfig:
    in_features: int
    out_features: int
    dim: int
    activation: str = 'swish'
    norm_position: Optional[str] = 'post'
    norm_type: Optional[str] = 'mixedfusedlayernorm'
    column_init_method: str = 'xavier'
    row_init_method: str = 'zero'
    gather_output: bool = True
    dropout: float = 0.0
    sequence_parallel: bool = False
    adapter_strategy: Optional[Any] = field(
        default_factory=lambda: adapter_mixin_strategies.ResidualAddAdapterStrategyConfig()
    )
    _target_: str = "{0}.{1}".format(ParallelLinearAdapter.__module__, ParallelLinearAdapter.__name__)


class LoraKQVAdapter(ParallelLinearAdapter):
    """
    Lora Adapters are the same arch as regular adapters but with potentially different input and output feature sizes 
    and they do not use an bottleneck activation function
    """

    pass


class LoraKVAdapter(ParallelLinearAdapter):
    """
    Lora Adapters are the same arch as regular adapters but with potentially different input and output feature sizes 
    and they do not use an bottleneck activation function
    """

    pass


class LoraQAdapter(ParallelLinearAdapter):
    """
    Lora Adapters are the same arch as regular adapters but with potentially different input and output feature sizes 
    and they do not use an bottleneck activation function
    """

    pass


@dataclass
class LoraKQVAdapterConfig(ParallelLinearAdapterConfig):
    _target_: str = "{0}.{1}".format(LoraKQVAdapter.__module__, LoraKQVAdapter.__name__)


@dataclass
class LoraQAdapterConfig(ParallelLinearAdapterConfig):
    _target_: str = "{0}.{1}".format(LoraQAdapter.__module__, LoraQAdapter.__name__)


@dataclass
class LoraKVAdapterConfig(ParallelLinearAdapterConfig):
    _target_: str = "{0}.{1}".format(LoraKVAdapter.__module__, LoraKVAdapter.__name__)


class ParallelLinearAdapterWeightTying(ParallelLinearAdapter):
    """
    Extends parallel linear adapter for weight tying by providing a position embedding and convenience methods for tying weights
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dim: int,
        activation: str = 'swish',
        norm_position: Optional[str] = 'post',
        norm_type: Optional[str] = 'mixedfusedlayernorm',
        column_init_method: str = 'xavier',  # TODO: should rename this to input_init_method to be more precise.
        row_init_method: str = 'zero',  # TODO: should rename this to output_init_method to be more precise.
        gather_output: bool = True,
        dropout: float = 0.0,
        num_position_embeddings: int = 1,
        dim_position_embeddings: int = 1024,
        position_embedding_strategy: Optional[str] = "add",
        model_parallel_config = None,
        **kwargs,
    ):
        self.position_embeddings = None
        self.mlp = None
        self.position_embedding_strategy = position_embedding_strategy
        assert self.position_embedding_strategy in ["add", "concat", "mlpconcat", "biasadd", None]
        if self.position_embedding_strategy == "concat":
            in_features += dim_position_embeddings
        elif self.position_embedding_strategy == "mlpconcat":
            in_features += dim_position_embeddings
        elif self.position_embedding_strategy == "biasadd":
            assert (
                out_features == dim_position_embeddings
            ), "adapter output feature size should match position emb size to bias add"
        elif self.position_embedding_strategy == "add":
            assert (
                in_features == dim_position_embeddings
            ), "adapter input feature size should match position emb size to add"
        super().__init__(
            in_features,
            out_features,
            dim,
            activation,
            norm_position,
            norm_type,
            column_init_method,
            row_init_method,
            gather_output,
            dropout,
            model_parallel_config,
            **kwargs,
        )
        if self.position_embedding_strategy:
            self.position_embeddings = torch.nn.Embedding(num_position_embeddings, dim_position_embeddings)
            self.position_embeddings.weight.data.fill_(0.0)
        if self.position_embedding_strategy == "mlpconcat":
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(dim_position_embeddings, dim_position_embeddings, bias=False),
                torch.nn.GELU(),
                torch.nn.Linear(dim_position_embeddings, dim_position_embeddings, bias=False),
            )
        self.register_buffer("position_id", torch.LongTensor([1]), persistent=False)

    def set_position(self, position_id):
        self.position_id *= position_id

    def tie_weights(self, position_id, adapter):

        self.set_position(position_id)
        if self.linear_in:
            self.linear_in.weight = adapter.linear_in.weight
        if self.linear_out:
            self.linear_out.weight = adapter.linear_out.weight
        if self.layer_norm:
            self.layer_norm.weight = adapter.layer_norm.weight
            self.layer_norm.bias = adapter.layer_norm.bias
        if self.mlp:
            self.mlp[0].weight = adapter.mlp[0].weight
            self.mlp[2].weight = adapter.mlp[2].weight
        if self.position_embeddings:
            self.position_embeddings.weight = adapter.position_embeddings.weight

        return True

    def forward(self, x):

        if self.position_embedding_strategy:
            pos: torch.nn.Embedding = self.position_embeddings(self.position_id).unsqueeze(0)
            if self.position_embedding_strategy == "add":
                pos = pos.expand_as(x)
                x = x + pos

            elif self.position_embedding_strategy == "concat":
                pos = pos.expand(x.shape[0], x.shape[1], pos.shape[2])
                x = torch.cat((x, pos), dim=2)
            elif self.position_embedding_strategy == "mlpconcat":
                pos = pos.expand(x.shape[0], x.shape[1], pos.shape[2])
                pos = self.mlp(pos)
                x = torch.cat((x, pos), dim=2)

        if self.norm_position == 'pre':
            x = self.layer_norm(x)

        x, _ = self.linear_in(x)  # ColumnLinear returns output and bias, we are ignoring the bias term.
        x = self.activation(x)
        x, _ = self.linear_out(x)
        if self.norm_position == 'post':
            x = self.layer_norm(x)

        if self.position_embedding_strategy == "biasadd":
            pos = pos.expand_as(x)
            x = x + pos

        # Add dropout if available
        if self.dropout is not None:
            x = self.dropout(x)

        return x


@dataclass
class ParallelLinearAdapterWeightTyingConfig:
    in_features: int
    out_features: int
    dim: int
    activation: str = 'swish'
    norm_position: Optional[str] = 'post'
    norm_type: Optional[str] = 'mixedfusedlayernorm'
    column_init_method: str = 'xavier'
    row_init_method: str = 'zero'
    gather_output: bool = True
    dropout: float = 0.0
    num_position_embeddings: int = 1
    dim_position_embeddings: int = 1024
    position_embedding_strategy: Optional[str] = "concat"
    _target_: str = "{0}.{1}".format(
        ParallelLinearAdapterWeightTying.__module__, ParallelLinearAdapterWeightTying.__name__
    )


class LoraKQVAdapterWeightTying(ParallelLinearAdapterWeightTying):
    """
    TODO
    """

    pass


@dataclass
class LoraKQVAdapterWeightTyingConfig(ParallelLinearAdapterWeightTyingConfig):
    _target_: str = "{0}.{1}".format(LoraKQVAdapterWeightTying.__module__, LoraKQVAdapterWeightTying.__name__)