# coding=utf-8
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

import torch
from einops import rearrange
from torch import einsum, nn
from apex.transformer import parallel_state
from nemo.utils import logging

__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']


def get_pos_emb_on_this_cp_rank(pos_emb, seq_dim):
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    
    seq_len = pos_emb.shape[seq_dim]
    assert seq_len%cp_size==0, f"seq_len {seq_len} is not divisible by CP size {cp_size}"
    pos_emb = pos_emb.view(
    *pos_emb.shape[0:seq_dim],
    cp_size, seq_len // cp_size,
    *pos_emb.shape[(seq_dim + 1) :],
    )
    index = torch.tensor([cp_rank], device=pos_emb.device)
    pos_emb = pos_emb.index_select(seq_dim, index)
    pos_emb = pos_emb.view(*pos_emb.shape[0:seq_dim], -1, *pos_emb.shape[(seq_dim + 2) :])
    return pos_emb


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, 
        dim, 
        max_position_embeddings=4096, 
        base=10000, 
        device=None, 
        position_interpolation_factor=1.0,
        position_abf_factor=1,
        rotary_percentage=1.0,
        ):
        super().__init__()
        if rotary_percentage<1:
            rot_dim = int(dim * rotary_percentage)
        else:
            rot_dim = dim
        pass_dim = dim - rot_dim
        base = base * position_abf_factor
        inv_freq = 1.0 / (base ** (torch.arange(0, rot_dim, 2).float() / rot_dim))
        self.register_buffer("inv_freq", inv_freq)
        self.position_interpolation_factor = position_interpolation_factor
        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        if pass_dim>0:
        # first part even vector components, second part odd vector components, third part passing dimensions (without rotation)
        # 2 * dim in dimension size + pass dimension
            freq_all = torch.cat((self.inv_freq, self.inv_freq, torch.zeros((pass_dim,)).float()))
        else:
            # first part even vector components, second part odd vector components,
            #  2 * dim in dimension size
            freq_all = torch.cat((self.inv_freq, self.inv_freq))
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        if self.position_interpolation_factor and self.position_interpolation_factor != 1:
            t /= self.position_interpolation_factor
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.einsum("i,j->ij", t, freq_all)
        # emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[:, None, None, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[:, None, None, :], persistent=False)

    def forward(self, x, seq_len=None):
        cos_cached = self.cos_cached[:seq_len, :, :, :].to(dtype=x.dtype)
        sin_cached = self.sin_cached[:seq_len, :, :, :].to(dtype=x.dtype)
        if parallel_state.get_context_parallel_world_size() > 1:
            # slice rotary_pos_emb along sequence dimension and select the parition of the current CP rank
            cos_cached = get_pos_emb_on_this_cp_rank(cos_cached, 0)
            sin_cached = get_pos_emb_on_this_cp_rank(sin_cached, 0)
        return (cos_cached,sin_cached,)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos = cos[offset: q.shape[0] + offset, :, :, :]
    sin = sin[offset: q.shape[0] + offset, :, :, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
