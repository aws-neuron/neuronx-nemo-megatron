import os

import neuronxcc.nki.language as nl
import numpy as np
import torch
import torch_xla.core.xla_model as xm
from collections import defaultdict

from neuronxcc.nki.kernels.attention import FlashConfig

import logging


logger = logging.getLogger(__name__)

def check_xla_bf16_flags():
    if os.getenv("XLA_USE_BF16") == '1' or os.getenv("XLA_DOWNCAST_BF16") == '1':
        return True
    return False

def _flash_attn_forward(q, k, v, causal, mixed_precision, seed, dropout_p, softmax_scale):
    try:
        from neuronxcc.nki.kernels import flash_fwd
    except ImportError:
        raise RuntimeError(
            "Flash Attention initialization failed!\n"
            "Please check and upgrade neuronx-cc and torch_neuronx.\n"
            "python3 -m pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com neuronx-cc torch_neuronx"
        )

    config = None
    if check_xla_bf16_flags():
        config = FlashConfig(lse_dtype="bfloat16")

    bs, num_heads, head_dim, seq = q.shape
    attn_output, lse = flash_fwd[bs, num_heads](q, k, v, seed,
                                                use_causal_mask=causal,
                                                mixed_precision=mixed_precision,
                                                dropout_p=dropout_p,
                                                softmax_scale=softmax_scale,
                                                config=config)
    
    if check_xla_bf16_flags():
        attn_output = attn_output.to(torch.bfloat16)
    
    return attn_output, lse


def _flash_attn_backward(q, k, v, o, dout, lse, seed, causal, mixed_precision, dropout_p, softmax_scale):
    try:
        from neuronxcc.nki.kernels import flash_attn_bwd
    except ImportError:
        raise RuntimeError(
            "Flash Attention initialization failed!\n"
            "Please check and upgrade neuronx-cc and torch_neuronx.\n"
            "python3 -m pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com neuronx-cc torch_neuronx"
        )

    bs, num_heads, _, _ = q.shape
    dq, dk, dv, = flash_attn_bwd[bs, num_heads](q, k, v, o,
                                                dout, lse, seed,
                                                use_causal_mask=causal,
                                                mixed_precision=mixed_precision,
                                                dropout_p=dropout_p,
                                                softmax_scale=softmax_scale)
    return dq, dk, dv


class NKIAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        softmax_scale,
        causal: bool,
        mixed_precision: bool,
        seed,
        dropout_p: float,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-2] ** (-0.5)

        if seed is None and dropout_p > 0.0:
            # NKI only supports 32bit seed
            seed = np.array([xm.get_rng_state()]).astype(np.int32)
            seed = torch.from_numpy(seed).to(q.device)

        attn_output, lse = _flash_attn_forward(
            q,
            k,
            v.permute(0, 1, 3, 2),
            causal=causal,
            mixed_precision=mixed_precision,
            seed=seed,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
        )
        ctx.save_for_backward(q, k, v, attn_output, lse, seed)
        ctx.causal = causal
        ctx.mixed_precision = mixed_precision
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale

        # Move seed manually if the dropout is used
        # https://github.com/pytorch/xla/blob/v1.13.0/torch_xla/csrc/tensor.cpp#L323
        if dropout_p > 0.0:
            orig_seed = xm.get_rng_state()
            running_seed = (orig_seed * 214013 + 2531011) & 0xFFFFFFFFFFFFFFFF
            xm.set_rng_state(int(running_seed))
        return attn_output

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, attn_output, lse, seed = ctx.saved_tensors
        dout = dout.permute(0, 1, 3, 2)
        attn_output = attn_output.permute(0, 1, 3, 2)
        dq, dk, dv = _flash_attn_backward(
            q,
            k,
            v,
            attn_output,
            dout,
            lse,
            seed=seed,
            causal=ctx.causal,
            mixed_precision=ctx.mixed_precision,
            dropout_p=ctx.dropout_p,
            softmax_scale=ctx.softmax_scale,
        )
        return dq, dk, dv, None, None, None, None, None


def nki_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=True,
    mixed_precision=True,
    seed=None,
):
    """
    Arguments:
        q: (batch_size, nheads, seqlen, headdim)
        k: (batch_size, nheads_k, seqlen, headdim)
        v: (batch_size, nheads_k, seqlen, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        mixed_precision: bool. Whether to enable the higher precisions on the softmax.
        seed: int32 torch.Tensor. The seed for the dropout.

    Return:
        out: (batch_size, seqlen, nheads, headdim).
    """

    _, _, seqlen, _ = q.shape
    if seqlen % 2048 != 0:
        raise NotImplementedError("Only support sequence as multiples of 2K")

    # Permute QKV to match the kernel required layouts
    q = q.permute(0, 1, 3, 2)
    k = k.permute(0, 1, 3, 2)
    v = v.permute(0, 1, 3, 2)

    if check_xla_bf16_flags():
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)

    return NKIAttnFunc.apply(q, k, v, softmax_scale, causal, mixed_precision, seed, dropout_p)