# Copyright (c) 2026, Flash Attention contributors.

"""Fused NVFP4 output tests, SM100/SM110.

The fused path quantizes attention output O to NVFP4 in the forward epilogue
(vLLM ``scaled_fp4_quant`` semantics): packed e2m1 codes, per-16-element dynamic
e4m3 block scales pre-multiplied by a static fp32 global scale, with
dequant = e2m1 * float(scale_e4m3) / global_scale.
"""

import math
import os
from typing import Tuple

import pytest
import torch

from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func
from flash_attn.cute.testing import (
    attention_ref,
    is_fake_mode,
    maybe_fake_tensor_mode,
)

USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", 0)) == 1
IS_NVFP4_SM_SUPPORTED = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] == 10
)

skip_if_no_nvfp4_sm = pytest.mark.skipif(
    not IS_NVFP4_SM_SUPPORTED,
    reason="Fused NVFP4 output requires SM100/SM110 (Blackwell).",
)

GROUP_SIZE = 16
FP4_MAX = 6.0
# e2m1 representable magnitudes, code order (sign in bit 3).
_E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _cast_to_e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even onto the e2m1 grid; returns uint8 codes in [0, 16)."""
    mag = x.abs().clamp(max=FP4_MAX)
    values = _E2M1_VALUES.to(x.device)
    dist = (mag.unsqueeze(-1) - values).abs()
    # Nearest value; on ties prefer the even code (matches cvt.rn PTX semantics).
    min_dist = dist.amin(dim=-1, keepdim=True)
    is_min = dist == min_dist
    codes = torch.arange(8, device=x.device)
    even_first = torch.where(codes % 2 == 0, codes, codes + 8)  # rank even codes first
    code = torch.where(is_min, even_first, torch.full_like(even_first, 32)).amin(dim=-1)
    code = torch.where(code >= 8, code - 8, code)
    return (code + torch.where(x < 0, 8, 0)).to(torch.uint8)


def _e2m1_codes_to_float(codes: torch.Tensor) -> torch.Tensor:
    values = _E2M1_VALUES.to(codes.device)
    mag = values[(codes & 0x7).long()]
    return torch.where(codes & 0x8 != 0, -mag, mag)


def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """(..., d/2) float4_e2m1fn_x2 -> (..., d) uint8 codes; low nibble = even element."""
    bytes_u8 = packed.view(torch.uint8)
    lo = bytes_u8 & 0xF
    hi = bytes_u8 >> 4
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def _quantize_nvfp4(
    x: torch.Tensor, global_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference NVFP4 quantizer (vLLM ``scaled_fp4_quant`` semantics).

    Returns ``(e2m1_codes, e4m3_scales)`` with codes unpacked to one uint8 per
    element. Dequant is ``e2m1 * scales.float() / global_scale`` (per 16-group).
    """
    head_dim = x.shape[-1]
    assert head_dim % GROUP_SIZE == 0
    x_fp32 = x.float()
    x_grp = x_fp32.unflatten(-1, (head_dim // GROUP_SIZE, GROUP_SIZE))
    amax = x_grp.abs().amax(dim=-1)
    sf_e4m3 = (amax / FP4_MAX * global_scale.float()).to(torch.float8_e4m3fn)
    sf_back = sf_e4m3.float()
    inv = torch.where(
        sf_back == 0, torch.zeros_like(sf_back), global_scale.float() / sf_back
    )
    codes = _cast_to_e2m1_codes(x_grp * inv.unsqueeze(-1)).flatten(-2)
    return codes, sf_e4m3


def _dequantize_nvfp4(
    codes: torch.Tensor, scales: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    """Inverse of `_quantize_nvfp4` (codes unpacked, one uint8 per element)."""
    vals = _e2m1_codes_to_float(codes)
    sf = scales.float().repeat_interleave(GROUP_SIZE, dim=-1)
    return vals * sf / global_scale.float()


def _global_scale_for(out_ref: torch.Tensor) -> torch.Tensor:
    """vLLM-style static global scale: (448 * 6) / amax of the tensor."""
    amax = out_ref.abs().max().clamp(min=1e-12)
    return (448.0 * FP4_MAX / amax).to(torch.float32)


def _assert_nvfp4_close(
    fused_fp4: torch.Tensor,
    fused_scales: torch.Tensor,
    global_scale: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rtol: float = 2.0,
    **ref_kwargs,
) -> None:
    """kernel_err <= rtol * eager_BF16+nvfp4-quant_err + ULP_atol, vs FP32 ref."""
    out_ref_fp32, _ = attention_ref(q, k, v, None, None, upcast=True, **ref_kwargs)
    out_pt_bf16, _ = attention_ref(
        q, k, v, None, None, upcast=False, reorder_ops=True, **ref_kwargs,
    )

    ref_codes, ref_scales = _quantize_nvfp4(out_ref_fp32, global_scale)
    pt_codes, pt_scales = _quantize_nvfp4(out_pt_bf16, global_scale)

    fused_deq = _dequantize_nvfp4(_unpack_fp4(fused_fp4), fused_scales, global_scale)
    ref_deq = _dequantize_nvfp4(ref_codes, ref_scales, global_scale)
    pt_deq = _dequantize_nvfp4(pt_codes, pt_scales, global_scale)

    fwd_atol = 2 * (ref_deq + 0.3 - 0.3 - ref_deq).abs().max().item()
    kernel_err = (fused_deq - ref_deq).abs().max().item()
    eager_err = (pt_deq - ref_deq).abs().max().item()

    assert kernel_err <= rtol * eager_err + fwd_atol, (
        f"fused NVFP4 kernel max-err vs FP32 ref ({kernel_err:.4f}) > "
        f"{rtol}x eager-BF16+nvfp4-quant max-err ({eager_err:.4f}) + "
        f"ULP atol ({fwd_atol:.4f})"
    )


@skip_if_no_nvfp4_sm
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "head_dim,head_dim_v",
    [
        (128, 128),  # standard MHA
        (192, 128),  # DeepSeek MLA prefill shape
        (64, 64),
    ],
)
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_nvfp4_output_matches_post_quant(
    dtype: torch.dtype,
    causal: bool,
    head_dim: int,
    head_dim_v: int,
    mha_type: str,
):
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, seqlen, num_heads = 2, 512, 16
    if mha_type == "mha":
        num_kv_heads = num_heads
    elif mha_type == "mqa":
        num_kv_heads = 1
    else:
        num_kv_heads = 4

    q = torch.randn(batch, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, seqlen, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, seqlen, num_kv_heads, head_dim_v, dtype=dtype, device=device)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    if is_fake_mode():
        global_scale = torch.tensor(448.0 * FP4_MAX, dtype=torch.float32, device=device)
    else:
        out_ref_fp32, _ = attention_ref(q, k, v, None, None, causal=causal, upcast=True)
        global_scale = _global_scale_for(out_ref_fp32).to(device)

    out_buf = torch.empty(
        batch, seqlen, num_heads, head_dim_v // 2, dtype=torch.float4_e2m1fn_x2, device=device
    )
    scales_buf = torch.empty(
        batch, seqlen, num_heads, head_dim_v // GROUP_SIZE,
        dtype=torch.float8_e4m3fn, device=device,
    )
    out, _ = flash_attn_func(
        q, k, v,
        softmax_scale=softmax_scale, causal=causal,
        out=out_buf, output_scale=global_scale, output_scales=scales_buf,
    )
    if is_fake_mode():
        return
    assert out.dtype == torch.float4_e2m1fn_x2
    assert scales_buf.dtype == torch.float8_e4m3fn
    assert scales_buf.shape == (batch, seqlen, num_heads, head_dim_v // GROUP_SIZE)

    _assert_nvfp4_close(out, scales_buf, global_scale, q, k, v, causal=causal)


@skip_if_no_nvfp4_sm
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_nvfp4_output_varlen():
    torch.manual_seed(0)
    device = torch.device("cuda")
    seqlens = [256, 384, 512, 192]
    total_q = sum(seqlens)
    num_heads, num_kv_heads = 16, 4
    head_dim = head_dim_v = 128
    dtype = torch.bfloat16

    q = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total_q, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(total_q, num_kv_heads, head_dim_v, dtype=dtype, device=device)
    cu_seqlens = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.tensor(seqlens, dtype=torch.int32, device=device).cumsum(0)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    global_scale = torch.tensor(448.0, dtype=torch.float32, device=device)

    out_buf = torch.empty(
        total_q, num_heads, head_dim_v // 2, dtype=torch.float4_e2m1fn_x2, device=device
    )
    scales_buf = torch.empty(
        total_q, num_heads, head_dim_v // GROUP_SIZE,
        dtype=torch.float8_e4m3fn, device=device,
    )
    out, _ = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(seqlens), max_seqlen_k=max(seqlens),
        softmax_scale=softmax_scale, causal=True,
        out=out_buf, output_scale=global_scale, output_scales=scales_buf,
    )
    if is_fake_mode():
        return
    assert out.dtype == torch.float4_e2m1fn_x2

    rtol = 2.0
    kernel_err = 0.0
    eager_err = 0.0
    fwd_atol = 0.0
    for i, sl in enumerate(seqlens):
        s, e = int(cu_seqlens[i].item()), int(cu_seqlens[i + 1].item())
        qi = q[s:e].unsqueeze(0)
        ki = k[s:e].unsqueeze(0)
        vi = v[s:e].unsqueeze(0)
        out_ref_fp32, _ = attention_ref(qi, ki, vi, None, None, causal=True, upcast=True)
        out_pt_bf16, _ = attention_ref(
            qi, ki, vi, None, None, causal=True, upcast=False, reorder_ops=True,
        )
        ref_codes, ref_scales = _quantize_nvfp4(out_ref_fp32, global_scale)
        pt_codes, pt_scales = _quantize_nvfp4(out_pt_bf16, global_scale)
        ref_deq = _dequantize_nvfp4(ref_codes, ref_scales, global_scale)
        pt_deq = _dequantize_nvfp4(pt_codes, pt_scales, global_scale)
        fused_deq = _dequantize_nvfp4(
            _unpack_fp4(out[s:e].unsqueeze(0)), scales_buf[s:e].unsqueeze(0), global_scale
        )
        fwd_atol = max(fwd_atol, 2 * (ref_deq + 0.3 - 0.3 - ref_deq).abs().max().item())
        kernel_err = max(kernel_err, (fused_deq - ref_deq).abs().max().item())
        eager_err = max(eager_err, (pt_deq - ref_deq).abs().max().item())

    assert kernel_err <= rtol * eager_err + fwd_atol, (
        f"varlen fused NVFP4 max-err vs FP32 ref ({kernel_err:.4f}) > "
        f"{rtol}x eager max-err ({eager_err:.4f}) + ULP atol ({fwd_atol:.4f})"
    )


@skip_if_no_nvfp4_sm
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_nvfp4_rejects_split_kv():
    device = torch.device("cuda")
    q = torch.randn(1, 128, 8, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, 4096, 8, 128, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, 4096, 8, 128, dtype=torch.bfloat16, device=device)
    out_buf = torch.empty(1, 128, 8, 64, dtype=torch.float4_e2m1fn_x2, device=device)
    scales_buf = torch.empty(1, 128, 8, 8, dtype=torch.float8_e4m3fn, device=device)
    global_scale = torch.tensor(448.0, dtype=torch.float32, device=device)
    with pytest.raises(AssertionError, match="SplitKV"):
        flash_attn_func(
            q, k, v, causal=False, num_splits=8,
            out=out_buf, output_scale=global_scale, output_scales=scales_buf,
        )


@skip_if_no_nvfp4_sm
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_nvfp4_rejects_wrong_group_size():
    device = torch.device("cuda")
    q = torch.randn(1, 128, 8, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, 128, 8, 128, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, 128, 8, 128, dtype=torch.bfloat16, device=device)
    out_buf = torch.empty(1, 128, 8, 64, dtype=torch.float4_e2m1fn_x2, device=device)
    # 32-element groups: unsupported (NVFP4 is 16).
    scales_buf = torch.empty(1, 128, 8, 4, dtype=torch.float8_e4m3fn, device=device)
    global_scale = torch.tensor(448.0, dtype=torch.float32, device=device)
    with pytest.raises(AssertionError, match="group_size 16"):
        flash_attn_func(
            q, k, v, causal=False,
            out=out_buf, output_scale=global_scale, output_scales=scales_buf,
        )
