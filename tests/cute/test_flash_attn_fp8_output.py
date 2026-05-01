# Copyright (c) 2025, the FlashAttention authors.
"""Correctness tests for the static per-tensor FP8 (e4m3fn) fused output
epilogue (SM100/SM110).

The kernel is exercised with an FP8 output buffer + an FP32 `out_scale`
(passed via `quant_kwargs={"quant_key": "kFp8StaticTensorSym", "out_scale": s}`);
the FP32 accumulator is multiplied by `1/s` and cast to FP8 in the
epilogue (`flash_fwd_sm100.py:correction_epilogue`; `flash_fwd_combine.py`
for split-KV).

Accuracy contract follows the convention in `test_flash_attn.py`: the
kernel result is compared against an FP32 ground-truth attention
(`attention_ref(..., upcast=True)`), and the allowed error is bounded by
how badly an eager-BF16 implementation (`upcast=False, reorder_ops=True`)
quantized to FP8 would diverge from the same FP32 ground truth. This
prevents tolerances from being a magic number while still tracking real
ULP-scale FP8 noise.

Skipped on hardware that doesn't expose an SM100/SM110 forward path
(the kernel constructor enforces this via `_get_dsl().get_arch_enum()`),
unless running under `FLASH_ATTENTION_FAKE_TENSOR=1` for the compile-only
fast pass — see `CLAUDE.md` for the two-pass workflow.
"""

import math
import os

import pytest
import torch

from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func
from flash_attn.cute.testing import (
    attention_ref,
    is_fake_mode,
    maybe_fake_tensor_mode,
)

USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", 0)) == 1
IS_FP8_SM_SUPPORTED = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] == 10
)

skip_if_no_fp8_sm = pytest.mark.skipif(
    not IS_FP8_SM_SUPPORTED,
    reason="Fused FP8 output requires SM100/SM110 (Blackwell).",
)

# Fixed scale for the fused-quant tests. Decoupled from the actual data
# peak so the same value works under FakeTensorMode (where `.amax()` is
# unavailable). Picked to cover the typical attention-output magnitude
# (~1-2 for unit-variance Q/K/V) within FP8 e4m3 range.
DEFAULT_OUT_SCALE = 0.005


def _quantize_fp8(x: torch.Tensor, out_scale: float) -> torch.Tensor:
    """Static-scaled FP8 cast — what `static_scaled_fp8_quant` would emit."""
    finfo = torch.finfo(torch.float8_e4m3fn)
    return (x.float() / out_scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)


def _assert_fp8_close(
    fused_fp8: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out_scale: float,
    rtol: float = 2.0,
    **ref_kwargs,
) -> None:
    """Assert the kernel's FP8 output is at most `rtol`x as inaccurate as
    eager-BF16-then-FP8 would be, both measured against an FP32 ground
    truth. Mirrors the accuracy check in `test_flash_attn.py:281`.

    `ref_kwargs` is forwarded to `attention_ref` (causal, window_size,
    softcap, learnable_sink, ...).
    """
    out_ref_fp32, _ = attention_ref(q, k, v, None, None, upcast=True, **ref_kwargs)
    out_pt_bf16, _ = attention_ref(
        q, k, v, None, None, upcast=False, reorder_ops=True, **ref_kwargs,
    )

    ref_fp8 = _quantize_fp8(out_ref_fp32, out_scale)
    pt_fp8 = _quantize_fp8(out_pt_bf16, out_scale)

    fused_deq = fused_fp8.float() * out_scale
    ref_deq = ref_fp8.float() * out_scale
    pt_deq = pt_fp8.float() * out_scale

    # ULP-style atol pinned to the ref's magnitude (same pattern as
    # `fwd_atol` in test_flash_attn_output).
    fwd_atol = 2 * (ref_deq + 0.3 - 0.3 - ref_deq).abs().max().item()
    kernel_err = (fused_deq - ref_deq).abs().max().item()
    eager_err = (pt_deq - ref_deq).abs().max().item()

    assert kernel_err <= rtol * eager_err + fwd_atol, (
        f"fused FP8 kernel max-err vs FP32 ref ({kernel_err:.4f}) > "
        f"{rtol}x eager-BF16+post-quant max-err ({eager_err:.4f}) + "
        f"ULP atol ({fwd_atol:.4f})"
    )


@skip_if_no_fp8_sm
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "head_dim,head_dim_v",
    [(64, 64), (128, 128), (192, 128)],  # small MHA + standard + DeepSeek MLA prefill
)
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_fp8_output_matches_post_quant(
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
    else:  # gqa
        num_kv_heads = 4

    q = torch.randn(batch, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, seqlen, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, seqlen, num_kv_heads, head_dim_v, dtype=dtype, device=device)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    out_scale = DEFAULT_OUT_SCALE

    fused_buffer = torch.empty(
        batch, seqlen, num_heads, head_dim_v, dtype=torch.float8_e4m3fn, device=device,
    )
    fused_out, _ = flash_attn_func(
        q, k, v,
        softmax_scale=softmax_scale, causal=causal,
        out=fused_buffer,
        quant_kwargs={"quant_key": "kFp8StaticTensorSym", "out_scale": out_scale},
    )
    if is_fake_mode():
        return  # compile-only pass; skip data-dependent comparison
    assert fused_out.dtype == torch.float8_e4m3fn

    _assert_fp8_close(fused_out, q, k, v, out_scale, causal=causal)


@skip_if_no_fp8_sm
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_fp8_output_varlen_deepseek_mla():
    """DeepSeek-V3 MLA prefill shape (qk=192, v=128) via the varlen API."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    seqlens = [256, 384, 512, 192]
    total_q = sum(seqlens)
    num_heads, num_kv_heads = 16, 1
    head_dim, head_dim_v = 192, 128
    dtype = torch.bfloat16
    out_scale = DEFAULT_OUT_SCALE

    q = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total_q, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(total_q, num_kv_heads, head_dim_v, dtype=dtype, device=device)
    cu_seqlens = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.tensor(seqlens, dtype=torch.int32, device=device).cumsum(0)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    fused_buffer = torch.empty(
        total_q, num_heads, head_dim_v, dtype=torch.float8_e4m3fn, device=device,
    )
    fused_out, _ = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(seqlens), max_seqlen_k=max(seqlens),
        softmax_scale=softmax_scale, causal=True,
        out=fused_buffer,
        quant_kwargs={"quant_key": "kFp8StaticTensorSym", "out_scale": out_scale},
    )
    if is_fake_mode():
        return
    assert fused_out.dtype == torch.float8_e4m3fn

    # Compare per-sequence: split fused output and inputs, run the standard
    # (non-varlen) reference for each, and accumulate errors. attention_ref
    # is 4D-only, so this is the cleanest way to handle varlen.
    finfo = torch.finfo(torch.float8_e4m3fn)
    fwd_atol = 0.0
    kernel_err = 0.0
    eager_err = 0.0
    for i, sl in enumerate(seqlens):
        s, e = int(cu_seqlens[i].item()), int(cu_seqlens[i + 1].item())
        qi = q[s:e].unsqueeze(0)
        ki = k[s:e].unsqueeze(0)
        vi = v[s:e].unsqueeze(0)
        out_ref_fp32, _ = attention_ref(qi, ki, vi, None, None, causal=True, upcast=True)
        out_pt_bf16, _ = attention_ref(
            qi, ki, vi, None, None, causal=True, upcast=False, reorder_ops=True,
        )
        ref_fp8 = _quantize_fp8(out_ref_fp32, out_scale)
        pt_fp8 = _quantize_fp8(out_pt_bf16, out_scale)
        ref_deq = ref_fp8.float() * out_scale
        pt_deq = pt_fp8.float() * out_scale
        fused_deq = fused_out[s:e].unsqueeze(0).float() * out_scale
        fwd_atol = max(fwd_atol, 2 * (ref_deq + 0.3 - 0.3 - ref_deq).abs().max().item())
        kernel_err = max(kernel_err, (fused_deq - ref_deq).abs().max().item())
        eager_err = max(eager_err, (pt_deq - ref_deq).abs().max().item())

    rtol = 2.0
    assert kernel_err <= rtol * eager_err + fwd_atol, (
        f"varlen fused FP8 max-err vs FP32 ref ({kernel_err:.4f}) > "
        f"{rtol}x eager max-err ({eager_err:.4f}) + ULP atol ({fwd_atol:.4f})"
    )


@skip_if_no_fp8_sm
def test_fp8_output_auto_allocate():
    """User passes quant_kwargs without `out`; library allocates FP8 buffer."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    q = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16, device=device)
    v = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16, device=device)

    fused_out, _ = flash_attn_func(
        q, k, v, causal=True,
        quant_kwargs={"quant_key": "kFp8StaticTensorSym", "out_scale": 0.05},
    )
    assert fused_out.dtype == torch.float8_e4m3fn
    assert fused_out.shape == (2, 256, 8, 128)


@skip_if_no_fp8_sm
def test_fp8_output_scale_as_tensor():
    """`out_scale` accepts a 0-d tensor, not just a Python float."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    q = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16, device=device)
    v = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16, device=device)

    scale_tensor = torch.tensor(0.05, dtype=torch.float32, device=device)
    out_a, _ = flash_attn_func(
        q, k, v, causal=True,
        quant_kwargs={"quant_key": "kFp8StaticTensorSym", "out_scale": scale_tensor},
    )
    out_b, _ = flash_attn_func(
        q, k, v, causal=True,
        quant_kwargs={"quant_key": "kFp8StaticTensorSym", "out_scale": 0.05},
    )
    # Tensor scalar coercion (`.cpu().item()`) must produce the same FP8 bytes
    # as the equivalent Python float.
    assert torch.equal(out_a.view(torch.uint8), out_b.view(torch.uint8))


@skip_if_no_fp8_sm
@pytest.mark.parametrize(
    "window_size",
    [(128, 0), (64, 64), (-1, 0)],  # left-only causal local, symmetric local, full causal
    ids=["causal_local_left", "symmetric_local", "causal_full"],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_fp8_output_sliding_window(window_size):
    """Local / sliding-window masking exercises a different mask_mod path."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, seqlen, num_heads = 2, 512, 8
    head_dim = 128
    dtype = torch.bfloat16
    out_scale = DEFAULT_OUT_SCALE

    q = torch.randn(batch, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    causal = window_size == (-1, 0)
    ws_kernel = (None, None) if causal else window_size
    # attention_ref converts (l, 0) when causal=True; pass the same shape.
    ws_ref = (None, None) if causal else window_size

    fused_buffer = torch.empty(
        batch, seqlen, num_heads, head_dim, dtype=torch.float8_e4m3fn, device=device,
    )
    fused_out, _ = flash_attn_func(
        q, k, v, softmax_scale=softmax_scale, causal=causal, window_size=ws_kernel,
        out=fused_buffer,
        quant_kwargs={"quant_key": "kFp8StaticTensorSym", "out_scale": out_scale},
    )
    if is_fake_mode():
        return

    _assert_fp8_close(
        fused_out, q, k, v, out_scale, causal=causal, window_size=ws_ref,
    )


@skip_if_no_fp8_sm
@pytest.mark.parametrize("softcap", [15.0, 30.0])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_fp8_output_softcap(softcap: float):
    """Softcap (Gemma/GLM) wraps logits through tanh before softmax."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, seqlen, num_heads = 2, 512, 8
    head_dim = 128
    dtype = torch.bfloat16
    out_scale = DEFAULT_OUT_SCALE

    q = torch.randn(batch, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    fused_buffer = torch.empty(
        batch, seqlen, num_heads, head_dim, dtype=torch.float8_e4m3fn, device=device,
    )
    fused_out, _ = flash_attn_func(
        q, k, v, softmax_scale=softmax_scale, causal=True, softcap=softcap,
        out=fused_buffer,
        quant_kwargs={"quant_key": "kFp8StaticTensorSym", "out_scale": out_scale},
    )
    if is_fake_mode():
        return

    # Softcap can lift the relative kernel-vs-eager error a touch; matches
    # `rtol = 3 if softcap > 0 else 2` in test_flash_attn_output.
    _assert_fp8_close(
        fused_out, q, k, v, out_scale, rtol=3.0, causal=True, softcap=softcap,
    )


@skip_if_no_fp8_sm
@pytest.mark.parametrize(
    "scale_factor",
    [0.1, 1.0, 4.0],
    ids=["scale_underuses_range", "scale_matches_peak", "scale_overuses_range"],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_fp8_output_scale_extremes(scale_factor: float):
    """Sweep `out_scale` away from the matched-peak choice.

    - small scale: values divide to >fp8_max → clamp.
    - matched scale: roughly fills the FP8 range.
    - large scale: values divide to << 1 → mantissa truncation.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, seqlen, num_heads = 2, 256, 8
    head_dim = 128
    dtype = torch.bfloat16
    out_scale = DEFAULT_OUT_SCALE * scale_factor

    q = torch.randn(batch, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    fused_buffer = torch.empty(
        batch, seqlen, num_heads, head_dim, dtype=torch.float8_e4m3fn, device=device,
    )
    fused_out, _ = flash_attn_func(
        q, k, v, softmax_scale=softmax_scale, causal=True,
        out=fused_buffer,
        quant_kwargs={"quant_key": "kFp8StaticTensorSym", "out_scale": out_scale},
    )
    if is_fake_mode():
        return

    _assert_fp8_close(fused_out, q, k, v, out_scale, causal=True)


@skip_if_no_fp8_sm
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_fp8_output_split_kv():
    """Split-KV + FP8: forward writes FP32 partials, combine emits FP8.

    Triggered by short Q + long K (decode-style); we force num_splits=4 to
    guarantee the split-KV combine path even if the auto-heuristic wouldn't
    pick it.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, seqlen_q, seqlen_k, num_heads = 2, 1, 4096, 16
    head_dim = 128
    out_scale = DEFAULT_OUT_SCALE

    q = torch.randn(batch, seqlen_q, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(batch, seqlen_k, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(batch, seqlen_k, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    fused_buffer = torch.empty(
        batch, seqlen_q, num_heads, head_dim, dtype=torch.float8_e4m3fn, device=device,
    )
    fused_out, _ = flash_attn_func(
        q, k, v,
        softmax_scale=softmax_scale, causal=True, num_splits=4,
        out=fused_buffer,
        quant_kwargs={"quant_key": "kFp8StaticTensorSym", "out_scale": out_scale},
    )
    if is_fake_mode():
        return
    assert fused_out.dtype == torch.float8_e4m3fn

    _assert_fp8_close(fused_out, q, k, v, out_scale, causal=True)


def test_fp8_output_validation_errors():
    """Validation paths fire on any GPU (no kernel launch needed)."""
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        pytest.skip("Validation calls into _flash_attn_fwd which expects CUDA tensors")
    from flash_attn.cute.interface import _flash_attn_fwd

    q = torch.randn(2, 64, 4, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(2, 64, 4, 128, dtype=torch.bfloat16, device=device)
    v = torch.randn(2, 64, 4, 128, dtype=torch.bfloat16, device=device)
    out_fp8 = torch.empty(2, 64, 4, 128, dtype=torch.float8_e4m3fn, device=device)

    fp8_kwargs = {"quant_key": "kFp8StaticTensorSym", "out_scale": 0.5}

    # FP8 output without quant_kwargs -> AssertionError (FP8 dtype but no key)
    with pytest.raises(AssertionError, match="no quant_kwargs"):
        _flash_attn_fwd(q, k, v, out=out_fp8, _arch=100)

    # quant_key set but out_scale missing -> AssertionError
    with pytest.raises(AssertionError, match="out_scale.*required"):
        _flash_attn_fwd(q, k, v, out=out_fp8,
                        quant_kwargs={"quant_key": "kFp8StaticTensorSym"}, _arch=100)

    # quant_kwargs on a BF16 output buffer -> AssertionError
    # (caught downstream by _validate_tensor which checks out.dtype matches
    # the dtype derived from quant_key, here torch.float8_e4m3fn).
    bf16_out = torch.empty(2, 64, 4, 128, dtype=torch.bfloat16, device=device)
    with pytest.raises(AssertionError,
                       match="torch.float8_e4m3fn"):
        _flash_attn_fwd(q, k, v, out=bf16_out, quant_kwargs=fp8_kwargs, _arch=100)

    # FP8 output on Ampere (SM80), Hopper (SM90), and consumer Blackwell
    # (SM120) -> rejected by the per-arch __init__ assert in each forward
    # class. SM90 plumbing is in place but the smem layouts / O copy atom
    # still use input dtype; tracked for follow-up.
    with pytest.raises(AssertionError, match="FP8 output not implemented"):
        _flash_attn_fwd(q, k, v, out=out_fp8, quant_kwargs=fp8_kwargs, _arch=80)
    with pytest.raises(AssertionError, match="FP8 output not implemented"):
        _flash_attn_fwd(q, k, v, out=out_fp8, quant_kwargs=fp8_kwargs, _arch=90)
    with pytest.raises(AssertionError, match="FP8 output not implemented"):
        _flash_attn_fwd(q, k, v, out=out_fp8, quant_kwargs=fp8_kwargs, _arch=120)

    # Wrong / unimplemented quant_key -> AssertionError
    with pytest.raises(AssertionError, match="not yet supported"):
        _flash_attn_fwd(q, k, v, out=out_fp8,
                        quant_kwargs={"quant_key": "kFp8Dynamic128Sym",
                                      "out_scale": 0.5}, _arch=100)

    # FP8 + num_splits > 1 is NOW SUPPORTED (combine kernel handles the
    # FP32-partials -> FP8 cast). Should not raise on the validation block.

    # Non-positive scale -> AssertionError
    with pytest.raises(AssertionError, match="positive"):
        _flash_attn_fwd(q, k, v, out=out_fp8,
                        quant_kwargs={"quant_key": "kFp8StaticTensorSym",
                                      "out_scale": -1.0}, _arch=100)
