"""Benchmark FA4 fused FP8 output epilogue vs. unfused (BF16 attn + post-quant).

Quantifies the win of writing FP8 directly from the attention epilogue instead
of running attention to BF16 and then quantizing in a separate kernel. FA4
itself ships no FP8 quant op, so the "unfused" baseline uses a torch.compile'd
eager `(out / scale).clamp.to(fp8)`. torch.compile fuses divide+clamp+cast
into a single kernel — close enough to a hand-tuned CUDA op that the numbers
are directionally accurate (a tuned op would be ~equal or marginally faster).

Output format mirrors `benchmark_attn.py`: per-shape lines with ms and TFLOPS
for each path, plus the saved-time delta.

Usage:

    python benchmarks/bench_flash_attention_fp8_output.py
    python benchmarks/bench_flash_attention_fp8_output.py --shape prefill_mla_4k
    python benchmarks/bench_flash_attention_fp8_output.py --rep 50

Requires SM100/SM110 (Blackwell) for the fused path. Other archs reject FP8
output in the per-arch __init__ assert and will fail to launch.
"""

import argparse
import time

import torch
from triton.testing import do_bench

from flash_attn.cute.bench_utils import flops
from flash_attn.cute.interface import flash_attn_func


# (name, batch, seqlen_q, seqlen_k, num_heads, num_kv_heads, head_dim, head_dim_v, causal)
# Naming convention: <mode>_<attn>_<seqlen>, where:
#   mode = prefill (sq == sk) | decode (sq == 1, sk large)
#   attn = mla (qk=192, v=128) | mha (h_q == h_kv) | gqa (h_q > h_kv)
#   seqlen = the K-side context length
SHAPES = {
    # DeepSeek-V3 MLA prefill — the primary target of this PR.
    "prefill_mla_4k":  (2,  4096, 4096, 16,   1, 192, 128, True),
    # Standard MHA prefill, 4K context.
    "prefill_mha_4k":  (2,  4096, 4096, 32,  32, 128, 128, True),
    # Llama-style GQA prefill (8:1 ratio), 8K context.
    "prefill_gqa_8k":  (2,  8192, 8192, 32,   4, 128, 128, True),
    # GQA decode (sq=1, h=16/1), 8K context — common decode shape.
    "decode_gqa_8k":   (16,    1, 8192, 16,   1, 128, 128, True),
    # MHA decode, 8K context.
    "decode_mha_8k":   (16,    1, 8192, 16,  16, 128, 128, True),
}


def static_fp8_quant_eager(out_bf16: torch.Tensor, inv_scale: float) -> torch.Tensor:
    """Reference post-attention static-FP8 cast — single kernel via torch.compile.

    Mirrors vLLM's `static_scaled_fp8_quant`: out / scale, clamp to [-fp8_max,
    +fp8_max], cast to e4m3fn. `inv_scale` is `1/scale` (precomputed to keep the
    hot path multiply-only).
    """
    finfo = torch.finfo(torch.float8_e4m3fn)
    return out_bf16.float().mul(inv_scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)


# Compile once for a fair per-call comparison; first call is excluded by do_bench warmup.
_static_fp8_quant_compiled = torch.compile(static_fp8_quant_eager, mode="reduce-overhead")


def bench_one(name, shape, warmup, rep):
    batch, sq, sk, nh, nkv, dq, dv, causal = shape
    device = torch.device("cuda")
    dtype = torch.bfloat16

    q = torch.randn(batch, sq, nh, dq, dtype=dtype, device=device)
    k = torch.randn(batch, sk, nkv, dq, dtype=dtype, device=device)
    v = torch.randn(batch, sk, nkv, dv, dtype=dtype, device=device)

    # Pick a representative scale (peak of one BF16 forward).
    ref_out, _ = flash_attn_func(q, k, v, causal=causal)
    finfo = torch.finfo(torch.float8_e4m3fn)
    out_scale = max(float(ref_out.float().abs().amax().item()) / finfo.max, 1e-4)
    inv_scale = 1.0 / out_scale

    fp8_buf = torch.empty(batch, sq, nh, dv, dtype=torch.float8_e4m3fn, device=device)

    # ── Path A: BF16 attn only (lower bound; what attention costs without quant) ──
    def fwd_bf16():
        return flash_attn_func(q, k, v, causal=causal)

    # ── Path B: BF16 attn + post-hoc fused FP8 cast (unfused baseline) ──
    def fwd_bf16_then_quant():
        out, _ = flash_attn_func(q, k, v, causal=causal)
        return _static_fp8_quant_compiled(out, inv_scale)

    # ── Path C: FA4 FP8 fused output (this PR) ──
    def fwd_fp8_fused():
        return flash_attn_func(
            q, k, v, causal=causal,
            out=fp8_buf,
            quant_kwargs={"quant_key": "kFp8StaticTensorSym", "out_scale": out_scale},
        )

    # `time.sleep(1.0)` between runs matches benchmark_attn.py — lets the GPU
    # cool / clock-stabilize between back-to-back do_bench windows.
    time.sleep(1.0)
    ms_bf16 = do_bench(fwd_bf16, warmup=warmup, rep=rep) * 1e-3
    time.sleep(1.0)
    ms_unfused = do_bench(fwd_bf16_then_quant, warmup=warmup, rep=rep) * 1e-3
    time.sleep(1.0)
    ms_fused = do_bench(fwd_fp8_fused, warmup=warmup, rep=rep) * 1e-3

    # FLOPS shared by all three paths (post-quant cast is negligible vs attn).
    n_flops = flops(batch, nh, sq, sk, dq, dv, causal=causal)
    def tflops(s): return n_flops / s * 1e-12

    saved = ms_unfused - ms_fused
    speedup = ms_unfused / ms_fused

    print(
        f"{name:<14} b={batch} sq={sq:>5} sk={sk:>5} h={nh:>3}/{nkv:<3} d={dq}-{dv:<3}  "
        f"bf16={ms_bf16*1e6:>7.1f}us/{tflops(ms_bf16):>4.0f}TF  "
        f"bf16+quant={ms_unfused*1e6:>7.1f}us/{tflops(ms_unfused):>4.0f}TF  "
        f"fused-fp8={ms_fused*1e6:>7.1f}us/{tflops(ms_fused):>4.0f}TF  "
        f"saved={saved*1e6:>+6.1f}us ({speedup:.2f}x)"
    )


def main():
    parser = argparse.ArgumentParser(description="FA4 fused FP8 output benchmark")
    parser.add_argument("--shape", action="append", choices=list(SHAPES) + ["all"],
                        default=None, help="Shape preset to run (repeatable). Default: all.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=10)
    args = parser.parse_args()

    cap = torch.cuda.get_device_capability()
    if cap[0] != 10:
        raise SystemExit(
            f"Fused FP8 output requires SM100/SM110 (Blackwell). "
            f"Detected sm{cap[0]}{cap[1]}; aborting."
        )

    shapes = list(SHAPES) if not args.shape or "all" in args.shape else args.shape
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Warmup={args.warmup}, rep={args.rep}\n")

    for name in shapes:
        torch.cuda.empty_cache()
        bench_one(name, SHAPES[name], args.warmup, args.rep)


if __name__ == "__main__":
    main()
