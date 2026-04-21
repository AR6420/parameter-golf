"""Compile-safety gate for fused QKV.

Per docs/research/compile_compatibility_rules.md, any kernel/refactor must
compose with torch.compile(fullgraph=True, dynamic=False) — the setting
pr-1493 uses unconditionally.

Run from repo root:
    python experiments/test_compile_qkv.py

Tests:
  1. Isolated CausalSelfAttention compiles under fullgraph.
  2. Isolated compiled forward runs to completion.
  3. Compiled forward+backward produces gradients.
  4. A single Block (which wraps the attention) compiles too.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from train_gpt import Block, CastedLinear, CausalSelfAttention


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    device = torch.device("cuda")
    B, T = 4, 1024

    # pr-1493 starter-kit defaults
    dim = 512
    num_heads = 8
    num_kv_heads = 4
    mlp_mult = 2
    rope_base = 10000.0
    qk_gain_init = 1.5

    torch.manual_seed(1337)

    # --- 1 & 2: Compile CausalSelfAttention under fullgraph, run forward ---
    attn = CausalSelfAttention(
        dim=dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        rope_base=rope_base,
        qk_gain_init=qk_gain_init,
    ).to(device)
    for m in attn.modules():
        if isinstance(m, CastedLinear):
            m.float()

    try:
        compiled_attn = torch.compile(attn, fullgraph=True, dynamic=False)
    except Exception as e:
        raise RuntimeError(f"torch.compile call raised: {e}") from e
    print("[PASS] torch.compile(fullgraph=True, dynamic=False) accepted CausalSelfAttention")

    x = torch.randn(B, T, dim, dtype=torch.bfloat16, device=device, requires_grad=True)
    try:
        out = compiled_attn(x)
    except Exception as e:
        raise RuntimeError(f"compiled forward raised: {e}") from e
    assert out.shape == (B, T, dim), f"Unexpected output shape {out.shape}"
    assert torch.isfinite(out).all(), "NaN/Inf in compiled forward output"
    print(f"[PASS] Compiled attention forward: shape {tuple(out.shape)}, finite")

    # --- 3: Compiled backward through the fused QKV param ---
    loss = out.pow(2).mean()
    try:
        loss.backward()
    except Exception as e:
        raise RuntimeError(f"compiled backward raised: {e}") from e
    assert attn.c_qkv.weight.grad is not None, "Fused QKV grad not populated"
    assert torch.isfinite(attn.c_qkv.weight.grad).all(), "NaN/Inf in fused QKV grad"
    assert x.grad is not None and torch.isfinite(x.grad).all(), "Input grad bad"
    print(f"[PASS] Compiled attention backward: c_qkv.weight.grad shape {tuple(attn.c_qkv.weight.grad.shape)}, finite")

    # --- 4: Full Block (attention + MLP) compiles ---
    torch.manual_seed(1337)
    block = Block(
        dim=dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        mlp_mult=mlp_mult,
        rope_base=rope_base,
        qk_gain_init=qk_gain_init,
    ).to(device)
    for m in block.modules():
        if isinstance(m, CastedLinear):
            m.float()

    try:
        compiled_block = torch.compile(block, fullgraph=True, dynamic=False)
    except Exception as e:
        raise RuntimeError(f"torch.compile call on Block raised: {e}") from e

    x2 = torch.randn(B, T, dim, dtype=torch.bfloat16, device=device, requires_grad=True)
    x0 = torch.randn(B, T, dim, dtype=torch.bfloat16, device=device)
    try:
        out2 = compiled_block(x2, x0)
    except Exception as e:
        raise RuntimeError(f"compiled Block forward raised: {e}") from e
    assert out2.shape == (B, T, dim)
    assert torch.isfinite(out2).all()
    out2.pow(2).mean().backward()
    print(f"[PASS] Compiled Block fwd+bwd: output shape {tuple(out2.shape)}, finite grads")

    print("\n[ALL TESTS PASS] Fused QKV is compile-safe under fullgraph=True, dynamic=False")


if __name__ == "__main__":
    main()
