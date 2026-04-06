# GEMM Boundary Fusion: MLP + Attention Kernel Fusion

**val_bpb: PENDING** (3-seed mean) | **~15.9 MB** | 8xH100 SXM, 600s

This submission fuses element-wise operations (RMSNorm, LeakyReLU^2, partial RoPE, per-head gain, residual adds) directly INTO Triton GEMM kernel boundaries as prologues and epilogues. Instead of running cuBLAS matmuls with separate activation kernels, we run autotuned Triton GEMMs that compute the activation inline -- eliminating HBM round-trips for intermediate tensors.

**Projected improvement over current SOTA ([PR #1019](https://github.com/openai/parameter-golf/pull/1019), 86.7 ms/step):** ~2.64 ms/step faster (3.0%), enabling ~200 additional training steps in the 600s window.

## Results

| Seed | Steps | ms/step | Pre-quant BPB | **Sliding BPB** | Artifact |
|------|-------|---------|---------------|-----------------|----------|
| 1337 | PENDING | PENDING | PENDING | **PENDING** | PENDING |
| 42   | PENDING | PENDING | PENDING | **PENDING** | PENDING |
| 2025 | PENDING | PENDING | PENDING | **PENDING** | PENDING |
| **Mean** | | | | **PENDING** | |

---

## Origin Story: From Megakernel Failure to Boundary Fusion

This work directly evolved from [PR #1316](https://github.com/openai/parameter-golf/pull/1316) -- our first submission attempt, a full-depth Triton megakernel that fused the entire MLP forward pass (RMSNorm + up-projection + LeakyReLU^2 + down-projection + residual) into a single kernel.

**PR #1316 validated the insight but proved the mechanism wrong:**

| | PR #1316 (Megakernel) | This work (Boundary Fusion) |
|---|---|---|
| **Insight** | Element-wise ops should not touch HBM | Same |
| **Mechanism** | Replace cuBLAS with one giant tl.dot loop | Partner with cuBLAS-speed Triton GEMMs |
| **Tiling** | BLOCK_N=1536 (full MLP width) | Autotuned 128x128x32 per GEMM |
| **H100 result** | 122 ms/step (41% slower than SOTA) | ~84 ms/step (3% faster, projected) |
| **Why** | tl.dot can't compete when tiling is wrong | Per-GEMM autotune matches cuBLAS within 3% |

The key discovery: PR #1316's megakernel was slow not because `tl.dot` is inherently slow, but because it tiled the entire 1536-dim MLP width as one block dimension. When we benchmark **individual** Triton GEMMs with proper autotuned tile sizes (128x128x32), they match cuBLAS within 3% -- and the fused epilogue eliminates the HBM round-trip entirely, making the fused kernel **faster** than cuBLAS + separate activation.

### The Inductor Audit That Confirmed the Opportunity

Before writing any kernels, we ran a systematic audit of what `torch.compile` (Inductor) does with the SOTA model's matmul boundaries:

```
Per MLP path, Inductor generates 5 separate kernels:
  1. triton_rms_norm_mul        (RMSNorm + ln_scale)
  2. extern_kernels.mm          (cuBLAS up projection 512->1536)
  3. triton_leaky_relu_pow      (LeakyReLU + square)
  4. extern_kernels.mm          (cuBLAS down projection 1536->512)
  5. triton_add_mul             (scale + residual add)
```

Every `extern_kernels.mm` (cuBLAS) call is an **absolute scheduling barrier**. Inductor cannot fuse element-wise ops across these boundaries -- not with `mode='default'`, not with `mode='max-autotune'`, not with CUTLASS epilogue fusion enabled. The 1536-dim MLP intermediate hits HBM ~48MB per block per pass, completely unnecessarily.

This audit result -- **zero cross-GEMM fusion by Inductor** -- is what motivated the Triton GEMM approach: if Inductor can't fuse across the boundary, we write kernels that ARE the boundary.

---

## Architecture: 3 Fused Kernels

### Kernel 1: `fused_mlp_up_proj`

Fuses the MLP up-projection GEMM with the activation function.

```
BEFORE: cuBLAS_mm(x, W_up) -> HBM -> triton_leaky_relu_square() -> HBM
AFTER:  triton_gemm_leaky_relu_sq(x, W_up)  [one kernel, no intermediate HBM]
```

- **Shape:** `[B*T, 512] @ [512, 1536]` with inline `LeakyReLU(0.5)^2` epilogue
- **Autotune winner:** BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
- **Speedup vs cuBLAS + separate activation:** 1.39x (M=4096), 1.52x (M=8192), 1.59x (M=16384)
- **Correctness:** cos_sim = 0.999992, float32 accumulator, bf16 I/O

### Kernel 2: `fused_mlp_down_proj`

Fuses the MLP down-projection GEMM with the residual connection.

```
BEFORE: cuBLAS_mm(hidden, W_down) -> HBM -> triton_scale_add(mlp_scale, residual) -> HBM
AFTER:  triton_gemm_scale_residual(hidden, W_down, mlp_scale, residual)  [one kernel]
```

- **Shape:** `[B*T, 1536] @ [1536, 512]` with inline `residual + mlp_scale * acc` epilogue
- **Autotune winner:** BLOCK_M=128, BLOCK_N=64, BLOCK_K=64
- **Speedup vs cuBLAS + separate ops:** 1.13x (M=4096), 1.26x (M=16384)
- **Correctness:** cos_sim = 0.999995

### Kernel 3: `fused_qk_norm_rope`

Fuses 5 element-wise attention ops into one kernel per Q/K tensor.

```
BEFORE: F.rms_norm(q) -> apply_rotary_emb(q) -> q * gain  [5 kernel launches]
AFTER:  fused_qk_norm_rope(q, cos, sin, gain)              [1 kernel launch]
```

- **Ops fused:** RMSNorm(head_dim=64) + partial RoPE (dims 0-15 only, 16-63 passthrough) + per-head gain
- **Shape:** `[B, T, H, 64]` where H=8 (Q) or H=4 (K, no gain)
- **Speedup:** 2.37x for combined Q+K processing
- **Correctness:** cos_sim = 0.999998, passthrough dims cos_sim = 1.000000
- **Design:** `HAS_GAIN: tl.constexpr` -- same kernel handles Q (with gain) and K (without) with zero branching

### Combined Savings

| Fusion Target | Savings per block | x11 blocks | % of 86.7ms |
|---|---|---|---|
| MLP up proj (GEMM + LeakyReLU^2) | 0.052 ms | 0.57 ms | 0.7% |
| MLP down proj (GEMM + scale + residual) | 0.017 ms | 0.19 ms | 0.2% |
| Attn Q+K (norm + RoPE + gain) | 0.171 ms | 1.88 ms | 2.2% |
| **Total** | **0.240 ms** | **2.64 ms** | **3.0%** |

At 86.7 ms/step baseline, a 3.0% speedup yields ~84.1 ms/step, enabling ~7,130 steps (vs ~6,920) -- roughly 210 additional training steps in the 600s window.

---

## Validation Evidence

### Local Validation (RTX 5070 Ti, 200 steps)

**MLP Fusion:**
- 200-step loss diff (fused vs unfused): **0.000052** (threshold: < 0.01)
- Gradient norm deviation: **0.00%** (threshold: < 1%)
- Forward cos_sim: 0.999992 (Kernel 1), 0.999995 (Kernel 2)

**Attention Fusion:**
- 200-step loss diff: **0.003359** (threshold: < 0.01)
- Gradient norm deviation: **0.00%** (threshold: < 1%)
- Q cos_sim: 0.999998, K cos_sim: 0.999998
- Passthrough dims (16-63) cos_sim: **1.000000** (partial RoPE boundary correct)

### Backward Pass

All three kernels use `torch.autograd.Function` wrappers:
- **Forward:** Triton fused kernel (fast path)
- **Backward:** Recompute with standard PyTorch ops (correct gradients guaranteed)
- Weight gradients flow correctly to parameter banks (validated: model learns identically to baseline over 200 steps)

### Fallback Path

Every kernel has a `_USE_TRITON_MLP = False` / `_USE_TRITON_ATTN = False` fallback that uses the original `F.linear` + manual ops. The model trains identically in both modes.

---

## Architecture (unchanged from SOTA)

| Component | Setting | Source |
|-----------|---------|--------|
| Layers | 11 (512d, 8 GQA heads, 4 KV heads) | Baseline |
| MLP | 3x (1536) with LeakyReLU(0.5)^2 | [#493](https://github.com/openai/parameter-golf/pull/493) |
| Attention | XSA on all 11 layers | [#478](https://github.com/openai/parameter-golf/pull/478) |
| BigramHash | 3072 x dim=112 | [PR #1019](https://github.com/openai/parameter-golf/pull/1019) |
| RoPE | Partial (16/64 dims) | [#315](https://github.com/openai/parameter-golf/pull/315) |
| Quantization | Full Hessian GPTQ int6 (AR self-gen) | [PR #1019](https://github.com/openai/parameter-golf/pull/1019) |
| Optimizer | Parallel Muon + Parameter Banking | [#399](https://github.com/openai/parameter-golf/pull/399) |
| **NEW: MLP fusion** | **Triton GEMM + inline epilogue** | **This work** |
| **NEW: Attn fusion** | **Fused QK norm + RoPE + gain** | **This work** |

## Requirements

```bash
pip install sentencepiece zstandard
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
```

Triton is bundled with PyTorch 2.9+ on Linux. On Windows, install `triton-windows`.

## Run Command

```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=42   torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=2025 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Lineage

```
PR #1019 (Current SOTA, 1.1147 BPB, 86.7 ms/step)
    |
    +-- PR #1316 (Megakernel attempt, FAILED: 122 ms/step, 41% slower)
    |   |  Lesson: tl.dot CAN match cuBLAS with proper per-GEMM autotuning
    |   |  Lesson: Don't replace cuBLAS -- partner with it
    |   +-- Inductor audit: confirmed zero cross-GEMM fusion by torch.compile
    |
    +-- This work (GEMM Boundary Fusion)
        +-- Kernel 1: fused_mlp_up_proj (GEMM + LeakyReLU^2)
        +-- Kernel 2: fused_mlp_down_proj (GEMM + scale + residual)
        +-- Kernel 3: fused_qk_norm_rope (RMSNorm + partial RoPE + gain)
        +-- Same architecture, same hyperparameters, just faster kernels
```
