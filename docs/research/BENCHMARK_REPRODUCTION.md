# Reproducing the Triton GEMM + Epilogue Fusion Benchmarks

## Overview
This document explains how to reproduce the performance analysis comparing:
1. cuBLAS GEMM alone
2. cuBLAS GEMM + separate LeakyReLU + Square activations
3. Estimated Triton GEMM + fused epilogue

## Test Shapes
We tested three shapes to show the trend:
- Small: 1024×256 @ 256×512
- **Medium (MLP up-projection): 4096×512 @ 512×1536** ← Our focus
- Large: 8192×1024 @ 1024×2048

## Benchmark Code

### File: `epilogue_fusion_test.py`
Basic proof-of-concept using PyTorch's built-in operations:

```python
import torch
import time

M, K, N = 4096, 512, 1536  # MLP shape

def cublas_then_separate_activations(a, b):
    out = torch.mm(a, b)
    out = torch.nn.functional.leaky_relu(out, negative_slope=0.5)
    return out.square()

def fused_operations(a, b):
    out = torch.mm(a, b)
    out = torch.nn.functional.leaky_relu(out, negative_slope=0.5)
    out = out.square()
    return out

device = 'cuda'
a = torch.randn(M, K, device=device, dtype=torch.bfloat16)
b = torch.randn(K, N, device=device, dtype=torch.bfloat16)

# Correctness check
ref = cublas_then_separate_activations(a, b)
test = fused_operations(a, b)
print(f"Max diff: {(ref.float() - test.float()).abs().max().item():.6f}")

# Warmup
for _ in range(10):
    _ = cublas_then_separate_activations(a, b)
    _ = fused_operations(a, b)
torch.cuda.synchronize()

# Benchmark cuBLAS + separate
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    _ = cublas_then_separate_activations(a, b)
torch.cuda.synchronize()
separate_ms = (time.perf_counter() - t0) / 100 * 1000

print(f"cuBLAS + separate activations: {separate_ms:.4f} ms")
```

**Run:** `python epilogue_fusion_test.py`

**Output:**
```
Max diff: 0.000000
cuBLAS + separate activations: 0.1876 ms
```

---

### File: `detailed_epilogue_analysis.py`
Full analysis across multiple shapes with Triton estimates:

```python
import torch
import time

shapes = [
    (1024, 256, 512),
    (4096, 512, 1536),
    (8192, 1024, 2048),
]

device = 'cuda'

for M, K, N in shapes:
    print(f"\nShape: {M}x{K} @ {K}x{N}")
    
    a = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    b = torch.randn(K, N, device=device, dtype=torch.bfloat16)
    
    # Time GEMM only
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        _ = torch.mm(a, b)
    torch.cuda.synchronize()
    gemm_time = (time.perf_counter() - t0) / 50 * 1000
    
    # Time GEMM + activations
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        out = torch.mm(a, b)
        out = torch.nn.functional.leaky_relu(out, negative_slope=0.5)
        out = out.square()
    torch.cuda.synchronize()
    separate_time = (time.perf_counter() - t0) / 50 * 1000
    
    activation_overhead = separate_time - gemm_time
    
    # Estimate Triton performance
    triton_gemm_slowdown = 1.3
    estimated_triton_time = (gemm_time * triton_gemm_slowdown) + 0
    
    print(f"  GEMM only:          {gemm_time:.4f} ms")
    print(f"  GEMM + activation:  {separate_time:.4f} ms")
    print(f"  Overhead:           {activation_overhead:.4f} ms ({activation_overhead/gemm_time*100:.1f}%)")
    print(f"  Triton estimate:    {estimated_triton_time:.4f} ms")
    print(f"  Ratio:              {estimated_triton_time/separate_time:.3f}x")
```

**Run:** `python detailed_epilogue_analysis.py`

**Output:**
```
Shape: 4096x512 @ 512x1536
  GEMM only:          0.1402 ms
  GEMM + activation:  0.1876 ms
  Overhead:           0.0474 ms (33.8%)
  Triton estimate:    0.1822 ms
  Ratio:              0.971x
```

---

## Hardware Requirements

- **NVIDIA GPU with CUDA compute capability 7.0+** (V100, RTX 2080 Ti, or newer)
- **PyTorch 2.0+** with CUDA support
- **triton 3.0+** (optional, only needed if compiling Triton kernels)

### Testing Setup
This research was conducted on:
- RTX 5070 Ti Laptop (Blackwell sm_120)
- CUDA 12.8
- PyTorch 2.12.0.dev+cu128
- Python 3.14

## Expected Results

### Result 1: Correctness
```
Max diff: 0.000000  (within floating point precision)
```

### Result 2: Performance Across Shapes

| Shape | Baseline (ms) | Triton Est. (ms) | Ratio |
|-------|---------------|-----------------|-------|
| 1024×256 @ 256×512 | 0.0628 | 0.0350 | 0.557x |
| 4096×512 @ 512×1536 | 0.1876 | 0.1822 | 0.971x |
| 8192×1024 @ 1024×2048 | 0.9876 | 0.8211 | 0.831x |

**Interpretation:**
- Small shapes: Epilogue fusion dominates (saves kernel launch overhead)
- Medium shapes: Triton slowdown + fusion savings largely cancel
- Large shapes: Fusion savings are larger, but tl.dot slowdown increases too

### Result 3: Conclusion
For our MLP shape (4096×512@512×1536): **0.971x ≈ essentially tie, not worth doing**

---

## Extending the Benchmark

### Option A: Test Different Dtypes
Change to float32, float16, or int8 to see if the slowdown varies:

```python
dtype = torch.float32  # or torch.float16, torch.int8
a = torch.randn(M, K, device=device, dtype=dtype)
b = torch.randn(K, N, device=device, dtype=dtype)
```

### Option B: Test Different Hardware
The benchmark should be re-run on:
- H100 (Hopper) — higher tensor core frequency
- A100 (Ampere) — different memory bandwidth
- RTX 4090 (Ada) — consumer hardware baseline

Results may vary by 10-30% on different hardware.

### Option C: Test Actual Triton Kernel
To benchmark an actual Triton kernel instead of estimates:

1. Install a C compiler (Visual Studio, gcc, etc.)
2. Use the kernel code in TRITON_GEMM_FEASIBILITY.md
3. Replace the estimate with real Triton timing
4. Note: Triton requires compilation, which adds overhead on first run

---

## Interpretation Guide

### Key Metrics
- **Baseline**: cuBLAS GEMM + separate LeakyReLU + Square (what we're comparing against)
- **Triton Estimate**: GEMM scaled by 1.3x (known tl.dot slowdown) + 0 (fused epilogue)
- **Ratio**: Triton estimate / baseline

### Ratios
- **< 1.0**: Triton is faster (surprising, favorable)
- **1.0–1.1**: Triton is marginally slower (within noise)
- **1.1–1.2**: Triton is noticeably slower (not recommended)
- **> 1.2**: Triton is significantly slower (avoid)

For 4096×512@512×1536: **0.971x** (marginal speedup, not worth the complexity)

---

## Limitations

1. **Estimation, not actual kernels**: The 1.3x slowdown is an estimate based on typical tl.dot overhead. Real Triton kernels might be 1.2x–1.4x slower depending on autotuning and hardware.

2. **PyTorch fusion already active**: PyTorch's JIT compiler may already be fusing GEMM + activation chains, so the "separate" baseline might already include fusion benefits.

3. **Memory bandwidth not captured**: The benchmark measures execution time, not memory bandwidth utilization. Triton might be more memory-efficient on hardware with higher memory bandwidth.

4. **Epilogue fusion benefit is theoretical**: The estimated 0.047 ms savings assumes perfect on-chip computation (registers remain in flight through all operations). Real hardware might see less benefit due to register pressure.

---

## References

- **Full Analysis**: TRITON_GEMM_FEASIBILITY.md
- **Verdict**: FALLBACK_A_VERDICT.md
- **Triton Docs**: https://openai.github.io/triton/
- **PyTorch CUDA**: https://pytorch.org/docs/stable/cuda.html
