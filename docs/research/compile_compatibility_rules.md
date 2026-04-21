# torch.compile Compatibility Rules for Custom Kernels

**Context:** `pr-1493` train_gpt.py wraps the whole model in `torch.compile(base_model, dynamic=False, fullgraph=True)` **unconditionally** — no `HAS_FLASH_ATTN_3` guard, no try/except fallback, no eager path. Any custom kernel we inject must compose with fullgraph tracing from the design stage.

**Why this matters:** On `forgefuse-phase3a` H100 seed=1337 run, our custom `_FusedQKNormRoPE.backward` called `x.detach().requires_grad_(True)` — dynamo rejected this with `Unsupported Tensor.requires_grad_() call`, which forced a full compile bypass (`TORCHDYNAMO_DISABLE=1`) and cost ~2× step time. The defensive try/except around the compile call *caught the object*, but dynamo's lazy trace on the first forward raised anyway. Environment-variable kill switch was the only reliable escape. Net result: submission trained in eager mode, losing the entire torch.compile performance envelope.

This document captures the rules to avoid repeating that mistake.

---

## Part 1 — The rules

### Rule 1: No `requires_grad_()` calls anywhere in the graph

Dynamo cannot symbolically track `requires_grad_()` mutations on tensors. This includes:

- `x.requires_grad_(True)` / `x.requires_grad_(False)`
- `x.detach().requires_grad_(True)` (the phase3a pattern)
- `torch.no_grad()` contexts that re-enable grad mid-graph

**Also forbidden:** `x.grad = ...` direct assignment, `x.grad.zero_()` inside forward.

If you need a tensor that flows through forward but doesn't propagate gradient, use `x.detach()` alone — no `requires_grad_` follow-up. For custom autograd where you want to control which inputs receive gradients, declare it explicitly via `ctx.save_for_backward` and return `None` for gradient slots you don't want.

### Rule 2: No in-place grad mutations in forward or backward

Even when compile succeeds, `ctx.saved_tensors` returned from backward as in-place-modified tensors trigger silent correctness bugs. Keep backward pure-functional:

- ❌ `ctx.saved[0].mul_(scale)` inside backward
- ✅ `grad_x = ctx.saved[0] * scale` (out-of-place)

This also rules out `Tensor.addcmul_`, `Tensor.add_` on saved tensors, `Tensor.copy_` into tensors dynamo might be tracing.

### Rule 3: Register Triton kernels via `torch.library.triton_op` (or `custom_op`)

Raw Triton launches inside `autograd.Function.forward` are an **eager-mode escape hatch** — dynamo doesn't know how to trace them. It either falls back (graph break = slow) or (under `fullgraph=True`) errors out.

Use the official registration path:

```python
from torch.library import triton_op, wrap_triton

@triton_op("my_lib::fused_mlp_up", mutates_args=())
def fused_mlp_up(x: Tensor, w: Tensor) -> Tensor:
    out = torch.empty_like(x @ w.T)
    def grid(meta): return (triton.cdiv(x.numel(), meta["BLOCK"]),)
    wrap_triton(_fused_mlp_up_kernel)[grid](x, w, out, BLOCK=128)
    return out
```

`triton_op` registers a **symbolic op** that dynamo recognizes and treats as a black box. `wrap_triton` lets you launch a Triton kernel from inside the op body.

### Rule 4: Add FakeTensor (meta) registrations for custom ops

Dynamo needs to trace shape/dtype of your op's output *without* actually running it. Provide a meta kernel:

```python
@fused_mlp_up.register_fake
def _(x, w):
    # Return an empty tensor with the correct shape/dtype/device.
    return torch.empty(x.shape[0], w.shape[0], dtype=x.dtype, device=x.device)
```

Without this, compile fails with "NotImplementedError: meta kernel not registered."

### Rule 5: No data-dependent control flow

Dynamo cannot trace Python `if`/`for` whose condition depends on a tensor's **value** (only shape-dependent is OK with dynamic=False + static shapes).

- ❌ `if x.sum() > 0: ... else: ...`
- ❌ `while loss > threshold: ...`
- ❌ `for i in range(int(n_iter_tensor)): ...`
- ✅ `if x.shape[0] == 8:` (static shape)
- ✅ `torch.where(condition, a, b)` (tensor-level branching)
- ✅ `torch.cond(predicate, true_fn, false_fn, (...))` (higher-order op)

### Rule 6: Out-of-place ops only, with a short exception list

Dynamo does support a narrow set of in-place ops that don't alias saved tensors (`+=` on a fresh buffer, `.copy_` into a pre-allocated output). But the failure modes are subtle. Default to out-of-place; only use in-place when profiling demonstrates a measurable win AND the target tensor is provably not part of any saved graph state.

### Rule 7: Avoid graph breaks

A graph break splits the compiled region at the break point, eagerly runs the broken section, then re-enters compile. Under `fullgraph=True`, breaks become hard errors. Common sources:

- `print(tensor)` — tensor→Python scalar implicit
- `tensor.item()`, `float(tensor)`, `bool(tensor)`
- `tensor.tolist()`, `tensor.numpy()`
- Calling a non-compiled function that dynamo cannot inline
- `torch.jit.trace` / `torch.jit.script` inside compiled region
- Raising exceptions on tensor conditions
- `hasattr`/`getattr` on tensor instances where the result is tensor-valued

**Important corollary:** Any logging or telemetry that touches a tensor value must happen **outside** the compiled forward. Push it to the training loop.

### Rule 8: Static shapes with `dynamic=False`

pr-1493 sets `dynamic=False`. This means: **every batch dim, sequence dim, and intermediate shape must be the same on every call** after the first. Don't pad to variable lengths; pre-pad to max and mask. Don't use `.item()` to derive shapes.

If you need dynamic batch size (e.g. for the last batch of an epoch), recompile per shape or pad.

### Rule 9: No `nn.Module.__init__` allocations inside forward

`nn.Parameter`, `nn.Buffer`, `register_buffer` — all must be created at `__init__` time. A `nn.Parameter(...)` constructed inside `forward` is untracked by the optimizer and triggers dynamo errors.

### Rule 10: Test with `fullgraph=True` from day one

A kernel that compiles under `fullgraph=False` may fail under `fullgraph=True` because graph breaks are now errors. Validate from the strictest setting during development, not at integration time.

---

## Part 2 — Template: compile-safe custom autograd.Function with fused Triton forward

Copy-paste starting point for any new fused kernel. Delete the parts that don't apply.

```python
# my_kernel.py
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.library import triton_op, wrap_triton
import triton
import triton.language as tl


# -----------------------------------------------------------------
# 1. Triton kernel (raw compute — no torch interaction here)
# -----------------------------------------------------------------
@triton.jit
def _my_fused_fwd_kernel(
    X_ptr, W_ptr, OUT_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_blk = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k * BLOCK_K < K), other=0.0)
        w_blk = tl.load(w_ptrs, mask=(offs_k[:, None] + k * BLOCK_K < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(x_blk, w_blk)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Fused epilogue goes here (e.g., activation, scale, residual-add).
    acc_bf16 = acc.to(tl.bfloat16)

    out_ptrs = OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc_bf16, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# -----------------------------------------------------------------
# 2. Register as triton_op so dynamo sees a black-box symbolic op
# -----------------------------------------------------------------
@triton_op("forgefuse::my_fused_fwd", mutates_args=())
def _my_fused_fwd(x: Tensor, w: Tensor) -> Tensor:
    # x: [M, K], w: [K, N] -> out: [M, N]
    assert x.is_cuda and w.is_cuda
    assert x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
    M, K = x.shape
    Kw, N = w.shape
    assert K == Kw

    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    wrap_triton(_my_fused_fwd_kernel)[grid](
        x, w, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


# -----------------------------------------------------------------
# 3. Register fake/meta kernel so dynamo can trace shape/dtype
# -----------------------------------------------------------------
@_my_fused_fwd.register_fake
def _(x, w):
    M, _ = x.shape
    _, N = w.shape
    return torch.empty((M, N), device=x.device, dtype=x.dtype)


# -----------------------------------------------------------------
# 4. autograd.Function wrapper (backward is a separate concern)
# -----------------------------------------------------------------
class MyFusedOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, w: Tensor) -> Tensor:
        # Rule 1: NO x.requires_grad_() or similar mutations.
        # Rule 2: save for backward out-of-place, no in-place ops.
        ctx.save_for_backward(x, w)
        return _my_fused_fwd(x, w)

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tuple[Tensor, Tensor]:
        x, w = ctx.saved_tensors
        # Pure-functional backward. Out-of-place ops only.
        # For simple matmul: grad_x = grad_out @ w.T, grad_w = x.T @ grad_out
        grad_x = grad_out @ w.T
        grad_w = x.T @ grad_out
        return grad_x, grad_w


# -----------------------------------------------------------------
# 5. Public API: module or functional wrapper
# -----------------------------------------------------------------
def my_fused_matmul(x: Tensor, w: Tensor) -> Tensor:
    """Drop-in replacement for x @ w with fused epilogue."""
    return MyFusedOp.apply(x, w)
```

---

## Part 3 — Validation checklist before merging any new kernel

Run these checks in order. **Each must pass before moving on.**

```python
# Test file: test_my_kernel.py
import torch
from my_kernel import my_fused_matmul

def test_correctness_fp32():
    # 1. Numerical correctness vs reference, fp32.
    x = torch.randn(32, 64, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(64, 128, device='cuda', dtype=torch.bfloat16)
    ref = x @ w
    out = my_fused_matmul(x, w)
    assert torch.allclose(out, ref, atol=1e-2, rtol=1e-2)

def test_correctness_grad():
    # 2. Gradient correctness vs reference.
    x = torch.randn(32, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(64, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)

    out = my_fused_matmul(x, w).sum()
    ref = (x_ref @ w_ref).sum()
    out.backward(); ref.backward()

    assert torch.allclose(x.grad, x_ref.grad, atol=1e-2, rtol=1e-2)
    assert torch.allclose(w.grad, w_ref.grad, atol=1e-2, rtol=1e-2)

def test_compile_fullgraph():
    # 3. THE critical test: does it compose with fullgraph compile?
    mod = torch.nn.Linear(64, 128, bias=False).cuda().bfloat16()
    def f(x): return my_fused_matmul(x, mod.weight.T)
    f_compiled = torch.compile(f, dynamic=False, fullgraph=True)

    x = torch.randn(32, 64, device='cuda', dtype=torch.bfloat16)
    # First call triggers the trace. It MUST NOT raise.
    out = f_compiled(x)
    assert out.shape == (32, 128)

def test_compile_gradient():
    # 4. Compile + grad composition (full training step pattern).
    mod = torch.nn.Linear(64, 128, bias=False).cuda().bfloat16()
    def loss_fn(x):
        return my_fused_matmul(x, mod.weight.T).sum()
    f_compiled = torch.compile(loss_fn, dynamic=False, fullgraph=True)

    x = torch.randn(32, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    loss = f_compiled(x)
    loss.backward()
    assert x.grad is not None
    assert mod.weight.grad is not None
```

If step 3 or 4 fails, **do not ship the kernel**. The failure will manifest as a full-training compile bypass, and the kernel's speed advantage will be swamped by the compile-bypass penalty (phase3a saw ~2× step-time regression).

---

## Part 4 — Known bad patterns from prior work

### `_FusedQKNormRoPE.backward` (forgefuse / forgefuse-phase3a)

```python
# BAD — this pattern broke fullgraph on phase3a.
class _FusedQKNormRoPE(torch.autograd.Function):
    @staticmethod
    def backward(ctx, grad_q, grad_k):
        q_in, k_in, cos, sin, ... = ctx.saved_tensors
        # Rewrap saved tensors to get gradients by running forward again:
        q_det = q_in.detach().requires_grad_(True)  # <-- dynamo rejects this
        k_det = k_in.detach().requires_grad_(True)
        with torch.enable_grad():
            out_q, out_k = _fused_qk_norm_rope_forward(q_det, k_det, cos, sin)
        torch.autograd.backward([out_q, out_k], [grad_q, grad_k])
        return q_det.grad, k_det.grad, None, None, ...
```

**Why it failed:** the "re-forward with grad enabled" pattern requires mid-graph `requires_grad_` mutations. Dynamo does not trace this.

**Fix:** implement backward analytically. For QK-norm + RoPE + gain, the math is well-known (RMSNorm backward is standard; RoPE backward is negation of sin term; gain backward is element-wise). Write it out explicitly rather than delegating to autograd.

### Raw Triton launch inside autograd.Function (pre-triton_op pattern)

```python
# BAD — compile either breaks or falls back silently.
class OldStyleFusedMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2):
        out = torch.empty(...)
        _fused_mlp_triton_kernel[grid](x, w1, w2, out, ...)  # raw launch
        ctx.save_for_backward(x, w1, w2)
        return out
```

**Fix:** wrap the Triton launch in a `@triton_op` (see template §2 above). Dynamo will treat the op as a symbolic black box and compile around it instead of trying to trace into it.

---

## Part 5 — When to break the rules

**Never**, unless you have explicit permission and an eager-fallback escape hatch with wallclock budget for the fallback path. The phase3a run's defensive try/except around `torch.compile()` caught the compile object but did NOT catch the lazy tracing error on first forward — so even the "defensive" pattern needs an env-var kill switch (`TORCHDYNAMO_DISABLE=1`) to reliably bypass.

The correct design: make the kernel compile-safe from day one. Budget a half-day of upfront `triton_op` + fake-kernel wiring rather than a full-day debug session at submission time.

---

## References

- `torch.library.triton_op`: [https://docs.pytorch.org/docs/stable/library.html](https://docs.pytorch.org/docs/stable/library.html) (API)
- `wrap_triton`: same page
- `register_fake` / meta kernels: [https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
- Dynamo graph-break catalog: check `torch._dynamo.config.verbose = True` + `TORCH_LOGS=+dynamo` in your dev environment.
- Prior incident: `experiments/h100_run1/SUMMARY.md` on `forgefuse-phase3a` branch (documented the `requires_grad_` failure mode).
