"""Compile smoke test — tries torch.compile(fullgraph=True) on each custom
autograd.Function in isolation. Reports per-Function PASS/FAIL and captures
the exact error message for failures.

Run:
    python test_compile_smoke.py
"""
from __future__ import annotations
import sys
import traceback
import torch
import torch.nn.functional as F

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[env] torch {torch.__version__}  device {device}  capability {torch.cuda.get_device_capability(0) if device.type=='cuda' else '-'}")

# Import our Functions. train_gpt.py has lots of module-level side effects
# (autotuned Triton configs, Hyperparameters), so we import narrowly.
import importlib
import train_gpt as tg  # noqa: E402
from w8a8_core import _MLPUp_W8A8, _QKV_W8A8, _OutProj_W8A8, _MLPDown_W8A8  # noqa: E402

_FusedMLPUpProj   = tg._FusedMLPUpProj
_FusedMLPDownProj = tg._FusedMLPDownProj
_FusedQKNormRoPE  = tg._FusedQKNormRoPE


def _mk_param(shape, scale=0.02, dtype=torch.bfloat16):
    t = torch.randn(*shape, device=device, dtype=dtype).mul_(scale)
    return t.requires_grad_(True)


def _mk_input(shape, dtype=torch.bfloat16):
    t = torch.randn(*shape, device=device, dtype=dtype)
    return t.detach().requires_grad_(True)


def _try_compile(label, fn, args):
    """Compile `fn` with fullgraph=True and run one forward + backward.
    Returns (ok, err_str). Clears dynamo cache between runs so each probe
    is independent."""
    try:
        import torch._dynamo as _dynamo
        _dynamo.reset()
    except Exception:
        pass
    try:
        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        out = compiled(*args)
        if isinstance(out, tuple):
            loss = sum(o.float().sum() for o in out if o is not None)
        else:
            loss = out.float().sum()
        loss.backward()
        return True, None
    except Exception as e:
        buf = f"{type(e).__name__}: {e}"
        return False, buf


def probe_mlp_up():
    B, T, K, N = 2, 64, 128, 384
    w = _mk_param((N, K))
    def call(x):
        return _FusedMLPUpProj.apply(x, w)
    x = _mk_input((B, T, K))
    return _try_compile("_FusedMLPUpProj", call, (x,))


def probe_mlp_down():
    B, T, N, D = 2, 64, 384, 128
    w = _mk_param((D, N))
    mlp_scale = torch.randn(D, device=device, dtype=torch.bfloat16).mul_(0.1).add_(1.0).requires_grad_(True)
    def call(hidden, residual):
        return _FusedMLPDownProj.apply(hidden, w, mlp_scale, residual)
    h = _mk_input((B, T, N))
    r = _mk_input((B, T, D))
    return _try_compile("_FusedMLPDownProj", call, (h, r))


def probe_qk_norm_rope_with_gain():
    B, T, H, D = 2, 64, 4, 64
    rope_dims = 16
    half = rope_dims // 2
    cos = torch.randn(T, half, device=device, dtype=torch.bfloat16).detach()
    sin = torch.randn(T, half, device=device, dtype=torch.bfloat16).detach()
    gain = torch.randn(H, device=device, dtype=torch.bfloat16).mul_(0.1).add_(1.0).requires_grad_(True)
    def call(x):
        return _FusedQKNormRoPE.apply(x, cos, sin, gain, rope_dims, True)
    x = _mk_input((B, T, H, D))
    return _try_compile("_FusedQKNormRoPE(has_gain=True)", call, (x,))


def probe_qk_norm_rope_no_gain():
    B, T, H, D = 2, 64, 4, 64
    rope_dims = 16
    half = rope_dims // 2
    cos = torch.randn(T, half, device=device, dtype=torch.bfloat16).detach()
    sin = torch.randn(T, half, device=device, dtype=torch.bfloat16).detach()
    def call(x):
        return _FusedQKNormRoPE.apply(x, cos, sin, x, rope_dims, False)
    x = _mk_input((B, T, H, D))
    return _try_compile("_FusedQKNormRoPE(has_gain=False)", call, (x,))


def probe_w8a8_qkv():
    B, T, K, N = 2, 64, 128, 384
    w = _mk_param((N, K))
    def call(x):
        return _QKV_W8A8.apply(x, w)
    x = _mk_input((B, T, K))
    return _try_compile("_QKV_W8A8", call, (x,))


def probe_w8a8_outproj():
    B, T, K, N = 2, 64, 128, 128
    w = _mk_param((N, K))
    attn_scale = torch.randn(N, device=device, dtype=torch.bfloat16).mul_(0.1).add_(1.0).requires_grad_(True)
    def call(x, residual):
        return _OutProj_W8A8.apply(x, w, attn_scale, residual)
    x = _mk_input((B, T, K))
    r = _mk_input((B, T, N))
    return _try_compile("_OutProj_W8A8", call, (x, r))


def probe_w8a8_mlp_up():
    B, T, K, N = 2, 64, 128, 384
    w = _mk_param((N, K))
    def call(x):
        return _MLPUp_W8A8.apply(x, w)
    x = _mk_input((B, T, K))
    return _try_compile("_MLPUp_W8A8", call, (x,))


def probe_w8a8_mlp_down():
    B, T, N, D = 2, 64, 384, 128
    w = _mk_param((D, N))
    mlp_scale = torch.randn(D, device=device, dtype=torch.bfloat16).mul_(0.1).add_(1.0).requires_grad_(True)
    def call(hidden, residual):
        return _MLPDown_W8A8.apply(hidden, w, mlp_scale, residual)
    h = _mk_input((B, T, N))
    r = _mk_input((B, T, D))
    return _try_compile("_MLPDown_W8A8", call, (h, r))


PROBES = [
    ("_FusedMLPUpProj",                probe_mlp_up),
    ("_FusedMLPDownProj",              probe_mlp_down),
    ("_FusedQKNormRoPE(has_gain=True)",  probe_qk_norm_rope_with_gain),
    ("_FusedQKNormRoPE(has_gain=False)", probe_qk_norm_rope_no_gain),
    ("_QKV_W8A8",                      probe_w8a8_qkv),
    ("_OutProj_W8A8",                  probe_w8a8_outproj),
    ("_MLPUp_W8A8",                    probe_w8a8_mlp_up),
    ("_MLPDown_W8A8",                  probe_w8a8_mlp_down),
]


def main():
    results = []
    print(f"\nRunning {len(PROBES)} compile probes ...\n")
    for name, probe in PROBES:
        try:
            ok, err = probe()
        except Exception as e:
            ok, err = False, f"probe setup failure: {type(e).__name__}: {e}"
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if err:
            # Keep error short in summary; dump full trace at end
            short = err.splitlines()[0][:200]
            print(f"         -> {short}")
        results.append((name, ok, err))
    print()
    npass = sum(1 for _, ok, _ in results if ok)
    print(f"=== Summary: {npass}/{len(results)} PASS ===")
    if npass < len(results):
        print("\n--- Full errors for failed probes ---")
        for name, ok, err in results:
            if not ok:
                print(f"\n[{name}]")
                print(err[:2000])
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
