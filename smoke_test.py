"""
Smoke test: Compare fused Triton MLP kernels vs unfused baseline.
Runs 10 training steps in both modes and checks:
  1. Loss matches within 0.01 at each step
  2. Grad norms don't diverge
  3. MLP output cos_sim > 0.9999 (via forward hook on step 1)
"""
import os, sys, copy, math, types
import torch
import torch.nn.functional as F

# Mock flash_attn_interface BEFORE importing train_gpt (module-level import)
mock_mod = types.ModuleType('flash_attn_interface')
def _sdpa_fallback(q, k, v, causal=False, **kwargs):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    H, Hkv = q.shape[1], k.shape[1]
    if H != Hkv:
        rep = H // Hkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.transpose(1, 2)
mock_mod.flash_attn_func = _sdpa_fallback
sys.modules['flash_attn_interface'] = mock_mod

import train_gpt

device = 'cuda'
torch.manual_seed(1337)

print("\n=== MLP FUSION SMOKE TEST (forward + backward, 10 steps) ===\n")
if True:  # Synthetic smoke test: validates MLP forward/backward in isolation

    model_dim = 512
    mlp_dim = 1536
    B, T = 2, 256

    # Create a standalone Block-like test
    mlp = train_gpt.MLP(model_dim, 3).to(device)
    mlp_norm = train_gpt.RMSNorm().to(device)
    mlp_scale = torch.nn.Parameter(torch.ones(model_dim, device=device, dtype=torch.float32))
    up_w = torch.randn(mlp_dim, model_dim, device=device, dtype=torch.bfloat16, requires_grad=False)
    down_w = torch.randn(model_dim, mlp_dim, device=device, dtype=torch.bfloat16, requires_grad=False)

    print("Step | Loss_fused | Loss_unfused | Loss_diff | GradNorm_f | GradNorm_u | MLP_cos_sim")
    print("-" * 95)

    for step in range(10):
        torch.manual_seed(42 + step)
        x = torch.randn(B, T, model_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
        target = torch.randn(B, T, model_dim, device=device, dtype=torch.bfloat16)

        # ── Fused path ──
        train_gpt._USE_TRITON_MLP = True
        x_fused = x.detach().clone().requires_grad_(True)
        x_norm_f = mlp_norm(x_fused) * (1.0 / math.sqrt(6))
        mlp_scale_f = mlp_scale.to(dtype=x_fused.dtype)
        out_fused = mlp.forward(x_norm_f, up_w, down_w, mlp_scale=mlp_scale_f, residual=x_fused)
        loss_fused = F.mse_loss(out_fused, target)
        loss_fused.backward()
        grad_norm_f = x_fused.grad.norm().item()

        # ── Unfused path ──
        train_gpt._USE_TRITON_MLP = False
        x_unfused = x.detach().clone().requires_grad_(True)
        x_norm_u = mlp_norm(x_unfused) * (1.0 / math.sqrt(6))
        mlp_scale_u = mlp_scale.to(dtype=x_unfused.dtype)
        out_unfused = mlp.forward(x_norm_u, up_w, down_w, mlp_scale=mlp_scale_u, residual=x_unfused)
        loss_unfused = F.mse_loss(out_unfused, target)
        loss_unfused.backward()
        grad_norm_u = x_unfused.grad.norm().item()

        # ── Compare ──
        loss_diff = abs(loss_fused.item() - loss_unfused.item())
        mlp_cos = F.cosine_similarity(
            out_fused.detach().flatten().float(),
            out_unfused.detach().flatten().float(), dim=0
        ).item()

        print(f"  {step:2d} | {loss_fused.item():10.6f} | {loss_unfused.item():12.6f} | "
              f"{loss_diff:9.6f} | {grad_norm_f:10.4f} | {grad_norm_u:10.4f} | {mlp_cos:.6f}")

        # ── Checks ──
        if loss_diff > 0.01:
            print(f"  ** FAIL: loss diff {loss_diff:.6f} > 0.01")
            sys.exit(1)
        if mlp_cos < 0.9999:
            print(f"  ** FAIL: MLP cos_sim {mlp_cos:.6f} < 0.9999")
            sys.exit(1)
        if grad_norm_f > 10 * grad_norm_u or grad_norm_u > 10 * grad_norm_f:
            print(f"  ** FAIL: grad norms diverged (fused={grad_norm_f:.4f}, unfused={grad_norm_u:.4f})")
            sys.exit(1)

    print("\n=== ALL 10 STEPS PASSED ===")
    print("Loss matches within 0.01, cos_sim > 0.9999, grad norms stable.")

    # Restore fused mode
    train_gpt._USE_TRITON_MLP = True
