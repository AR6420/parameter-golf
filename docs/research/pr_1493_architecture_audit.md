# pr-1493 Architecture Audit

**Source:** `pr-1493` branch, commit `857de47` ("Record: SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT — val_bpb 1.0810").
**File audited:** `train_gpt.py` (root), 1126 lines.
**Produced:** pre-implementation reconnaissance for kernel retargeting and harness updates. No code changes.

---

## 0. Critical caveat — "record" vs "starter kit"

The `train_gpt.py` at repo root on pr-1493 is **the starter kit / baseline**, not the SOTA record config. The record implementation lives at:

```
records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py
```

and is **2 lines** of LZMA-compressed, base85-encoded exec (obfuscated competition submission). The name-string of the record implies features the starter kit does NOT include:

| Feature | Starter kit `train_gpt.py` | Record artifact (obfuscated) |
|---|---|---|
| Parallel residuals | ✅ (via `resid_mix`) | ✅ |
| U-Net-style skip connections | ✅ (encoder/decoder halves) | ✅ |
| 3-Layer Recurrence | ❌ | ✅ (name implies weight-sharing across 3 layers) |
| Legal TTT (test-time training) | ❌ | ✅ (name implies) |
| QK-Gain 5.25 | ❌ (default 1.5) | ✅ (per name) |
| SP8192 (context length?) | ❌ (default TRAIN_SEQ_LEN=1024) | ✅ (per name) |

**For our kernel work, target the starter kit.** The record is an opaque blob and competition-specific. A new submission will rebuild these features atop the starter kit if needed.

All findings below describe the starter kit unless stated otherwise.

---

## 1. GPT class signatures

### `GPT.__init__` (line 648–691)

```python
GPT(
    vocab_size: int,
    num_layers: int,
    model_dim: int,
    num_heads: int,
    num_kv_heads: int,
    mlp_mult: int,
    tie_embeddings: bool,
    tied_embed_init_std: float,
    logit_softcap: float,      # must be > 0
    rope_base: float,
    qk_gain_init: float,
)
```

All 11 parameters **required** (no defaults in the signature — defaults live in `Hyperparameters`). Raises `ValueError` if `logit_softcap <= 0`.

### `GPT.forward` (line 700–724)

```python
def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor
```

- **Inputs:** `input_ids` shape `[B, T]`, `target_ids` shape `[B, T]` (both int64).
- **Return:** scalar `Tensor` — `F.cross_entropy(logits.float(), targets, reduction="mean")` over the flattened batch.
- **Not** logits — the forward fuses the cross-entropy loss. This is important: any kernel that wraps `forward` must return a scalar loss, not logits.
- Targets are **shifted-inputs** (next-token prediction), passed in by the caller (not derived internally).

---

## 2. Default Hyperparameters (line 39–87)

Every hyperparameter is an env-var override of a baked default. **Defaults below are the "out-of-the-box" values when no env vars are set.**

### Model shape

| Hyper | Default | Env var | Notes |
|---|---:|---|---|
| `vocab_size` | 1024 | `VOCAB_SIZE` | must match SentencePiece `sp.vocab_size()` or raises |
| `num_layers` | **9** | `NUM_LAYERS` | *not 11* — starter kit is 9L. Prior Submission #2 was 11L. |
| `model_dim` | 512 | `MODEL_DIM` | same as pr-1019 |
| `num_heads` | 8 | `NUM_HEADS` | same |
| `num_kv_heads` | 4 | `NUM_KV_HEADS` | GQA, same as pr-1019 |
| `mlp_mult` | **2** | `MLP_MULT` | *not 3* — hidden = 1024, not 1536 |
| `tie_embeddings` | `True` | `TIE_EMBEDDINGS` | 1/0 env flag |
| `rope_base` | **10000.0** | `ROPE_BASE` | *not 1024.0* — harness hard-codes 1024.0, which is wrong |
| `logit_softcap` | 30.0 | `LOGIT_SOFTCAP` | applied as `softcap * tanh(logits/softcap)` |
| `qk_gain_init` | **1.5** | `QK_GAIN_INIT` | *not 1.0, not 5.25* — record used 5.25 |

Derived:
- `head_dim = model_dim / num_heads = 64`
- `kv_dim = num_kv_heads * head_dim = 256`
- `hidden (MLP) = mlp_mult * model_dim = 1024`

### Training schedule

| Hyper | Default | Env var |
|---|---:|---|
| `iterations` | 20000 | `ITERATIONS` |
| `warmup_steps` | 20 | `WARMUP_STEPS` |
| `warmdown_iters` | 1200 | `WARMDOWN_ITERS` |
| `train_batch_tokens` | 524288 (2^19) | `TRAIN_BATCH_TOKENS` |
| `train_seq_len` | **1024** | `TRAIN_SEQ_LEN` |
| `max_wallclock_seconds` | 600.0 | `MAX_WALLCLOCK_SECONDS` |
| `val_batch_size` | 524288 | `VAL_BATCH_SIZE` |
| `val_loss_every` | 1000 | `VAL_LOSS_EVERY` |
| `train_log_every` | 200 | `TRAIN_LOG_EVERY` |
| `seed` | 1337 | `SEED` |

### Optimizer (line 73–87)

Four param-group split:
| Group | LR default | Env var | Optimizer |
|---|---:|---|---|
| Token embedding (untied) | 0.6 | `EMBED_LR` | Adam (fused) |
| Token embedding (tied) | 0.05 | `TIED_EMBED_LR` | Adam (fused), used when `tie_embeddings=True` |
| LM head (untied only) | 0.008 | `HEAD_LR` | Adam (fused) |
| Matrix params in blocks | 0.04 | `MATRIX_LR` | Muon |
| Vectors/scalars | 0.04 | `SCALAR_LR` | Adam (fused) |

Shared Adam knobs: `beta1=0.9`, `beta2=0.95`, `adam_eps=1e-8`.

Muon knobs:
- `muon_momentum = 0.95` (steady-state)
- `muon_momentum_warmup_start = 0.85`
- `muon_momentum_warmup_steps = 500` (linear ramp 0.85→0.95 over first 500 steps, line 1021–1024)
- `muon_backend_steps = 5` (Newton-Schulz iterations)
- `nesterov = True` (hard-coded in `Muon.__init__`)

Other:
- `tied_embed_init_std = 0.005`
- `grad_clip_norm = 0.0` (**disabled by default**; line 1030 only clips when > 0)

### Warmdown LR schedule (line 924–933)

**Wallclock-adaptive**, not iteration-based. Divides remaining wallclock by estimated `warmdown_ms = warmdown_iters * step_ms`. Returns ratio in `[0, 1]`. When `max_wallclock_seconds <= 0`, falls back to iteration-based.

### EMA

**None.** The starter kit has no EMA — single-model weights throughout. The `forgefuse-phase3a` run log mentioned `post_ema` metrics, but those come from the *record* train_gpt.py, not the starter kit.

---

## 3. Parameter structure

### Embedding

- `tok_emb = nn.Embedding(vocab_size, model_dim)` (line 669).
- **Tied by default**: same weight used for input (`tok_emb(ids)`) and output (`F.linear(x, tok_emb.weight)` at line 718). No separate `lm_head` parameter when tied.
- Untied mode creates `lm_head = CastedLinear(model_dim, vocab_size, bias=False)` with `_zero_init = True`.

### Block layout (line 620–645)

Each `Block` owns:
- `attn_norm: RMSNorm` (before attn)
- `mlp_norm: RMSNorm` (before MLP)
- `attn: CausalSelfAttention`
- `mlp: MLP`
- `attn_scale: Parameter(dim,)` — per-dim gate on attn output (init ones, fp32)
- `mlp_scale: Parameter(dim,)` — per-dim gate on MLP output (init ones, fp32)
- `resid_mix: Parameter(2, dim)` — parallel residual mixer (init `[[1,...], [0,...]]`, fp32)

Block `forward`:
```python
mix = resid_mix.to(x.dtype)
x = mix[0] * x + mix[1] * x0            # parallel residual w/ post-embedding state
attn_out = attn(attn_norm(x))
x = x + attn_scale * attn_out           # per-dim gated add
x = x + mlp_scale * mlp(mlp_norm(x))    # per-dim gated add
return x
```

### GPT-level params beyond blocks (line 669–690)

- `tok_emb.weight` (embedding)
- `skip_weights: Parameter(num_skip, dim)` — U-Net skip connection per-dim weights, init ones, fp32. `num_skip = min(num_encoder_layers, num_decoder_layers) = num_layers // 2` for even-split.
- `final_norm: RMSNorm`
- `lm_head` (only when untied)

### Skip-connection structure (line 704–714)

**U-Net style**: first `num_layers // 2` layers push post-block output onto a stack; remaining layers pop and add with per-dim `skip_weights[i]`. For `num_layers=9`: 4 encoder layers, 5 decoder layers, 4 skip weights (skips[3] → decoder[0], etc.).

### Attention (line 555–603)

Uses **4 separate `CastedLinear` projections**, NOT fused, NOT banked:

| Name | Shape | Zero-init? |
|---|---|---|
| `c_q` | `(dim, dim)` = (512, 512) | No |
| `c_k` | `(dim, kv_dim)` = (512, 256) | No |
| `c_v` | `(dim, kv_dim)` = (512, 256) | No |
| `proj` | `(dim, dim)` = (512, 512) | **Yes** (`_zero_init=True`, line 579) |

Also:
- `q_gain: Parameter(num_heads,)` — per-head multiplicative gain on Q after RMSNorm+RoPE (init `qk_gain_init=1.5`, fp32).
- `rotary: Rotary` — instance per layer (not shared), caches `cos/sin` of `head_dim=64` for current `seq_len`. `inv_freq` has `head_dim // 2 = 32` entries.

Attention `forward`:
```python
q = c_q(x).reshape(B, T, H, Hd).transpose(1, 2)     # [B, H, T, Hd]
k = c_k(x).reshape(B, T, Hkv, Hd).transpose(1, 2)
v = c_v(x).reshape(B, T, Hkv, Hd).transpose(1, 2)
q = F.rms_norm(q, (Hd,))                            # QK-norm
k = F.rms_norm(k, (Hd,))
q = apply_rotary_emb(q, cos, sin)                   # FULL RoPE (not partial)
k = apply_rotary_emb(k, cos, sin)
q = q * q_gain[None, :, None, None]                 # per-head gain
y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(Hkv != H))
y = y.transpose(1, 2).contiguous().reshape(B, T, dim)
return proj(y)
```

**No FA3.** No `flash_attn_interface`. Uses `F.scaled_dot_product_attention` with backend controlled by `torch.backends.cuda.enable_flash_sdp(True)` (line 767). Math/cuDNN/mem-efficient SDP are **disabled** — only Flash SDP backend is allowed.

### RoPE (line 524–552)

- **Full RoPE** over all `head_dim=64` dims, *not* partial. The `apply_rotary_emb` splits `x` into halves `x1, x2` (dim 32 each) and rotates; this is the "non-interleaved half-rotation" layout.
- `rope_base=10000.0` default.
- `inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))` — stride-2 arange over `head_dim`.
- `cos/sin` cached per-instance, keyed by `(seq_len, device)`. Invalidated when any changes.

### MLP (line 606–617)

- `fc: CastedLinear(dim, hidden)` — shape `(512, 1024)` at default.
- `proj: CastedLinear(hidden, dim)` — shape `(1024, 512)`, `_zero_init=True`.
- **Activation:** `torch.relu(x).square()` (relu² / squared ReLU). NOT `leaky_relu(0.5).pow(2)` from pr-1019.

MLP `forward`:
```python
x = torch.relu(fc(x))
return proj(x.square())
```

Note: `relu` is applied FIRST, then `.square()` inside `proj(...)`. So intermediate tensor goes through `fc → relu → square → proj` without materializing the squared tensor between square and proj.

### CastedLinear (line 509–513)

Keeps `weight` in **fp32** for optimizer state quality, casts to input dtype at `F.linear` time:

```python
class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)
```

**All matrix parameters in blocks + lm_head are `CastedLinear` (fp32 storage, bf16 compute).** This is a key detail for kernel work: the weight `.data` is fp32 on-disk; the cast happens inside `forward`. Any Triton kernel fusing the matmul must match this cast timing.

### RMSNorm (line 500–506)

```python
class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        ...
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)
```

- No learnable `weight` — pure normalization. pr-1019 had per-layer learnable RMSNorm weights; pr-1493 does NOT.
- `eps=None` uses PyTorch's default.
- **Per-block count:** 2 (attn_norm, mlp_norm). Plus `final_norm`, plus 2 inline `F.rms_norm` calls on Q and K inside attention, plus 1 inline `F.rms_norm(x)` post-embedding at line 702. **Total per forward: 2 × num_layers + 3 + 2 × num_layers = 4L + 3 RMSNorm calls.** At 9L: 39 calls; at 11L: 47 calls.

### Parameter dtype mapping (line 826–842)

Model init sequence:
1. Build `base_model = GPT(...).to(device).bfloat16()` — everything becomes bf16.
2. For each `CastedLinear`, call `.float()` — restores its `weight` to fp32.
3. `restore_low_dim_params_to_fp32(base_model)` (line 516–521): any parameter with `ndim < 2` OR matching `CONTROL_TENSOR_NAME_PATTERNS` is promoted back to fp32.

Result at training time:
- **bf16**: Embedding weight (`tok_emb.weight`)
- **fp32**: all `CastedLinear.weight` (Q, K, V, proj, fc, proj, lm_head if untied)
- **fp32**: `q_gain`, `attn_scale`, `mlp_scale`, `resid_mix`, `skip_weights` (control/low-dim params)
- **fp32**: Rotary buffers (`inv_freq`, cached `cos/sin` — buffers, not params)

---

## 4. Training loop

### Env-var knobs read in `main()` (line 731–1123)

Beyond `Hyperparameters`:
- `RANK`, `WORLD_SIZE`, `LOCAL_RANK` — distributed setup; absent → single-GPU, world_size=1.
- `CONTROL_TENSOR_NAME_PATTERNS` — comma-sep list of name substrings treated as "control" (line 288, default: `"attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights"`).
- `INT8_KEEP_FLOAT_FP32_NAME_PATTERNS` — same patterns by default, kept as fp32 in the int8 roundtrip.
- `DATA_PATH`, `TOKENIZER_PATH`, `RUN_ID` — environment paths.

### Critical distributed-setup constraint (line 748–751)

```python
if 8 % world_size != 0:
    raise ValueError(...)
grad_accum_steps = 8 // world_size
```

**WORLD_SIZE must divide 8.** `grad_accum_steps = 8 // world_size` is integer-exact. Single-GPU → `grad_accum=8`. 8-GPU → `grad_accum=1`.

### torch.compile setup (CRITICAL for kernel work)

**Two compile calls, both unconditional:**

| Line | Call | Notes |
|---|---|---|
| 736 | `zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)` | Muon's Newton-Schulz orthogonalization is compiled; default compile mode (no `fullgraph`). |
| 843 | `compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)` | **fullgraph=True, dynamic=False**. No HAS_FLASH_ATTN_3 guards. No try/except fallback. Fails hard if the graph can't be traced. |

**No custom compile-compatibility workarounds** in the starter kit (no custom ops, no `@torch.library.custom_op`, no `allow_in_graph`). This means:
- Any Triton kernel we inject must compose with `torch.compile(fullgraph=True)`. The phase3a issue where `x.detach().requires_grad_(True)` broke fullgraph tracing would recur in identical form.
- SDPA in `F.scaled_dot_product_attention` is Inductor-supported natively; no issue.
- `F.rms_norm` has native Inductor support.
- `F.cross_entropy`, `F.linear`, `torch.relu`, `.square()` — all compile-clean.

The forward path as written is **compile-clean with no workarounds required** — this is free infrastructure we inherit.

### SDP backend (line 763–769)

```python
enable_cudnn_sdp(False)
enable_flash_sdp(True)
enable_mem_efficient_sdp(False)
enable_math_sdp(False)
```

**Only Flash SDP enabled**; all other backends disabled. On H100 this routes through Flash v2 (via PyTorch's native Flash impl, not the `flash_attn_interface` C extension). On compute capability without Flash support (e.g. local 5070 Ti Blackwell without Flash build), `F.scaled_dot_product_attention` will raise — a known failure mode. The harness' SDP-math-backend mock is a workaround for that.

### Training step (line 1007–1036)

```python
scale = lr_mul(step, elapsed_ms)     # wallclock-adaptive warmdown
zero_grad_all()
train_loss = 0
for micro_step in range(grad_accum_steps):
    x, y = train_loader.next_batch(...)
    with autocast("cuda", bf16):
        loss = model(x, y)
    train_loss += loss.detach()
    (loss * grad_scale).backward()

# Muon momentum warmup (linear 0.85→0.95 over first 500 steps)
muon_momentum = lerp(start, steady, min(step/500, 1))

# LR scale (from warmdown)
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["base_lr"] * scale

if grad_clip_norm > 0:
    torch.nn.utils.clip_grad_norm_(base_model.parameters(), grad_clip_norm)
for opt in optimizers:
    opt.step()
zero_grad_all()
```

**Order:** forward+backward grad-accumulated → update Muon momentum → scale LRs → clip (if enabled) → step all optimizers → zero grads.

### Warmup protocol (line 937–961)

Primes the compiled graph by running `warmup_steps=20` full forward+backward+optimizer iterations, then **restores** pre-warmup weights and optimizer state. This ensures measured training starts from the true init, while the compiled kernels are already cached. The data loader is also re-initialized to reset the position.

### Autocast

bf16 autocast on `"cuda"` device, enabled for both forward and `(loss * grad_scale).backward()`. Validation also uses bf16 autocast (line 258).

### Post-training (line 1062–1119)

1. Save raw `final_model.pt` (fp32+bf16 state dict).
2. Quantize state dict to int8 (per-row for 2D matrices, per-tensor for scalars; `INT8_CLIP_Q = 99.99984` percentile).
3. Zlib-compress the quantized state dict → `final_model.int8.ptz`.
4. Round-trip decompress + dequant → reload into model.
5. Validate round-tripped weights: log `final_int8_zlib_roundtrip val_bpb`.

**Note:** pr-1493 uses int8 + zlib. `forgefuse-phase3a` used int6 + lzma — that's a record-level modification, not starter-kit behavior.

---

## 5. Data pipeline

### Paths

- Default `DATA_PATH = "./data/datasets/fineweb10B_sp1024"` (line 41)
  - train: `fineweb_train_*.bin`
  - val: `fineweb_val_*.bin`
- Default `TOKENIZER_PATH = "./data/tokenizers/fineweb_1024_bpe.model"` (line 44)

### Shard format (line 429–443)

**CRITICAL:** The shard header is **256 int32** (= 1024 bytes), NOT 256 uint16 (= 512 bytes).

```python
def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * 4                         # 1024 bytes
    token_bytes = 2
    header = np.fromfile(file, dtype="<i4", count=256)
    # Magic: header[0] == 20240520, version: header[1] == 1
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))
```

**Header validation**: `header[0] == 20240520` (magic), `header[1] == 1` (version), `header[2] == num_tokens`, and file size must equal `1024 + num_tokens * 2` bytes.

### Vocabulary constraint (line 808–811)

```python
if int(sp.vocab_size()) != args.vocab_size:
    raise ValueError(...)
```

The SentencePiece model's vocab size MUST equal `VOCAB_SIZE` env var (default 1024). Can't swap tokenizers without updating `VOCAB_SIZE`.

### Token stream (line 446–474)

`TokenStream` reads shards **sequentially** by sorted filename, wrapping forever. No sampling, no shuffling, no multi-worker loaders. `DistributedTokenLoader` slices the stream into per-rank disjoint spans per step.

### Validation (line 207–278)

`load_validation_tokens` concatenates ALL `fineweb_val_*.bin` shards into one long tensor, trims to `((numel - 1) // seq_len) * seq_len + 1` tokens. Val loop runs over disjoint sequence-aligned chunks per rank.

BPB computed as: `(val_loss_in_nats / ln 2) * (tokens_per_byte)` where `tokens_per_byte` comes from SentencePiece piece-length LUTs (accounting for `▁` leading-space markers).

---

## 6. Kernel injection points — where kernels would hook

For future kernel work against the starter kit:

### 6.1 MLP forward (line 606–617)

**Location:** `MLP` class, class-level `forward` method on an `nn.Module`.

**Current code path:**
```python
x = torch.relu(self.fc(x))            # fc: CastedLinear(512, 1024), bf16 compute
return self.proj(x.square())          # proj: CastedLinear(1024, 512), bf16 compute
```

**Fusion opportunities:**
- Pre-norm `mlp_norm(x)` (called in `Block.forward`, line 644) → `fc` matmul → `relu` → `.square()` → `proj` matmul → `* mlp_scale` (line 644) → residual add.
- Per-dim `mlp_scale` is applied post-MLP in the block, not inside MLP. Bringing it into MLP kernel is possible.
- Unlike pr-1019's `leaky_relu(0.5).pow(2)`, the activation is `relu(x).square()` (cheaper — no negative slope).
- Intermediate dim: 1024 at 9L default, 1536 if we push `mlp_mult=3`. Both are small enough to consider full megakernel fusion (pr-1019's tile-H=1536 approach scales down naturally).

**Watch-out:** `CastedLinear` stores weight in fp32, casts at matmul. A Triton kernel must load weight as fp32 and cast inside, OR the kernel expects pre-cast bf16 weights (would need a wrapper that maintains the cast).

### 6.2 Attention Q/K/V projections (line 555–603)

**Location:** `CausalSelfAttention` class, 4 separate `CastedLinear` instances.

**Current code path:**
- `c_q: Linear(512, 512)`, `c_k: Linear(512, 256)`, `c_v: Linear(512, 256)` — three separate matmuls from same input `x`.
- `proj: Linear(512, 512)` — separate from the three.

**Fusion opportunities:**
- Fuse Q/K/V into a single `Linear(512, 512+256+256) = Linear(512, 1024)` + slice. This is a classic QKV-fusion and directly analogous to the `qkv_bank` refactor in Submission #2, but without the banking (just a normal concat'd weight). **Would cut 2 kernel launches per block.**
- Q-side has `rms_norm → rope → * q_gain` chain before SDPA. The `q_gain` is a per-head broadcast (shape `[1, H, 1, 1]` after broadcast-style indexing) — fusable into the rope kernel as a final scale.
- K-side has `rms_norm → rope` only.

**Watch-out:** SDPA at line 594 is the hard barrier — `F.scaled_dot_product_attention` is a black box we can't fuse through. Pre-SDPA fusion (QKV projection + QK-norm + RoPE + q_gain) is viable; post-SDPA (output projection) is a separate kernel.

### 6.3 Output projection + residual (line 603 + Block.forward line 643)

**Location:** `CausalSelfAttention.proj(y)` returns → `Block.forward` applies `attn_scale * attn_out` and residual-adds.

**Current code path:**
```python
return self.proj(y)                                          # in attn.forward
...
attn_out = self.attn(...)                                    # in block.forward
x = x + self.attn_scale.to(x.dtype)[None, None, :] * attn_out
```

**Fusion opportunities:**
- `proj` matmul epilogue: apply `attn_scale` and residual-add inline. Similar to Sub #2's MLP output fusion pattern.

### 6.4 RMSNorm call sites

**Starter kit uses `F.rms_norm` (no learnable weight) everywhere:**
1. `RMSNorm.forward` (line 506): used by `attn_norm`, `mlp_norm`, `final_norm` — 2L+1 calls per forward.
2. Inline in attention (line 588–589): `q = F.rms_norm(q, ...)`, `k = F.rms_norm(k, ...)` — 2L calls.
3. Inline post-embedding (line 702): `x = F.rms_norm(x, (x.size(-1),))` — 1 call.

Total: **4L + 2 calls per forward**. At 9L: 38 calls. Each is independently compile-fusable by Inductor (typically into the next consumer's prologue). No learnable gamma means no extra parameter to thread through.

### 6.5 Existing custom autograd / Triton kernels

**None.** The starter kit uses only:
- Native PyTorch ops (`F.linear`, `F.rms_norm`, `F.cross_entropy`, `F.scaled_dot_product_attention`)
- Inline ops (`torch.relu`, `.square()`, element-wise `*` and `+`)
- `torch.compile` (relies on Inductor)

**No `torch.autograd.Function` subclasses.** **No custom ops via `torch.library`.** **No `@torch.compile` decorator on sub-modules.** **No Triton imports.**

This means: when we inject kernels, we have no pre-existing `@torch.compile` contract to respect or match. We're writing on a blank slate w.r.t. custom-op infrastructure.

### 6.6 Muon compile wrapper

**Worth noting:** `zeropower_via_newtonschulz5` is compiled at line 736 (default mode, not fullgraph). If we ever swap the orthogonalizer, or if our kernels need access to Muon's update buffer layout, this compile wrapper is the one existing compile hook in the code.

---

## 7. Summary table: pr-1019 (our old target) → pr-1493 (new target)

| Attribute | pr-1019 | pr-1493 starter kit |
|---|---|---|
| File size | ~2135 lines | 1126 lines |
| Layers default | 11 | **9** |
| MLP mult | 3× | **2×** |
| MLP activation | leaky_relu(0.5)² | **relu²** |
| Parameter banking | `qo_bank`, `kv_bank`, `qkv_bank`, `mlp_up_bank`, `mlp_down_bank` (3D tensors, hand-indexed) | **None** — plain `CastedLinear` per layer |
| RMSNorm learnable weight | Yes | **No** (pure F.rms_norm) |
| Attention backend | FA3 via `flash_attn_interface` | `F.scaled_dot_product_attention` with flash SDP backend |
| RoPE | Partial (via `rope_dims`) | **Full** (over entire head_dim=64) |
| QK-norm | Yes | Yes |
| QK-gain | Yes (init 1.0) | Yes (init **1.5**, record used 5.25) |
| Parallel residuals | No | **Yes** (`resid_mix` per block) |
| U-Net skip connections | No | **Yes** (encoder/decoder halves) |
| Per-block per-dim attn/mlp gates | No | **Yes** (`attn_scale`, `mlp_scale`) |
| `torch.compile` mode | fullgraph + HAS_FLASH_ATTN_3 guards | **fullgraph unconditionally**, no guards, no fallback |
| Custom autograd.Function | Yes (fused kernels) | **None** |
| Post-training quant | int6 + lzma (record-level) | int8 + zlib (starter kit default) |
| EMA | Yes (record-level) | **No** in starter kit |
| Optimizer | Muon + Adam | Same structure, 4 param groups |
| `grad_clip_norm` default | nonzero | **0.0 (disabled)** |

---

## 8. Harness retargeting notes (for next phase, not done yet)

From this audit, the **concrete changes** `experiments/harness.py` needs to become pr-1493-compatible:

| harness.py line | Issue | Fix |
|---|---|---|
| 44 `rope_base=1024.0` | Wrong value — pr-1493 default is 10000.0 | Change to `10000.0` or add CLI override |
| 44 `mlp_mult=3`; 118 `--mlp-mult` default `3` | pr-1493 default is 2 | Change to 2 |
| 44 `qk_gain_init=1.0` | pr-1493 default is 1.5; record used 5.25 | Change to 1.5 (or 5.25 for record-matching) |
| 44 `rope_dims=16` | Not a pr-1493 `GPT.__init__` arg | `inspect.signature` filter silently drops; cosmetically stale, remove |
| 55–57 `load_shard` skips `data[256:]` (uint16 offset = 512 bytes) | pr-1493 shard header is 256 int32 = **1024 bytes** | Fix to skip 512 uint16 values, or use `load_data_shard`-style int32 header read |
| 78–81 bank-cast loop | pr-1493 has no banks | Remove (dead code) |
| 27–37 `flash_attn_interface` mock | pr-1493 doesn't import FA3 | Remove (dead code) |
| 117 `--num-layers` default `11` | pr-1493 default is 9 | Change to 9 (or leave as 11 if harness is intentionally testing a bigger config) |

**Separate scientific question (not a harness bug):** the harness runs with `B=8, T=1024, BT=8192` across all branches for fair comparison. pr-1493 was trained at `T=1024` natively (matching!), so this is fine — the "SP8192" in the record name refers to some other feature, not training sequence length.

---

## 9. Canonical references

- **Starter kit:** `train_gpt.py` @ pr-1493 (SHA 857de47).
- **Record artifact (obfuscated):** `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py` — not audited; out of scope for kernel work.
- **Related branch kept for reference:** `forgefuse-salvage` (has `train_gpt_sota_base.py` = pr-1019 baseline copy if needed for diff).

Kernel work from prior phases (on `forgefuse-salvage`, not to be ported directly):
- `w8a16_core.py` — INT8 weight-quant Triton primitive (reusable concept, non-reusable wiring).
- `w8a8_core.py` — INT8 W+A Triton primitive (same — reusable concept).
- `fused_mlp_kernels.py` — built around 3× MLP + leaky_relu² + bank args. Reference only; rewrite for pr-1493's 2× + relu² + CastedLinear.
- `fused_attn_kernel.py` — built around QKV-banked layout. Reference only; rewrite for pr-1493's 4-linear layout.

Kernel work from Submission #1 (on `megakernel-fusion-stash`):
- `sub1_kernel_iterations/triton_megakernel.py` — tile-H megakernel design. Conceptual reference for MLP megakernel at 1024-hidden.
