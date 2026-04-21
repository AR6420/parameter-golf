### Attempted: Fully Fused Attention Preprocessing Kernel

We initially designed a single-kernel fusion of the entire attention preprocessing chain: RMSNorm(x) -> Q/K/V projection -> QK RMSNorm -> RoPE -> q_gain scaling. This would eliminate all HBM round-trips between the input activations and FlashAttention's Q/K/V inputs.

However, RoPE requires splitting each 64-dim head vector into two halves (dims 0:31 and 32:63) for independent cos/sin rotation. Triton's block tensor model does not support arbitrary register-level slicing -- tensors loaded as tiles must be operated on as complete blocks. While offset-based loading of separate half-tiles partially works, integrating this with the tiled QKV projection matmul within the same kernel creates register pressure that exceeds practical limits on current hardware.

We instead fuse the post-projection operations (QK RMSNorm + RoPE + q_gain) into a single kernel, reducing attention preprocessing from 6+ kernel launches to 2. The fully fused QKV preprocessing kernel remains a promising direction -- likely achievable with raw CUDA (which allows arbitrary register indexing) or future Triton versions with richer indexing support.
