#!/usr/bin/env python3
"""
Tensor rank analysis of neural network weight matrices.

Analyzes the weight tensors of pre-trained models to determine their
multilinear rank structure. Low-rank structure = compressibility,
directly connecting to LoRA and other parameter-efficient methods.

Uses randomly initialized networks of varying architectures as baseline,
then compares against trained weights if available.
"""

from __future__ import annotations

import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

try:
    import tensorly
    HAS_TENSORLY = True
except ImportError:
    HAS_TENSORLY = False


def analyze_weight_tensor(W: np.ndarray, name: str) -> dict:
    """Analyze a weight matrix/tensor."""
    shape = W.shape
    sv = np.linalg.svd(W.reshape(W.shape[0], -1), compute_uv=False)

    matrix_rank = int(np.sum(sv / (sv[0] + 1e-30) > 1e-6))
    full_rank = min(W.shape[0], int(np.prod(W.shape[1:])))

    # Rank for 90%, 95%, 99% variance
    total_var = np.sum(sv**2)
    cumvar = np.cumsum(sv**2) / (total_var + 1e-30)
    rank_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    rank_95 = int(np.searchsorted(cumvar, 0.95)) + 1
    rank_99 = int(np.searchsorted(cumvar, 0.99)) + 1

    # Participation ratio
    sv_pos = sv[sv > 1e-30]
    pr = np.sum(sv_pos)**2 / np.sum(sv_pos**2) if len(sv_pos) > 0 else 0

    # Compression ratio at 95% variance
    original_params = int(np.prod(shape))
    lora_params = rank_95 * (shape[0] + int(np.prod(shape[1:])))
    compression = original_params / (lora_params + 1e-30)

    log.info("%-35s shape=%-20s rank=%d/%d  r90=%d r95=%d r99=%d  PR=%.1f  compress=%.1fx",
             name, str(shape), matrix_rank, full_rank,
             rank_90, rank_95, rank_99, pr, compression)

    return {
        "name": name,
        "shape": shape,
        "matrix_rank": matrix_rank,
        "full_rank": full_rank,
        "rank_90": rank_90,
        "rank_95": rank_95,
        "rank_99": rank_99,
        "participation_ratio": pr,
        "compression_95": compression,
    }


def create_mlp_weights(
    layer_sizes: list[int],
    init: str = "xavier",
    seed: int = 42,
) -> list[tuple[str, np.ndarray]]:
    """Create MLP weight matrices with specified initialization."""
    rng = np.random.RandomState(seed)
    weights = []

    for i in range(len(layer_sizes) - 1):
        fan_in = layer_sizes[i]
        fan_out = layer_sizes[i + 1]

        if init == "xavier":
            std = np.sqrt(2.0 / (fan_in + fan_out))
            W = rng.randn(fan_out, fan_in) * std
        elif init == "kaiming":
            std = np.sqrt(2.0 / fan_in)
            W = rng.randn(fan_out, fan_in) * std
        elif init == "orthogonal":
            A = rng.randn(fan_out, fan_in)
            U, _, Vt = np.linalg.svd(A, full_matrices=False)
            W = U if fan_out <= fan_in else Vt
        elif init == "low_rank":
            # Simulate a trained network with low-rank structure
            rank = min(fan_in, fan_out) // 4
            A = rng.randn(fan_out, rank) * 0.1
            B = rng.randn(rank, fan_in) * 0.1
            W = A @ B
        else:
            W = rng.randn(fan_out, fan_in)

        weights.append((f"layer_{i}_{fan_in}x{fan_out}_{init}", W))

    return weights


def create_transformer_weights(
    d_model: int = 256,
    n_heads: int = 4,
    d_ff: int = 1024,
    seed: int = 42,
) -> list[tuple[str, np.ndarray]]:
    """Create transformer-like weight matrices."""
    rng = np.random.RandomState(seed)
    d_head = d_model // n_heads

    weights = []

    # QKV projections (can be viewed as 3D tensor: heads x d_model x d_head)
    for name in ["Q", "K", "V"]:
        W = rng.randn(n_heads, d_model, d_head) * np.sqrt(2.0 / d_model)
        weights.append((f"attn_{name}_{d_model}x{n_heads}x{d_head}", W))

    # Output projection
    W_o = rng.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
    weights.append((f"attn_out_{d_model}x{d_model}", W_o))

    # FFN
    W1 = rng.randn(d_ff, d_model) * np.sqrt(2.0 / d_model)
    W2 = rng.randn(d_model, d_ff) * np.sqrt(2.0 / d_ff)
    weights.append((f"ffn_up_{d_model}x{d_ff}", W1))
    weights.append((f"ffn_down_{d_ff}x{d_model}", W2))

    return weights


def main():
    log.info("=" * 70)
    log.info("Neural Network Weight Tensor Rank Analysis")
    log.info("=" * 70)

    all_results = []

    # MLP with different initializations
    log.info("\n--- MLP (784 -> 512 -> 256 -> 128 -> 10) ---")
    sizes = [784, 512, 256, 128, 10]
    for init in ["xavier", "kaiming", "orthogonal", "low_rank"]:
        weights = create_mlp_weights(sizes, init=init)
        for name, W in weights:
            r = analyze_weight_tensor(W, name)
            r["init"] = init
            all_results.append(r)

    # Transformer weights
    log.info("\n--- Transformer (d=256, 4 heads, ff=1024) ---")
    weights = create_transformer_weights(256, 4, 1024)
    for name, W in weights:
        r = analyze_weight_tensor(W, name)
        r["init"] = "transformer"
        all_results.append(r)

    # Larger transformer
    log.info("\n--- Transformer (d=768, 12 heads, ff=3072) — BERT-scale ---")
    weights = create_transformer_weights(768, 12, 3072)
    for name, W in weights:
        r = analyze_weight_tensor(W, name)
        r["init"] = "transformer_large"
        all_results.append(r)

    # Summary: compression potential
    log.info("\n" + "=" * 70)
    log.info("SUMMARY: Compression potential by initialization")
    log.info("=" * 70)

    for init in ["xavier", "kaiming", "orthogonal", "low_rank", "transformer", "transformer_large"]:
        subset = [r for r in all_results if r.get("init") == init]
        if subset:
            mean_compress = np.mean([r["compression_95"] for r in subset])
            mean_pr = np.mean([r["participation_ratio"] for r in subset])
            mean_rank_ratio = np.mean([r["rank_95"] / r["full_rank"] for r in subset])
            log.info("%-20s  mean_compress=%.1fx  mean_PR=%.1f  mean_rank95_ratio=%.3f",
                     init, mean_compress, mean_pr, mean_rank_ratio)


if __name__ == "__main__":
    main()
