#!/usr/bin/env python3
"""Wide-beam depth 3-4 search with TurboQuant compression.

Usage: python run_wide_d3.py <DatasetName> <gpu_id>

Uses 3-bit beam to enable 500-wide search at depth 3, 300 at depth 4.
NO deeper — avoids overfitting seen in depth 7-9 experiments.
"""
import os
import sys
import json
import logging
import time

DATASET = sys.argv[1]
GPU = sys.argv[2] if len(sys.argv) > 2 else "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

with open("run_full_pipeline.py") as f:
    code = f.read().split("def main")[0]
code = code.replace('os.environ["CUDA_VISIBLE_DEVICES"] = "0"',
                     '# CUDA device set by caller')
exec(code)  # noqa: S102

import cupy as cp
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from dataset_loader import load_dataset
from cuml.ensemble import RandomForestClassifier as cuRF
from turbo_beam import TurboBeam

GSB = {
    "+": lambda a, b: a + b, "-": lambda a, b: a - b,
    "*": lambda a, b: a * b, "/": lambda a, b: a / (b + 1e-30),
    "max": lambda a, b: cp.maximum(a, b), "min": lambda a, b: cp.minimum(a, b),
    "hypot": lambda a, b: cp.sqrt(a**2 + b**2),
    "diff_sq": lambda a, b: (a - b) ** 2,
    "harmonic": lambda a, b: 2 * a * b / (a + b + 1e-30),
    "geometric": lambda a, b: cp.sign(a * b) * cp.sqrt(cp.abs(a * b)),
}
GSU = {
    "log": lambda x: cp.log(cp.abs(x) + 1e-30),
    "sqrt": lambda x: cp.sqrt(cp.abs(x)),
    "sq": lambda x: x**2, "abs": lambda x: cp.abs(x),
    "sigmoid": lambda x: 1.0 / (1.0 + cp.exp(-cp.clip(x, -500, 500))),
    "tanh": lambda x: cp.tanh(cp.clip(x, -500, 500)),
}


def gpu_trace_eval(trace, X_gpu):
    v = None
    for o in trace.ops:
        if o[0] == "l":
            v = X_gpu[:, o[1]].copy()
        elif o[0] == "b":
            v = GSB[o[1]](v, X_gpu[:, o[2]])
            v = cp.nan_to_num(v, nan=0.0, posinf=1e10, neginf=-1e10)
        elif o[0] == "u":
            v = GSU[o[1]](v)
            v = cp.nan_to_num(v, nan=0.0, posinf=1e10, neginf=-1e10)
    return v


def gpu_opt_thresh(vals_gpu, y_gpu):
    P = int(y_gpu.sum())
    N = vals_gpu.shape[0]
    if P == 0 or P == N:
        return 0.0, 1, 0.0
    best_f1, best_thresh, best_dir = 0.0, 0.0, 1
    for d in [1, -1]:
        sv = d * vals_gpu
        idx = cp.argsort(-sv)
        sorted_y = y_gpu[idx].astype(cp.bool_)
        tp = cp.cumsum(sorted_y, dtype=cp.float64)
        fp = cp.cumsum(~sorted_y, dtype=cp.float64)
        f1 = 2.0 * tp / (tp + P + fp + 1e-30)
        bi = int(cp.argmax(f1))
        if float(f1[bi]) > best_f1:
            best_f1 = float(f1[bi])
            best_thresh = float(sv[idx[bi]])
            best_dir = d
    return best_thresh, best_dir, best_f1


def gpu_f1(y_true, y_pred):
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def gpu_pls(X_gpu, y_gpu, n_components=8):
    N, d = X_gpu.shape
    k = min(n_components, d, N - 1)
    y = y_gpu.reshape(-1, 1).astype(cp.float64)
    X_mean = X_gpu.mean(axis=0)
    Xc = X_gpu - X_mean
    yc = y - y.mean()
    W = cp.zeros((d, k), dtype=cp.float64)
    for i in range(k):
        w = Xc.T @ yc
        w = w / (cp.linalg.norm(w) + 1e-30)
        t = Xc @ w
        tt = float(cp.asnumpy(t.T @ t).item())
        if tt < 1e-30:
            break
        p = (Xc.T @ t) / tt
        q = (yc.T @ t) / tt
        Xc = Xc - t @ p.T
        yc = yc - t * q
        W[:, i:i+1] = w
    return W, X_mean


def gpu_tucker(X_gpu, y_gpu, n_components=4):
    """Top-k discriminative pairwise interaction features on GPU."""
    N, d = X_gpu.shape
    yf = y_gpu.astype(cp.float64)
    k = min(n_components, d * (d - 1) // 2)
    pairs = []
    pair_idx = []
    for i in range(min(d, 20)):
        for j in range(i + 1, min(d, 20)):
            prod = X_gpu[:, i] * X_gpu[:, j]
            prod = cp.nan_to_num(prod, nan=0.0, posinf=1e10, neginf=-1e10)
            pairs.append(prod)
            pair_idx.append((i, j))
    if not pairs:
        return cp.empty((N, 0), dtype=cp.float64), []
    pair_matrix = cp.stack(pairs, axis=1)
    f1s = gpu_batch_f1(pair_matrix, yf)
    top = cp.argsort(-f1s)[:k].get()
    return pair_matrix[:, top], [f"tk{ii}({pair_idx[idx][0]}x{pair_idx[idx][1]})" for ii, idx in enumerate(top)]


def wide_beam_search(X_gpu, y_gpu, max_depth=4, bits=3):
    """Wide-beam shallow search. 500 candidates at d2, 300 at d3, 150 at d4."""
    N, d = X_gpu.shape
    yf = y_gpu.astype(cp.float64)

    # Scale beam to fit memory: B * N * d * 8 * 10 < 4 GB
    max_beam = int(4e9 / (N * d * 8 * 10))
    max_beam = max(100, min(500, max_beam))

    beam = TurboBeam(N, bits=bits)
    f1s = gpu_batch_f1(X_gpu, yf)
    for fi in range(d):
        beam.add(float(f1s[fi]), FT(fi), X_gpu[:, fi])
    beam.top_k(min(max_beam, d))

    schedule = {2: max_beam, 3: max(60, max_beam * 3 // 5), 4: max(30, max_beam * 3 // 10)}

    for depth in range(2, max_depth + 1):
        bw = schedule.get(depth, 30)
        beam.top_k(bw)
        B = len(beam)
        if B == 0:
            break

        beam_vals = cp.stack([beam.get_values(i) for i in range(B)])
        bv = beam_vals[:, :, None]
        xf = X_gpu[None, :, :]
        new_entries = []

        for opn, opf in VBIN:
            try:
                cands = opf(bv, xf)
                cands = cp.nan_to_num(cands, nan=0.0, posinf=1e10, neginf=-1e10)
                flat = cands.reshape(B * d, N)
                CHUNK = 8000
                parts = []
                for ci in range(0, flat.shape[0], CHUNK):
                    parts.append(gpu_batch_f1(flat[ci:ci + CHUNK].T, yf))
                f1_all = cp.concatenate(parts)
                for bi in range(B):
                    for fi in range(d):
                        k_idx = bi * d + fi
                        new_entries.append((float(f1_all[k_idx]), beam[bi][1].b(opn, fi), flat[k_idx]))
                del cands, flat, f1_all
            except cp.cuda.memory.OutOfMemoryError:
                cp.get_default_memory_pool().free_all_blocks()

        for opn, opf in VUN:
            try:
                cands = opf(beam_vals)
                cands = cp.nan_to_num(cands, nan=0.0, posinf=1e10, neginf=-1e10)
                f1u = gpu_batch_f1(cands.T, yf)
                for bi in range(B):
                    new_entries.append((float(f1u[bi]), beam[bi][1].u(opn), cands[bi]))
            except Exception:
                pass

        del bv, xf, beam_vals

        new_beam = TurboBeam(N, bits=bits)
        for i in range(len(beam)):
            f1_v, trace, qidx, norm = beam[i]
            new_beam.entries.append((f1_v, trace, qidx, norm))
        for f1_v, trace, vals in new_entries:
            new_beam.add(f1_v, trace, vals)
        new_beam.top_k(bw)
        beam = new_beam

        log.info("  d%d: best=%.4f K=%d beam=%d (%.0fx savings)",
                 depth, beam[0][0], len(new_entries), len(beam),
                 beam.memory_savings())

    if len(beam) > 0:
        return beam[0][0], beam[0][1]
    return 0.0, None


# ---- Main ----
X, y, feats = load_dataset(DATASET)
N, d = X.shape
# Depth 4 only if N/d > 10, else depth 3
max_d = 4 if N / d > 10 else 3
log.info("=== %s: N=%d d=%d N/d=%.0f depth=%d GPU=%s ===",
         DATASET, N, d, N / d, max_d, GPU)
log.info("  Wide TurboBeam: 500-wide d2, 300-wide d3, 150-wide d4")

X_gpu = cp.asarray(X, dtype=cp.float64)
y_gpu = cp.asarray(y, dtype=cp.float64)

results_deep = []
results_rf = []
t0 = time.time()

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)
for fold_i, (tr, te) in enumerate(cv.split(X, y)):
    X_tr, y_tr = X_gpu[tr], y_gpu[tr]
    X_te, y_te = X_gpu[te], y_gpu[te]

    # PLS + Tucker augmentation
    k = min(8, d - 1, len(tr) - 1)
    W, X_mean = gpu_pls(X_tr, y_tr, n_components=k)
    pls_tr = (X_tr - X_mean) @ W
    pls_te = (X_te - X_mean) @ W

    tucker_tr, tucker_names = gpu_tucker(X_tr, y_tr, n_components=4)
    if tucker_tr.shape[1] > 0:
        tucker_te_parts = []
        for tn in tucker_names:
            inner = tn.split("(")[1].rstrip(")")
            i, j = int(inner.split("x")[0]), int(inner.split("x")[1])
            tucker_te_parts.append(X_te[:, i] * X_te[:, j])
        tucker_te = cp.stack(tucker_te_parts, axis=1)
        X_aug_tr = cp.hstack([X_tr, pls_tr, tucker_tr])
        X_aug_te = cp.hstack([X_te, pls_te, tucker_te])
    else:
        X_aug_tr = cp.hstack([X_tr, pls_tr])
        X_aug_te = cp.hstack([X_te, pls_te])

    best_f1, best_trace = wide_beam_search(X_aug_tr, y_tr, max_depth=max_d)

    if best_trace is not None:
        train_vals = gpu_trace_eval(best_trace, X_aug_tr)
        thresh, direction, _ = gpu_opt_thresh(train_vals, y_tr)
        test_vals = gpu_trace_eval(best_trace, X_aug_te)
        pred = cp.where(direction * test_vals > thresh, 1, 0).astype(cp.int32)
        results_deep.append(gpu_f1(y_te, pred))
    else:
        results_deep.append(0.0)

    rf = cuRF(n_estimators=100, max_depth=8, random_state=42 + fold_i)
    rf.fit(X_tr.astype(cp.float32), y_tr.astype(cp.int32))
    rf_pred = rf.predict(X_te.astype(cp.float32))
    results_rf.append(gpu_f1(y_te, rf_pred.astype(cp.float64)))

    if fold_i % 10 == 0:
        log.info("fold %d/%d: deep=%.4f rf=%.4f (%.0fs)",
                 fold_i, 100, results_deep[-1], results_rf[-1],
                 time.time() - t0)

dm = np.mean(results_deep)
rfm = np.mean(results_rf)
diffs = np.array(results_deep) - np.array(results_rf)
se = np.std(diffs) / np.sqrt(len(diffs))
sigma = float(np.mean(diffs) / se) if se > 1e-10 else 0.0

out = {
    "name": DATASET, "N": N, "d": d, "max_depth": max_d,
    "method": "wide_turbo_d3",
    "deep_f1": round(dm, 4), "cuml_rf_f1": round(rfm, 4),
    "sigma_vs_rf": round(sigma, 1),
    "seconds": round(time.time() - t0),
}
with open(f"wide_d3_{DATASET.lower()}.json", "w") as f:
    json.dump(out, f, indent=2)

log.info("DONE %s: deep=%.4f rf=%.4f sigma=%.1f (%.0fs)",
         DATASET, dm, rfm, sigma, time.time() - t0)
