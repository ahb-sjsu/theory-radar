#!/usr/bin/env python3
"""Depth-7 100% GPU pilot. NOTHING on CPU except data loading.
PLS, beam search, threshold, F1, RF baseline — all CuPy/cuML on GPU 1.
"""
import os
import sys
import json
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

with open("run_full_pipeline.py") as f:
    code = f.read().split("def main")[0]
# Strip the CUDA_VISIBLE_DEVICES override so our "1" sticks
code = code.replace('os.environ["CUDA_VISIBLE_DEVICES"] = "0"',
                     '# CUDA_VISIBLE_DEVICES set by caller')
exec(code)  # noqa: S102 — imports gpu_batch_f1, VBIN, VUN, FT, SB, SU

import cupy as cp
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from dataset_loader import load_dataset
from cuml.ensemble import RandomForestClassifier as cuRF

# GPU versions of SB/SU ops (CuPy, not numpy)
GSB = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / (b + 1e-30),
    "max": lambda a, b: cp.maximum(a, b),
    "min": lambda a, b: cp.minimum(a, b),
    "hypot": lambda a, b: cp.sqrt(a**2 + b**2),
    "diff_sq": lambda a, b: (a - b) ** 2,
    "harmonic": lambda a, b: 2 * a * b / (a + b + 1e-30),
    "geometric": lambda a, b: cp.sign(a * b) * cp.sqrt(cp.abs(a * b)),
}
GSU = {
    "log": lambda x: cp.log(cp.abs(x) + 1e-30),
    "sqrt": lambda x: cp.sqrt(cp.abs(x)),
    "sq": lambda x: x**2,
    "abs": lambda x: cp.abs(x),
    "sigmoid": lambda x: 1.0 / (1.0 + cp.exp(-cp.clip(x, -500, 500))),
    "tanh": lambda x: cp.tanh(cp.clip(x, -500, 500)),
}


def gpu_trace_eval(trace, X_gpu):
    """Evaluate FT trace entirely on GPU using CuPy ops."""
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
    """Find optimal threshold on GPU. Returns (thresh, direction, f1)."""
    N = vals_gpu.shape[0]
    P = int(y_gpu.sum())
    if P == 0 or P == N:
        return 0.0, 1, 0.0

    best_f1 = 0.0
    best_thresh = 0.0
    best_dir = 1

    for d in [1, -1]:
        sv = d * vals_gpu
        idx = cp.argsort(-sv)
        sorted_v = sv[idx]
        sorted_y = y_gpu[idx].astype(cp.bool_)
        tp = cp.cumsum(sorted_y, dtype=cp.float64)
        fp = cp.cumsum(~sorted_y, dtype=cp.float64)
        prec = tp / (tp + fp + 1e-30)
        rec = tp / P
        f1 = 2 * prec * rec / (prec + rec + 1e-30)
        best_idx = int(cp.argmax(f1))
        f1_val = float(f1[best_idx])
        if f1_val > best_f1:
            best_f1 = f1_val
            best_thresh = float(sorted_v[best_idx])
            best_dir = d

    return best_thresh, best_dir, best_f1


def gpu_f1_score(y_true_gpu, y_pred_gpu):
    """F1 score on GPU."""
    tp = float(((y_pred_gpu == 1) & (y_true_gpu == 1)).sum())
    fp = float(((y_pred_gpu == 1) & (y_true_gpu == 0)).sum())
    fn = float(((y_pred_gpu == 0) & (y_true_gpu == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


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
        tt = float((t.T @ t))
        if tt < 1e-30:
            break
        p = (Xc.T @ t) / tt
        q = (yc.T @ t) / tt
        Xc = Xc - t @ p.T
        yc = yc - t * q
        W[:, i:i+1] = w
    return W, X_mean


def gpu_beam_search(X_gpu, y_gpu, max_depth=7):
    """Beam search: vectorized binary ops, traces kept for replay."""
    N, d = X_gpu.shape
    yf = y_gpu.astype(cp.float64)

    f1s = gpu_batch_f1(X_gpu, yf)
    beam = [(float(f1s[fi]), FT(fi)) for fi in range(d)]
    beam.sort(key=lambda x: -x[0])

    schedule = {2: 100, 3: 60, 4: 30, 5: 15, 6: 7, 7: 3}

    for depth in range(2, max_depth + 1):
        bw = schedule.get(depth, 3)
        beam = beam[:bw]
        B = len(beam)
        if B == 0:
            break

        # Get current beam values on GPU
        beam_vals = cp.stack([gpu_trace_eval(b[1], X_gpu) for b in beam])

        new_entries = []
        bv = beam_vals[:, :, None]       # (B, N, 1)
        xf = X_gpu[None, :, :]           # (1, N, d)

        for opn, opf in VBIN:
            try:
                cands = opf(bv, xf)      # (B, N, d)
                cands = cp.nan_to_num(cands, nan=0.0, posinf=1e10, neginf=-1e10)
                flat = cands.reshape(B * d, N)
                # Chunked F1 on GPU
                CHUNK = 8000
                parts = []
                for ci in range(0, flat.shape[0], CHUNK):
                    parts.append(gpu_batch_f1(flat[ci:ci+CHUNK].T, yf))
                f1_all = cp.concatenate(parts)

                for bi in range(B):
                    for fi in range(d):
                        idx = bi * d + fi
                        new_entries.append((float(f1_all[idx]), beam[bi][1].b(opn, fi)))
                del cands, flat, f1_all
            except cp.cuda.memory.OutOfMemoryError:
                cp.get_default_memory_pool().free_all_blocks()

        for opn, opf in VUN:
            try:
                cands = opf(beam_vals)
                cands = cp.nan_to_num(cands, nan=0.0, posinf=1e10, neginf=-1e10)
                f1u = gpu_batch_f1(cands.T, yf)
                for bi in range(B):
                    new_entries.append((float(f1u[bi]), beam[bi][1].u(opn)))
            except Exception:
                pass

        del bv, xf, beam_vals

        combined = beam + new_entries
        combined.sort(key=lambda x: -x[0])
        seen = set()
        beam = []
        for item in combined:
            key = round(item[0], 5)
            if key not in seen:
                seen.add(key)
                beam.append(item)
            if len(beam) >= bw:
                break

        log.info("  d%d: best=%.4f K=%d beam=%d",
                 depth, beam[0][0], len(new_entries), len(beam))

    return beam[0] if beam else (0.0, None)


DATASETS = ["Spambase", "Vehicle", "Ionosphere",
            "QSAR", "Wilt", "German", "Banknote", "Mammography"]

for name in DATASETS:
    X, y, feats = load_dataset(name)  # only CPU touch: load from disk
    N, d = X.shape
    n_over_d = N / d
    log.info("=== %s: N=%d d=%d N/d=%.0f ===", name, N, d, n_over_d)

    max_d = 7 if n_over_d > 10 else (5 if n_over_d > 5 else 3)
    log.info("  max_depth=%d", max_d)

    # Move to GPU once, stay there
    X_gpu = cp.asarray(X, dtype=cp.float64)
    y_gpu = cp.asarray(y, dtype=cp.float64)

    results_deep = []
    results_rf = []

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)
    for fold_i, (tr, te) in enumerate(cv.split(X, y)):
        tr_gpu = cp.asarray(tr)
        te_gpu = cp.asarray(te)
        X_tr = X_gpu[tr_gpu]
        y_tr = y_gpu[tr_gpu]
        X_te = X_gpu[te_gpu]
        y_te = y_gpu[te_gpu]

        # GPU PLS
        k = min(8, d - 1, len(tr) - 1)
        W, X_mean = gpu_pls(X_tr, y_tr, n_components=k)
        X_aug_tr = cp.hstack([X_tr, (X_tr - X_mean) @ W])
        X_aug_te = cp.hstack([X_te, (X_te - X_mean) @ W])

        # GPU beam search (returns trace for replay)
        best_f1, best_trace = gpu_beam_search(X_aug_tr, y_tr, max_depth=max_d)

        if best_trace is not None:
            # GPU trace eval on train + test
            train_vals = gpu_trace_eval(best_trace, X_aug_tr)
            thresh, direction, _ = gpu_opt_thresh(train_vals, y_tr)
            test_vals = gpu_trace_eval(best_trace, X_aug_te)
            pred = cp.where(direction * test_vals > thresh, 1, 0).astype(cp.int32)
            results_deep.append(gpu_f1_score(y_te, pred))
        else:
            results_deep.append(0.0)

        # GPU RF baseline
        rf = cuRF(n_estimators=100, max_depth=8, random_state=42 + fold_i)
        rf.fit(X_tr.astype(cp.float32), y_tr.astype(cp.int32))
        rf_pred = rf.predict(X_te.astype(cp.float32))
        results_rf.append(gpu_f1_score(y_te, rf_pred.astype(cp.float64)))

        if fold_i % 25 == 0:
            log.info("  fold %d: deep=%.4f rf=%.4f",
                     fold_i, results_deep[-1], results_rf[-1])

    dm = np.mean(results_deep)
    rfm = np.mean(results_rf)
    diffs = np.array(results_deep) - np.array(results_rf)
    se = np.std(diffs) / np.sqrt(len(diffs))
    sigma = float(np.mean(diffs) / se) if se > 1e-10 else 0.0

    out = {
        "name": name, "N": N, "d": d, "max_depth": max_d,
        "deep_f1": round(dm, 4),
        "cuml_rf_f1": round(rfm, 4),
        "sigma_vs_rf": round(sigma, 1),
    }
    with open(f"pilot_d7_{name.lower()}.json", "w") as f:
        json.dump(out, f, indent=2)
    log.info("DONE %s: deep(d%d)=%.4f rf=%.4f sigma=%.1f",
             name, max_d, dm, rfm, sigma)

    del X_gpu, y_gpu
    cp.get_default_memory_pool().free_all_blocks()

log.info("=== 100%% GPU COMPLETE ===")
