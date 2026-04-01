#!/usr/bin/env python3
"""Depth-7 GPU pilot: 5 biggest losers, 20x5 CV, full GPU beam search.

Puts those GV100s to work with CuPy batched F1 evaluation at every depth.
Funnel beam: 100 -> 60 -> 30 -> 15 -> 7 -> 3 candidates.
"""
import os
import sys
import json
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

# Import GPU primitives from run_full_pipeline
with open("run_full_pipeline.py") as f:
    code = f.read().split("def main")[0]
exec(code)  # noqa: S102 — gives us gpu_batch_f1, VBIN, VUN, FT, opt_thresh, SB, SU

import cupy as cp
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.cross_decomposition import PLSRegression
from dataset_loader import load_dataset


def gpu_beam_funnel(X_gpu, y_gpu, max_depth=7):
    """GPU beam search with funnel narrowing to depth 7."""
    N, d = X_gpu.shape
    yf = y_gpu.astype(cp.float64)

    # Depth 1: all features
    f1s = gpu_batch_f1(X_gpu, yf)
    beam = [(float(f1s[fi]), FT(fi)) for fi in range(d)]
    beam.sort(key=lambda x: -x[0])

    schedule = {2: 100, 3: 60, 4: 30, 5: 15, 6: 7, 7: 3}

    for depth in range(2, max_depth + 1):
        bw = schedule.get(depth, 3)
        beam = beam[:min(bw, len(beam))]
        if not beam:
            break

        vals_list = []
        trace_list = []

        for _score, trace in beam:
            v = cp.asarray(trace.ev(X_gpu.get()), dtype=cp.float64)
            if v.ndim == 0:
                continue

            # Binary ops: v (op) feature_j — all on GPU
            for opn, opf in VBIN:
                for fi in range(d):
                    try:
                        cv = opf(v, X_gpu[:, fi])
                        cv = cp.nan_to_num(cv, nan=0.0, posinf=1e10, neginf=-1e10)
                        # Skip constant candidates
                        if float(cp.std(cv)) < 1e-12:
                            continue
                        vals_list.append(cv)
                        trace_list.append(trace.b(opn, fi))
                    except Exception:
                        pass

            # Unary ops
            for opn, opf in VUN:
                try:
                    cv = opf(v)
                    cv = cp.nan_to_num(cv, nan=0.0, posinf=1e10, neginf=-1e10)
                    if float(cp.std(cv)) < 1e-12:
                        continue
                    vals_list.append(cv)
                    trace_list.append(trace.u(opn))
                except Exception:
                    pass

        if not vals_list:
            break

        # Chunked batch F1 on GPU to avoid OOM on large datasets
        CHUNK = 5000
        all_f1_parts = []
        for ci in range(0, len(vals_list), CHUNK):
            chunk = cp.stack(vals_list[ci:ci + CHUNK], axis=1)
            all_f1_parts.append(gpu_batch_f1(chunk, yf))
            del chunk
            cp.get_default_memory_pool().free_all_blocks()
        all_f1 = cp.concatenate(all_f1_parts)

        new_beam = [(float(all_f1[i]), trace_list[i]) for i in range(len(trace_list))]
        new_beam.sort(key=lambda x: -x[0])

        # Keep best across all depths
        seen = set()
        merged = []
        for item in sorted(beam + new_beam, key=lambda x: -x[0]):
            key = round(item[0], 6)
            if key not in seen:
                seen.add(key)
                merged.append(item)
            if len(merged) >= bw:
                break
        beam = merged

        log.info("  d%d: best=%.4f candidates=%d beam=%d",
                 depth, beam[0][0], len(vals_list), len(beam))

    return beam[0] if beam else (0.0, None)


DATASETS = ["EEG", "Magic", "Electricity", "HIGGS", "Adult"]

for name in DATASETS:
    X, y, feats = load_dataset(name)
    N, d = X.shape
    log.info("=== %s: N=%d d=%d N/d=%.0f ===", name, N, d, N / d)

    results_d7 = []
    results_d3 = []
    results_gb = []

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)
    for fold_i, (tr, te) in enumerate(cv.split(X, y)):
        # PLS projection (fit on train only)
        k = min(8, d - 1, len(tr) - 1)
        pls = PLSRegression(n_components=k)
        pls.fit(X[tr], y[tr])
        X_aug_tr = np.hstack([X[tr], pls.transform(X[tr])])
        X_aug_te = np.hstack([X[te], pls.transform(X[te])])

        X_gpu = cp.asarray(X_aug_tr, dtype=cp.float64)
        y_gpu = cp.asarray(y[tr], dtype=cp.float64)

        # --- Depth 7 ---
        best_f1, best_trace = gpu_beam_funnel(X_gpu, y_gpu, max_depth=7)
        if best_trace is not None:
            train_vals = best_trace.ev(X_aug_tr)
            thresh, direction, _ = opt_thresh(train_vals, y[tr])
            test_vals = best_trace.ev(X_aug_te)
            pred = ((test_vals > thresh) if direction == 1
                    else (test_vals < thresh)).astype(int)
            results_d7.append(f1_score(y[te], pred))
        else:
            results_d7.append(0.0)

        # --- Depth 3 for comparison ---
        f3, t3 = gpu_beam_funnel(X_gpu, y_gpu, max_depth=3)
        if t3 is not None:
            tv = t3.ev(X_aug_tr)
            th, dr, _ = opt_thresh(tv, y[tr])
            tp = t3.ev(X_aug_te)
            p3 = ((tp > th) if dr == 1 else (tp < th)).astype(int)
            results_d3.append(f1_score(y[te], p3))
        else:
            results_d3.append(0.0)

        # --- GB baseline ---
        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=42 + fold_i)
        gb.fit(X[tr], y[tr])
        results_gb.append(f1_score(y[te], gb.predict(X[te])))

        if fold_i % 25 == 0:
            log.info("  fold %d: d7=%.4f d3=%.4f gb=%.4f",
                     fold_i, results_d7[-1], results_d3[-1], results_gb[-1])

    d7m = np.mean(results_d7)
    d3m = np.mean(results_d3)
    gbm = np.mean(results_gb)
    diffs = np.array(results_d7) - np.array(results_gb)
    se = np.std(diffs) / np.sqrt(len(diffs))
    sigma = float(np.mean(diffs) / se) if se > 1e-10 else 0.0

    out = {
        "name": name, "N": N, "d": d,
        "depth7_f1": round(d7m, 4),
        "depth3_f1": round(d3m, 4),
        "gb_f1": round(gbm, 4),
        "sigma_vs_gb": round(sigma, 1),
        "improvement_d3_to_d7": round(d7m - d3m, 4),
    }
    with open(f"pilot_d7_{name.lower()}.json", "w") as f:
        json.dump(out, f, indent=2)
    log.info("DONE %s: d7=%.4f (+%.4f vs d3) gb=%.4f sigma=%.1f",
             name, d7m, d7m - d3m, gbm, sigma)

log.info("=== ALL DONE ===")
