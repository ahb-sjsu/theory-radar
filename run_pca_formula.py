#!/usr/bin/env python3
"""PCA + Formula Search: use ALL features via projections.

Key idea: depth-3 formulas touch at most 3 raw features, losing to
linear models that use all d features. Fix: project to k principal
components first, then search formulas over the projections. Each
PC is a linear combination of ALL features, so the formula implicitly
uses the full feature set.

Three search modes compared:
  1. Raw features only (baseline)
  2. PCA components only
  3. Raw + PCA combined (formula can use either)

Fair test evaluation: PCA fit on train, transform test, replay formula.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cupy as cp
import numpy as np
import logging
import time
import json
from collections import Counter
from scipy import stats
from joblib import Parallel, delayed

from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

# ─── GPU ops (same core) ──────────────────────────────────────────


def gpu_batch_f1(vals, labels):
    N, K = vals.shape
    pos = labels.sum()
    if float(pos) == 0 or float(pos) == N:
        return cp.zeros(K, dtype=cp.float64)
    return cp.maximum(_sweep(vals, labels, pos), _sweep(-vals, labels, pos))


def _sweep(vals, labels, pos):
    N, K = vals.shape
    idx = cp.argsort(-vals, axis=0)
    sv = cp.take_along_axis(vals, idx, axis=0)
    sl = labels[idx]
    tp = cp.cumsum(sl, axis=0, dtype=cp.float64)
    fp = cp.cumsum(1.0 - sl, axis=0, dtype=cp.float64)
    denom = tp + pos + fp
    f1 = cp.where(denom > 0, 2.0 * tp / denom, 0.0)
    valid = cp.ones((N, K), dtype=cp.bool_)
    valid[:-1, :] = (sv[:-1, :] - sv[1:, :]) > 1e-15
    return cp.max(cp.where(valid, f1, 0.0), axis=0)


VBINARY = [
    ("+", lambda B, F: B + F),
    ("-", lambda B, F: B - F),
    ("*", lambda B, F: B * F),
    ("/", lambda B, F: B / (F + 1e-30)),
    ("max", lambda B, F: cp.maximum(B, F)),
    ("min", lambda B, F: cp.minimum(B, F)),
    ("hypot", lambda B, F: cp.sqrt(B**2 + F**2)),
    ("diff_sq", lambda B, F: (B - F) ** 2),
    ("harmonic", lambda B, F: 2 * B * F / (B + F + 1e-30)),
    ("geometric", lambda B, F: cp.sign(B * F) * cp.sqrt(cp.abs(B * F))),
]
VUNARY = [
    ("log", lambda V: cp.log(cp.abs(V) + 1e-30)),
    ("sqrt", lambda V: cp.sqrt(cp.abs(V))),
    ("sq", lambda V: V**2),
    ("abs", lambda V: cp.abs(V)),
    ("sigmoid", lambda V: 1.0 / (1.0 + cp.exp(-cp.clip(V, -500, 500)))),
    ("tanh", lambda V: cp.tanh(cp.clip(V, -500, 500))),
]
SCALAR_BIN = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / (b + 1e-30),
    "max": lambda a, b: np.maximum(a, b),
    "min": lambda a, b: np.minimum(a, b),
    "hypot": lambda a, b: np.sqrt(a**2 + b**2),
    "diff_sq": lambda a, b: (a - b) ** 2,
    "harmonic": lambda a, b: 2 * a * b / (a + b + 1e-30),
    "geometric": lambda a, b: np.sign(a * b) * np.sqrt(np.abs(a * b)),
}
SCALAR_UN = {
    "log": lambda x: np.log(np.abs(x) + 1e-30),
    "sqrt": lambda x: np.sqrt(np.abs(x)),
    "sq": lambda x: x**2,
    "abs": lambda x: np.abs(x),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
    "tanh": lambda x: np.tanh(np.clip(x, -500, 500)),
}


class FormulaTrace:
    """Records operations. Feature indices refer to the augmented feature matrix."""

    def __init__(self, fi):
        self.ops = [("leaf", fi)]

    def binary(self, op, fi):
        t = FormulaTrace.__new__(FormulaTrace)
        t.ops = self.ops + [("binary", op, fi)]
        return t

    def unary(self, op):
        t = FormulaTrace.__new__(FormulaTrace)
        t.ops = self.ops + [("unary", op)]
        return t

    def evaluate(self, X):
        v = None
        for s in self.ops:
            if s[0] == "leaf":
                v = X[:, s[1]].copy()
            elif s[0] == "binary":
                v = SCALAR_BIN[s[1]](v, X[:, s[2]])
                v = np.nan_to_num(v, nan=0.0, posinf=1e10, neginf=-1e10)
            elif s[0] == "unary":
                v = SCALAR_UN[s[1]](v)
                v = np.nan_to_num(v, nan=0.0, posinf=1e10, neginf=-1e10)
        return v


def find_optimal_threshold(vals, y):
    best_f1, best_t, best_d = 0.0, 0.0, 1
    P = int(y.astype(bool).sum())
    N = len(y)
    if P == 0 or P == N:
        return 0.0, 1, 0.0
    for d in [1, -1]:
        o = np.argsort(d * vals)[::-1]
        sv = (d * vals)[o]
        sa = y.astype(bool)[o]
        tp = fp = 0
        for i in range(N):
            if sa[i]:
                tp += 1
            else:
                fp += 1
            if tp + fp > 0:
                p = tp / (tp + fp)
                r = tp / P
                if p + r > 0:
                    f = 2 * p * r / (p + r)
                    if f > best_f1:
                        best_f1, best_t, best_d = f, sv[i], d
    return best_t, best_d, best_f1


def gpu_beam_search(X_gpu, y_gpu, feat_names, max_depth=3, beam_width=100):
    """Standard vectorized beam search."""
    N, d = X_gpu.shape
    y_f = y_gpu.astype(cp.float64)

    f1s = gpu_batch_f1(X_gpu, y_f)
    order = cp.argsort(-f1s).get()
    B = min(beam_width, d)

    beam_vals = cp.empty((N, B), dtype=cp.float64)
    beam_names, beam_traces = [], []
    for rank, i in enumerate(order[:B]):
        beam_vals[:, rank] = X_gpu[:, i]
        beam_names.append(feat_names[i])
        beam_traces.append(FormulaTrace(i))

    best_f1 = float(f1s[order[0]])
    best_name = beam_names[0]
    best_trace = beam_traces[0]

    for depth in range(2, max_depth + 1):
        B_cur = beam_vals.shape[1]
        bv3 = beam_vals[:, :, None]
        fv3 = X_gpu[:, None, :]

        chunks, name_chunks, trace_chunks = [], [], []
        for opname, opfn in VBINARY:
            try:
                out = opfn(bv3, fv3).reshape(N, B_cur * d)
                out = cp.nan_to_num(out, nan=0.0, posinf=1e10, neginf=-1e10)
                chunks.append(out)
                names, traces = [], []
                for bi in range(B_cur):
                    for fi in range(d):
                        names.append(f"({beam_names[bi]} {opname} {feat_names[fi]})")
                        traces.append(beam_traces[bi].binary(opname, fi))
                name_chunks.append(names)
                trace_chunks.append(traces)
            except:
                pass
        for opname, opfn in VUNARY:
            try:
                out = opfn(beam_vals)
                out = cp.nan_to_num(out, nan=0.0, posinf=1e10, neginf=-1e10)
                chunks.append(out)
                name_chunks.append([f"{opname}({beam_names[bi]})" for bi in range(B_cur)])
                trace_chunks.append([beam_traces[bi].unary(opname) for bi in range(B_cur)])
            except:
                pass

        if not chunks:
            break
        all_vals = cp.concatenate(chunks, axis=1)
        all_names = [n for nc in name_chunks for n in nc]
        all_traces = [t for tc in trace_chunks for t in tc]

        var = cp.var(all_vals, axis=0)
        f1s = gpu_batch_f1(all_vals, y_f)
        f1s = cp.where(var < 1e-20, 0.0, f1s)

        top_k = min(beam_width, all_vals.shape[1])
        top_idx = cp.argsort(-f1s)[:top_k].get()
        top_f1 = f1s[cp.asarray(top_idx)].get()

        beam_vals = cp.empty((N, top_k), dtype=cp.float64)
        beam_names, beam_traces = [], []
        for rank, i in enumerate(top_idx):
            beam_vals[:, rank] = all_vals[:, i]
            beam_names.append(all_names[i])
            beam_traces.append(all_traces[i])
            if float(top_f1[rank]) > best_f1:
                best_f1 = float(top_f1[rank])
                best_name = all_names[i]
                best_trace = all_traces[i]

        del all_vals, chunks
        cp.get_default_memory_pool().free_all_blocks()

    return best_f1, best_name, best_trace


def run_sklearn_fold(X, y, tr, te, fi):
    out = {}
    for nm, clf in [
        ("GB", GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42 + fi)),
        ("RF", RandomForestClassifier(n_estimators=100, random_state=42 + fi)),
        ("LR", LogisticRegression(max_iter=1000, random_state=42)),
    ]:
        clf.fit(X[tr], y[tr])
        out[nm] = f1_score(y[te], clf.predict(X[te]))
    return out


# ─── PCA + Formula Search Pipeline ────────────────────────────────


def run_dataset(
    name, X, y, raw_feat_names, n_repeats=200, n_folds=5, n_pca=8, max_depth=3, beam_width=100
):
    total = n_repeats * n_folds
    d = X.shape[1]
    log.info("=" * 70)
    log.info(
        "PCA+FORMULA: %s (N=%d, d=%d) — %d folds, %d PCs, depth=%d",
        name,
        X.shape[0],
        d,
        total,
        n_pca,
        max_depth,
    )
    log.info("=" * 70)

    cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
    splits = list(cv.split(X, y))

    modes = {
        "raw": {"use_pca": False, "use_raw": True},
        "pca": {"use_pca": True, "use_raw": False},
        "raw+pca": {"use_pca": True, "use_raw": True},
    }

    all_mode_results = {}

    for mode_name, mode_cfg in modes.items():
        astar_f1s, astar_train_f1s, formulas = [], [], []
        t0 = time.time()

        for fold_i, (tr, te) in enumerate(splits):
            # Fit PCA on training data
            if mode_cfg["use_pca"]:
                k = min(n_pca, X[tr].shape[1])
                pca = PCA(n_components=k)
                pca_tr = pca.fit_transform(X[tr])
                pca_te = pca.transform(X[te])
                pca_names = [f"pc{i}" for i in range(k)]
            else:
                pca_tr = np.empty((len(tr), 0))
                pca_te = np.empty((len(te), 0))
                pca_names = []

            # Build augmented feature matrix
            if mode_cfg["use_raw"] and mode_cfg["use_pca"]:
                X_aug_tr = np.hstack([X[tr], pca_tr])
                X_aug_te = np.hstack([X[te], pca_te])
                aug_names = raw_feat_names + pca_names
            elif mode_cfg["use_pca"]:
                X_aug_tr = pca_tr
                X_aug_te = pca_te
                aug_names = pca_names
            else:
                X_aug_tr = X[tr]
                X_aug_te = X[te]
                aug_names = raw_feat_names

            # GPU beam search on augmented features
            X_gpu = cp.asarray(X_aug_tr, dtype=cp.float64)
            y_gpu = cp.asarray(y[tr], dtype=cp.float64)

            train_f1, formula_name, trace = gpu_beam_search(
                X_gpu, y_gpu, aug_names, max_depth=max_depth, beam_width=beam_width
            )

            astar_train_f1s.append(train_f1)
            formulas.append(formula_name)

            # Fair test eval on augmented test features
            vals_tr = trace.evaluate(X_aug_tr)
            vals_te = trace.evaluate(X_aug_te)
            thresh, dirn, _ = find_optimal_threshold(vals_tr, y[tr])
            preds = (dirn * vals_te >= thresh).astype(int)
            astar_f1s.append(f1_score(y[te], preds))

            del X_gpu, y_gpu
            cp.get_default_memory_pool().free_all_blocks()

            if (fold_i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (fold_i + 1) / elapsed
                log.info(
                    "  [%s] %d/%d  train=%.3f test=%.3f  %.1f/s",
                    mode_name,
                    fold_i + 1,
                    total,
                    np.mean(astar_train_f1s[-100:]),
                    np.mean(astar_f1s[-100:]),
                    rate,
                )

        gpu_time = time.time() - t0
        astar_f1s = np.array(astar_f1s)
        astar_train_f1s = np.array(astar_train_f1s)

        fc = Counter(formulas)
        top = fc.most_common(1)[0]
        log.info(
            "  [%s] TRAIN=%.4f TEST=%.4f gap=%.4f formula=%s (%.0f%%)",
            mode_name,
            astar_train_f1s.mean(),
            astar_f1s.mean(),
            astar_train_f1s.mean() - astar_f1s.mean(),
            top[0][:40],
            100 * top[1] / len(formulas),
        )

        all_mode_results[mode_name] = {
            "test_f1s": astar_f1s,
            "train_mean": float(astar_train_f1s.mean()),
            "test_mean": float(astar_f1s.mean()),
            "formula": top[0],
            "stability": float(top[1] / len(formulas)),
        }

    # CPU baselines (same for all modes)
    t1 = time.time()
    n_jobs = min(os.cpu_count() or 1, 20)
    baselines = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(run_sklearn_fold)(X, y, tr, te, fi) for fi, (tr, te) in enumerate(splits)
    )
    cpu_time = time.time() - t1

    # Compare each mode vs baselines
    log.info("")
    log.info(
        "  %-10s  %-8s  %-8s  %-30s  %-12s %-12s %-12s",
        "Mode",
        "TrainF1",
        "TestF1",
        "Formula",
        "vs GB",
        "vs RF",
        "vs LR",
    )
    log.info("  " + "-" * 100)

    results = {}
    for mode_name, mr in all_mode_results.items():
        mode_results = {}
        for bn in ["GB", "RF", "LR"]:
            bf = np.array([r[bn] for r in baselines])
            diff = mr["test_f1s"] - bf
            t_stat, p_val = stats.ttest_1samp(diff, 0)
            mode_results[bn] = {
                "mean": float(bf.mean()),
                "diff": float(diff.mean()),
                "sigma": float(abs(t_stat)),
                "dir": "A*>" if diff.mean() > 0 else f"{bn}>",
            }

        log.info(
            "  %-10s  %.4f   %.4f   %-30s  %.1f%s %s  %.1f%s %s  %.1f%s %s",
            mode_name,
            mr["train_mean"],
            mr["test_mean"],
            mr["formula"][:30],
            mode_results["GB"]["sigma"],
            "s",
            mode_results["GB"]["dir"],
            mode_results["RF"]["sigma"],
            "s",
            mode_results["RF"]["dir"],
            mode_results["LR"]["sigma"],
            "s",
            mode_results["LR"]["dir"],
        )

        results[mode_name] = {
            "test_f1": mr["test_mean"],
            "formula": mr["formula"],
            "baselines": mode_results,
        }

    return results


def main():
    log.info("PCA + FORMULA SEARCH: Using ALL features via projections")

    props = cp.cuda.runtime.getDeviceProperties(0)
    log.info("GPU: %s (%.1f GB free)", props["name"].decode(), cp.cuda.Device(0).mem_info[0] / 1e9)
    _ = cp.sort(cp.random.rand(10000, 100), axis=0)
    cp.cuda.Stream.null.synchronize()

    datasets = []

    # Datasets where raw formula LOST to baselines
    bc = load_breast_cancer()
    X = StandardScaler().fit_transform(bc.data)
    datasets.append(("BreastCancer", X, bc.target, [f"f{i}" for i in range(X.shape[1])]))

    try:
        ion = fetch_openml("ionosphere", version=1, as_frame=False, parser="auto")
        X = StandardScaler().fit_transform(ion.data.astype(float))
        y = (ion.target == "g").astype(int)
        datasets.append(("Ionosphere", X, y, [f"i{i}" for i in range(X.shape[1])]))
    except Exception as e:
        log.warning("Ionosphere: %s", e)

    # Also test on datasets where raw formula WON (should still win or improve)
    try:
        pima = fetch_openml("diabetes", version=1, as_frame=False, parser="auto")
        X = StandardScaler().fit_transform(pima.data.astype(float))
        y = (pima.target == "tested_positive").astype(int)
        datasets.append(("Diabetes", X, y, [f"d{i}" for i in range(8)]))
    except Exception as e:
        log.warning("Diabetes: %s", e)

    all_results = {}
    for name, X, y, feats in datasets:
        n_pca = min(8, X.shape[1])
        r = run_dataset(
            name, X, y, feats, n_repeats=200, n_folds=5, n_pca=n_pca, max_depth=3, beam_width=100
        )
        all_results[name] = r

    with open("pca_formula_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    log.info("\nSaved pca_formula_results.json")


if __name__ == "__main__":
    main()
