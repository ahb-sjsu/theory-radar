#!/usr/bin/env python3
"""Full Theory Radar pipeline on ALL 9 datasets.
PCA + A* beam + meta-learned pruning + fuzzed subspaces + fair eval.

Small datasets (N<2K): beam=100, 200x5=1000 folds, 15 subspaces
Large datasets (N>10K): beam=50, 50x5=250 folds, 8 subspaces
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, os.path.dirname(__file__))

# Import pipeline from run_full_pipeline.py
with open(os.path.join(os.path.dirname(__file__), "run_full_pipeline.py")) as f:
    code = f.read().split("def main")[0]
exec(code)

import logging
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

from sklearn.datasets import load_breast_cancer, load_wine, fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder


def main():
    log.info("=" * 70)
    log.info("FULL PIPELINE — ALL 9 DATASETS")
    log.info("PCA + A* Beam + Meta Pruning + Fuzzed Subspaces + Fair Eval")
    log.info("=" * 70)

    props = cp.cuda.runtime.getDeviceProperties(0)
    log.info("GPU: %s (%.1f GB free)", props["name"].decode(), cp.cuda.Device(0).mem_info[0] / 1e9)
    _ = cp.sort(cp.random.rand(10000, 100), axis=0)
    cp.cuda.Stream.null.synchronize()

    datasets = []

    # Small UCI datasets
    bc = load_breast_cancer()
    X = StandardScaler().fit_transform(bc.data)
    datasets.append(("BreastCancer", X, bc.target, [f"f{i}" for i in range(X.shape[1])]))

    wine = load_wine()
    X = StandardScaler().fit_transform(wine.data)
    y = (wine.target == 0).astype(int)
    datasets.append(("Wine", X, y, [f"w{i}" for i in range(13)]))

    for oml, ver, short, tfn in [
        ("diabetes", 1, "Diabetes", lambda t: (t == "tested_positive").astype(int)),
        ("banknote-authentication", 1, "Banknote", None),
        ("ionosphere", 1, "Ionosphere", lambda t: (t == "g").astype(int)),
    ]:
        try:
            ds = fetch_openml(oml, version=ver, as_frame=False, parser="auto")
            X = StandardScaler().fit_transform(ds.data.astype(float))
            if tfn:
                y = tfn(ds.target)
            else:
                yr = ds.target.astype(int)
                y = (yr == yr.max()).astype(int)
            datasets.append((short, X, y, [f"v{i}" for i in range(X.shape[1])]))
        except Exception as e:
            log.warning("%s: %s", oml, e)

    # Large datasets
    for oml, ver, short in [
        ("eeg-eye-state", 1, "EEG"),
        ("MagicTelescope", 1, "Magic"),
        ("electricity", 1, "Electricity"),
        ("MiniBooNE", 1, "MiniBooNE"),
    ]:
        try:
            ds = fetch_openml(oml, version=ver, as_frame=False, parser="auto")
            X = StandardScaler().fit_transform(ds.data.astype(float))
            le = LabelEncoder()
            y = le.fit_transform(ds.target)
            if len(set(y)) != 2:
                y = (y == y.max()).astype(int)
            datasets.append((short, X, y, [f"v{i}" for i in range(X.shape[1])]))
        except Exception as e:
            log.warning("%s: %s", oml, e)

    for name, X, y, _ in datasets:
        log.info("  %s: N=%d d=%d prev=%.2f", name, X.shape[0], X.shape[1], y.mean())

    all_res = []
    for name, X, y, feats in datasets:
        N, d = X.shape
        # Scale parameters by dataset size
        if N > 50000:
            nr, bw, ns, sk = 20, 50, 5, min(8, d + 4)
        elif N > 10000:
            nr, bw, ns, sk = 50, 50, 8, min(10, d + 4)
        elif N > 2000:
            nr, bw, ns, sk = 100, 100, 10, min(10, d + 8)
        else:
            nr, bw, ns, sk = 200, 100, 15, min(12, d + 8)

        r = run_dataset(
            name,
            X,
            y,
            feats,
            n_repeats=nr,
            n_folds=5,
            n_pca=8,
            max_depth=3,
            beam_width=bw,
            n_subspaces=ns,
            subspace_k=sk,
        )
        all_res.append(r)

        # Save incrementally
        with open("all_full_results.json", "w") as f:
            json.dump(all_res, f, indent=2)

    log.info("\n" + "=" * 110)
    log.info("COMPLETE RESULTS — Full Pipeline (PCA + A* Beam + Meta + Fuzz, Fair Eval)")
    log.info(
        "%-15s %5s %3s  %-8s %-8s %-5s  %-30s  %-10s %-10s %-10s",
        "Dataset",
        "N",
        "d",
        "Train",
        "Test",
        "Gap",
        "Formula",
        "vs GB",
        "vs RF",
        "vs LR",
    )
    log.info("-" * 110)
    for r in all_res:
        bl = r["baselines"]
        log.info(
            "%-15s %5d %3d  %.4f  %.4f  %.3f  %-30s  %5.1f %s  %5.1f %s  %5.1f %s",
            r["name"],
            r.get("N", 0),
            r.get("d", 0),
            r["train_f1"],
            r["test_f1"],
            r["train_f1"] - r["test_f1"],
            r["formula"][:30],
            bl["GB"]["sigma"],
            bl["GB"]["dir"],
            bl["RF"]["sigma"],
            bl["RF"]["dir"],
            bl["LR"]["sigma"],
            bl["LR"]["dir"],
        )
    log.info("=" * 110)


if __name__ == "__main__":
    main()
