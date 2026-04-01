#!/usr/bin/env python3
"""Run full Theory Radar pipeline on a single dataset.

Usage: python run_one.py <Name> <n_repeats> <beam_width> <n_subspaces> <subspace_k>

Handles dataset loading, categorical encoding, and target binarization
for all 17 benchmark datasets.
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, os.path.dirname(__file__))

with open(os.path.join(os.path.dirname(__file__), "run_full_pipeline.py")) as f:
    code = f.read().split("def main")[0]
exec(code)

import logging
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

from sklearn.datasets import load_breast_cancer, load_wine, fetch_openml, fetch_covtype
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder

name = sys.argv[1]
nr, bw, ns, sk = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])


def load_dataset(name):
    """Load and preprocess dataset. Returns (X, y, feature_names).

    All features are numeric (float64) after preprocessing.
    Target is binary (0/1).
    """
    if name == "BreastCancer":
        bc = load_breast_cancer()
        X = StandardScaler().fit_transform(bc.data)
        return X, bc.target, [f"f{i}" for i in range(X.shape[1])]

    if name == "Wine":
        wine = load_wine()
        X = StandardScaler().fit_transform(wine.data)
        y = (wine.target == 0).astype(int)
        return X, y, [f"w{i}" for i in range(13)]

    # OpenML datasets with known target encodings
    OPENML = {
        "Diabetes": ("diabetes", 1, lambda t: (t == "tested_positive").astype(int)),
        "Banknote": ("banknote-authentication", 1, None),
        "Ionosphere": ("ionosphere", 1, lambda t: (t == "g").astype(int)),
        "EEG": ("eeg-eye-state", 1, None),
        "Magic": ("MagicTelescope", 1, None),
        "Electricity": ("electricity", 1, None),
        "MiniBooNE": ("MiniBooNE", 1, None),
        "Heart": ("heart-statlog", 1, None),
        "Sonar": ("sonar", 1, None),
        "Spambase": ("spambase", 1, None),
        "Australian": ("australian", 1, None),
        "Mammography": ("mammography", 1, None),
        "Breast-w": ("breast-w", 1, None),
        "Steel": ("steel-plates-fault", 1, None),
        "QSAR": ("qsar-biodeg", 1, None),
        "Wilt": ("wilt", 2, None),
        "JM1": ("jm1", 1, None),
        "KC1": ("kc1", 1, None),
        "PC4": ("pc4", 1, None),
        "Vehicle": ("vehicle", 1, None),
    }

    # Datasets that need special handling
    SPECIAL = {
        "German": ("credit-g", 1),
        "Adult": ("adult", 1),
        "HIGGS": ("higgs", 1),
        "Covertype": None,  # sklearn built-in
    }

    if name in OPENML:
        oml_name, ver, target_fn = OPENML[name]
        ds = fetch_openml(oml_name, version=ver, as_frame=False, parser="auto")

        import numpy as np
        import scipy.sparse
        if scipy.sparse.issparse(ds.data):
            X = ds.data.toarray().astype(float)
        else:
            X = np.array(ds.data, dtype=float)
        X = np.nan_to_num(X, nan=0.0)
        feat_names = [f"v{i}" for i in range(X.shape[1])]

        X = StandardScaler().fit_transform(X)

        # Target encoding
        if target_fn is not None:
            y = target_fn(ds.target.values if hasattr(ds.target, "values") else ds.target)
        else:
            le = LabelEncoder()
            y = le.fit_transform(ds.target)
            if len(set(y)) != 2:
                y = (y == y.max()).astype(int)

        return X, y.astype(int), feat_names

    if name in SPECIAL:
        if name == "Covertype":
            ds = fetch_covtype()
            X = StandardScaler().fit_transform(ds.data)
            y = (ds.target == 2).astype(int)  # lodgepole pine vs rest
            return X, y, [f"v{i}" for i in range(X.shape[1])]

        oml_name, ver = SPECIAL[name]
        ds = fetch_openml(oml_name, version=ver, as_frame=False, parser="auto")
        df = ds.data

        # Encode mixed types: numeric stays, categorical gets ordinal encoded
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

        parts = []
        feat_names = []
        if numeric_cols:
            parts.append(df[numeric_cols].values.astype(float))
            feat_names.extend([f"v{i}" for i in range(len(numeric_cols))])
        if cat_cols:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            cat_encoded = enc.fit_transform(df[cat_cols]).astype(float)
            n_offset = len(numeric_cols)
            parts.append(cat_encoded)
            feat_names.extend([f"c{i}" for i in range(cat_encoded.shape[1])])

        import numpy as np

        X = np.hstack(parts) if len(parts) > 1 else parts[0]

        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)
        X = StandardScaler().fit_transform(X)

        # Target
        le = LabelEncoder()
        y = le.fit_transform(ds.target)
        if len(set(y)) != 2:
            y = (y == y.max()).astype(int)

        return X, y.astype(int), feat_names

    raise ValueError(f"Unknown dataset: {name}")


# Load dataset
X, y, feats = load_dataset(name)
log.info("%s: N=%d d=%d prev=%.2f", name, X.shape[0], X.shape[1], y.mean())

# Step 1: Autotune to find the best projection/config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from symbolic_search import TheoryRadar

log.info("Autotuning (max 120s)...")
import time as _time
t0 = _time.time()
best_radar, autotune_result = TheoryRadar.autotune(X, y, feature_names=feats, max_time=120)
log.info("Autotune: %.0fs, projection=%s, F1=%.4f, formula=%s",
         _time.time() - t0, best_radar.projection,
         autotune_result.f1, autotune_result.formula[:30])

# Step 2: Run full CV with the winning config
r = run_dataset(
    name, X, y, feats,
    n_repeats=nr, n_folds=5,
    max_depth=3, beam_width=bw,
    n_subspaces=ns, subspace_k=sk,
)

# Add metadata
r["N"] = int(X.shape[0])
r["d"] = int(X.shape[1])
r["autotune_projection"] = str(best_radar.projection)
r["autotune_f1"] = float(autotune_result.f1)

with open(f"result_{name.lower()}.json", "w") as f:
    json.dump(r, f, indent=2)
log.info("Done: %s", name)
