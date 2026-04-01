import numpy as np
import scipy.sparse
from sklearn.datasets import load_breast_cancer, load_wine, fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_dataset(name):
    if name == "BreastCancer":
        bc = load_breast_cancer()
        X = StandardScaler().fit_transform(bc.data)
        return X, bc.target, [f"f{i}" for i in range(X.shape[1])]

    if name == "Wine":
        wine = load_wine()
        X = StandardScaler().fit_transform(wine.data)
        y = (wine.target == 0).astype(int)
        return X, y, [f"w{i}" for i in range(13)]

    OPENML = {
        "Diabetes": ("diabetes", 1, lambda t: (t == "tested_positive").astype(int)),
        "Banknote": ("banknote-authentication", 1, None),
        "Ionosphere": ("ionosphere", 1, lambda t: (t == "g").astype(int)),
        "EEG": ("eeg-eye-state", 1, None),
        "Magic": ("MagicTelescope", 1, None),
        "Electricity": ("electricity", 1, None),
        "Heart": ("heart-statlog", 1, None),
        "Sonar": ("sonar", 1, None),
        "Spambase": ("spambase", 1, None),
        "Australian": ("australian", 1, None),
        "Mammography": ("mammography", 1, None),
        "QSAR": ("qsar-biodeg", 1, None),
        "Wilt": ("wilt", 2, None),
        "Vehicle": ("vehicle", 1, None),
        "German": ("credit-g", 1, None),
        "Adult": ("adult", 1, None),
        "HIGGS": ("higgs", 1, None),
        # New candidates — medical/clinical, moderate N, likely linear structure
        "Parkinsons": ("parkinsons", 1, None),
        "Haberman": ("haberman", 1, None),
        "Liver": ("ilpd", 1, None),
        "Thyroid": ("sick", 1, None),
        "Transfusion": ("blood-transfusion-service-center", 1, None),
        "Fertility": ("fertility", 1, None),
        "Hepatitis": ("hepatitis", 1, None),
        "Spectf": ("spect", 1, None),
        "Vertebral": ("vertebra-column", 2, None),
        "Dermatology": ("dermatology", 1, None),
        "Cylinder": ("cylinder-bands", 1, None),
    }

    if name not in OPENML:
        raise ValueError(f"Unknown dataset: {name}")

    oml_name, ver, target_fn = OPENML[name]
    ds = fetch_openml(oml_name, version=ver, as_frame=True, parser="auto")

    # Separate numeric and categorical columns
    from sklearn.preprocessing import OrdinalEncoder
    df = ds.data if hasattr(ds.data, 'dtypes') else ds.frame.drop(columns=[ds.target_names[0]] if hasattr(ds, 'target_names') else [])

    if hasattr(df, 'dtypes'):
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
        parts = []
        if num_cols:
            parts.append(np.array(df[num_cols], dtype=float))
        if cat_cols:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            parts.append(enc.fit_transform(df[cat_cols]).astype(float))
        X = np.hstack(parts) if len(parts) > 1 else parts[0]
    else:
        if scipy.sparse.issparse(ds.data):
            X = ds.data.toarray().astype(float)
        else:
            X = np.array(ds.data, dtype=float)

    X = np.nan_to_num(X, nan=0.0)
    feat_names = [f"v{i}" for i in range(X.shape[1])]
    X = StandardScaler().fit_transform(X)
    if target_fn is not None:
        t = ds.target.values if hasattr(ds.target, "values") else ds.target
        y = target_fn(t)
    else:
        le = LabelEncoder()
        t = ds.target.values if hasattr(ds.target, "values") else ds.target
        y = le.fit_transform(t)
        if len(set(y)) != 2:
            y = (y == y.max()).astype(int)
    return X, y.astype(int), feat_names
