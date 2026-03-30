#!/usr/bin/env python3
"""Full Theory Radar on LARGE datasets — multiprocess fold parallelism.

4 worker processes handle folds in parallel:
- CPU work (PCA, meta-search, threshold) runs concurrently across workers
- GPU calls (beam search batch F1) serialize naturally through CuPy
- Each worker: PCA → meta → GPU search → CPU threshold → test eval

This keeps 4+ cores busy and feeds GPU with overlapping requests.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cupy as cp
import numpy as np
import logging
import time
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import stats
from joblib import Parallel, delayed

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

# ─── GPU core (imported into each worker) ──────────────────────────

def gpu_batch_f1(vals, labels):
    N, K = vals.shape
    pos = labels.sum()
    if float(pos)==0 or float(pos)==N: return cp.zeros(K, dtype=cp.float64)
    return cp.maximum(_sw(vals,labels,pos), _sw(-vals,labels,pos))

def _sw(vals, labels, pos):
    N,K = vals.shape
    idx = cp.argsort(-vals, axis=0)
    sv = cp.take_along_axis(vals,idx,axis=0); sl = labels[idx]
    tp = cp.cumsum(sl,axis=0,dtype=cp.float64)
    fp = cp.cumsum(1.0-sl,axis=0,dtype=cp.float64)
    denom = tp+pos+fp; f1 = cp.where(denom>0, 2.0*tp/denom, 0.0)
    v = cp.ones((N,K),dtype=cp.bool_); v[:-1,:] = (sv[:-1,:]-sv[1:,:])>1e-15
    return cp.max(cp.where(v,f1,0.0), axis=0)

VBIN = [
    ("+",lambda B,F:B+F),("-",lambda B,F:B-F),("*",lambda B,F:B*F),
    ("/",lambda B,F:B/(F+1e-30)),("max",lambda B,F:cp.maximum(B,F)),
    ("min",lambda B,F:cp.minimum(B,F)),("hypot",lambda B,F:cp.sqrt(B**2+F**2)),
    ("diff_sq",lambda B,F:(B-F)**2),("harmonic",lambda B,F:2*B*F/(B+F+1e-30)),
    ("geometric",lambda B,F:cp.sign(B*F)*cp.sqrt(cp.abs(B*F))),
]
VUN = [
    ("log",lambda V:cp.log(cp.abs(V)+1e-30)),("sqrt",lambda V:cp.sqrt(cp.abs(V))),
    ("sq",lambda V:V**2),("abs",lambda V:cp.abs(V)),
    ("sigmoid",lambda V:1.0/(1.0+cp.exp(-cp.clip(V,-500,500)))),
    ("tanh",lambda V:cp.tanh(cp.clip(V,-500,500))),
]
SB={"+":lambda a,b:a+b,"-":lambda a,b:a-b,"*":lambda a,b:a*b,"/":lambda a,b:a/(b+1e-30),
    "max":lambda a,b:np.maximum(a,b),"min":lambda a,b:np.minimum(a,b),
    "hypot":lambda a,b:np.sqrt(a**2+b**2),"diff_sq":lambda a,b:(a-b)**2,
    "harmonic":lambda a,b:2*a*b/(a+b+1e-30),"geometric":lambda a,b:np.sign(a*b)*np.sqrt(np.abs(a*b))}
SU={"log":lambda x:np.log(np.abs(x)+1e-30),"sqrt":lambda x:np.sqrt(np.abs(x)),
    "sq":lambda x:x**2,"abs":lambda x:np.abs(x),
    "sigmoid":lambda x:1.0/(1.0+np.exp(-np.clip(x,-500,500))),
    "tanh":lambda x:np.tanh(np.clip(x,-500,500))}

class FT:
    def __init__(s,fi): s.ops=[("l",fi)]
    def b(s,op,fi): t=FT.__new__(FT); t.ops=s.ops+[("b",op,fi)]; return t
    def u(s,op): t=FT.__new__(FT); t.ops=s.ops+[("u",op)]; return t
    def ev(s,X):
        v=None
        for o in s.ops:
            if o[0]=="l": v=X[:,o[1]].copy()
            elif o[0]=="b": v=SB[o[1]](v,X[:,o[2]]); v=np.nan_to_num(v,nan=0.,posinf=1e10,neginf=-1e10)
            elif o[0]=="u": v=SU[o[1]](v); v=np.nan_to_num(v,nan=0.,posinf=1e10,neginf=-1e10)
        return v

def opt_thresh(vals,y):
    bf,bt,bd=0.,0.,1; P=int(y.astype(bool).sum()); N=len(y)
    if P==0 or P==N: return 0.,1,0.
    for d in [1,-1]:
        o=np.argsort(d*vals)[::-1]; sv=(d*vals)[o]; sa=y.astype(bool)[o]
        tp=fp=0
        for i in range(N):
            if sa[i]: tp+=1
            else: fp+=1
            if tp+fp>0:
                p=tp/(tp+fp); r=tp/P
                if p+r>0:
                    f=2*p*r/(p+r)
                    if f>bf: bf,bt,bd=f,sv[i],d
    return bt,bd,bf


def beam_search(X_gpu, y_gpu, feat_names, feat_idx, max_depth=3, beam_width=100):
    """GPU beam search on a feature subset."""
    Xs = X_gpu[:, feat_idx]
    ln = [feat_names[i] for i in feat_idx]
    N, d = Xs.shape; yf = y_gpu.astype(cp.float64)

    f1s = gpu_batch_f1(Xs, yf)
    order = cp.argsort(-f1s).get()
    B = min(beam_width, d)
    beam_vals = cp.empty((N, B), dtype=cp.float64)
    beam_names, beam_traces = [], []
    for rank, li in enumerate(order[:B]):
        gi = feat_idx[li]
        beam_vals[:, rank] = Xs[:, li]
        beam_names.append(ln[li]); beam_traces.append(FT(gi))
    best_f1 = float(f1s[order[0]]); best_nm = beam_names[0]; best_tr = beam_traces[0]

    for depth in range(2, max_depth+1):
        Bc = beam_vals.shape[1]
        bv3 = beam_vals[:,:,None]; fv3 = Xs[:,None,:]
        chunks, nm_ch, tr_ch = [], [], []
        for on, of in VBIN:
            try:
                out = of(bv3,fv3).reshape(N,Bc*d)
                out = cp.nan_to_num(out,nan=0.,posinf=1e10,neginf=-1e10)
                chunks.append(out)
                nms, trs = [], []
                for bi in range(Bc):
                    for li in range(d):
                        nms.append(f"({beam_names[bi]} {on} {ln[li]})")
                        trs.append(beam_traces[bi].b(on, feat_idx[li]))
                nm_ch.append(nms); tr_ch.append(trs)
            except: pass
        for on, of in VUN:
            try:
                out = of(beam_vals)
                out = cp.nan_to_num(out,nan=0.,posinf=1e10,neginf=-1e10)
                chunks.append(out)
                nm_ch.append([f"{on}({beam_names[bi]})" for bi in range(Bc)])
                tr_ch.append([beam_traces[bi].u(on) for bi in range(Bc)])
            except: pass
        if not chunks: break
        av = cp.concatenate(chunks,axis=1)
        an = [n for nc in nm_ch for n in nc]
        at = [t for tc in tr_ch for t in tc]
        var = cp.var(av,axis=0)
        f1s = gpu_batch_f1(av,yf); f1s = cp.where(var<1e-20,0.0,f1s)
        tk = min(beam_width, av.shape[1])
        ti = cp.argsort(-f1s)[:tk].get()
        tf = f1s[cp.asarray(ti)].get()
        beam_vals = cp.empty((N,tk),dtype=cp.float64)
        beam_names, beam_traces = [], []
        for rank, i in enumerate(ti):
            beam_vals[:,rank] = av[:,i]
            beam_names.append(an[i]); beam_traces.append(at[i])
            if float(tf[rank]) > best_f1:
                best_f1 = float(tf[rank]); best_nm = an[i]; best_tr = at[i]
        del av, chunks
        try: del bv3, fv3
        except: pass
        cp.get_default_memory_pool().free_all_blocks()
    return best_f1, best_nm, best_tr


# ─── Single fold worker ───────────────────────────────────────────

import threading
_gpu_sem = threading.Semaphore(1)  # 1 GPU user at a time; others do CPU work

def process_fold(args):
    """Process one fold: PCA (CPU) → GPU search (semaphore) → threshold (CPU) → test F1."""
    fold_i, tr, te, X, y, feat_names, n_pca, max_depth, beam_width, n_subspaces, subspace_k = args

    d = X.shape[1]
    k = min(n_pca, d)

    # CPU: PCA
    pca = PCA(n_components=k)
    pca_tr = pca.fit_transform(X[tr])
    pca_te = pca.transform(X[te])

    X_aug_tr = np.hstack([X[tr], pca_tr])
    X_aug_te = np.hstack([X[te], pca_te])
    aug_names = feat_names + [f"pc{i}" for i in range(k)]
    d_aug = len(aug_names)

    # GPU: beam search with fuzzed subspaces (semaphore limits concurrent GPU use)
    rng = np.random.RandomState(42 + fold_i)
    best_f1, best_nm, best_tr = 0., "", None

    with _gpu_sem:
        X_gpu = cp.asarray(X_aug_tr, dtype=cp.float64)
        y_gpu = cp.asarray(y[tr], dtype=cp.float64)

        for trial in range(n_subspaces):
            sk = min(subspace_k, d_aug)
            fi = list(range(d_aug)) if d_aug <= sk else sorted(rng.choice(d_aug, sk, replace=False).tolist())
            f1, nm, tr_obj = beam_search(X_gpu, y_gpu, aug_names, fi,
                                          max_depth=max_depth, beam_width=beam_width)
            if f1 > best_f1:
                best_f1, best_nm, best_tr = f1, nm, tr_obj

        del X_gpu, y_gpu
        cp.get_default_memory_pool().free_all_blocks()

    # CPU: fair test eval
    vtr = best_tr.ev(X_aug_tr)
    vte = best_tr.ev(X_aug_te)
    th, dr, _ = opt_thresh(vtr, y[tr])
    preds = (dr * vte >= th).astype(int)
    test_f1 = f1_score(y[te], preds)

    return fold_i, best_f1, test_f1, best_nm


def run_bl(X, y, tr, te, fi):
    o = {}
    for nm, clf in [
        ("GB", GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42+fi)),
        ("RF", RandomForestClassifier(n_estimators=100, random_state=42+fi)),
        ("LR", LogisticRegression(max_iter=1000, random_state=42)),
    ]:
        clf.fit(X[tr], y[tr]); o[nm] = f1_score(y[te], clf.predict(X[te]))
    return o


def run_dataset(name, X, y, feat_names, n_repeats=50, n_folds=5,
                n_pca=8, max_depth=3, beam_width=100,
                n_subspaces=15, subspace_k=10, n_workers=4):
    total = n_repeats * n_folds
    d = X.shape[1]
    log.info("=" * 70)
    log.info("PARALLEL PIPELINE: %s (N=%d, d=%d) — %d folds, %d workers",
             name, X.shape[0], d, total, n_workers)
    log.info("  PCs=%d, depth=%d, beam=%d, subspaces=%d, k=%d",
             n_pca, max_depth, beam_width, n_subspaces, subspace_k)
    log.info("=" * 70)

    cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
    splits = list(cv.split(X, y))

    # Phase 1: Parallel fold processing (GPU search + CPU eval)
    t0 = time.time()

    # Build args for each fold
    fold_args = [
        (fold_i, tr, te, X, y, feat_names, n_pca, max_depth,
         beam_width, n_subspaces, subspace_k)
        for fold_i, (tr, te) in enumerate(splits)
    ]

    # Process folds with thread pool (shares GPU via CuPy, no pickle issues)
    from concurrent.futures import ThreadPoolExecutor
    astar_train_f1s = [0.] * total
    astar_f1s = [0.] * total
    formulas = [""] * total
    completed = 0

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(process_fold, args): args[0] for args in fold_args}
        for future in as_completed(futures):
            fold_i, train_f1, test_f1, formula = future.result()
            astar_train_f1s[fold_i] = train_f1
            astar_f1s[fold_i] = test_f1
            formulas[fold_i] = formula
            completed += 1

            if completed % 25 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed
                log.info("  %d/%d  train=%.3f test=%.3f  %.2f/s  ETA %.0fs",
                         completed, total,
                         np.mean([x for x in astar_train_f1s[:completed] if x > 0]),
                         np.mean([x for x in astar_f1s[:completed] if x > 0]),
                         rate, (total - completed) / rate)

    gpu_time = time.time() - t0
    log.info("  Search: %.0fs (%.2f folds/s)", gpu_time, total / gpu_time)

    # Phase 2: CPU baselines (parallel with joblib)
    t1 = time.time()
    nj = min(os.cpu_count() or 1, 16)
    baselines = Parallel(n_jobs=nj, backend="loky", verbose=0)(
        delayed(run_bl)(X, y, tr, te, fi) for fi, (tr, te) in enumerate(splits))
    cpu_time = time.time() - t1
    log.info("  Baselines: %.0fs", cpu_time)

    af = np.array(astar_f1s); at = np.array(astar_train_f1s)
    res = {}
    for bn in ["GB", "RF", "LR"]:
        bf = np.array([r[bn] for r in baselines])
        diff = af - bf
        ts, pv = stats.ttest_1samp(diff, 0)
        res[bn] = {"mean": float(bf.mean()), "diff": float(diff.mean()),
                    "sigma": float(abs(ts)), "dir": "A*>" if diff.mean() > 0 else f"{bn}>"}

    fc = Counter(formulas); top = fc.most_common(1)[0]
    log.info("  FORMULA: %s (%.0f%%)", top[0][:60], 100*top[1]/len(formulas))
    log.info("  TRAIN: %.4f  TEST: %.4f  gap=%.4f", at.mean(), af.mean(), at.mean()-af.mean())
    for bn in ["GB","RF","LR"]:
        b = res[bn]
        log.info("  vs %-3s: %.4f  diff=%+.4f  %.1fs  %s", bn, b["mean"], b["diff"], b["sigma"], b["dir"])

    return {"name": name, "N": X.shape[0], "d": X.shape[1],
            "test_f1": float(af.mean()), "train_f1": float(at.mean()),
            "formula": top[0], "baselines": res}


def main():
    log.info("PARALLEL FULL PIPELINE ON LARGE DATASETS")
    props = cp.cuda.runtime.getDeviceProperties(0)
    log.info("GPU: %s (%.1f GB free)", props["name"].decode(), cp.cuda.Device(0).mem_info[0]/1e9)
    _ = cp.sort(cp.random.rand(10000,100),axis=0); cp.cuda.Stream.null.synchronize()

    datasets = []
    for oml_name, ver, short in [
        ("eeg-eye-state", 1, "eeg"),
        ("MagicTelescope", 1, "magic"),
        ("electricity", 1, "elec"),
        ("MiniBooNE", 1, "miniboone"),
    ]:
        try:
            ds = fetch_openml(oml_name, version=ver, as_frame=False, parser="auto")
            X = StandardScaler().fit_transform(ds.data.astype(float))
            le = LabelEncoder(); y = le.fit_transform(ds.target)
            if len(set(y)) != 2: y = (y == y.max()).astype(int)
            datasets.append((short, X, y, [f"v{i}" for i in range(X.shape[1])]))
            log.info("Loaded %s: N=%d d=%d prev=%.2f", short, X.shape[0], X.shape[1], y.mean())
        except Exception as e:
            log.warning("%s: %s", oml_name, e)

    all_res = []
    for name, X, y, feats in datasets:
        d = X.shape[1]
        if X.shape[0] > 50000:
            nr, ns, sk, nw, bw = 20, 5, min(8, d+4), 2, 50
        elif X.shape[0] > 10000:
            nr, ns, sk, nw, bw = 50, 8, min(10, d+4), 4, 50
        else:
            nr, ns, sk, nw, bw = 100, 10, min(10, d+8), 4, 100
        r = run_dataset(name, X, y, feats, n_repeats=nr, n_folds=5,
                        n_pca=8, max_depth=3, beam_width=bw,
                        n_subspaces=ns, subspace_k=sk, n_workers=nw)
        all_res.append(r)

    with open("large_full_results.json","w") as f:
        json.dump(all_res, f, indent=2)

    log.info("\n" + "="*100)
    log.info("PARALLEL FULL PIPELINE — LARGE DATASETS")
    for r in all_res:
        log.info("%-12s N=%-6d test=%.4f  %s  GB:%.1f%s RF:%.1f%s LR:%.1f%s",
                 r["name"], r["N"], r["test_f1"], r["formula"][:25],
                 r["baselines"]["GB"]["sigma"], r["baselines"]["GB"]["dir"],
                 r["baselines"]["RF"]["sigma"], r["baselines"]["RF"]["dir"],
                 r["baselines"]["LR"]["sigma"], r["baselines"]["LR"]["dir"])


if __name__ == "__main__":
    main()
