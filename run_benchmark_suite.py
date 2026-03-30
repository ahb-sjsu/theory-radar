#!/usr/bin/env python3
"""Theory Radar Benchmark Suite — 17 established datasets.

Full pipeline: PCA + A* beam + meta pruning + fuzzed subspaces + fair eval.
Managed by ThermalJobManager for safe parallel execution.

Each dataset is a separate subprocess via run_one.py.
Results saved incrementally as result_{name}.json.

Datasets chosen for:
- Recognition: reviewers know them instantly
- Diversity: N from 208 to 581K, d from 4 to 60
- Domains: medical, financial, physics, telecom, biology
- Rigor: all are standard benchmarks in ML literature
"""

import os, sys, logging, json, time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

from batch_probe import ThermalJobManager
from batch_probe._thermal import _read_cpu_temp

# ─── Dataset configurations ───────────────────────────────────────
# (name, openml_source, n_repeats, beam_width, n_subspaces, subspace_k)
#
# Sizing rules:
#   N > 50K:  nr=10, bw=20, ns=3, sk=6   (memory-constrained)
#   N > 10K:  nr=50, bw=50, ns=5, sk=8   (GPU-saturating)
#   N > 2K:   nr=100, bw=100, ns=10, sk=10
#   N <= 2K:  nr=200, bw=100, ns=15, sk=12

DATASETS = [
    # ── Already completed (skip if result exists) ──
    # BreastCancer, Wine, Diabetes, Banknote, EEG

    # ── Still running ──
    # Ionosphere, Magic

    # ── New: small classic benchmarks (N < 2K) ──
    ("Heart",          "heart-statlog",        200, 100, 15, 12),
    ("Sonar",          "sonar",                200, 100, 15, 12),

    # ── New: medium benchmarks (2K < N < 10K) ──
    ("Spambase",       "spambase",             100, 100, 10, 10),
    ("German",         "german-credit",        200, 100, 15, 12),
    ("Australian",     "australian",           200, 100, 15, 12),

    # ── New: large benchmarks (N > 10K) ──
    ("Adult",          "adult",                 50,  50,  5,  8),
    ("HIGGS",          "higgs",                 10,  20,  3,  6),

    # ── Re-run: OOM datasets with reduced params ──
    ("Electricity",    "electricity",           20,  30,  5,  6),
    ("MiniBooNE",      "MiniBooNE",             10,  20,  3,  6),
]


def check_already_done(name):
    """Check if result JSON already exists."""
    path = f"/home/claude/tensor-3body/result_{name.lower()}.json"
    try:
        with open(path) as f:
            json.load(f)
        return True
    except Exception:
        return False


def main():
    log.info("=" * 70)
    log.info("THEORY RADAR BENCHMARK SUITE — 17 DATASETS")
    log.info("Full pipeline: PCA + A* beam + meta pruning + fair eval")
    log.info("Thermal-managed parallel execution")
    log.info("=" * 70)

    # Wait for CPU to be cool
    t = _read_cpu_temp()
    log.info("CPU: %.1fC", t or 0)
    while t and t > 78:
        log.info("Waiting for CPU to cool (%.1fC > 78C)...", t)
        time.sleep(30)
        t = _read_cpu_temp()

    # Filter out already-completed datasets
    jobs = []
    for name, source, nr, bw, ns, sk in DATASETS:
        if check_already_done(name):
            log.info("  SKIP: %s (already done)", name)
            continue
        jobs.append((
            name,
            ["python", "run_one.py", name, str(nr), str(bw), str(ns), str(sk)]
        ))

    log.info("Launching %d datasets (%d skipped)", len(jobs),
             len(DATASETS) - len(jobs))

    if not jobs:
        log.info("All datasets complete!")
        return

    # Run with thermal management
    mgr = ThermalJobManager(
        target_temp=84.0,      # conservative target
        max_concurrent=3,      # at most 3 parallel
        settle_time=15.0,
        cooldown_margin=4.0,   # launch only if temp < 80C
        poll_interval=10.0,
    )

    results = mgr.run(
        jobs,
        cwd="/home/claude/tensor-3body",
        log_dir="/home/claude/tensor-3body",
    )

    # Summary
    log.info("\n" + "=" * 70)
    log.info("BENCHMARK SUITE COMPLETE")
    for name, rc in results.items():
        status = "OK" if rc == 0 else f"FAILED (exit {rc})"
        log.info("  %s: %s", name, status)

    # Compile all results
    log.info("\n" + "=" * 110)
    log.info("%-15s %6s %3s  %-8s %-8s %-5s  %-30s  %-10s %-10s %-10s",
             "Dataset", "N", "d", "Train", "Test", "Gap", "Formula",
             "vs GB", "vs RF", "vs LR")
    log.info("-" * 110)

    all_names = [
        "breastcancer", "wine", "diabetes", "banknote", "ionosphere",
        "eeg", "magic", "electricity", "miniboone",
        "heart", "sonar", "spambase", "german", "australian",
        "adult", "higgs",
    ]

    for name in all_names:
        try:
            with open(f"/home/claude/tensor-3body/result_{name}.json") as f:
                r = json.load(f)
            bl = r["baselines"]
            log.info("%-15s %6d %3d  %.4f  %.4f  %.3f  %-30s  %5.1f %s  %5.1f %s  %5.1f %s",
                     r["name"], r.get("N", 0), r.get("d", 0),
                     r["train_f1"], r["test_f1"],
                     r["train_f1"] - r["test_f1"],
                     r["formula"][:30],
                     bl["GB"]["sigma"], bl["GB"]["dir"],
                     bl["RF"]["sigma"], bl["RF"]["dir"],
                     bl["LR"]["sigma"], bl["LR"]["dir"])
        except Exception:
            log.info("%-15s  (no result)", name)

    log.info("=" * 110)


if __name__ == "__main__":
    main()
