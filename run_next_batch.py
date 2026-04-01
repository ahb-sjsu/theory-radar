#!/usr/bin/env python3
"""Next batch: re-run OOM datasets + new large datasets.
Uses ThermalJobManager for temperature-safe parallel execution.
Reduced beam widths for large N to avoid GPU OOM.
"""

import os
import logging
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

from batch_probe import ThermalJobManager
from batch_probe._thermal import _read_cpu_temp

# Wait for current thermal manager jobs to finish
log.info("Waiting for current jobs to finish (CPU < 76C)...")
while True:
    t = _read_cpu_temp()
    if t and t < 76.0:
        break
    log.info("  CPU: %.1fC", t or 0)
    time.sleep(30)

log.info("CPU cool (%.1fC). Launching next batch.", _read_cpu_temp())

jobs = [
    # Re-run OOM datasets with smaller beams
    (
        "Electricity",
        [
            "python",
            "run_one.py",
            "Electricity",
            "20",
            "30",
            "5",
            "6",  # nr=20, bw=30, ns=5, sk=6
        ],
    ),
    (
        "MiniBooNE",
        [
            "python",
            "run_one.py",
            "MiniBooNE",
            "10",
            "20",
            "3",
            "6",  # nr=10, bw=20, ns=3, sk=6
        ],
    ),
    # Tucker experiment on BreastCancer
    ("Tucker", ["python", "run_tucker_formula.py"]),
]

mgr = ThermalJobManager(
    target_temp=85.0,
    max_concurrent=2,
    settle_time=15.0,
    cooldown_margin=5.0,
)

log.info("Launching %d jobs with ThermalJobManager", len(jobs))
results = mgr.run(jobs, cwd="/home/claude/tensor-3body")

log.info("All done:")
for name, rc in results.items():
    log.info("  %s: exit %d", name, rc)
