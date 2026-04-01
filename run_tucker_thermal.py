#!/usr/bin/env python3
"""Tucker Formula Search — launched via ThermalJobManager.

Runs Tucker experiments as managed subprocesses, throttled by CPU temp.
Waits for the thermal manager's other jobs to finish first.
"""

import os
import sys
import logging
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

# Update batch-probe to v0.4.0 thermal features
sys.path.insert(0, "/home/claude/env/lib/python3.12/site-packages")
from batch_probe import ThermalJobManager

# Wait for other experiments to finish (thermal manager's jobs)
from batch_probe._thermal import _read_cpu_temp

log.info("TUCKER FORMULA SEARCH — THERMAL MANAGED")
log.info("Waiting for CPU to cool below 80C before starting...")

while True:
    t = _read_cpu_temp()
    if t and t < 80.0:
        log.info("CPU at %.1fC — starting Tucker jobs", t)
        break
    log.info("CPU at %.1fC — waiting...", t or 0)
    time.sleep(15)

# Launch Tucker experiments via ThermalJobManager
jobs = [
    ("tucker_bc", ["python", "run_tucker_formula.py"]),
]

mgr = ThermalJobManager(
    target_temp=85.0,
    max_concurrent=1,  # one at a time — Tucker is heavy
    settle_time=15.0,
    cooldown_margin=5.0,
)

results = mgr.run(jobs, cwd="/home/claude/tensor-3body")
log.info("Results: %s", results)
