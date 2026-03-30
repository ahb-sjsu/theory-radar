#!/usr/bin/env python3
"""Thermal-managed parallel dataset runner.

Uses batch_probe.ThermalController to monitor CPU temperature and
dynamically adjust how many dataset jobs run concurrently.

- Launches jobs one at a time
- After each launch, waits for temp to stabilize
- If under target, launches another
- If over target, waits for a job to finish before launching next
"""

import os, sys, subprocess, time, logging

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

from batch_probe._thermal import _read_cpu_temp

TARGET_TEMP = 85.0  # max acceptable CPU temp
SETTLE_TIME = 15     # seconds to wait after launch before checking temp

DATASETS = [
    # (name, args for run_one.py)
    ("BreastCancer", "BreastCancer 200 100 15 12"),
    ("Wine", "Wine 200 100 15 12"),
    ("Diabetes", "Diabetes 200 100 15 12"),
    ("Banknote", "Banknote 200 100 15 10"),
    ("Ionosphere", "Ionosphere 200 100 15 12"),
    ("EEG", "EEG 50 50 8 10"),
    ("Magic", "Magic 50 50 8 10"),
    ("Electricity", "Electricity 20 50 5 8"),
    ("MiniBooNE", "MiniBooNE 20 50 5 8"),
]


def main():
    log.info("THERMAL-MANAGED PARALLEL RUNNER")
    log.info("Target: %.0f°C, %d datasets", TARGET_TEMP, len(DATASETS))

    t0 = _read_cpu_temp()
    log.info("Baseline CPU temp: %.1f°C", t0)

    active = {}   # name -> Popen
    finished = {} # name -> result file
    queue = list(DATASETS)

    while queue or active:
        # Check for finished jobs
        done = []
        for name, proc in active.items():
            if proc.poll() is not None:
                done.append(name)
                log.info("  DONE: %s (exit %d)", name, proc.returncode)
        for name in done:
            del active[name]
            finished[name] = f"result_{name.lower()}.json"

        # Try to launch next job if temp allows
        if queue:
            temp = _read_cpu_temp()
            if temp is None:
                temp = 70.0

            if temp < TARGET_TEMP and len(active) < 4:
                name, args = queue.pop(0)
                short = name.lower()[:4]
                log.info("  LAUNCH: %s (temp=%.1f°C, %d active)",
                         name, temp, len(active))

                proc = subprocess.Popen(
                    ["python", "run_one.py"] + args.split(),
                    stdout=open(f"result_{short}.log", "w"),
                    stderr=subprocess.STDOUT,
                    cwd="/home/claude/tensor-3body",
                )
                active[name] = proc

                # Wait for temp to settle
                time.sleep(SETTLE_TIME)
                new_temp = _read_cpu_temp() or 70.0
                log.info("  After settle: %.1f°C (+%.1f)", new_temp, new_temp - temp)

                if new_temp > TARGET_TEMP:
                    log.info("  Too hot — pausing launches")
            else:
                if temp >= TARGET_TEMP:
                    log.info("  WAITING: %.1f°C > %.1f°C target, %d active",
                             temp, TARGET_TEMP, len(active))
                time.sleep(10)
        else:
            # Queue empty, wait for remaining jobs
            time.sleep(10)

        # Status every 30s
        if int(time.time()) % 30 < 2:
            temp = _read_cpu_temp() or 0
            log.info("  STATUS: %.1f°C, %d active [%s], %d queued, %d done",
                     temp, len(active), ",".join(active.keys()),
                     len(queue), len(finished))

    log.info("ALL DONE: %d datasets", len(finished))
    for name, path in finished.items():
        log.info("  %s -> %s", name, path)


if __name__ == "__main__":
    main()
