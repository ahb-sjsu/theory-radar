#!/usr/bin/env python3
"""Launch all 9 datasets in parallel subprocesses.
Each subprocess runs the full pipeline on one dataset.
Small datasets share CPU; large datasets share GPU via CUDA driver serialization.
"""

import time

DATASETS = [
    # (name, script_args)
    ("wine", "Wine,sklearn_wine,200,100,15,12"),
    ("diab", "Diabetes,diabetes,200,100,15,12"),
    ("bank", "Banknote,banknote-authentication,200,100,15,10"),
    ("iono", "Ionosphere,ionosphere,200,100,15,12"),
    ("eeg", "EEG,eeg-eye-state,50,50,8,10"),
    ("magic", "Magic,MagicTelescope,50,50,8,10"),
    ("elec", "Electricity,electricity,20,50,5,8"),
    ("mini", "MiniBooNE,MiniBooNE,20,50,5,8"),
]

# Write the single-dataset runner
RUNNER = r'''#!/usr/bin/env python3
"""Run full pipeline on a single dataset."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, os.path.dirname(__file__))

with open(os.path.join(os.path.dirname(__file__), "run_full_pipeline.py")) as f:
    code = f.read().split("def main")[0]
exec(code)

import logging, json
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

from sklearn.datasets import load_wine, fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder

name, source, nr, bw, ns, sk = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])

if source == "sklearn_wine":
    wine = load_wine()
    X = StandardScaler().fit_transform(wine.data)
    y = (wine.target == 0).astype(int)
    feats = [f"w{i}" for i in range(13)]
else:
    ds = fetch_openml(source, version=1, as_frame=False, parser="auto")
    X = StandardScaler().fit_transform(ds.data.astype(float))
    # Binary encode target
    try:
        y = LabelEncoder().fit_transform(ds.target)
    except:
        y = ds.target.astype(int)
    if len(set(y)) != 2:
        y = (y == y.max()).astype(int)
    # Known targets
    if source == "diabetes":
        y = (ds.target == "tested_positive").astype(int)
    elif source == "ionosphere":
        y = (ds.target == "g").astype(int)
    elif source == "banknote-authentication":
        yr = ds.target.astype(int); y = (yr == yr.max()).astype(int)
    feats = [f"v{i}" for i in range(X.shape[1])]

log.info("Running %s: N=%d d=%d nr=%d bw=%d ns=%d sk=%d", name, X.shape[0], X.shape[1], nr, bw, ns, sk)

r = run_dataset(name, X, y, feats, n_repeats=nr, n_folds=5,
                n_pca=8, max_depth=3, beam_width=bw,
                n_subspaces=ns, subspace_k=sk)

with open(f"result_{name.lower()}.json", "w") as f:
    json.dump(r, f, indent=2)
log.info("Saved result_%s.json", name.lower())
'''

with open("/tmp/run_one_dataset.py", "w") as f:
    f.write(RUNNER)

# Upload runner
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("100.68.134.21", username="claude", password="roZes9090!~")

sftp = ssh.open_sftp()
sftp.put("/tmp/run_one_dataset.py", "/home/claude/tensor-3body/run_one_dataset.py")
sftp.close()

# Kill the sequential run
ssh.exec_command("tmux kill-session -t all 2>/dev/null")
time.sleep(1)

# Launch all 8 datasets (BreastCancer is already in all_full.log from earlier runs)
# Also launch BreastCancer fresh
ALL = [("bc", "BreastCancer,sklearn_bc,200,100,15,12")] + DATASETS

for short, args in ALL:
    name_arg = args.split(",")
    cmd = f'tmux new-session -d -s {short} "cd /home/claude/tensor-3body && source /home/claude/env/bin/activate && python run_one_dataset.py {" ".join(name_arg)} 2>&1 | tee result_{short}.log"'
    ssh.exec_command(cmd)
    time.sleep(0.5)
    print(f"Launched {short}: {name_arg[0]}")

time.sleep(5)

# Check
stdin, stdout, stderr = ssh.exec_command("tmux list-sessions 2>/dev/null")
print("\nSessions:", stdout.read().decode().strip())

stdin, stdout, stderr = ssh.exec_command("sensors 2>/dev/null | grep Package")
print("CPU:", stdout.read().decode().strip())

stdin, stdout, stderr = ssh.exec_command(
    "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader"
)
print("GPU:", stdout.read().decode().strip())

stdin, stdout, stderr = ssh.exec_command("ps aux --sort=-%cpu | head -15")
print(stdout.read().decode())

ssh.close()
