#!/bin/bash
# Wide-beam depth 3-4 (PLS+Tucker+TurboBeam) across both GPUs.
cd /home/claude/tensor-3body
PY=/home/claude/env/bin/python

# Kill d9 runs
tmux kill-session -t gpu0-turbo-d9 2>/dev/null
tmux kill-session -t gpu1-turbo-d9 2>/dev/null
sleep 2

GPU0="HIGGS Electricity Adult Magic EEG Mammography Wilt"
GPU1="Spambase Vehicle Ionosphere QSAR German Banknote Sonar Heart"

tmux new-session -d -s "gpu0-wide-d3" "\
  export CUDA_VISIBLE_DEVICES=0; \
  for ds in $GPU0; do \
    echo \"=== \$ds wide-d3 GPU 0 ===\"; \
    $PY run_wide_d3.py \$ds 0 2>&1 | tee /tmp/\${ds}-wide-d3.log; \
    sleep 2; \
  done; \
  echo '=== GPU 0 DONE ==='; bash"

tmux new-session -d -s "gpu1-wide-d3" "\
  export CUDA_VISIBLE_DEVICES=1; \
  for ds in $GPU1; do \
    echo \"=== \$ds wide-d3 GPU 1 ===\"; \
    $PY run_wide_d3.py \$ds 1 2>&1 | tee /tmp/\${ds}-wide-d3.log; \
    sleep 2; \
  done; \
  echo '=== GPU 1 DONE ==='; bash"

echo "Wide-beam d3-4: 15 datasets, PLS+Tucker+TurboBeam 500-wide"
