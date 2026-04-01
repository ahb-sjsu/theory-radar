#!/bin/bash
cd /home/claude/tensor-3body
PY=/home/claude/env/bin/python
tmux kill-server 2>/dev/null
sleep 1

GPU0="Haberman Liver Transfusion Hepatitis Fertility Parkinsons"
GPU1="Spectf Vertebral Dermatology Cylinder Thyroid"

tmux new-session -d -s "gpu0-unified" "\
  export CUDA_VISIBLE_DEVICES=0; \
  for ds in $GPU0; do \
    echo \"=== \$ds unified 200x5 CV ===\"; \
    $PY run_unified.py \$ds 0 2>&1 | tee /tmp/\${ds}-unified.log; \
    sleep 2; \
  done; \
  echo '=== GPU 0 DONE ==='; bash"

tmux new-session -d -s "gpu1-unified" "\
  export CUDA_VISIBLE_DEVICES=1; \
  for ds in $GPU1; do \
    echo \"=== \$ds unified 200x5 CV ===\"; \
    $PY run_unified.py \$ds 1 2>&1 | tee /tmp/\${ds}-unified.log; \
    sleep 2; \
  done; \
  echo '=== GPU 1 DONE ==='; bash"

echo "11 datasets, 200x5 CV vs GB/RF/LR, unified protocol"
