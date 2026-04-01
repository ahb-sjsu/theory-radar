#!/bin/bash
# Run Theory Radar depth-3 PLS on new candidate datasets.
# These are moderate-N medical/clinical datasets likely to favor formulas.
cd /home/claude/tensor-3body
PY=/home/claude/env/bin/python

GPU0="Parkinsons Haberman Liver Thyroid Transfusion Fertility"
GPU1="Hepatitis Spectf Vertebral Dermatology Cylinder"

tmux new-session -d -s "gpu0-new" "\
  export CUDA_VISIBLE_DEVICES=0; \
  for ds in $GPU0; do \
    echo \"=== \$ds depth-3 PLS GPU 0 ===\"; \
    $PY run_depth7.py \$ds 0 2>&1 | tee /tmp/\${ds}-d3.log; \
    sleep 2; \
  done; \
  echo '=== GPU 0 DONE ==='; bash"

tmux new-session -d -s "gpu1-new" "\
  export CUDA_VISIBLE_DEVICES=1; \
  for ds in $GPU1; do \
    echo \"=== \$ds depth-3 PLS GPU 1 ===\"; \
    $PY run_depth7.py \$ds 1 2>&1 | tee /tmp/\${ds}-d3.log; \
    sleep 2; \
  done; \
  echo '=== GPU 1 DONE ==='; bash"

echo "11 new datasets queued across 2 GPUs (depth-3 PLS)"
echo "Monitor: tmux list-sessions"
