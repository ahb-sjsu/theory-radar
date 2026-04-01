#!/bin/bash
# Launch TurboQuant depth-9 (PLS+Tucker) across both GPUs.
# Sequential per GPU to avoid OOM. batch-probe monitors temps.
cd /home/claude/tensor-3body
PY=/home/claude/env/bin/python

# Kill any old sessions
for s in gpu0-depth7 gpu1-depth7 EEG-turbo-tucker turbo-test; do
    tmux kill-session -t "$s" 2>/dev/null
done
sleep 2

# Split datasets by estimated GPU memory (large N on GPU with more VRAM headroom)
# GPU 0: large-N datasets (more memory available, no other jobs)
# GPU 1: medium-N datasets
# Ordered by estimated runtime (largest first for better bin packing)

GPU0_DATASETS="HIGGS Electricity Adult Magic EEG Mammography Wilt"
GPU1_DATASETS="Spambase Vehicle Ionosphere QSAR German Banknote Sonar Heart"

tmux new-session -d -s "gpu0-turbo-d9" "\
  export CUDA_VISIBLE_DEVICES=0; \
  for ds in $GPU0_DATASETS; do \
    echo \"=== Starting \$ds on GPU 0 (TurboBeam d9 PLS+Tucker) ===\"; \
    $PY run_depth9.py \$ds 0 2>&1 | tee /tmp/\${ds}-turbo-d9.log; \
    echo \"=== Done \$ds ===\"; \
    sleep 2; \
  done; \
  echo '=== GPU 0 TURBO D9 ALL DONE ==='; bash"
echo "GPU 0: $GPU0_DATASETS"

tmux new-session -d -s "gpu1-turbo-d9" "\
  export CUDA_VISIBLE_DEVICES=1; \
  for ds in $GPU1_DATASETS; do \
    echo \"=== Starting \$ds on GPU 1 (TurboBeam d9 PLS+Tucker) ===\"; \
    $PY run_depth9.py \$ds 1 2>&1 | tee /tmp/\${ds}-turbo-d9.log; \
    echo \"=== Done \$ds ===\"; \
    sleep 2; \
  done; \
  echo '=== GPU 1 TURBO D9 ALL DONE ==='; bash"
echo "GPU 1: $GPU1_DATASETS"

echo ""
echo "15 datasets queued across 2 GPUs (PLS+Tucker, TurboBeam 3-bit, depth 9)"
echo "Monitor: tmux list-sessions"
echo "Logs:    tail -f /tmp/*-turbo-d9.log"
echo "Temps:   sensors | grep Package"
