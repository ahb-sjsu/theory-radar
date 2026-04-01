#!/bin/bash
# Launch depth-7 pilots — one dataset at a time per GPU.
# Each runs sequentially in its GPU's tmux session.
cd /home/claude/tensor-3body
PY=/home/claude/env/bin/python

# Kill any old pilots
tmux kill-session -t gpu0-depth7 2>/dev/null
tmux kill-session -t gpu1-depth7 2>/dev/null
sleep 1

# GPU 0: sequential queue of 7 datasets
tmux new-session -d -s "gpu0-depth7" "\
  for ds in EEG Magic Electricity HIGGS Adult Sonar Heart; do \
    echo '=== Starting \$ds on GPU 0 ===' ; \
    $PY run_depth7.py \$ds 0 2>&1 | tee /tmp/\${ds}-depth7.log ; \
    echo '=== Done \$ds ===' ; \
  done ; \
  echo '=== GPU 0 ALL DONE ===' ; bash"
echo "GPU 0 queue launched: EEG Magic Electricity HIGGS Adult Sonar Heart"

# GPU 1: sequential queue of 7 datasets
tmux new-session -d -s "gpu1-depth7" "\
  for ds in Spambase Vehicle Ionosphere QSAR Wilt German Banknote; do \
    echo '=== Starting \$ds on GPU 1 ===' ; \
    $PY run_depth7.py \$ds 1 2>&1 | tee /tmp/\${ds}-depth7.log ; \
    echo '=== Done \$ds ===' ; \
  done ; \
  echo '=== GPU 1 ALL DONE ===' ; bash"
echo "GPU 1 queue launched: Spambase Vehicle Ionosphere QSAR Wilt German Banknote"

echo ""
echo "Monitor: tmux list-sessions"
echo "Logs:    tail -f /tmp/*-depth7.log"
