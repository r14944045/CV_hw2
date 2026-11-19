#!/bin/bash
# Run a command once one of N GPU slots is free (default 3), so parallel experiments don't oversubscribe the GPU.
#   tools/dev/gpu_slot.sh python evaluate.py ...        (N via GPU_SLOTS)
N=${GPU_SLOTS:-3}
LOCKDIR=${GPU_SLOT_DIR:-/tmp/hw2_gpu_slots}
mkdir -p "$LOCKDIR"
while true; do
  for i in $(seq 0 $((N - 1))); do
    exec 9>"$LOCKDIR/slot$i.lock"
    if flock -n 9; then
      "$@"; rc=$?
      flock -u 9; exit $rc
    fi
    exec 9>&-
  done
  sleep 5
done
