#!/bin/bash
# Transient watcher: waits for the FA waiter chain to finish
# (baselines → FreezeAlign), then launches BA on GPU 1.
# Sequential execution required: torch.load(mmap=True) uses MAP_PRIVATE
# which reserves full file size against Committed_AS, and the ~87 GB
# all-layer CLS+avg caches loaded by each CLS training run would blow
# Server B's CommitLimit (~135 GB) if two processes ran concurrently.
cd /workspace/STRUCTURE
FA_WAITER_PID="${1:?usage: _ba_after_fa_waiter.sh <fa_waiter_pid>}"
while kill -0 "$FA_WAITER_PID" 2>/dev/null; do
    sleep 60
done
echo "[$(date)] FA chain done (fa waiter $FA_WAITER_PID exited), launching BA on GPU 1"
bash scripts/vitl_roberta/02_train_ba.sh 1
