#!/bin/bash
# Transient watcher: waits for baselines bash PID to exit, then launches
# FreezeAlign on GPU 0. NOTE: fires regardless of baselines success/fail,
# matching the Server A pattern. Caller is responsible for supplying a
# valid PID.
cd /workspace/STRUCTURE
BASELINES_PID="${1:?usage: _fa_waiter.sh <baselines_pid>}"
while kill -0 "$BASELINES_PID" 2>/dev/null; do
    sleep 60
done
echo "[$(date)] Baselines done (pid $BASELINES_PID exited), launching FreezeAlign on GPU 0"
bash scripts/vitl_roberta/03_train_freezealign.sh 0
