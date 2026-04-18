#!/bin/bash
# Wait for csa_d512 python to finish, kill baselines bash
# (prevents csa_d512_struct from starting), then launch BA.
cd /workspace/STRUCTURE
CSA_PID="${1:?usage: _ba_after_csa_waiter.sh <csa_pid> <baselines_bash_pid>}"
BASELINES_PID="${2:?}"
echo "[$(date)] Waiting for csa_d512 (pid $CSA_PID) to finish..."
while kill -0 "$CSA_PID" 2>/dev/null; do
    sleep 15
done
echo "[$(date)] csa_d512 done. Killing baselines bash $BASELINES_PID to skip csa_d512_struct"
kill "$BASELINES_PID" 2>/dev/null
sleep 2
echo "[$(date)] Launching BA on GPU 0"
bash scripts/vitl_roberta/02_train_ba.sh 0
