#!/bin/bash
# Wait for the BA-after-CSA waiter to finish (which itself waits for
# csa_d512 → kills baselines → runs full BA queue), then run
# FreezeAlign x2 and csa_d512_struct on the same GPU.
cd /workspace/STRUCTURE
BA_WAITER_PID="${1:?usage: _fa_csa_after_ba_waiter.sh <ba_waiter_pid>}"
echo "[$(date)] Waiting for BA chain (waiter pid $BA_WAITER_PID) to finish..."
while kill -0 "$BA_WAITER_PID" 2>/dev/null; do
    sleep 60
done
echo "[$(date)] BA chain done. Launching FreezeAlign x2"
bash scripts/vitl_roberta/03_train_freezealign.sh 0
echo "[$(date)] FreezeAlign done. Launching csa_d512_struct"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python src/train_alignment.py \
    --config_path configs/csa/vitl_roberta/csa_d512_struct.yaml \
    2>&1 | tee logs/vitl_roberta_csa_struct_standalone_$(date +%Y%m%d_%H%M).log
echo "[$(date)] All done."
