#!/bin/bash
# After the current BA queue finishes, rerun the two token BA configs
# that crashed due to features.num_workers: 4 (now patched to 0),
# then run FreezeAlign x2.
cd /workspace/STRUCTURE
BA_WAITER_PID="${1:?usage: _token_rerun_then_fa.sh <ba_waiter_pid>}"
echo "[$(date)] Waiting for BA queue (waiter pid $BA_WAITER_PID) to finish..."
while kill -0 "$BA_WAITER_PID" 2>/dev/null; do
    sleep 60
done

echo "[$(date)] BA queue done. Rerunning token_k128 on GPU 0"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python src/train_alignment.py \
    --config_path configs/ba/vitl_roberta/token_k128.yaml \
    2>&1 | tee logs/vitl_roberta_token_k128_rerun_$(date +%Y%m%d_%H%M).log

echo "[$(date)] Rerunning token_k256 on GPU 0"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python src/train_alignment.py \
    --config_path configs/ba/vitl_roberta/token_k256.yaml \
    2>&1 | tee logs/vitl_roberta_token_k256_rerun_$(date +%Y%m%d_%H%M).log

echo "[$(date)] Token reruns done. Launching FreezeAlign x2"
bash scripts/vitl_roberta/03_train_freezealign.sh 0

echo "[$(date)] All done."
