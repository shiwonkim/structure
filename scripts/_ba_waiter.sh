#!/bin/bash
# Transient watcher: waits for baselines training to start epoch 1
# (meaning extraction + layer selection finished), then launches BA
# on GPU 1 which reuses the feature cache.
# Watches only the specific log file passed in to avoid stale-log hits.
cd /workspace/STRUCTURE
LOG_FILE="${1:-logs/vitl_roberta_baselines_20260415_1612.log}"
# Trigger: baselines tqdm prints "Train loss: X, Val loss: Y: N/1000"
# on every epoch update. Match "Train loss:" + ".*/1000" to confirm
# training has started — extraction is done, cache is stable, safe for
# GPU 1 to read.
while ! grep -q "Train loss:.*/1000" "$LOG_FILE" 2>/dev/null; do
    sleep 30
done
echo "[$(date)] Extraction done, launching BA on GPU 1"
bash scripts/vitl_roberta/02_train_ba.sh 1
