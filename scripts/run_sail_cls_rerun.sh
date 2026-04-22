#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate structure

GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

cd /home/shiwon/STRUCTURE

CONFIGS=(
    "01_sail_cls_vits         configs/sail/vits_minilm/sail_star_cls_cliploss.yaml"
    "02_sail_cls_str_vits     configs/sail/vits_minilm/sail_star_cls_struct_cliploss.yaml"
    "03_sail_cls_vitb         configs/sail/vitb_mpnet/sail_star_cls_cliploss.yaml"
    "04_sail_cls_str_vitb     configs/sail/vitb_mpnet/sail_star_cls_struct_cliploss.yaml"
)

for entry in "${CONFIGS[@]}"; do
    name=$(echo "$entry" | awk '{print $1}')
    config=$(echo "$entry" | awk '{print $2}')
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] START $name ($config)"
    PYTHONPATH=. python src/train_alignment.py --config_path "$config" || {
        echo "[$(date +%Y-%m-%d\ %H:%M:%S)] FAILED $name exit=$?"
        continue
    }
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] DONE  $name exit=0"
done

echo "[$(date +%Y-%m-%d\ %H:%M:%S)] ALL DONE"
