#!/usr/bin/env bash
set -euo pipefail
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate structure

GPU="${1:-1}"
CKPT='results/alignment-sentence_transformers_all_MiniLM_L6_v2-vit_small_patch14_dinov2.lvd142m-glowing-lake-14/(11, 6)_0.2739/checkpoints/checkpoint-epoch551.pth'
CONFIG="configs/freezealign/vits_minilm/fa_d512.yaml"

echo "=== WITH patch L2 norm (current) ==="
PYTHONPATH=. python src/evaluation/zero_shot_segmentation.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --layer-img 11 --layer-txt 6 \
    --dataset voc2012 --data-root data/pascal_voc \
    --methods freezealign \
    --text-strategies ensemble \
    --gpu "$GPU" 2>&1 | grep -E "^(Method|freezealign|----)"

echo ""
echo "=== WITHOUT patch L2 norm (match original FA) ==="
PYTHONPATH=. SEG_NO_PATCH_NORM=1 python src/evaluation/zero_shot_segmentation.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --layer-img 11 --layer-txt 6 \
    --dataset voc2012 --data-root data/pascal_voc \
    --methods freezealign \
    --text-strategies ensemble \
    --gpu "$GPU" 2>&1 | grep -E "^(Method|freezealign|----)"
