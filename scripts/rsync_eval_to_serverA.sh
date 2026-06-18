#!/bin/bash
# Rsync materials needed to run eval on a new dataset on Server A.
# Usage: bash scripts/rsync_eval_to_serverA.sh <user@serverA:/path/to/STRUCTURE>
#
# This syncs: codebase, alignment checkpoints, encoder cache.
# CSA is skipped (requires 187GB COCO training features to refit).

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <user@serverA:/path/to/STRUCTURE>"
    echo "Example: $0 shiwon@serverA:/home/shiwon/STRUCTURE"
    exit 1
fi

DEST="$1"
ROOT="/workspace/STRUCTURE"

echo "=== Syncing alignment checkpoints ==="

CKPTS=(
    # CLS Linear+STR
    "results/alignment-sentence_transformers_all_roberta_large_v1-vit_large_patch14_dinov2.lvd142m-ethereal-sunset-40/(23, 24)_0.2903/checkpoints/checkpoint-epoch402.pth"
    # CLS MLP+STR
    "results/alignment-sentence_transformers_all_roberta_large_v1-vit_large_patch14_dinov2.lvd142m-hardy-haze-42/(23, 24)_0.2903/checkpoints/checkpoint-epoch381.pth"
    # SAIL concat
    "results/alignment-sentence_transformers_all_roberta_large_v1-vit_large_patch14_dinov2.lvd142m-visionary-night-137/(23, 24)_0.2903/checkpoints/checkpoint-epoch224.pth"
    # Token FA
    "results/alignment-sentence_transformers_all_roberta_large_v1-vit_large_patch14_dinov2.lvd142m-visionary-yogurt-51/(23, 24)_0.2903/checkpoints/checkpoint-epoch205.pth"
    # Token BA K=512 CAP (tau=0.05)
    "results/alignment-sentence_transformers_all_roberta_large_v1-vit_large_patch14_dinov2.lvd142m-generous-elevator-48/(23, 24)_0.2903/checkpoints/checkpoint-epoch400.pth"
    # Token BA K=512 CAP (tau=0.02, best)
    "results/alignment-sentence_transformers_all_roberta_large_v1-vit_large_patch14_dinov2.lvd142m-playful-feather-168/(23, 24)_nan/checkpoints/checkpoint-epoch302.pth"
)

for ckpt in "${CKPTS[@]}"; do
    # Create parent dir on remote
    parent_dir=$(dirname "$ckpt")
    echo "  Syncing: $ckpt"
    rsync -avz --progress --relative "$ROOT/./$ckpt" "$DEST/"
done

echo ""
echo "=== Syncing encoder model cache (ViT-L DINOv2) ==="
rsync -avz --progress \
    /root/.cache/huggingface/hub/models--timm--vit_large_patch14_dinov2.lvd142m/ \
    "${DEST%:*}:/root/.cache/huggingface/hub/models--timm--vit_large_patch14_dinov2.lvd142m/" \
    2>/dev/null || echo "  (ViT-L cache sync failed — will auto-download on Server A)"

echo ""
echo "=== Summary ==="
echo "Synced to: $DEST"
echo ""
echo "On Server A, RoBERTa will auto-download on first run (~1.3 GB)."
echo ""
echo "CSA was NOT synced (requires 187GB COCO training features to refit)."
echo "To run CSA on Server A, you need COCO data and must re-extract training features."
echo ""
echo "To run eval on a new dataset (e.g. 'newdata'):"
echo "  PYTHONPATH=. python rerun_eval.py \\"
echo "    --config_path configs/ba/vitl_roberta/token_k512.yaml \\"
echo "    --ckpt 'results/alignment-.../checkpoint-epochN.pth' \\"
echo "    --label my_eval \\"
echo "    --zs newdata --rt newdata"
