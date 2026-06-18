#!/bin/bash
# Rsync materials needed to run eval on a new dataset on Server A.
# Usage: bash scripts/rsync_eval_to_serverA.sh <user@serverA:/path/to/STRUCTURE>
#
# This syncs: codebase, alignment checkpoints, encoder cache.
# CSA is skipped (requires 187GB COCO training features to refit).

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <user@serverA:/path/to/STRUCTURE>"
    echo "Example: $0 shiwon@61.252.54.79:/home/shiwon/STRUCTURE"
    exit 1
fi

DEST="$1"
ROOT="/home/shiwon/structure"
SSH_PORT=30033
RSYNC_RSH="ssh -p ${SSH_PORT}"

echo "=== Syncing alignment checkpoints ==="

CKPTS=(
    # Token BA K=512 Mean Pool
    "results/alignment-sentence_transformers_all_roberta_large_v1-vit_large_patch14_dinov2.lvd142m-cosmic-butterfly-158/(23, 24)_0.2873/checkpoints/checkpoint-epoch315.pth"
    # CLS BA K=512
    "results/alignment-sentence_transformers_all_roberta_large_v1-vit_large_patch14_dinov2.lvd142m-scarlet-galaxy-31/(23, 24)_0.2903/checkpoints/checkpoint-epoch407.pth"
    # Token BA K=512 CAP tau=0.01
    "results/alignment-sentence_transformers_all_roberta_large_v1-vit_large_patch14_dinov2.lvd142m-hopeful-durian-167/(23, 24)_nan/checkpoints/checkpoint-epoch283.pth"
    # Token BA K=512 CAP tau=0.1
    "results/alignment-sentence_transformers_all_roberta_large_v1-vit_large_patch14_dinov2.lvd142m-devout-disco-169/(23, 24)_nan/checkpoints/checkpoint-epoch404.pth"
    # Token BA K=512 CAP tau=0.2
    "results/alignment-sentence_transformers_all_roberta_large_v1-vit_large_patch14_dinov2.lvd142m-frosty-capybara-174/(23, 24)_nan/checkpoints/checkpoint-epoch325.pth"
    # Token BA K=512 CAP tau=0.03 + 80K LAION
    "results/alignment-sentence_transformers_all_roberta_large_v1-vit_large_patch14_dinov2.lvd142m-leafy-planet-191/(23, 24)_nan/checkpoints/checkpoint-epoch487.pth"
    # Token BA K=512 CAP tau=0.03 + 30K LAION
    "results/alignment-sentence_transformers_all_roberta_large_v1-vit_large_patch14_dinov2.lvd142m-pious-violet-198/(23, 24)_nan/checkpoints/checkpoint-epoch400.pth"
    # Token BA K=1024 CAP tau=0.05
    "results/alignment-sentence_transformers_all_roberta_large_v1-vit_large_patch14_dinov2.lvd142m-glamorous-disco-105/(23, 24)_0.2903/checkpoints/checkpoint-epoch403.pth"
)

for ckpt in "${CKPTS[@]}"; do
    echo "  Syncing: $ckpt"
    rsync -avz --progress -e "$RSYNC_RSH" --relative "$ROOT/./$ckpt" "$DEST/"
done

echo ""
echo "=== Syncing encoder model cache (ViT-L DINOv2) ==="
rsync -avz --progress -e "$RSYNC_RSH" \
    /root/.cache/huggingface/hub/models--timm--vit_large_patch14_dinov2.lvd142m/ \
    "${DEST%:*}:/root/.cache/huggingface/hub/models--timm--vit_large_patch14_dinov2.lvd142m/" \
    2>/dev/null || echo "  (ViT-L cache sync failed — will auto-download on Server A)"

echo ""
echo "=== Summary ==="
echo "Synced to: $DEST"
echo "SSH port: $SSH_PORT"
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