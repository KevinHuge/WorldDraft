#!/usr/bin/env bash
# Download the WorldDraft benchmark data from Hugging Face into data/ (or $WORLDDRAFT_DATA):
#   gt/      ground truth (visible-instance masks, projected entity boxes, event and camera GT)
#   drafts/  the 432 draft videos with their preprocessing metadata and target prompts
#   scenes/  the 432 Blender scenes (optional, ~0.6 GB; only needed to re-render or regenerate GT)
# Usage: bash eval/download_data.sh [--with-scenes]
set -euo pipefail
DATASET=${WORLDDRAFT_DATASET:-KevinHuge/WorldDraft}
DEST=${WORLDDRAFT_DATA:-$(cd "$(dirname "$0")/.." && pwd)/data}
PATTERNS=("gt/*" "drafts/*")
[ "${1:-}" = "--with-scenes" ] && PATTERNS+=("scenes/*")
hf download "$DATASET" --repo-type dataset --local-dir "$DEST" --include "${PATTERNS[@]}"
echo "Benchmark data in $DEST"
