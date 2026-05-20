#!/bin/bash
set -euo pipefail

DATASETS_DIR="${1:-./datasets/tusz}"
CHECKPOINT_DIR="${2:-./ckpt/jet_tusz}"
OUTPUT_DIR="${3:-./output/eval_tusz}"

python inference.py \
  --dataset tusz \
  --datasets_dir "$DATASETS_DIR" \
  --resume "$CHECKPOINT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --num_images 0 \
  --gen_bsz 64 \
  --eval_split train \
  --eval_label_mode match_gt
