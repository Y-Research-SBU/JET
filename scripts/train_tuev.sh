#!/bin/bash
set -euo pipefail

DATASETS_DIR="${1:-./datasets/tuev}"
OUTPUT_DIR="${2:-./output/tuev}"

python train_eeg.py \
  --dataset tuev \
  --datasets_dir "$DATASETS_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --model JiT-B/16 \
  --num_eeg_channels 16 \
  --target_length 1000 \
  --eeg_patch_size 200 \
  --batch_size 256 \
  --epochs 200 \
  --blr 5e-5 \
  --loss_type mix \
  --loss_weight_stat 1.0 \
  --loss_weight_tv 0.1 \
  --loss_weight_corr 0.1
