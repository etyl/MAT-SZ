#!/usr/bin/env bash
# Train the GNN predictor. Usage: ./train.sh /path/to/images [extra args...]
# Picks CUDA automatically if available (override with --device cpu).
set -euo pipefail

shift || true

python scripts/train_gnn.py \
    --data /data/parietal/store/data/div2k \
    --out data/gnn_predictor.pt \
    --steps 6000 \
    --batch 8 \
    --crop 256 \
    --d 32 \
    "$@"

# checkpoint  -> data/gnn_predictor.pt
# loss curve  -> data/gnn_predictor.csv (+ .png if matplotlib is installed)
