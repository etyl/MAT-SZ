#!/bin/bash
set -euo pipefail

# Submit the production H100 training configuration under torch.profiler.
# Run this script directly (do not pass it to sbatch):
#
#   ./profile_train_h100.sh
#   ./profile_train_h100.sh --profile 5
#
# The Slurm log contains the operator table and trace.json contains the CPU/CUDA
# timeline for https://ui.perfetto.dev/. Run one profiling job at a time because
# trace.json has a fixed name.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"

exec sbatch --job-name=gnn-prof train_h100.sh \
    --profile 1 \
    --wandb-mode disabled \
    "$@"
