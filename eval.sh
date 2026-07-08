#!/bin/bash
#SBATCH --job-name=gnn-eval
#SBATCH --qos=qos_gpu-dev
#SBATCH --time=1:00:00
#SBATCH --partition=gpu_p13
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --account=lzs@v100
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# Usage:
#   ./eval.sh                              # defaults below
#   ./eval.sh --data /path/to/kodak --gnn-checkpoint data/gnn_predictor.pt
#   sbatch eval.sh --eb 1 2 4 8 --csv eval.csv
# Extra flags after the defaults ("$@") override them.

module purge
module load pytorch-gpu

DATA=${DATA:-/lustre/fswork/projects/rech/lzs/uhq13gg/data/kodak}
CKPT=${CKPT:-data/gnn_predictor.pt}

python scripts/eval_predictors.py \
    --data "$DATA" \
    --gnn-checkpoint "$CKPT" \
    --methods gnn interp sz3 \
    --eb 1 2 4 \
    --levels 4 \
    --anchor-stride 16 \
    --anchor-block 1 \
    --gnn-tile 64 \
    --device cuda \
    --csv eval.csv \
    --plot eval_rd.png \
    "$@"

# per-image + per-method table -> job log
# machine-readable results        -> eval.csv
# rate-distortion curves          -> eval_rd.png
