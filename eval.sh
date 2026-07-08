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

module purge
module load pytorch-gpu

DATA=${DATA:-/lustre/fswork/projects/rech/lzs/uhq13gg/data/kodak}
CKPT=${CKPT:-/lustre/fswork/projects/rech/lzs/uhq13gg/MAT-SZ/data/runs/20260708-112905-8b8203/gnn_predictor.pt}

python scripts/eval_predictors.py \
    --data "$DATA" \
    --gnn-checkpoint "$CKPT" \
    --methods gnn interp sz3 \
    --eb 0.03 0.06 0.1 0.2 \
    --levels 6 \
    --anchor-stride 64 \
    --anchor-block 1 \
    --device cuda \
    --csv eval.csv \
    --plot eval_rd.png \
    "$@"

# per-image + per-method table -> job log
# machine-readable results        -> eval.csv
# rate-distortion curves          -> eval_rd.png
