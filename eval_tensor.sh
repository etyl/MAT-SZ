#!/bin/bash
#SBATCH --job-name=gnn-eval-tensor
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

export PYTHONUNBUFFERED=1   # flush progress to the SLURM .out live
# export DEEPSZ_M_TILE=$((16**4))   # M-tiling off (chunk-batch 1 fits without it)
unset DEEPSZ_M_TILE
export DEEPSZ_COMPILE_MODE=${DEEPSZ_COMPILE_MODE:-reduce-overhead}

# D128 checkpoint
# CKPT=${CKPT:-/lustre/fswork/projects/rech/lzs/uhq13gg/MAT-SZ/data/runs/20260710-115201-7bbb4e/gnn_predictor.pt}
# D32 checkpoint
# CKPT=${CKPT:-/lustre/fswork/projects/rech/lzs/uhq13gg/MAT-SZ/data/runs/20260714-095239-e16d09/gnn_predictor.pt}
# CKPT=${CKPT:-/lustre/fswork/projects/rech/lzs/uhq13gg/MAT-SZ/data/runs/20260714-182442-d79742/gnn_predictor.pt}
# D64 checkpoint
CKPT=${CKPT:-/lustre/fswork/projects/rech/lzs/uhq13gg/MAT-SZ/data/runs/20260715-165121-dca9a5/gnn_predictor.pt}

DATA=${DATA:-/lustre/fswork/projects/rech/lzs/uhq13gg/benchmark-scientific-data-compression/rti_75_density.npy}
# DATA=${DATA:-./data/rti_rough.npy}

EB=${EB:-0.0001}
TUNE=${TUNE:-fast}               # MAT: fast (1 encode) | size/rd (4 encodes)
CODEC=${CODEC:-gnn}             # gnn | skel
LINE_ORDER=${LINE_ORDER:-cubic}   # skel: cubic | linear
# SZ3 independently uses its tuned INTERP_LORENZO hybrid at every error bound.
TUNE_SIZE_SLACK=${TUNE_SIZE_SLACK:-1.05}

python scripts/eval_tensor.py "$DATA" \
    --gnn-checkpoint "$CKPT" \
    --predictor gnn \
    --codec "$CODEC" \
    --line-order "$LINE_ORDER" \
    --eb "$EB" \
    --levels 5 \
    --anchor-stride 32 \
    --chunk-size 32 \
    --anchor-block 1 \
    --agg-level 2 \
    --chunk-batch 1 \
    --tune "$TUNE" \
    --tune-size-slack "$TUNE_SIZE_SLACK" \
    --normalize \
    --fp16 \
    --compile \
    --interfaces \
    "$@"
