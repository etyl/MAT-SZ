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
export DEEPSZ_M_TILE=$((1 << 30))   # M-tiling off (chunk-batch 1 fits without it)

CKPT=${CKPT:-/lustre/fswork/projects/rech/lzs/uhq13gg/MAT-SZ/data/runs/20260710-115201-7bbb4e/gnn_predictor.pt}
# DATA=${DATA:-/lustre/fswork/projects/rech/lzs/uhq13gg/benchmark-scientific-data-compression/rti_75_density.npy}
DATA=${DATA:-./data/rti_normal.npy}
EB=${EB:-0.01}
TUNE=${TUNE:-fast}               # fast (1 encode) | size/rd (4 encodes)
TUNE_SIZE_SLACK=${TUNE_SIZE_SLACK:-1.05}

python scripts/eval_tensor.py "$DATA" \
    --gnn-checkpoint "$CKPT" \
    --predictor gnn \
    --eb "$EB" \
    --levels 4 \
    --anchor-stride 16 \
    --anchor-block 1 \
    --agg-level 2 \
    --chunk-batch 1 \
    --tune "$TUNE" \
    --tune-size-slack "$TUNE_SIZE_SLACK" \
    --fp16 \
    --compile \
    "$@"
