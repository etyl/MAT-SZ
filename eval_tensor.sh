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

CKPT=${CKPT:-/lustre/fswork/projects/rech/lzs/uhq13gg/MAT-SZ/.worktrees/axis-embeddings/data/runs/20260709-173909-ffd4f2/gnn_predictor.pt}
EB=${EB:-0.01}
TUNE=${TUNE:-fast}               # fast (1 encode) | size/rd (4 encodes)
TUNE_SIZE_SLACK=${TUNE_SIZE_SLACK:-1.05}

python scripts/eval_tensor.py "/lustre/fswork/projects/rech/lzs/uhq13gg/benchmark-scientific-data-compression/rti_75_density.npy" \
    --gnn-checkpoint "$CKPT" \
    --predictor gnn \
    --eb "$EB" \
    --levels 6 \
    --anchor-stride 64 \
    --anchor-block 1 \
    --tune "$TUNE" \
    --tune-size-slack "$TUNE_SIZE_SLACK" \
    "$@"
