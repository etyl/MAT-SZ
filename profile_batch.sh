#!/bin/bash
#SBATCH --job-name=gnn-profile-batch
#SBATCH --qos=qos_gpu-dev
#SBATCH --time=0:20:00
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

export PYTHONUNBUFFERED=1   # flush the table to the SLURM .out live

: "${CKPT:?Set CKPT to a GNN checkpoint before running this launcher}"

python scripts/profile_gnn.py \
    --gnn-checkpoint "$CKPT" \
    --levels 4 \
    --anchor-stride 16 \
    --batches 1,2,4,8,16 \
    --target-shape 119,128,128,128 \
    "$@"
