#!/bin/bash
#SBATCH --job-name=gnn-prof
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

export PYTHONUNBUFFERED=1

: "${CKPT:?Set CKPT to a GNN checkpoint before running this launcher}"
DATA=${DATA:-./data/rti_normal.npy}

# Operator-level trace of one worst-case chunk's GNN forward: shows which ops
# (matmul/gelu/softmax/index/copy...) dominate CUDA time -> what to optimize.
python scripts/profile_chunked_tensor.py "$DATA" \
    --gnn-checkpoint "$CKPT" \
    --device cuda \
    --levels 4 \
    --chunk-size 16 \
    --row-limit 30 \
    --fp16 \
    --compile \
    "$@"
