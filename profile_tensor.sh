#!/bin/bash
#SBATCH --job-name=gnn-profile-tensor
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

# Detailed end-to-end profile of the *chunked* GNN codec on a 4-D tensor.
# Same knobs as eval_tensor.sh; prints a phase table + a forward-sublayer table
# instead of the roundtrip quality line. See scripts/profile_chunked_tensor.py.

module purge
module load pytorch-gpu

export PYTHONUNBUFFERED=1
export DEEPSZ_M_TILE=$((1 << 30))   # M-tiling off, to match eval_tensor.sh

# D32 checkpoint
CKPT=${CKPT:-/lustre/fswork/projects/rech/lzs/uhq13gg/MAT-SZ/data/runs/20260710-115201-7bbb4e/gnn_predictor.pt}

DATA=${DATA:-/lustre/fswork/projects/rech/lzs/uhq13gg/benchmark-scientific-data-compression/rti_75_density.npy}
# DATA=${DATA:-./data/rti_normal.npy}

EB=${EB:-0.01}

python scripts/profile_chunked_tensor.py "$DATA" \
    --gnn-checkpoint "$CKPT" \
    --eb "$EB" \
    --levels 4 \
    --anchor-stride 16 \
    --chunk-size 16 \
    --anchor-block 1 \
    --agg-level 2 \
    --chunk-batch 1 \
    --fp16 \
    --compile \
    "$@"
