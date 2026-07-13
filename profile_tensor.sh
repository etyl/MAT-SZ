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
# Tile = one chunk's worth of queries: chunk_size**ndim = 16**4 for the 4-D rti
# tensor. This is the largest M any stage in a chunk produces, so it's one tile
# per chunk (no sub-tiling) while capping the transient buffers to exactly the
# chunk. (Same result as 1<<30 for chunked runs, just the honest ceiling.)
export DEEPSZ_M_TILE=$((16 ** 4))
# reduce-overhead (CUDA graphs): -17% wall (600->500ms) even though self-CUDA
# rises (extra graph input-staging copies). The pass is launch-bound, so closing
# the inter-kernel gaps beats the added copies. Grade this by wall time, not
# self-CUDA-total. Set empty to compare the default (uncompiled-mode) path.
# export DEEPSZ_COMPILE_MODE=${DEEPSZ_COMPILE_MODE:-reduce-overhead}

# D32 checkpoint
CKPT=${CKPT:-/lustre/fswork/projects/rech/lzs/uhq13gg/MAT-SZ/data/runs/20260710-115201-7bbb4e/gnn_predictor.pt}

DATA=${DATA:-/lustre/fswork/projects/rech/lzs/uhq13gg/benchmark-scientific-data-compression/rti_75_density.npy}
# DATA=${DATA:-./data/rti_normal.npy}

EB=${EB:-0.01}

python scripts/profile_chunked_tensor.py "$DATA" \
    --gnn-checkpoint "$CKPT" \
    --device cuda \
    --eb "$EB" \
    --levels 5 \
    --anchor-stride 32 \
    --chunk-size 32 \
    --anchor-block 1 \
    --agg-level 2 \
    --chunk-batch 1 \
    --fp16 \
    --compile \
    --no-stack \
    "$@"
