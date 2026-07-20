#!/bin/bash
#SBATCH --job-name=gnn-bench-subset
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

# Benchmark the GNN codec on a fixed 64^4 (default) centred subset of a large
# tensor, reporting PSNR / bpp / time / max RAM / mean GPU utilisation so an
# optimisation can be judged before-vs-after. Every run appends a JSON record
# to $JSON_OUT, so re-run this on two commits and diff the two lines.
#
#   sbatch bench_subset.sh                       # defaults below
#   LABEL=baseline sbatch bench_subset.sh        # tag the JSON record
#   EDGE=32 EB=1e-3 sbatch bench_subset.sh       # override any knob via env

module purge
module load pytorch-gpu

export PYTHONUNBUFFERED=1          # flush progress to the SLURM .out live
export DEEPSZ_M_TILE=$((32**4))    # M-tiling off (chunk-batch 1 fits without it)

# GNN checkpoint (same one eval_tensor.sh uses; override with CKPT=...).
CKPT=${CKPT:-/lustre/fswork/projects/rech/lzs/uhq13gg/MAT-SZ/data/runs/20260716-190203-07b676/gnn_predictor.pt}

# Large source tensor; a centred EDGE^ndim hypercube is cropped out of it.
DATA=${DATA:-/lustre/fswork/projects/rech/lzs/uhq13gg/benchmark-scientific-data-compression/rti_75_density.npy}

EDGE=${EDGE:-64}                   # subset edge (capped per axis, floored to stride)
EB=${EB:-0.0001}
LEVELS=${LEVELS:-5}
ANCHOR_STRIDE=${ANCHOR_STRIDE:-32}
AGG=${AGG:-1}                      # neighbourhood aggregation level (1 or 2)
CHUNK=${CHUNK:-32}
CHUNK_BATCH=${CHUNK_BATCH:-1}
TUNE=${TUNE:-fast}
LABEL=${LABEL:-}
JSON_OUT=${JSON_OUT:-bench_results.jsonl}

# V100 supports Triton, so --compile pays off here (unlike the local Titan Xp).
# Set COMPILE=0 to disable; add extra flags (e.g. --fp16) via EXTRA=... or "$@".
COMPILE_FLAG=""
[ "${COMPILE:-1}" = "1" ] && COMPILE_FLAG="--compile"

python scripts/bench_gnn_subset.py "$DATA" \
    --gnn-checkpoint "$CKPT" \
    --subset-edge "$EDGE" \
    --eb "$EB" \
    --levels "$LEVELS" \
    --anchor-stride "$ANCHOR_STRIDE" \
    --anchor-block 1 \
    --agg-level "$AGG" \
    --chunk-size "$CHUNK" \
    --chunk-batch "$CHUNK_BATCH" \
    --tune "$TUNE" \
    --normalize \
    --json-out "$JSON_OUT" \
    --label "$LABEL" \
    $COMPILE_FLAG \
    ${EXTRA:-} \
    "$@"
