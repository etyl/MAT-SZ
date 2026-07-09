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

CKPT=${CKPT:-/lustre/fswork/projects/rech/lzs/uhq13gg/MAT-SZ/.worktrees/context-predictor/data/runs/20260708-180745-f15497/gnn_predictor.pt}
IMAGE=${IMAGE:-/lustre/fswork/projects/rech/lzs/uhq13gg/data/kodak/17.png}
EB=${EB:-2}
LEVELS=${LEVELS:-6}
ANCHOR_STRIDE=${ANCHOR_STRIDE:-64}
WARMUP=${WARMUP:-1}
REPEATS=${REPEATS:-3}

python scripts/profile_gnn_inference.py \
  --checkpoint "$CKPT" \
  --input "$IMAGE" \
  --eb "$EB" \
  --levels "$LEVELS" \
  --anchor-stride "$ANCHOR_STRIDE" \
  --device cuda \
  --mode codec \
  --warmup "$WARMUP" \
  --repeats "$REPEATS" \
  "$@"

# Add --profile --trace inference_trace.json to the sbatch command when an
# operator-level trace is needed; normal runs report the codec phase timings.
