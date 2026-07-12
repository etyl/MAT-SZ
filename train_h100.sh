#!/bin/bash
#SBATCH --job-name=gnn-sz
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=5:00:00
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --account=lzs@h100
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

module purge
module load arch/h100
module load pytorch-gpu

# Small n-D (e.g. 4-D) tensor roundtripped through the full codec during training
# (distortion / rate / peak-RAM / enc+dec time -> wandb). Set EVAL_TENSOR to your
# 4-D field. Unset -> the default path below; pass EVAL_TENSOR= (empty) to skip.
EVAL_TENSOR=${EVAL_TENSOR-/lustre/fswork/projects/rech/lzs/uhq13gg/benchmark-scientific-data-compression/rti_75_density.npy}
EVAL_TENSOR_EB=${EVAL_TENSOR_EB:-0.01}
EVAL_TENSOR_EVERY=${EVAL_TENSOR_EVERY:-2000}
eval_tensor_args=()
if [ -n "$EVAL_TENSOR" ]; then
    eval_tensor_args=(--eval-tensor "$EVAL_TENSOR" \
                      --eval-tensor-eb "$EVAL_TENSOR_EB" \
                      --eval-tensor-every "$EVAL_TENSOR_EVERY")
fi

python scripts/train_gnn.py \
    --data /lustre/fswork/projects/rech/lzs/uhq13gg/data/div2k \
    --out data/gnn_predictor.pt \
    --steps 50000 \
    --batch 32 \
    --crop 128 \
    --d 32 \
    --lr 0.0005 \
    --noise-range 0.0001 0.05 \
    --eval-image /lustre/fswork/projects/rech/lzs/uhq13gg/data/kodak/17.png \
    --eval-eb 0.01 \
    --eval-every 100 \
    --img-every 1000 \
    "${eval_tensor_args[@]}" \
    --device cuda \
    --wandb-mode offline \
    --run-name gnn-axis \
    "$@"

# per-run dir -> data/runs/<date>-<hash>/ (checkpoint + config.json)
