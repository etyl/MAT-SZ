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

# 16*128^2 image + 4*16^4 synthetic = 524288 total points (50% synthetic),
# preserving the old 32*128^2 per-step scalar-point budget.
python scripts/train_gnn.py \
    --data /lustre/fswork/projects/rech/lzs/uhq13gg/data/div2k \
    --out data/gnn_predictor.pt \
    --steps 50000 \
    --batch 16 \
    --crop 128 \
    --synthetic-frac 0.5 \
    --synthetic-shape 16 16 16 16 \
    --synthetic-correlation 6 3 1.5 0.75 \
    --synthetic-batch 4 \
    --synthetic-stride 16 \
    --agg-level 2 \
    --d 32 \
    --lr 0.0005 \
    --noise-range 0.0001 0.05 \
    --eval-image /lustre/fswork/projects/rech/lzs/uhq13gg/data/kodak/17.png \
    --eval-eb 0.01 \
    --eval-every 500 \
    --img-every 10000 \
    --eval-tensor ./data/rti_normal.npy \
    --eval-tensor-eb 0.01 \
    --eval-tensor-every 500 \
    --device cuda \
    --wandb-mode offline \
    --run-name gnn-axis-4d \
    --fp16 \
    --compile \
    "$@"

# per-run dir -> data/runs/<date>-<hash>/ (checkpoint + config.json)
