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

python scripts/train_gnn.py \
    --data /lustre/fswork/projects/rech/lzs/uhq13gg/data/div2k \
    --out data/gnn_predictor.pt \
    --batch 4 \
    --crop 128 \
    --d 64 \
    --levels 4 \
    --stride 16 \
    --device cuda \
    --wandb-mode disabled \
    --profile 1 \
    "$@"

# op table -> job log; chrome trace -> trace.json (open in perfetto.dev)
