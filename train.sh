#!/bin/bash
#SBATCH --job-name=gnn-sz
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=8:00:00
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
    --steps 10000 \
    --batch 4 \
    --crop 128 \
    --d 64 \
    --lr 0.0001 \
    --eval-every 100 \
    --img-every 500 \
    --device cuda \
    --wandb-mode offline \
    "$@"

# checkpoint  -> data/gnn_predictor.pt
# loss curve  -> data/gnn_predictor.csv (+ .png if matplotlib is installed)
