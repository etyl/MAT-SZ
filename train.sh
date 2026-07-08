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
    --batch 12 \
    --crop 128 \
    --d 128 \
    --lr 0.0001 \
    --noise-range 0.0039215686 0.0313725490 \
    --eval-eb 0.0078431373 \
    --eval-every 100 \
    --img-every 500 \
    --device cuda \
    --wandb-mode offline \
    --run-name gnn-sz \
    "$@"

# per-run dir -> data/runs/<date>-<hash>/ (checkpoint + config.json)
