#!/bin/bash
#SBATCH --job-name=gnn-sz
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=10:00:00
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

# The profile is launch-bound (thousands of ~10 us kernels per step). Ask
# torch.compile to use CUDA graphs where shapes permit, reducing CPU dispatch
# overhead without changing the batch or model.
export DEEPSZ_COMPILE_MODE=reduce-overhead

# Training must run embed untiled (max-M kernels, fewest launches). sbatch
# inherits the submitting shell's env, so drop any stray DEEPSZ_M_TILE.
unset DEEPSZ_M_TILE

# 16*128^2 image + 4*16^4 synthetic = 524,288 total points (50% synthetic).
# synthetic-stride 8 (< field 16) makes each synthetic field span 2^4 chunks, so
# the chunked/coarse path (CoarseProj) trains on n-D, not just 2-D images. Each
# step splits the synthetic batch in half: half full-level dense, half chunked
# coarse. Same field size / batch / balance as before; only the stride changed.
# Synthetic fields are generated concurrently with GPU work by train_gnn.py.
# --synthetic-correlation is a *band* (MIN MAX): each field draws a base
# smoothness in [MIN/2, MAX], a random per-axis spread (-> isotropic..
# anisotropic), and a random value-marginal warp (unimodal..bimodal), so the
# prior spans diverse scientific fields, not one texture a wide model overfits.
python scripts/train_gnn.py \
    --data /lustre/fswork/projects/rech/lzs/uhq13gg/data/div2k \
    --out data/gnn_predictor.pt \
    --steps 50000 \
    --save-every 5000 \
    --batch 16 \
    --crop 128 \
    --synthetic-frac 0.5 \
    --synthetic-shape 16 16 16 16 \
    --synthetic-correlation 1.0 8 \
    --synthetic-batch 4 \
    --synthetic-stride 8 \
    --workers 4 \
    --agg-level 1 \
    --d 64 \
    --lr 0.0005 \
    --noise-range 0.0001 0.05 \
    --eval-image /lustre/fswork/projects/rech/lzs/uhq13gg/data/kodak/17.png \
    --eval-eb 0.001 \
    --eval-every 500 \
    --img-every 10000 \
    --eval-tensor ./data/rti_normal.npy \
    --eval-tensor-normalize \
    --eval-tensor-eb 0.001 \
    --eval-tensor-every 500 \
    --device cuda \
    --wandb-mode offline \
    --run-name gnn-agg1 \
    --compile \
    "$@"

# per-run dir -> data/runs/<date>-<hash>/ (checkpoint + config.json)
