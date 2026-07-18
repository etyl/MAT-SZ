#!/bin/bash
#SBATCH --job-name=gnn-bench-levels
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


# Per-level bpp + MSE breakdown, interp vs chunked GNN (vs skel), in fp32.
#
# fp32 on this workstation (TITAN Xp, driver 450) hits a hard torch assert:
#   "Can't find nvmlDeviceGetNvLinkRemoteDeviceType" -- torch 2.6 was built
# against a newer nvml than the driver ships, and c10's DriverAPI asserts on the
# missing symbol instead of tolerating the NVML error. We fix it by LD_PRELOADing
# a forwarding shim: a copy of the real driver lib renamed to a private soname,
# plus a thin libnvidia-ml.so.1 that adds a stub for the missing symbol and
# forwards everything else to the renamed copy. Built once into ./nvml_shim/.
# (fp16 sidesteps the assert on its own, so FP16=1 needs none of this.)
set -euo pipefail

module purge
module load pytorch-gpu

# ---- benchmark knobs (see scripts/bench_levels.py) ----
export CKPT=${CKPT:-/lustre/fswork/projects/rech/lzs/uhq13gg/MAT-SZ/data/runs/20260717-225540-c29fd5/gnn_predictor.pt}
export EB=${EB:-1e-4}
export N=${N:-64}
export LEVELS=${LEVELS:-5}
export STRIDE=${STRIDE:-32}
export BLOCK=${BLOCK:-1}
export CHUNK=${CHUNK:-32}
export AGG=${AGG:-2}
export TUNE=${TUNE:-fast}
export FP16=${FP16:-0}                 # 1 = fp16 (no shim needed), 0 = fp32
export CODECS=${CODECS:-interp,gnn}
export DATA=${DATA:-/lustre/fswork/projects/rech/lzs/uhq13gg/benchmark-scientific-data-compression/rti_75_density.npy}
export PYTHONUNBUFFERED=1


echo "[bench_levels] FP16=$FP16 N=$N EB=$EB CHUNK=$CHUNK CODECS=$CODECS"
python scripts/bench_levels.py "$@"
