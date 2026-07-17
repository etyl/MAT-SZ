#!/bin/bash
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
cd "$(dirname "$0")"

PY=${PY:-/home/mind/hvernina/miniconda3/bin/python}
GPU=${CUDA_VISIBLE_DEVICES:-0}

# ---- benchmark knobs (see scripts/bench_levels.py) ----
export CKPT=${CKPT:-checkpoints/d64.pt}
export EB=${EB:-1e-4}
export N=${N:-64}
export LEVELS=${LEVELS:-5}
export STRIDE=${STRIDE:-32}
export BLOCK=${BLOCK:-1}
export CHUNK=${CHUNK:-32}
export AGG=${AGG:-2}
export TUNE=${TUNE:-fast}
export FP16=${FP16:-0}                 # 1 = fp16 (no shim needed), 0 = fp32
export CODECS=${CODECS:-interp,gnn,skel}
export DATA=${DATA:-}                   # optional 4-D .npy (else synthetic RTI)
export PYTHONUNBUFFERED=1

PRELOAD=""
if [ "$FP16" != "1" ]; then
    # ---- build the fp32 nvml shim on first run ----
    SHIM_DIR="$PWD/nvml_shim"
    SHIM="$SHIM_DIR/libnvidia-ml.so.1"
    if [ ! -f "$SHIM" ]; then
        echo "[bench_levels] building fp32 nvml shim in $SHIM_DIR ..."
        REAL=$(ldconfig -p | awk '/libnvidia-ml\.so\.1/{print $NF; exit}')
        REAL=${REAL:-/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1}
        PATCHELF=${PATCHELF:-$(dirname "$PY")/patchelf}
        command -v "$PATCHELF" >/dev/null 2>&1 || PATCHELF=patchelf
        command -v "$PATCHELF" >/dev/null 2>&1 || { "$PY" -m pip install -q patchelf; PATCHELF=$(dirname "$PY")/patchelf; }
        mkdir -p "$SHIM_DIR"
        cp "$REAL" "$SHIM_DIR/libnvidia-ml-real.so.1"
        "$PATCHELF" --set-soname libnvidia-ml-real.so.1 "$SHIM_DIR/libnvidia-ml-real.so.1"
        cat > "$SHIM_DIR/nvml_shim.c" <<'EOF'
/* nvmlDeviceGetNvLinkRemoteDeviceType is absent from driver 450's nvml; torch
   2.6 hard-asserts on the null symbol. Return NVML_ERROR_NOT_SUPPORTED (3) so
   torch takes the graceful "no NVLink" path. All other nvml calls forward to
   libnvidia-ml-real.so.1 via DT_NEEDED. */
int nvmlDeviceGetNvLinkRemoteDeviceType(void *device, unsigned int link, void *out) {
    return 3;
}
EOF
        gcc -shared -fPIC -o "$SHIM" "$SHIM_DIR/nvml_shim.c" \
            -Wl,-soname,libnvidia-ml.so.1 -Wl,--no-as-needed \
            -L "$SHIM_DIR" -l:libnvidia-ml-real.so.1 -Wl,-rpath,'$ORIGIN'
        echo "[bench_levels] shim built."
    fi
    PRELOAD="$SHIM"
fi

echo "[bench_levels] GPU=$GPU FP16=$FP16 N=$N EB=$EB CHUNK=$CHUNK CODECS=$CODECS"
CUDA_VISIBLE_DEVICES="$GPU" LD_PRELOAD="${PRELOAD}${LD_PRELOAD:+:$LD_PRELOAD}" \
    "$PY" scripts/bench_levels.py "$@"
