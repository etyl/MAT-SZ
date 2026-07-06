"""Risk spike for MAT-SZ: validate every assumption the codec is built on.

Checks, in order:
1. Checkpoint loads via spandrel (+extras), casts to fp32, runs on CPU.
2. Timing of one 512x512 forward.
3. Mask polarity: known pixels (mask=0) pass through unchanged, holes (mask=1) change.
4. Determinism across process restarts (the codec requirement):
     python sanity_check_mat.py --stage save    # writes prediction .npy
     python sanity_check_mat.py --stage compare # recomputes, asserts bitwise equal
5. OOD probe: prediction PSNR over holes for anchor-block geometries at equal
   coverage, vs a nearest-neighbor-fill baseline -> picks anchor_block default.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "models" / "MAT_Places512_G_fp16.safetensors"
TEST_IMG = ROOT / "tests" / "data" / "kodim23.png"
OUT_DIR = ROOT / "scripts" / "_spike_out"
SEED = 1234


def load_model():
    import spandrel
    import spandrel_extra_arches

    spandrel_extra_arches.install()
    desc = spandrel.ModelLoader().load_from_file(str(CKPT))
    print(f"descriptor: {type(desc).__name__}, purpose={desc.purpose}, "
          f"size_req={desc.size_requirements}")
    model = desc.model.float().eval()
    # CRITICAL: MAT.__init__ draws self.z unseeded; pin it for reproducibility.
    model.z = torch.from_numpy(np.random.RandomState(SEED).randn(1, 512)).float()
    return model


def forward(model, img_hw3: np.ndarray, hole_hw: np.ndarray) -> np.ndarray:
    """img in [0,1] float32 (H,W,3); hole True where pixel must be predicted."""
    x = torch.from_numpy(img_hw3.transpose(2, 0, 1)[None]).float()
    m = torch.from_numpy(hole_hw.astype(np.float32)[None, None])
    torch.manual_seed(SEED)  # F.dropout(training=True) lives in the synthesis net
    with torch.inference_mode():
        y = model(x, m)
    return y[0].numpy().transpose(1, 2, 0)


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")


def anchor_known_mask(h, w, stride, block):
    known = np.zeros((h, w), bool)
    for di in range(block):
        for dj in range(block):
            known[di::stride, dj::stride] = True
    return known


def nearest_fill(img, known):
    from scipy.ndimage import distance_transform_edt

    _, (ii, jj) = distance_transform_edt(~known, return_indices=True)
    return img[ii, jj]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["save", "compare", "probe", "all"], default="all")
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    torch.set_num_threads(torch.get_num_threads())
    print(f"torch {torch.__version__}, threads={torch.get_num_threads()}, cpu only")

    t0 = time.time()
    model = load_model()
    print(f"load+fp32: {time.time() - t0:.1f}s")

    rng = np.random.RandomState(0)
    img = rng.rand(512, 512, 3).astype(np.float32)
    hole = np.zeros((512, 512), bool)
    hole[:, 256:] = True  # right half is hole

    if args.stage in ("save", "compare", "all"):
        t0 = time.time()
        pred = forward(model, img * (~hole[..., None]), hole)
        dt = time.time() - t0
        print(f"one 512x512 forward: {dt:.1f}s")

        # polarity: left half (known) must pass through, right half must change
        left_diff = np.abs(pred[:, :256] - img[:, :256]).max()
        right_diff = np.abs(pred[:, 256:] - img[:, 256:]).mean()
        print(f"polarity: known-half max|out-in|={left_diff:.2e} (expect ~0), "
              f"hole-half mean|out-in|={right_diff:.3f} (expect >>0)")
        assert left_diff < 1e-5, "mask polarity wrong: known pixels were modified"
        assert right_diff > 0.01, "mask polarity wrong: holes were not inpainted"

        ref_path = OUT_DIR / "determinism_ref.npy"
        if args.stage == "compare":
            ref = np.load(ref_path)
            bitwise = np.array_equal(
                pred.view(np.uint32) if pred.dtype == np.float32 else pred,
                ref.view(np.uint32) if ref.dtype == np.float32 else ref)
            print(f"determinism across processes: bitwise_equal={bitwise}, "
                  f"max|diff|={np.abs(pred - ref).max():.2e}")
            assert bitwise, "DETERMINISM FAILURE: predictions differ across processes"
        else:
            np.save(ref_path, pred)
            print(f"saved reference prediction to {ref_path}")
            # in-process repeat as a first check
            pred2 = forward(model, img * (~hole[..., None]), hole)
            print(f"in-process repeat: bitwise_equal={np.array_equal(pred, pred2)}")

    if args.stage in ("probe", "all"):
        from PIL import Image

        real = np.asarray(Image.open(TEST_IMG).convert("RGB"), np.float32) / 255.0
        h0 = (real.shape[0] - 512) // 2
        w0 = (real.shape[1] - 512) // 2
        real = real[h0:h0 + 512, w0:w0 + 512]

        print("\nOOD probe (prediction PSNR over holes, higher is better):")
        print(f"{'coverage':>9} {'block':>5} {'stride':>6} {'MAT':>7} {'nearest':>8}")
        for cov_name, configs in [
            ("1/16", [(1, 4), (4, 16), (8, 32)]),
            ("1/64", [(1, 8), (4, 32), (8, 64)]),
        ]:
            for block, stride in configs:
                known = anchor_known_mask(512, 512, stride, block)
                masked = real * known[..., None]
                pred = forward(model, masked, ~known)
                p_mat = psnr(pred[~known], real[~known])
                p_nn = psnr(nearest_fill(real, known)[~known], real[~known])
                print(f"{cov_name:>9} {block:>5} {stride:>6} {p_mat:>7.2f} {p_nn:>8.2f}")

    print("\nSPIKE OK")


if __name__ == "__main__":
    main()
