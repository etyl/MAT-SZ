# DeepSZ

An SZ-style **error-bounded lossy compressor** in which the classic prediction
stage (Lorenzo in SZ1/2, spline interpolation in SZ3) is replaced by **MAT** —
the *Mask-Aware Transformer for Large Hole Image Inpainting* (CVPR 2022).
Already-reconstructed pixels form the known context; MAT inpaints everything
else; residuals against those predictions are quantized to the error bound,
Huffman-coded, and zstd-compressed — exactly the classic SZ pipeline with a
deep inpainting model as the predictor.

```
prediction (MAT inpainting) → linear-scaling quantization → canonical Huffman → zstd
```

Current target: natural images (RGB/grayscale, uint8), where the pretrained
Places512 checkpoint predicts well. The codec core is dtype-general
(float32 arrays work through the same path).

## Install

From the repository root:

```bash
pip install .
```

For GNN checkpoint compression:

```bash
pip install ".[gnn]"
```

For the image CLI and tests:

```bash
pip install ".[image,test]"
```

Editable development install:

```bash
pip install -e ".[test]"
```

## Python API

```python
from deepsz import GNNCodec

codec = GNNCodec("path/to/gnn_checkpoint.pt", error_bound=1e-3)

compressed = codec.compress(array_or_tensor)  # numpy array or torch tensor, any rank
reconstructed = codec.uncompress(compressed)  # torch.Tensor
```

`GNNCodec` stores the tensor shape, dtype, error bound, codec parameters, and
checkpoint hash in the binary stream. Decoding verifies the checkpoint hash by
default.

## GNN training with synthetic 4-D fields

The GNN trainer mixes natural-image crops with generated smooth 4-D fields by
default. The four correlation values are Gaussian smoothing lengths in grid
cells: larger values make an axis smoother and more predictable. Their axis
assignment is randomly permuted for every generated field, so the model must
infer which axes are most useful instead of relying on a fixed axis ordering.

```bash
python scripts/train_gnn.py \
    --data /path/to/images \
    --synthetic-frac 0.5 \
    --synthetic-shape 16 16 16 16 \
    --synthetic-correlation 6 3 1.5 0.75 \
    --synthetic-batch 1 \
    --synthetic-stride 8 \
    --agg-level 2
```

Every optimizer step processes both sources sequentially: an image batch using
the existing mixture of whole-image and 2-D chunked training, followed by a 4-D
batch using the rank-generic whole-field schedule. `--synthetic-frac` is the
target fraction of scalar points from the synthetic batch. For example, the
default combines one `16^4` field (65,536 points) with four `128^2` crops
(65,536 points), making exactly 50% of the 131,072 values synthetic. The image
batch size is derived from this target and `--synthetic-batch`; fractions that
cannot be represented by whole examples are rounded to the nearest achievable
ratio and the actual ratio is printed. Source losses are weighted by those
actual point counts before one optimizer update. Sequential backward passes
avoid holding both computation graphs in memory. Set `--synthetic-frac 0` for
image-only training or `1` for synthetic-only training.

`--synthetic-stride 8` controls the synthetic field's prediction schedule, not
its correlation length. Directly coded anchors are spaced every eight cells on
each axis; three dyadic refinement levels then reduce that spacing through
`8 -> 4 -> 2 -> 1`. `--agg-level 2` limits GNN neighbourhood lines to
axis-aligned directions and two-axis diagonals, excluding directions that vary
along three or four axes at once.

On CUDA, add `--fp16` to autocast the GNN message pass with gradient scaling,
and `--compile` to compile the dynamic-shape embedding pass. The prediction
readout and loss remain FP32. These settings are also used by periodic
`--eval-tensor` codec evaluations.

## Legacy Setup Notes

```bash
# conda base env already has torch 2.6.0+cu118, numpy, scipy, zstandard, pillow
pip install spandrel spandrel_extra_arches pytest
# torchvision must match the torch CUDA variant (cu118):
pip install --force-reinstall --no-deps torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu118
python scripts/download_checkpoint.py     # 125 MB, models/MAT_Places512_G_fp16.safetensors
```

The MAT weights are **CC-BY-NC** (research use only, see
https://github.com/fenglinglwb/MAT).

## Usage

```bash
deepsz compress photo.png photo.msz --eb 2        # eb in 0-255 units
deepsz decompress photo.msz reconstructed.png
deepsz eval photo.png --eb 2                      # roundtrip + metrics
```

Useful flags: `--levels N` (progressive refinement stages), `--anchor-stride`
/ `--anchor-block` (geometry of the directly-coded anchor set), `--rel`
(relative error bound), `--mock` (torch-free nearest-neighbor predictor — fast,
for development), `-v`.

This machine is CPU-only (~40 s per 512×512 MAT forward), so a real compress
or decompress takes `levels × n_tiles × ~40 s`. `--mock` runs in seconds.

## How it works

1. The image is edge-padded to a multiple of 512 and cut into 512×512 tiles
   (MAT's fixed operating resolution). The padded canvas is compressed whole —
   replicated padding costs almost nothing after entropy coding.
2. **Stage 0 (anchors)**: a sparse grid of `anchor_block × anchor_block` pixel
   blocks (defaults: 4×4 blocks every 16 pixels → 6.25 % coverage) is
   quantized directly (prediction = 0) and entropy-coded.
3. **Stages 1..levels**: the sampling grid halves each stage. All
   already-reconstructed pixels are fed to MAT as known context (`mask = 1` on
   the rest); MAT inpaints; predictions at the new grid positions are corrected
   by quantized residuals `code = round((x − pred)/2eb)` and join the known
   set. The final stage covers every remaining pixel.
4. Quantization codes (per stage) go through a canonical Huffman coder; code 0
   is reserved for *unpredictable* values stored exactly (classic SZ outlier
   handling). Everything is wrapped in one zstd frame.

The encoder is a **closed loop**: predictor inputs are built exclusively from
dequantized values, never from the original data, so the decoder — running the
identical schedule — reproduces the exact same predictions and reconstruction.

### Error-bound guarantee

`|original − decoded| ≤ eb` holds **unconditionally**: after quantizing, the
encoder re-checks every value with the decoder's own reconstruction arithmetic
(including the final round-to-integer for uint8 sources) and demotes any
violator (float rounding at bound edges, `|pred| ≫ eb` absorption, radius
overflow) to an exactly-stored outlier.

### Determinism

MAT is StyleGAN2-based and stochastic in two places that we pin down:

- `MAT.__init__` draws a random style latent `z` — overwritten with an RNG
  seeded from the header `seed`;
- the synthesis network calls `F.dropout(training=True)` even at inference —
  `torch.manual_seed(seed)` is issued before every forward, identically at
  encode and decode.

Verified: predictions are **bitwise identical across process restarts**
(`scripts/sanity_check_mat.py`). Compression and decompression must run on the
same platform (CPU fp32 here) — the usual caveat for every DL-based compressor;
a platform mismatch is flagged via the checkpoint hash in the header and, in
the worst case, shows up as a bound violation measurable with `eval`.

## Layout

```
deepsz/quantizer.py   linear-scaling quantization + outliers (pure numpy)
deepsz/huffman.py     canonical Huffman coder (numpy + heapq)
deepsz/levels.py      progressive stage schedule shared by encoder/decoder
deepsz/bitstream.py   header + tile payloads + zstd frame
deepsz/predictor.py   MATPredictor (spandrel) and MockPredictor (scipy EDT)
deepsz/codec.py       closed-loop compress/decompress, tiling
deepsz/cli.py         compress / decompress / eval
scripts/sanity_check_mat.py  spike: load, timing, polarity, determinism, OOD probe
```

## Tests

```bash
pytest                  # fast suite, mock predictor, no checkpoint needed (~3 s)
pytest -m slow          # real-checkpoint end-to-end roundtrip (minutes, CPU)
```

## Spike findings (this machine)

- Load + fp32 cast: ~12 s; one 512×512 forward: ~40 s (36 CPU threads).
- Mask polarity: known pixels pass through (≤1.5e-8 float noise — the codec
  only consumes predictions at hole positions).
- Determinism: bitwise-equal predictions in-process and across processes.
- Anchor-geometry probe (prediction PSNR over holes vs a nearest-neighbor
  baseline): see table below.

| coverage | block | stride | MAT PSNR | nearest PSNR |
|---------:|------:|-------:|---------:|-------------:|
| 1/16     | 1     | 4      | 20.72    | **24.16**    |
| 1/16     | 4     | 16     | **21.12**| 20.64        |
| 1/16     | 8     | 32     | **20.00**| 18.76        |
| 1/64     | 1     | 8      | 21.19    | **21.72**    |
| 1/64     | 4     | 32     | **18.75**| 18.06        |
| 1/64     | 8     | 64     | 15.15    | **15.69**    |

Scattered single-pixel anchors are out-of-distribution for MAT (trained on
large contiguous holes) — a nearest-neighbor fill beats it there. Contiguous
4×4 anchor blocks flip the comparison in MAT's favor, hence the defaults
`--anchor-block 4 --anchor-stride 16 --levels 4` (6.25 % anchor coverage).
These numbers are for the *first* refinement stage only; later stages see much
denser context and predict correspondingly better.

## Results (kodim23, 512×768 RGB, defaults, this machine)

| codec | eb | ratio | bpp | PSNR (dB) | max err | time c/d |
|-----------|---:|------:|----:|----------:|--------:|---------:|
| DeepSZ (MAT) | 1 | 1.21 | 19.80 | 51.19 | 1 PASS | 420 s / 482 s |
| DeepSZ (MAT) | 2 | 1.54 | 15.56 | 46.41 | 2 PASS | 408 s / 495 s |
| DeepSZ (MAT) | 4 | 2.14 | 11.22 | 40.78 | 4 PASS | 452 s / 429 s |
| DeepSZ (mock NN) | 2 | 3.40 | 7.05 | 46.48 | 2 PASS | 1.6 s / 5.0 s |
| **SZ3** | 1 | 2.68 | 8.96 | 51.18 | 1 | ~0.1 s |
| **SZ3** | 2 | 3.96 | 6.06 | 46.51 | 2 | ~0.1 s |
| **SZ3** | 4 | 6.61 | 3.63 | 41.54 | 4 | ~0.1 s |
| zstd raw  | —  | 1.45  | 16.51 | ∞ (lossless) | 0  | — |

SZ3 (interpolation predictor, per-channel 2D) is the reference the `eval`
command prints automatically. It is used through the official `pysz` binding
(`pip install pysz`) with a fallback to a locally built CLI
(`tools/sz3/bin/sz3`); `imagecodecs` also ships SZ3 but its wheels need
glibc ≥ 2.28 (this machine has 2.27). Caveat on this machine: import numpy
before pysz — the pysz wheel otherwise loads the old system libstdc++ first
and breaks numpy's C extensions.

**Honest finding**: with the *pretrained* Places512 checkpoint, MAT is a worse
predictor for residual coding than plain nearest-neighbor interpolation. The
error bound and the pipeline hold up perfectly, but the compression ratio
barely beats raw zstd. The cause is structural: MAT is a GAN trained for
perceptual realism — given dense scattered context (the fine refinement
stages) it hallucinates *plausible* texture rather than converging to the true
pixel values, so residuals stay large exactly where most of the pixels are.
The probe table above shows the same effect at the sparse stages.

This is the expected starting point for the "use a checkpoint for now"
prototype. Ways forward, in increasing order of effort:
1. **Hybrid schedule** — MAT for the sparse early stages (where it beats
   interpolation), a deterministic interpolator for the dense final stages.
2. **Fine-tune MAT** with an L1/L2 objective (no adversarial loss) on the
   codec's actual masking distribution (scattered grids + anchor blocks).
3. Train a mask-aware transformer predictor from scratch on the target data
   domain (the original goal — the codec, bitstream and closed loop are all
   predictor-agnostic via the `Predictor` protocol).
