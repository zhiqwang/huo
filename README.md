# Huo

A CT image reconstruction library implementing the [Algebraic Reconstruction Technique (ART)](https://en.wikipedia.org/wiki/Algebraic_reconstruction_technique) with both a **Python** (PyTorch) back-end and a **TypeScript** (jax-js) browser front-end.

> **Note:** This project was written mainly to validate that PyTorch can work
> in the general scientific-computing field. The current implementation has no
> deep-learning components — if you want to apply it in a DL context you will
> need to add those parts manually.

## What is ART?

The **Algebraic Reconstruction Technique** is a classical iterative algorithm
for reconstructing a 2-D image from its projections (a *sinogram*), as
collected by a Computed Tomography (CT) scanner.

At each iteration the algorithm:

1. **Forward-projects** the current image estimate to predict what a single
   detector measurement should look like.
2. Computes the **residual** between the predicted and the actually measured
   sinogram column.
3. **Back-projects** that residual into image space to correct the estimate.
4. **Clamps** negative values to zero.

Angles are visited in random order (the Kaczmarz method). After one pass
through all angles the reconstruction is typically close to the original image.

## Installation

### Python

```bash
git clone https://github.com/zhiqwang/huo.git
cd huo

# with uv (recommended)
uv pip install -e .

# or with pip
pip install -e .
```

**Requirements:** Python ≥ 3.9, NumPy ≥ 1.18.5, PyTorch (see
[pytorch.org](https://pytorch.org/get-started/locally/) for install instructions).

### TypeScript / Browser Demo

```bash
cd js
npm install
npm run build
npx serve .
```

Then open `http://localhost:3000/demo/` in a modern browser (Chrome / Edge with
WebGPU recommended).

## Quick Start

### Python — CLI Reconstruction

After installing the package (`pip install -e .` or `uv pip install -e .`), the
`huo-art` command is available system-wide:

```bash
huo-art ./data/sinogram.npy -o reconstruction.npy \
    --img-pixels 512 \
    --img-len 144 \
    --detr-num 500 \
    --detr-len 180 \
    --sdd 1200 \
    --sod 981 \
    --rotate-step 1.0
```

The first positional argument is the path to a `.npy` sinogram file. Use
`-o` / `--output` to save the reconstructed image. All geometry flags are
optional and default to the values shown above.

See the [tutorial](docs/tutorial.md) for a detailed walk-through of every
parameter and the full reconstruction pipeline.

### Python — Library Usage

```python
import torch
from huo import RadonFanbeam

# Create a fan-beam CT operator (all geometry is pre-computed once)
angles = torch.arange(0, 360, step=1.0)
radon = RadonFanbeam(
    resolution=512,
    angles=angles,
    source_distance=981,    # SOD in mm
    det_distance=219,       # SDD − SOD in mm
    det_count=500,
    det_spacing=0.36,       # 180 mm / 500 elements
    volume_size=144,        # FOV diameter in mm
)

# Forward projection (image → sinogram)
sinogram = radon.forward(img)       # [det_count, num_angles]

# Back-projection (sinogram → image)
bp = radon.backprojection(sinogram) # [resolution, resolution]

# ART reconstruction (sinogram → image)
reconstructed = radon.art(sinogram) # [resolution, resolution]
```

The low-level functions in `huo.art` (``forward_propagation``,
``backward_propagation``, ``scan``, ``art``) are still available for
backward compatibility.

### Browser — Interactive Demo

The demo visualises both the **forward projection** (Radon transform — building
the sinogram angle by angle) and the **ART reconstruction** (back-projection —
iteratively recovering the image). A *Frame delay* slider lets you control the
animation speed so that each step can be observed.

## Project Structure

```
huo/
├── huo/                    # Python package
│   ├── __init__.py
│   ├── radon.py            # RadonFanbeam class (torch-radon-style API)
│   ├── art.py              # Low-level functions (forward / backward / scan / art)
│   └── cli.py              # CLI entry-point (huo-art command)
├── tools/
│   └── projection.py       # Thin wrapper → huo.cli.main
├── js/                     # TypeScript / browser implementation
│   ├── src/
│   │   └── art.ts          # Core CT algorithms (jax-js)
│   ├── demo/
│   │   ├── index.html      # Interactive visualisation page
│   │   └── main.ts         # Demo logic: phantom, rendering, UI
│   ├── package.json
│   └── tsconfig.json
├── docs/
│   └── tutorial.md         # Usage tutorial & algorithm walk-through
├── pyproject.toml
├── LICENSE
└── README.md
```

## API Overview

### Python — `RadonFanbeam` class (`huo.radon`)

The recommended interface.  Follows the
[torch-radon](https://torch-radon.readthedocs.io/en/latest/modules/radon.html)
convention: geometry is configured once in the constructor, then ``forward`` /
``backprojection`` / ``art`` operate on plain tensors.

| Method | Description |
|---|---|
| `RadonFanbeam(resolution, angles, source_distance, det_distance, det_count, det_spacing, volume_size, lat_sampling)` | Constructor — pre-computes fan-beam gantry coordinates |
| `forward(img)` | Radon transform over all angles (image → sinogram) |
| `backprojection(sinogram)` | Back-projection over all angles (sinogram → image) |
| `art(sinogram)` | ART iterative reconstruction (sinogram → image) |

### Python — low-level functions (`huo.art`)

| Function | Description |
|---|---|
| `forward_propagation(img, gantry_coor_x, gantry_coor_y, view, param)` | Forward-project the image at a single angle (Radon transform) |
| `backward_propagation(sinogram, img_end, detr_end, view, param)` | Back-project a sinogram column at a single angle |
| `scan(img, gantry_coor_x, gantry_coor_y, gantry_view, param)` | Full forward projection over all angles → sinogram |
| `art(sinogram, img_end, detr_end, gantry_coor_x, gantry_coor_y, gantry_view, param)` | ART iterative reconstruction → image |

### TypeScript (`js/src/art.ts`)

| Function | Description |
|---|---|
| `forward(img, gantryCoordX, gantryCoordY, angle, param)` | Radon transform for one angle |
| `backprojection(sinogramData, imgEnd, detEnd, angle, param)` | Back-projection for one angle |
| `scan(img, gantryCoordX, gantryCoordY, angles, param, ...)` | Full forward projection → sinogram |
| `art(sinogram, imgEnd, detEnd, gantryCoordX, gantryCoordY, angles, param, ...)` | ART reconstruction → image |

Terminology follows the [torch-radon](https://torch-radon.readthedocs.io/en/latest/) convention.

## CT Geometry Parameters

Both implementations share the same set of geometry parameters:

| Parameter | Python name | TypeScript name | Default | Description |
|---|---|---|---|---|
| Image pixels | `--img-pixels` | `imgPixels` | 512 | Number of pixels along each image axis |
| Image length | `--img-len` | `imgLen` | 144 mm | Diameter of the field of view |
| Detector count | `--detr-num` | `detrNum` | 500 | Number of detector elements |
| Detector length | `--detr-len` | `detrLen` | 180 mm | Detector panel length |
| Lateral sampling | `--lat-sampling` | `latSampling` | 2 | Lateral sampling grid multiplier |
| SDD | `--sdd` | `sdd` | 1200 mm | Source-to-detector distance |
| SOD | `--sod` | `sod` | 981 mm | Source-to-object distance |
| Rotation step | `--rotate-step` | `rotateStep` | 1.0° | Angular step between projections |

## References

- [Algebraic Reconstruction Technique — Wikipedia](https://en.wikipedia.org/wiki/Algebraic_reconstruction_technique)
- [torch-radon documentation](https://torch-radon.readthedocs.io/en/latest/) (API naming convention)
- [jax-js](https://github.com/ekzhang/jax-js) (TypeScript numerical computing library)

## Contributing

See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for development setup,
coding guidelines, and pull-request instructions.

## License

This project is released under the [GPL-3.0 License](LICENSE).
