# Huo

Fan-beam CT forward projection and ART-inspired residual reconstruction with
Python / PyTorch and TypeScript / jax-js implementations.

Both implementations use **the same geometry fields, radians, pixel units,
and `[angle, detector]` sinograms**. The public names and scanner orientation
follow [torch-radon](https://github.com/matteo-ronchetti/torch-radon).
See [Geometry](docs/geometry.md) for the common notation and diagrams, and the
[Tutorial](docs/tutorial.md) for runnable examples.

## Iterative update

A reconstruction starts from zero and visits each angle once in random order:

1. Forward-project the current image to predict one detector row.
2. Subtract that row from the measured sinogram row.
3. Divide by the ray-window length, `resolution` in pixel units.
4. Backproject the residual and clamp negative image values to zero.

The implementation uses uniform residual backfill. Its equivalence to a
Kaczmarz projection and its convergence require additional assumptions. The
[mathematical analysis](docs/backward_propagation_proof.md) separates these
claims and includes executable Lean proofs.

## Installation

Use Python 3.11 or newer for current PyTorch releases. Install PyTorch
explicitly, then Huo:

```bash
git clone https://github.com/zhiqwang/huo.git
cd huo
python -m pip install torch
python -m pip install -e .
```

Open the [online browser demo](https://zhiqwang.github.io/huo/demo/) to run forward
projection and ART reconstruction in your browser, or build it locally:

```bash
cd packages
npm ci
npm run build
python -m http.server 8080 --bind 0.0.0.0
```

Open `http://localhost:8080/demo/`, or use the server's address from another
machine. jax-js loads through the demo's CDN import map.

To use the npm static server, run
`npx serve . --listen tcp://0.0.0.0:3000 --no-clipboard`
and open `http://localhost:3000/demo/`.

## Python example

```python
import math
import torch
from huo import RadonFanbeam

radon = RadonFanbeam(
    resolution=64,
    angles=torch.arange(24, dtype=torch.float32) * (math.pi / 12),
    source_distance=436,
    det_distance=97.33333333333333,
    det_count=80,
    det_spacing=1,
    ray_sampling=2,
)
image = torch.zeros(64, 64, dtype=torch.float32)
image[20:44, 24:40] = 1
sinogram = radon.forward(image)           # [24, 80]: angle, detector
backprojected = radon.backprojection(sinogram)
reconstructed = radon.art(sinogram)       # [64, 64]
```

Use CPU `float32` tensors. Geometry is immutable and cached in `radon.geometry`.
The TypeScript class accepts an object with these exact field names; its
methods return promises of jax-js arrays.

## Command line

```bash
huo-art sinogram.npy --angles angles.npy \
  --resolution 64 --source-distance 436 --det-distance 97.33333333333333 \
  --det-count 80 --det-spacing 1 --ray-sampling 2 --output reconstruction.npy
```

`sinogram.npy` must have shape `[V, M]`. `angles.npy` is a one-dimensional
radian array in row order. Alternatively, `--angle-step` generates uniform
angles in `[0, 2π)`; its default is `π/180`. The two angle options are mutually
exclusive. The CLI casts NumPy values to `float32` and runs one sweep.

This is an API migration. Earlier Huo data needs angle, axis, detector-order,
and length-unit conversion; follow [Migrating earlier data](docs/geometry.md#migrating-earlier-huo-data).

## Shared API

Both languages expose `RadonFanbeam` with these methods:

| Method | Input → output |
| --- | --- |
| `forward(image)` | `[N, N]` → `[V, M]` |
| `backprojection(sinogram)` | `[V, M]` → `[N, N]`, summed over angles |
| `art(sinogram)` | `[V, M]` → `[N, N]`, one clipped residual sweep |
| `forward_angle(image, angle)` | `[N, N]` → `[M]` |
| `backprojection_angle(projection, angle)` | `[M]` → `[N, N]` |

| Field | Meaning |
| --- | --- |
| `resolution` | $N$: image pixels per axis |
| `angles` | $V$ angles in radians, in sinogram row order |
| `source_distance` | $d_s$: source to image centre, in pixels |
| `det_distance` | $d_d$: image centre to detector plane, in pixels |
| `det_count` | $M$: detector elements |
| `det_spacing` | $\Delta u$: detector spacing, in pixels |
| `ray_sampling` | $q_{\rm ray}$: integration intervals per pixel; defaults to 2 |

`FanbeamGeometry` derives `source_detector_distance`, `pixel_extent`,
`det_extent`, `ray_sample_count`, and `ray_step`. At angle zero the source is
`(0, source_distance)` and the detector plane is `y = -det_distance`.

## Repository structure

```text
huo/
  geometry.py                 Shared Python geometry
  art.py                      Per-angle operators
  radon.py                    RadonFanbeam
  cli.py                      CLI and NumPy I/O
packages/
  src/geometry.ts             Same geometry fields and coordinate rules
  src/art.ts                  TypeScript RadonFanbeam
  demo/                       Interactive browser demo
  tests/                      TypeScript regressions
tests/                        Python regressions and shared fixtures
docs/                         English Sphinx documentation
lean/Huo/BackwardPropagation.lean  Formal algebra and convergence lemmas
```

## Checks

With PyTorch, pytest, and the JavaScript dependencies installed:

```bash
npm --prefix packages test
python -m pytest tests
```

The Python suite also compares the built TypeScript operators on shared signed
inputs when Node and the JavaScript dependencies are available.

For documentation previews and Lean verification, see
[Maintaining and Publishing Documentation](docs/publishing.md).

## Contributing

Issues and pull requests are welcome. Keep geometry notation and conventions
consistent across Python, TypeScript, documentation, diagrams, and proofs.

## License

[GPL-3.0](LICENSE).
