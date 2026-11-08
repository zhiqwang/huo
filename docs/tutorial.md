---
tocdepth: 2
---

# Huo ART Tutorial

Huo provides a sampled fan-beam projector and a residual reconstruction loop.
Python and TypeScript use the same `RadonFanbeam` interface and geometry:
**radians, pixel units, and `[angle, detector]` sinograms**.
Use [Geometry](geometry.md) for the diagrams and notation reference.

(background)=
## Overview

Forward projection interpolates an image along source-to-detector rays and
sums the samples with a distance weight. Repeating this for each angle creates
a sinogram. Reconstruction compares each measured row with a predicted row
and backprojects the residual to update the image.

```text
image [N, N] ── forward ──> sinogram [V, M] ── art ──> estimate [N, N]
```

$N$ is `resolution`, $M$ is `det_count`, and $V$ is the number of `angles`.
Raw detector intensities must be calibrated and converted to line integrals
before using Huo; the loader does not perform this preprocessing.

| File | Responsibility |
| --- | --- |
| `huo/geometry.py` | Immutable `FanbeamGeometry` and zero-angle ray coordinates. |
| `huo/art.py` | Shared per-angle `forward_angle` and `backprojection_angle` operators. |
| `huo/radon.py` | `RadonFanbeam`, cached geometry, and multi-angle methods. |
| `huo/cli.py` | NumPy input/output and the command-line entry point. |
| `packages/src/geometry.ts` | The same geometry fields and coordinate construction. |
| `packages/src/art.ts` | TypeScript `RadonFanbeam` and progress callbacks. |
| `packages/demo/main.ts` | Phantom generation and canvas animation. |

(python-usage)=
## Python quick start

### Install

From a repository checkout, use an activated Python environment:

```bash
python -m pip install torch
python -m pip install -e .
```

Huo declares Python 3.9 or newer and NumPy as requirements. Use Python 3.11 or
newer with current PyTorch releases, and install PyTorch separately. The
documentation build environment is described in
[Maintaining and Publishing Documentation](publishing.md).

### Reconstruct a phantom

Save this as a Python script and run it from a writable directory. All scanner
lengths are in pixels, and the 24 angles are in radians.

(forward-projection-scan)=
(art-reconstruction)=
```python
import math
import numpy as np
import torch

from huo import RadonFanbeam

resolution = 64
angles = torch.arange(24, dtype=torch.float32) * (math.pi / 12)
radon = RadonFanbeam(
    resolution=resolution,
    angles=angles,
    source_distance=436.0,
    det_distance=97.33333333333333,
    det_count=80,
    det_spacing=1.0,
    ray_sampling=2,
)

axis = (torch.arange(resolution, dtype=torch.float32) + 0.5) / resolution - 0.5
y, x = torch.meshgrid(axis, axis, indexing="ij")
image = ((x / 0.25) ** 2 + (y / 0.175) ** 2 <= 1.0).to(torch.float32)

sinogram = radon.forward(image)
backprojected = radon.backprojection(sinogram)
torch.manual_seed(0)
reconstructed = radon.art(sinogram)

predicted = radon.forward(reconstructed)
relative_residual = (predicted - sinogram).norm() / sinogram.norm()
print("sinogram:", tuple(sinogram.shape))            # (24, 80)
print("backprojection:", tuple(backprojected.shape)) # (64, 64)
print("reconstruction:", tuple(reconstructed.shape)) # (64, 64)
print("relative projection residual:", relative_residual.item())

np.save("phantom_sinogram.npy", sinogram.numpy())
np.save("phantom_angles.npy", angles.numpy())
np.save("phantom_reconstruction.npy", reconstructed.numpy())
```

| Method | Result |
| --- | --- |
| `forward(image)` | A sinogram in the supplied angle order. |
| `backprojection(sinogram)` | The sum of uniform backfills over all angle rows. No filter, view-count division, residual correction, or nonnegative clamp. |
| `art(sinogram)` | One randomly ordered sweep from zero, using normalized residual backfill and nonnegative clipping. |
| `forward_angle(image, angle)` | One detector row `[M]`. |
| `backprojection_angle(projection, angle)` | One image `[N, N]` from a detector row `[M]`. |

The sum returned by `backprojection` grows with the number of views. Its scale
is not directly comparable to the ART estimate. A small projection residual
measures data agreement, not exact image recovery; see the
[proof article](backward_propagation_proof.md).

## Data layout

The Python implementation accepts finite **CPU `torch.float32` tensors**.
It validates input shape, dtype, and device. Moving only an input to CUDA is
insufficient; the current operators use cached CPU coordinates.

| Value | Shape | Interpretation |
| --- | --- | --- |
| Image | `[N, N]` | Rows are y, columns are x. |
| `angles` | `[V]` | Radians, in sinogram row order. |
| Sinogram | `[V, M]` | `sinogram[j, :]` belongs to `angles[j]`. |
| One projection | `[M]` | Detector values in increasing $u$ order. |

Using the `radon` object above:

```python
values = np.asarray(np.load("phantom_sinogram.npy"), dtype=np.float32)
expected = (radon.angles.numel(), radon.geometry.det_count)
if values.shape != expected:
    raise ValueError(f"Expected angle-by-detector shape {expected}, got {values.shape}")
if not np.isfinite(values).all():
    raise ValueError("The sinogram contains NaN or infinity")
torch.manual_seed(0)
reloaded = radon.art(torch.from_numpy(values))
```

Keep geometry and the angle array with each acquisition. For data from older
Huo versions, use the [migration procedure](geometry.md#migrating-earlier-huo-data),
which includes orientation and length-unit conversion as well as axis order.

(geometry)=
(ct-geometry-setup)=
(coordinate-preparation)=
(sampling-grid)=
(parameter-reference)=
(parameters)=
See [Geometry](geometry.md) for scanner diagrams, the
[Sampling grid](geometry.md#sampling-grid), and [Parameters](geometry.md#parameters).

(algorithm-overview)=
## Iterative update

Both implementations perform the same per-angle update. The ray-window length
is $L=N$ pixels:

```text
image = zeros(N, N)
for j in a random permutation of the angle indices:
    predicted = forward_angle(image, angles[j])
    residual = (sinogram[j, :] - predicted) / N
    correction = backprojection_angle(residual, angles[j])
    image = maximum(image + correction, 0)
```

Each call starts from zero and visits each angle once. There is no initial-image
argument, epoch count, stopping tolerance, or exposed relaxation parameter.
A second call starts a new reconstruction. Repeated sweeps would need to retain
and update the image between passes.

### Forward projection

`compute_ray_coordinates` constructs $K+1=q_{\rm ray}N+1$ samples per ray, with spacing
$1/q_{\rm ray}$. `forward_angle` rotates them by $R_{-\theta}$, samples the image with
bilinear interpolation and zero padding, and sums along each ray. It integrates
a fixed distance window of length $N$ rather than a complete square chord.

### Backprojection

`backprojection_angle` rotates pixel centres by $R_\theta$ and computes
$u=Dx'/(d_s-y')$, then linearly interpolates the 1-D detector row at fractional
index $u/\Delta u+(M-1)/2$, with zero padding. Python constructs the 2-D pixel
grid with `meshgrid`; no synthetic image or batch dimensions are needed for
this detector interpolation. It performs uniform residual backfill. Its weights
differ from the transpose of the sampled forward operator and from a weighted
fan-beam backprojection.
The [mathematical analysis](backward_propagation_proof.md) states the assumptions
needed for Kaczmarz equivalence or convergence.

(cli-tool)=
## CLI

The command uses the same parameter names and units as the API. This reconstructs
the data generated above, preserving the saved angle order:

```bash
huo-art phantom_sinogram.npy \
  --angles phantom_angles.npy \
  --resolution 64 \
  --source-distance 436 \
  --det-distance 97.33333333333333 \
  --det-count 80 \
  --det-spacing 1 \
  --ray-sampling 2 \
  --output phantom_cli_reconstruction.npy
```

For uniform acquisitions, `--angle-step` generates angles in $[0,2\pi)$.
The default step is $\pi/180$ radians, giving 360 rows. Use either `--angles`
or `--angle-step`. `det_distance` is measured from the image centre, so the
source-to-detector distance is `source_distance + det_distance`.

The CLI converts numeric NumPy input to `float32` and validates its shape. It
runs one ART sweep. With no `--output`, it computes the image without saving it.
`python tools/projection.py` delegates to this same CLI. Run `huo-art --help`
for the full option list.

(browser-demo-typescript)=
## Browser demo

From the repository root:

```bash
cd packages
npm ci
npm run build
python -m http.server 8080 --bind 0.0.0.0
```

Open `http://localhost:8080/demo/`, or replace `localhost` with the server's
address from another device. This serves the browser demo; the Sphinx preview
has its own [instructions](publishing.md#local-preview-and-strict-build).
The import map loads jax-js from a CDN, so the demo needs network access.

The demo builds one `RadonFanbeam`, generates a phantom on pixel centres,
then calls `forward` and `art` with progress callbacks. The sinogram canvas
has detector position on its horizontal axis and angle index vertically,
matching `[angle, detector]` storage. The default is `128 × 128`, 200 detectors,
and 180 angles. Frame delay is read when each phase starts. ART callbacks
arrive in random angle order, so the progress bar counts completed updates.

## TypeScript API

The same class and parameter fields are available from `packages/src/art.ts`.
Methods are asynchronous and use jax-js arrays:

```typescript
import { init, numpy as np } from "@jax-js/jax";
import { RadonFanbeam } from "../src/art.js";

await init();
const radon = new RadonFanbeam({
  resolution: 64,
  angles: Array.from({ length: 24 }, (_, j) => j * Math.PI / 12),
  source_distance: 436,
  det_distance: 97.33333333333333,
  det_count: 80,
  det_spacing: 1,
  ray_sampling: 2,
});
const image = np.array(new Float32Array(64 * 64).fill(1)).reshape([64, 64]);
const sinogram = await radon.forward(image);           // [24, 80]
const backprojected = await radon.backprojection(sinogram);
const reconstructed = await radon.art(sinogram);       // [64, 64]

image.dispose();
sinogram.dispose();
backprojected.dispose();
reconstructed.dispose();
```

`forward` and `art` accept an optional second argument
`{on_progress, delay}`. `on_progress(data, angle_index)` receives the current
`Float32Array` and the original angle index. Callbacks are awaited; buffers
are reused, so copy with `.slice()` when retaining a snapshot. `delay` is
additional milliseconds after each callback.

jax-js arrays have explicit ownership. `await array.data()` consumes a
reference; use `await array.ref.data()` to inspect an array you will reuse.
The operator retains caller-owned inputs during reads. Dispose inputs and
returned arrays when finished, as the example does.

## Practical limits

Geometry fields, rotations, pixel centres, array axes, padding, and normalization
are shared by Python and TypeScript. Their floating-point evaluation paths
and random generators differ: Python uses `torch.randperm`, while TypeScript
uses Fisher–Yates with `Math.random`. Matching geometry does not promise
bitwise-identical reconstructions or an adjoint identity.

The TypeScript sampling and update loops run over typed arrays in JavaScript.
jax-js provides array storage and backend initialization; choosing a GPU
backend does not move these loops onto the GPU.

For an unexpected result, check radians, angle order, `[V, M]` layout, and
pixel-length units first. Keep source-to-centre and centre-to-detector distances
separate. The [code analysis](backward_propagation_proof.md#code-limits) and
[Lean proofs](backward_propagation_proof.md#lean-proofs) explain the mathematical
coverage and remaining reconstruction limits.
