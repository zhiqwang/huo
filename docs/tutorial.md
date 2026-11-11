---
tocdepth: 2
---

# Tutorial

Huo's Python workflow is an ordinary PyTorch optimization loop. A fixed
`FanbeamProjector` maps the current image to predicted measurements. A scalar
loss compares the prediction with a fixed measured sinogram. `loss.backward()`
computes image gradients, and `optimizer.step()` updates the image.

Raw detector intensities must first be calibrated and converted to line
integrals. Huo reads line integrals; it does not perform that preprocessing.

## Install

Use Python 3.11 or newer. From the repository root:

```bash
uv sync --locked --python 3.12
source .venv/bin/activate
```

PyTorch `2.15.0+cpu` is pinned. CPU float32 and float64 are supported. Scanner
distances are in pixels and angles are in radians. See [Geometry](geometry.md)
for the source location, detector orientation, and ray sampling convention.

## A complete reconstruction loop

This script creates synthetic measurements and fits a nonnegative image.
The target is computed once and detached before optimization.

```python
import math
import numpy as np
import torch
from huo import FanbeamGeometry, FanbeamProjector

geometry = FanbeamGeometry(
    resolution=32,
    source_distance=128,
    det_distance=64,
    det_count=40,
    det_spacing=1.3,
    ray_sampling=2,
)
angles = torch.arange(24, dtype=torch.float32) * (math.pi / 12)
projector = FanbeamProjector(geometry, angles)

phantom = torch.zeros(32, 32)
phantom[10:22, 12:20] = 1
with torch.no_grad():
    measured = projector(phantom)

image = torch.nn.Parameter(torch.zeros_like(phantom))
optimizer = torch.optim.Adam([image], lr=0.03)
initial_loss = 0.5 * measured.square().mean()

for step in range(200):
    optimizer.zero_grad(set_to_none=True)
    predicted = projector(image)
    loss = 0.5 * (predicted - measured).square().mean()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        image.clamp_min_(0)

reconstructed = image.detach()
with torch.no_grad():
    final_loss = 0.5 * (projector(reconstructed) - measured).square().mean()
print("loss:", initial_loss.item(), "->", final_loss.item())
np.save("measured.npy", measured.numpy())
np.save("angles.npy", angles.numpy())
np.save("reconstruction.npy", reconstructed.numpy())
```

The learning rate and iteration count are example settings, not a convergence
criterion. Monitor the loss for the acquisition and choose regularization when
needed. A small projection loss establishes data agreement; it does not by
itself establish recovery of a unique image.

`FanbeamProjector` contains no trainable parameters. The optimizer above owns
`image`, which is a leaf `nn.Parameter`. Each iteration builds a fresh graph;
no `retain_graph=True` or explicit backprojection call is needed.

## The reconstruction helper

The helper performs the same Adam loop, using half mean squared error:

```python
from huo import reconstruct

result = reconstruct(projector, measured, steps=200, lr=0.03)
```

It starts from zeros unless `initial_image` is provided. Nonnegativity is
applied before the first forward and after every optimizer step. Set
`nonnegative=False` to fit signed images. Measurements and the initial image
are copied; the helper does not change their values or gradients. Optimization
works inside an outer `torch.no_grad()` or `torch.inference_mode()` context.

The result is detached. The helper is an image solver, with no gradient through
its optimizer trajectory. Use the projector in your own training loop when
learning model parameters or implementing a differentiable unrolled solver.
See the [API reference](api.md#reconstruct) for the full argument contract.

## Batches and dtype

The projector accepts arbitrary leading batch dimensions:

| Value | Shape |
| --- | --- |
| One image | `[N, N]` |
| One sinogram | `[V, M]` |
| Images with leading batch dimensions | `[..., N, N]` |
| Corresponding measurements | `[..., V, M]` |

`N` is the image resolution, `V` is the number of angles, and `M` is the detector
count. Angle order is preserved, including unsorted or repeated angles.

```python
images = torch.stack([phantom, 0.5 * phantom])
measurements = projector(images)                    # [2, 24, 40]
results = reconstruct(projector, measurements, steps=100, lr=0.03)
assert results.shape == (2, 32, 32)

projector64 = FanbeamProjector(geometry, angles, dtype=torch.float64)
projections64 = projector64(phantom.double())
```

Inputs must be finite and match the module's dtype and device. Constructing with
`dtype=torch.float64` generates ray coordinates in double precision. Calling
`projector.double()` converts existing buffers, including their stored sampling
coordinates. Module state contains geometry, angles, and those coordinates.
Reload state into a projector with matching geometry and angle count.

## Train a neural network

For a learned image estimate, give the optimizer the network's parameters:

```python
network = torch.nn.Sequential(
    torch.nn.Linear(8, 32 * 32),
    torch.nn.Unflatten(-1, (32, 32)),
    torch.nn.Softplus(),
)
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
latent = torch.ones(1, 8)
target = measured.detach().unsqueeze(0)

for _ in range(100):
    optimizer.zero_grad(set_to_none=True)
    image_prediction = network(latent)
    prediction = projector(image_prediction)
    loss = 0.5 * (prediction - target).square().mean()
    loss.backward()
    optimizer.step()
```

`Softplus` enforces positive network outputs. Do not detach the predicted image
between the network and projector: autograd follows that connection to the
network parameters. Geometry and angles remain fixed.

## Custom losses and derivatives

Use standard PyTorch operations to add penalties. For example, squared image
finite differences give a differentiable smoothness penalty:

```python
image = torch.nn.Parameter(torch.zeros(32, 32))
optimizer = torch.optim.Adam([image], lr=0.03)
for _ in range(100):
    optimizer.zero_grad(set_to_none=True)
    data_loss = 0.5 * (projector(image) - measured).square().mean()
    smoothness = (image[1:] - image[:-1]).square().mean()
    smoothness = smoothness + (image[:, 1:] - image[:, :-1]).square().mean()
    loss = data_loss + 1e-3 * smoothness
    loss.backward()
    optimizer.step()
```

Input gradients, higher-order gradients, `torch.func.grad`, `jvp`, `jacrev`,
`jacfwd`, `hessian`, and `vmap` are supported. Native batches normally suffice
for independent images; `vmap` also composes with these transforms. The
[autograd article](autograd.md) derives the gradient and explains memory usage.

## CLI

Reconstruct the files written by the first example:

```bash
huo-reconstruct measured.npy --angles angles.npy \
  --resolution 32 --source-distance 128 --det-distance 64 --det-spacing 1.3 \
  --ray-sampling 2 --steps 200 --lr 0.03 --output cli-reconstruction.npy
```

The angle file and output path are required. Detector count is inferred from
the last sinogram dimension. The CLI converts numeric NumPy inputs to float32,
checks geometry and measurement shape, reports the initial/final loss, and
writes the image. `--allow-negative` disables nonnegative projection.
Run `huo-reconstruct --help` for all options.

## Checks and benchmarks

```bash
uv sync --locked --group test
npm --prefix packages ci
npm --prefix packages test
uv run --locked --group test pytest tests -q
uv run --locked python -m benchmarks.autograd --batch-size 2
```

Tests check the actual sampled matrix, finite differences, gradient transforms,
module state, optimizer behavior, and forward-projection parity with TypeScript.
Memory regressions bound retained sampling storage. Benchmarks report timings
and unique live tensor storage; wall-clock thresholds are not enforced in CI.

## Browser demo

The [browser demo](demo/index.md) is a separate TypeScript ART-style example.
It shares coordinates, ray sampling, and sinogram layout with the Python
projector. Its reconstruction backfills one angle's residual at a time and
uses a random angle order. That update is not the Python Adam loss loop.

Build the standalone example with:

```bash
npm --prefix packages ci
npm --prefix packages run build
python -m http.server 8080 --bind 0.0.0.0 --directory packages
```

Open `http://localhost:8080/demo/`. Use the
[Sphinx preview](publishing.md#local-preview-and-strict-build) for the complete
site and themed demo. The demo loads jax-js from `esm.sh` and requires access
to that CDN. Its typed-array sampling loops run in JavaScript.
