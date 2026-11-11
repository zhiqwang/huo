# Huo

Differentiable fan-beam CT projection and image reconstruction with PyTorch.
Huo maps an image to predicted line integrals; a loss compares them with fixed
measurements, and a PyTorch optimizer updates the image or a neural network.

```text
image → FanbeamProjector → predicted sinogram → loss against measured sinogram
  ↑                                              │
  └──────── optimizer.step() ← loss.backward() ───┘
```

The public Python API contains `FanbeamGeometry`, `FanbeamProjector`, and
`reconstruct`. The projector is an `nn.Module` with fixed scanner buffers.
Its autograd implementation handles the discrete transpose internally.

## Installation

Python 3.11 or newer is required. Install the locked CPU environment with uv:

```bash
git clone https://github.com/zhiqwang/huo.git
cd huo
uv sync --locked --python 3.12
source .venv/bin/activate
```

The project pins PyTorch `2.15.0+cpu`. Images and measurements use CPU float32
or float64 and must match the projector's dtype. Angles are radians and all
scanner distances are in pixels.

## Optimize an image

```python
import math
import torch
from huo import FanbeamGeometry, FanbeamProjector

geometry = FanbeamGeometry(
    resolution=32,
    source_distance=128,
    det_distance=64,
    det_count=40,
    det_spacing=1.3,
)
projector = FanbeamProjector(geometry, torch.arange(24) * (math.pi / 12))

# Synthetic measurements; replace these with calibrated line-integral data.
phantom = torch.zeros(32, 32)
phantom[10:22, 12:20] = 1
measured = projector(phantom).detach()

image = torch.nn.Parameter(torch.zeros_like(phantom))
optimizer = torch.optim.Adam([image], lr=0.03)
for _ in range(200):
    optimizer.zero_grad(set_to_none=True)
    predicted = projector(image)
    loss = 0.5 * (predicted - measured).square().mean()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        image.clamp_min_(0)

reconstructed = image.detach()
```

The same loop is available as a convenience function:

```python
from huo import reconstruct
reconstructed = reconstruct(projector, measured, steps=200, lr=0.03)
```

`reconstruct` returns a detached image. Use the projector directly to choose
another optimizer, add regularization, or train a network. Native batch shapes
`[..., N, N] → [..., V, M]`, `torch.func` transforms, higher-order input
derivatives, module dtype conversion, and `state_dict` are supported.

## Command line

```bash
huo-reconstruct measured.npy --angles angles.npy \
  --resolution 32 --source-distance 128 --det-distance 64 --det-spacing 1.3 \
  --steps 200 --lr 0.03 --output reconstruction.npy
```

The detector count comes from the last measurement dimension. The angle file
must match the preceding dimension. Nonnegative image updates are enabled by
default; `--allow-negative` disables the constraint.

## Documentation and browser example

- [Tutorial](docs/tutorial.md): complete PyTorch loops, batches, networks, and CLI.
- [API](docs/api.md): geometry, module state, dtype, and optimizer contracts.
- [Autograd reconstruction](docs/autograd.md): the loss and its discrete gradient.
- [Geometry](docs/geometry.md): scanner coordinates and sampling weights.
- [Browser demo](https://zhiqwang.github.io/huo/demo/): a separate TypeScript ART
  example. It shares forward geometry with Python and uses its own residual
  backfill reconstruction algorithm.

The browser example and its [proof article](docs/backward_propagation_proof.md)
illustrate ART-style updates; Python reconstruction uses loss minimization.

## Checks

```bash
uv sync --locked --group test
npm --prefix packages ci
npm --prefix packages test
uv run --locked --group test pytest tests -q
uv run --locked python -m benchmarks.autograd
```

Tests cover gradients, higher derivatives, batches, optimizer behavior, module
state, and Python/TypeScript forward-projection parity. See
[documentation publishing](docs/publishing.md) for Sphinx builds and Lean checks.

## Repository structure

```text
huo/
  geometry.py                 Immutable scanner geometry and ray coordinates
  projector.py                FanbeamProjector(nn.Module)
  _autograd.py                Private projection/transpose autograd pair
  _validation.py              Tensor checks compatible with torch.func
  reconstruction.py           Adam image optimization
  cli.py                      NumPy input/output
packages/                     TypeScript ART browser example
tests/                        Python regressions and shared geometry fixtures
benchmarks/                   CPU timing and retained autograd storage
docs/                         English Sphinx documentation
lean/                         Formal results for residual backprojection
```

## License

[GPL-3.0](LICENSE).
