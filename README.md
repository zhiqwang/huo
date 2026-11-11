# Huo

Differentiable fan-beam CT projection and image reconstruction with PyTorch, JAX and browser jax-js.
Huo maps an image to predicted line integrals; a loss compares them with fixed
measurements, and an optimizer updates the image or a neural network.

```text
image → FanbeamProjector → predicted sinogram → loss against measured sinogram
  ↑                                              │
  └──────── parameter update ← automatic gradient ┘
```

`huo.FanbeamGeometry` is shared. Choose `huo.torch` for an `nn.Module` projector,
or `huo.jax` for an immutable PyTree projector. Both backends expose
`FanbeamProjector` and `reconstruct`, and compute the sampled transpose through
automatic differentiation. Only the selected backend is imported.

## Installation

Python 3.11 or newer is required. Install the locked CPU environment with uv:

```bash
git clone https://github.com/zhiqwang/huo.git
cd huo
uv sync --locked --python 3.12 --extra torch
source .venv/bin/activate
```

Use `--extra jax` for JAX/Optax, or `--all-extras` for both. The base package
requires only NumPy. Python 3.11 and 3.12 are tested; the lockfile selects the
appropriate JAX version. PyTorch is pinned to `2.15.0+cpu` from its CPU index.
For a wheel installation outside uv, install the PyTorch extra with
`pip install 'huo[torch]' --extra-index-url https://download.pytorch.org/whl/cpu`,
or use `pip install 'huo[jax]'` for JAX.

Python images and measurements are float32/float64 and must match the projector's
dtype; the browser backend supports float32. Angles are radians and scanner
distances are in pixels. The PyTorch backend is CPU-only; JAX uses its configured
device, with CPU correctness covered by this project's tests.

## Optimize an image with PyTorch

```python
import math
import torch
from huo import FanbeamGeometry
from huo.torch import FanbeamProjector

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
from huo.torch import reconstruct
reconstructed = reconstruct(projector, measured, steps=200, lr=0.03)
```

`reconstruct` returns a detached image. Use the projector directly to choose
another optimizer, add regularization, or train a network. Native batch shapes
`[..., N, N] → [..., V, M]`, `torch.func` transforms, higher-order input
derivatives, module dtype conversion, and `state_dict` are supported.

## Reconstruct with JAX

```python
import jax
import jax.numpy as jnp
from huo import FanbeamGeometry
from huo.jax import FanbeamProjector, reconstruct

geometry = FanbeamGeometry(32, 128, 64, 40, 1.3)
angles = jnp.arange(24, dtype=jnp.float32) * (jnp.pi / 12)
projector = FanbeamProjector.create(geometry, angles, method="scan")
phantom = jnp.zeros((32, 32), dtype=jnp.float32).at[10:22, 12:20].set(1)
measured = jax.jit(lambda state, image: state(image))(projector, phantom)
reconstructed = reconstruct(projector, measured, steps=200, lr=0.03)
```

The JAX helper compiles the complete Adam loop. Pass the projector explicitly
to JIT-compiled losses and training steps; its arrays are dynamic PyTree leaves.
Native AD supports JVPs, VJPs, Hessians, and `vmap`. The default `scan` method
recomputes sampling intermediates per view; `vmap` is an explicit alternative
with higher memory demand. See the [JAX guide](docs/jax.md) for custom losses,
device placement, float64, and compiled validation.

## Command line

```bash
huo-reconstruct measured.npy --backend torch --angles angles.npy \
  --resolution 32 --source-distance 128 --det-distance 64 --det-spacing 1.3 \
  --steps 200 --lr 0.03 --output reconstruction.npy
```

The detector count comes from the last measurement dimension. The angle file
must match the preceding dimension. Nonnegative image updates are enabled by
default; `--allow-negative` disables the constraint.
Select JAX with `--backend jax`; `--method scan|vmap` applies to that backend.

## Documentation and browser example

- [Tutorial](docs/tutorial.md): complete PyTorch loops, batches, networks, and CLI.
- [JAX guide](docs/jax.md): PyTree state, JIT, Optax, and input contracts.
- [jax-js guide](docs/jax_js.md): native browser gradients, ownership, and chunked Adam.
- [API](docs/api.md): geometry, module state, dtype, and optimizer contracts.
- [Autograd reconstruction](docs/autograd.md): the loss and its discrete gradient.
- [JAX comparison](docs/jax_benchmark.md): the original JAX reproduction with
  numerical tests, CPU benchmarks, compilation costs, and memory measurements.
- [Geometry](docs/geometry.md): scanner coordinates and sampling weights.
- [Browser demo](https://zhiqwang.github.io/huo/demo/): TypeScript loss optimization
  using the same forward model and Adam objective, with progress and cancellation.

The [proof article](docs/backward_propagation_proof.md) preserves mathematical
analysis of the former ART-style residual-backfill algorithm. Current Python
and browser reconstruction both use loss minimization.

## Checks

```bash
uv sync --locked --all-extras --group test
npm --prefix packages ci
npm --prefix packages test
uv run --locked --all-extras --group test pytest tests -q
uv run --locked --extra torch python -m benchmarks.autograd
```

Tests cover gradients, higher derivatives, batches, optimizer behavior, module
state, and Python/TypeScript projection, derivative, and reconstruction parity. See
[documentation publishing](docs/publishing.md) for Sphinx builds and Lean checks.

## Repository structure

```text
huo/
  geometry.py                 Shared geometry and NumPy ray preparation
  torch/                      nn.Module, private autograd pair, Adam solver
  jax/                        PyTree, native AD sampling, compiled Optax solver
  cli.py                      Backend selection and NumPy input/output
packages/                     Differentiable jax-js operators and browser demo
tests/                        Python regressions and shared geometry fixtures
benchmarks/                   CPU timing and retained autograd storage
docs/                         English Sphinx documentation
lean/                         Formal results for residual backprojection
```

## License

[GPL-3.0](LICENSE).
