# JAX Guide

Huo's JAX backend is a callable, immutable PyTree with native automatic
differentiation. Build a fixed scanner, evaluate its forward model, and
differentiate your loss with respect to an image or network parameters.
JAX derives the sampled transpose and higher derivatives from that model;
there is no public backprojection method or custom derivative rule.

## Install and project

From the repository root, install the JAX extra. This does not install PyTorch:

```bash
uv sync --locked --python 3.12 --extra jax
```

Python 3.11 is also supported. Import shared geometry from `huo`, and
backend-specific operators from `huo.jax`. Angles are radians and distances
are pixels; the [geometry specification](geometry.md) defines the conventions.
The following Python blocks can be run in sequence.

```python
import jax
import jax.numpy as jnp
import numpy as np

from huo import FanbeamGeometry
from huo.jax import FanbeamProjector, reconstruct

geometry = FanbeamGeometry(32, 128, 64, 40, 1.3)
angles = np.linspace(0, 2 * np.pi, 24, endpoint=False, dtype=np.float32)
projector = FanbeamProjector.create(geometry, angles, method="scan")
phantom = jnp.zeros((32, 32), dtype=jnp.float32).at[10:22, 12:20].set(1)

forward = jax.jit(lambda state, image: state(image))
measured = forward(projector, phantom)
assert measured.shape == (24, 40)
```

Images have shape `[..., N, N]` and projections have shape `[..., V, M]`.
Any leading batch dimensions are retained. Calling `projector(image)` eagerly
is supported; compile the enclosing computation for repeated work.

`create()` validates and prepares geometry outside JAX transformations. The
default float32 dtype must match the input. For float64, enable
`jax.config.update("jax_enable_x64", True)` before creating and evaluating
float64 arrays and pass `dtype=jnp.float64` to `create()`.

## Differentiate a loss

Pass scanner state as an explicit function argument. The following function
compiles loss evaluation and its image gradient together:

```python
def objective(image, state, target):
    residual = state(image) - target
    return 0.5 * jnp.mean(residual**2)

loss_and_grad = jax.jit(jax.value_and_grad(objective))
initial = jnp.zeros_like(phantom)
loss, gradient = loss_and_grad(initial, projector, measured)
assert gradient.shape == initial.shape

direction = jnp.ones_like(initial)
projection, tangent = jax.jvp(
    lambda image: forward(projector, image), (initial,), (direction,)
)
gradient, hvp = jax.jvp(
    jax.grad(lambda image: objective(image, projector, measured)),
    (initial,), (direction,),
)
assert tangent.shape == measured.shape and hvp.shape == initial.shape
```

For a network, form the image from its parameter PyTree inside the objective
and differentiate with respect to those parameters. Geometry and angles are
fixed: their array gradients are deliberately stopped. JVPs, VJPs, Jacobians,
and Hessians refer to image or network values, not scanner calibration.

## Reconstruct with Adam

The convenience helper compiles the complete Optax Adam loop with `lax.scan`.
It minimizes half mean squared projection residual, starting from zeros unless
an `initial_image` is supplied. Nonnegative projection is enabled by default.

```python
image = reconstruct(projector, measured, steps=200, lr=0.03)
final_loss = objective(image, projector, measured)
assert float(final_loss) < float(loss)
print("projection loss:", float(loss), "->", float(final_loss))
```

The compiled solver is cached across calls with matching shapes, dtypes, and
static settings. If wrapping `reconstruct` in another `jax.jit`, make `steps`,
`lr`, and `nonnegative` static. The result has the measurement's batch dimensions
followed by `[N, N]`. Inputs remain unchanged. Gradients through the helper's
optimization trajectory are stopped; use your own loop for differentiable
unrolling or a custom objective.

For example, this loop adds an image penalty and updates using Optax directly:

```python
import optax

optimizer = optax.adam(0.03)

def regularized_objective(image, state, target):
    return objective(image, state, target) + 1e-4 * jnp.mean(image**2)

@jax.jit
def train_step(image, optimizer_state, state, target):
    value, gradient = jax.value_and_grad(regularized_objective)(image, state, target)
    updates, optimizer_state = optimizer.update(gradient, optimizer_state, image)
    image = jnp.maximum(optax.apply_updates(image, updates), 0)
    return image, optimizer_state, value

fitted = jnp.zeros_like(phantom)
optimizer_state = optimizer.init(fitted)
for _ in range(50):
    fitted, optimizer_state, value = train_step(fitted, optimizer_state, projector, measured)
assert float(objective(fitted, projector, measured)) < float(loss)
```

This example dispatches one compiled step per Python iteration. Use `lax.scan`
around the updates to compile a complete fixed-length loop, as the helper does.
See [Optax's optimizer reference](https://optax.readthedocs.io/en/latest/api/optimizers.html)
for optimizer settings and alternatives.

## State, compilation, and devices

The projector has three dynamic array leaves: `angles`, `ray_x`, and `ray_y`.
Its static metadata is `resolution`, `ray_step`, and `method`. Distances and
detector spacing are encoded in the ray arrays. Changing array values at fixed
shapes and dtypes can reuse a compiled function; changing shapes or static
metadata creates another compiled signature. The caller retains the original
`FanbeamGeometry` configuration.

```python
leaves, structure = jax.tree.flatten(projector)
restored = jax.tree.unflatten(structure, leaves)
np.testing.assert_array_equal(forward(restored, phantom), measured)

images = jnp.stack((phantom, 0.5 * phantom))
batched = forward(projector, images)
mapped = jax.jit(jax.vmap(lambda image: projector(image)))(images)
np.testing.assert_allclose(batched, mapped, atol=1e-5, rtol=1e-5)

device = jax.devices()[0]
placed_projector = jax.device_put(projector, device)
placed_image = jax.device_put(phantom, device)
placed_projection = forward(placed_projector, placed_image)
```

You can also pass `device=` to `create()` or map array conversions over the
PyTree. Converting float32 state to float64 preserves its already rounded
coordinates; use a fresh float64 `create()` for double-precision preparation.
State uses JAX's ordinary device placement. This project's correctness and
performance validation currently covers CPU; no GPU speedup is claimed.
JAX's [dataclass PyTree reference](https://docs.jax.dev/en/latest/_autosummary/jax.tree_util.register_dataclass.html)
describes the distinction between array leaves and static metadata.

The default `method="scan"` iterates over views with checkpointing to reduce
retained interpolation data. `method="vmap"` vectorizes views and may improve
complete-loop performance at a substantial memory cost. Choose using your
workload; the [CPU comparison](jax_benchmark.md) records both strategies.
Explicit `custom_jvp` remains a benchmark experiment, not a production option.
For timings, separate compilation from steady-state execution and call
`jax.block_until_ready(result)` before stopping the timer.

## Validation

Shapes and dtypes are checked both eagerly and while tracing transformed code.
Ordinary eager calls also reject nonfinite images, measurements, initial images,
and reconstruction results. These checks synchronize values with the host.

Inside JAX transformations, finite-value checks use `checkify.debug_check`.
They are inactive unless you wrap the computation with `checkify.checkify`;
ordinary `jax.jit` does not raise Python exceptions for runtime NaN/Inf values.
Apply checkify before JIT, then inspect or throw its returned error:

```python
from jax.experimental import checkify

checked_forward = jax.jit(checkify.checkify(lambda state, image: state(image)))
error, checked_projection = checked_forward(projector, phantom)
error.throw()
np.testing.assert_array_equal(checked_projection, measured)

checked_solve = jax.jit(checkify.checkify(
    lambda state, target: reconstruct(state, target, steps=30, lr=0.03)
))
error, checked_image = checked_solve(projector, measured)
error.throw()
```

Checkify also works with mapped batches. Its checks add work; opt in when the
compiled program needs value validation. See JAX's
[checkify guide](https://docs.jax.dev/en/latest/debugging/checkify_guide.html).

## Command line

Save the example measurements and angles:

```python
np.save("measured.npy", np.asarray(measured))
np.save("angles.npy", angles)
```

Then run the same solver from the CLI:

```bash
uv run --locked --extra jax huo-reconstruct measured.npy --backend jax --method scan \
  --angles angles.npy --resolution 32 --source-distance 128 --det-distance 64 \
  --det-spacing 1.3 --steps 200 --lr 0.03 --output reconstructed.npy
```

The CLI imports only the selected backend and reports its name and the initial
and final projection losses. `--allow-negative` disables the nonnegative
constraint. Input files and output images use float32.
