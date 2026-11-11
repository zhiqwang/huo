# Python API

```python
from huo import FanbeamGeometry
```

Choose `from huo.torch import FanbeamProjector, reconstruct` or
`from huo.jax import FanbeamProjector, reconstruct`. The base package has only
a NumPy dependency; `huo[torch]` and `huo[jax]` install the respective backend.
The top-level namespace exports only `FanbeamGeometry`.

Both projectors handle the forward physical model. Their framework owns losses,
gradients, and parameter updates. The transpose is internal to differentiation;
there is no separate backprojection, adjoint, or ART entry point.

## FanbeamGeometry

`FanbeamGeometry(resolution, source_distance, det_distance, det_count,
det_spacing, ray_sampling=2)` is an immutable dataclass.

| Field | Contract |
| --- | --- |
| `resolution` | Integer `N >= 2`; pixels along each image axis. |
| `source_distance` | Positive finite source-to-centre distance, greater than `N/2`. |
| `det_distance` | Positive finite centre-to-detector-plane distance. |
| `det_count` | Integer `M >= 2`; number of detector elements. |
| `det_spacing` | Positive finite detector-centre spacing. |
| `ray_sampling` | Positive integer; ray integration intervals per pixel. |

All lengths are in pixels. Derived properties are `source_detector_distance`,
`pixel_extent`, `det_extent`, `ray_sample_count`, and `ray_step`.
[Geometry](geometry.md) defines their coordinates and formulas.

## FanbeamProjector

`huo.torch.FanbeamProjector(geometry, angles, *, dtype=torch.float32)` is a
`torch.nn.Module`. Call `projector(image)` to evaluate its forward model.

| Argument or result | Contract |
| --- | --- |
| `geometry` | A `FanbeamGeometry` instance. |
| `angles` | Nonempty, finite 1-D sequence of `V` radians. Input storage and any input gradient graph are detached and copied. |
| `dtype` | `torch.float32` or `torch.float64`. |
| Input image | Finite CPU tensor of shape `[..., N, N]`, matching the module's dtype. Empty batches are rejected. |
| Output | Tensor of shape `[..., V, M]`, preserving all leading dimensions and angle order. |

The module has no trainable parameters. Its `geometry` property is immutable;
`angles` and the ray coordinates are buffers. Do not optimize scanner buffers:
only image values participate in differentiation. Construct a new module to
configure a different acquisition geometry.

First and higher-order image derivatives use a private pair of custom autograd
Functions. `torch.func.grad`, `jvp`, `jacrev`, `jacfwd`, `hessian`, and `vmap`
compose with the module. Finite-input checks also validate mapped batches.

### Dtype and state

Use `dtype=torch.float64` in the constructor to compute double-precision ray
coordinates. Standard module conversions such as `.double()` convert existing
buffers. Input tensors must be converted separately.

`state_dict()` contains angles, ray coordinates, and scanner geometry. Load it
into a module with the same geometry and angle count. Geometry mismatches raise
`ValueError`. Saving the coordinates preserves the actual sampled operator,
including any earlier dtype conversion.

```python
import torch
from huo import FanbeamGeometry
from huo.torch import FanbeamProjector

geometry = FanbeamGeometry(8, 32, 16, 12, 1.3)
projector = FanbeamProjector(geometry, [0.0, 0.5])
torch.save(projector.state_dict(), "projector.pt")
restored = FanbeamProjector(geometry, [0.0, 0.0])
restored.load_state_dict(torch.load("projector.pt", weights_only=True))
```

## JAX FanbeamProjector

`huo.jax.FanbeamProjector.create(geometry, angles, *, dtype=jnp.float32,
method="scan", device=None)` prepares an immutable PyTree outside JIT.
Call `projector(image)` for the same `[..., N, N] -> [..., V, M]` mapping.

| Part | Contract |
| --- | --- |
| Array leaves | Owned `angles`, `ray_x`, and `ray_y` arrays. They are dynamic JIT inputs; geometry gradients are stopped. |
| Static metadata | `resolution`, `ray_step`, and `method`. Physical distances are encoded in the rays, so changing their values at fixed shapes can reuse compiled code. |
| `method` | `"scan"` (default) uses checkpointed per-view sampling; `"vmap"` vectorizes angles and can require much more memory. |
| `dtype` | float32 or float64; float64 requires JAX x64 mode to be enabled before construction and evaluation. |
| `device` | A JAX device/sharding or None for default placement. Use `jax.device_put` to move the whole PyTree. |
| Input | JAX/NumPy floating array with nonempty trailing shape `[N, N]`, matching the projector dtype. |
| Properties | `views`, `det_count`, `dtype`, plus the state fields above. |

Pass the projector as an explicit argument to a compiled function. PyTree
flatten/unflatten preserves all sampled-operator state. Array conversions via
`jax.tree.map` convert existing coordinates; a new float64 `create()` computes
coordinates in double precision. The original physical configuration is kept
by the caller, not stored as a static JIT cache key.

Eager calls reject nonfinite images. Under JAX transformations, shape/dtype
checks remain active, while finite-value checks require `checkify.checkify`
before JIT. See the [JAX guide](jax.md#validation) for checked compiled calls.
Native AD supports `grad`, `jvp`, `vjp`, Jacobians, Hessians and `vmap`; no
custom derivative rule is used by the production backend.

## reconstruct

`reconstruct(projector, measured, *, steps=300, lr=1e-2,
initial_image=None, nonnegative=True)` performs Adam image optimization in
either `huo.torch` or `huo.jax`, using that backend's projector and arrays.

| Argument or result | Contract |
| --- | --- |
| `projector` | A configured `FanbeamProjector`. |
| `measured` | Fixed finite measurements `[..., V, M]`, matching the projector dtype. PyTorch requires CPU tensors; JAX uses its native placement. |
| `steps` | Positive integer number of optimizer updates. |
| `lr` | Positive finite Adam learning rate. |
| `initial_image` | Optional array `[..., N, N]` with exactly the measurement batch dimensions. Defaults to zeros. |
| `nonnegative` | Clamp the initial image and each updated image at zero when true. |
| Return value | Backend-native image `[..., N, N]`, with gradients through the solver stopped. |

The objective is half the mean squared projection residual, averaged over all
batch, angle, and detector entries. The helper copies measurements and the
initial image in PyTorch, and treats JAX arrays as immutable. It does not modify
caller-owned arrays, gradients, scanner state, or random-number generators.
The PyTorch helper enables gradients locally even inside outer `no_grad` or
`inference_mode` contexts.

The JAX helper JIT-compiles and caches the full Optax/`lax.scan` loop. Its
`steps`, `lr`, and `nonnegative` settings must be static when called under an
outer JIT. Eager calls check measurements, the initial image, and the final
result for finite values; transformed calls use the same opt-in checkify
contract as the projector. JAX controls device placement.

This helper does not differentiate through optimizer steps. Use the projector
directly to train a network, introduce a custom loss, choose another optimizer,
or implement an unrolled optimization method. The [PyTorch tutorial](tutorial.md)
and [JAX guide](jax.md) show custom optimization workflows.

## Command line

`huo-reconstruct` uses the same solver and reads/writes NumPy files. Its required
arguments are the measurement file, `--angles`, `--output`, `--resolution`,
`--source-distance`, `--det-distance`, and `--det-spacing`. Detector count is
inferred from the final measurement dimension. Optional arguments are
`--ray-sampling`, `--steps`, `--lr`, `--allow-negative`, and `--backend torch|jax`
(default `torch`). `--method scan|vmap` selects JAX angle traversal and is
rejected for the PyTorch backend. Only the selected framework is imported;
missing extras produce an installation hint instead of backend fallback.

Numeric inputs are converted to float32. Geometry and shape errors are reported
as command-line errors. The output includes image shape, iteration count, and
initial/final projection loss.
