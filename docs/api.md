# Python API

```python
from huo import FanbeamGeometry, FanbeamProjector, reconstruct
```

The projector handles the forward physical model. PyTorch owns losses,
gradients, and parameter updates. Its transpose is an internal autograd detail;
the Python API has no separate backprojection, adjoint, or ART entry point.

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

`FanbeamProjector(geometry, angles, *, dtype=torch.float32)` is a
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
from huo import FanbeamGeometry, FanbeamProjector

geometry = FanbeamGeometry(8, 32, 16, 12, 1.3)
projector = FanbeamProjector(geometry, [0.0, 0.5])
torch.save(projector.state_dict(), "projector.pt")
restored = FanbeamProjector(geometry, [0.0, 0.0])
restored.load_state_dict(torch.load("projector.pt", weights_only=True))
```

## reconstruct

`reconstruct(projector, measured, *, steps=300, lr=1e-2,
initial_image=None, nonnegative=True)` performs Adam image optimization.

| Argument or result | Contract |
| --- | --- |
| `projector` | A configured `FanbeamProjector`. |
| `measured` | Fixed finite measurements `[..., V, M]`, matching the projector dtype and device. |
| `steps` | Positive integer number of optimizer updates. |
| `lr` | Positive finite Adam learning rate. |
| `initial_image` | Optional tensor `[..., N, N]` with exactly the measurement batch dimensions. Defaults to zeros. |
| `nonnegative` | Clamp the initial image and each updated image at zero when true. |
| Return value | Detached image `[..., N, N]` with the input dtype and device. |

The objective is half the mean squared projection residual, averaged over all
batch, angle, and detector entries. The helper copies measurements and the
initial image; it does not modify caller-owned tensors, gradients, module
buffers, or the random-number generator. It enables gradients locally even
inside outer `no_grad` or `inference_mode` contexts.

This helper does not differentiate through optimizer steps. Use the module
directly to train a network, introduce a custom loss, choose another optimizer,
or implement an unrolled optimization method. The [tutorial](tutorial.md)
shows those standard PyTorch workflows.

## Command line

`huo-reconstruct` uses the same solver and reads/writes NumPy files. Its required
arguments are the measurement file, `--angles`, `--output`, `--resolution`,
`--source-distance`, `--det-distance`, and `--det-spacing`. Detector count is
inferred from the final measurement dimension. Optional arguments are
`--ray-sampling`, `--steps`, `--lr`, and `--allow-negative`.

Numeric inputs are converted to float32. Geometry and shape errors are reported
as command-line errors. The output includes image shape, iteration count, and
initial/final projection loss.
