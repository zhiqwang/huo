"""An immutable PyTree describing a fixed fan-beam projector."""

from dataclasses import dataclass, field
from functools import partial
import math
from numbers import Integral, Real

import jax
import jax.numpy as jnp
import numpy as np

from huo.geometry import FanbeamGeometry, compute_ray_coordinates
from huo.jax._project import project
from huo.jax._validation import check_array, check_dtype


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("angles", "ray_x", "ray_y"),
    meta_fields=("resolution", "ray_step", "method"),
)
@dataclass(frozen=True, eq=False)
class FanbeamProjector:
    """Callable PyTree: [..., N, N] -> [..., V, M]. Build with create().

    Angles/rays are dynamic array leaves, but their gradients are stopped.
    Shape/strategy metadata is static. Pass the object as a JIT argument so
    different scanner values with matching shapes can reuse compiled code.
    """

    angles: jax.Array = field(repr=False)
    ray_x: jax.Array = field(repr=False)
    ray_y: jax.Array = field(repr=False)
    resolution: int
    ray_step: float
    method: str = "scan"

    def __post_init__(self):
        # Do not inspect array leaves here: PyTree transforms can reconstruct
        # objects using tracers or placeholders. Host validation belongs in create.
        if (
            isinstance(self.resolution, bool)
            or not isinstance(self.resolution, Integral)
            or self.resolution < 2
        ):
            raise ValueError("resolution must be an integer >= 2")
        if not isinstance(self.ray_step, Real) or not math.isfinite(self.ray_step) or self.ray_step <= 0:
            raise ValueError("ray_step must be positive and finite")
        if self.method not in ("scan", "vmap"):
            raise ValueError("method must be 'scan' or 'vmap'")

    @classmethod
    def create(cls, geometry: FanbeamGeometry, angles, *, dtype=jnp.float32, method="scan", device=None):
        """Validate host configuration and prepare owned arrays outside JIT.

        Float64 requires JAX x64 mode. ``device`` is a JAX device or sharding;
        None uses JAX's default placement. Geometry is in pixels and radians.
        """
        if not isinstance(geometry, FanbeamGeometry):
            raise TypeError("geometry must be a FanbeamGeometry")
        dtype = check_dtype(dtype)
        if isinstance(angles, jax.core.Tracer):
            raise ValueError("create() prepares scanner state outside JAX transformations")
        values = np.array(angles, dtype=dtype, copy=True)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("angles must be a nonempty finite 1-D sequence in radians")
        rays = compute_ray_coordinates(geometry, dtype=dtype)
        return cls(
            *(jax.device_put(value, device) for value in (values, *rays)),
            resolution=int(geometry.resolution),
            ray_step=float(geometry.ray_step),
            method=method,
        )

    def __call__(self, image):
        image = check_array(image, (self.resolution,) * 2, self.angles.dtype, "image")
        return project(image, self.ray_x, self.ray_y, self.angles, ray_step=self.ray_step, method=self.method)

    @property
    def views(self):
        return self.angles.shape[0]

    @property
    def det_count(self):
        return self.ray_x.shape[1]

    @property
    def dtype(self):
        return self.angles.dtype
