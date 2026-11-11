"""The fan-beam geometry shared by all Huo projection operators.

Angles are radians; distances are measured in pixels. At angle zero the
source is (0, source_distance) and the detector plane is y=-det_distance.
"""

from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class FanbeamGeometry:
    """Fixed scanner parameters, using torch-radon's public parameter names.

    Pixel centres are index-(resolution-1)/2. Detector centres are
    (index-(det_count-1)/2)*det_spacing. ``ray_sampling`` sets the number of
    integration intervals per pixel along a ray.
    """

    resolution: int
    source_distance: float
    det_distance: float
    det_count: int
    det_spacing: float
    ray_sampling: int = 2

    def __post_init__(self):
        for name, minimum in (("resolution", 2), ("det_count", 2), ("ray_sampling", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("source_distance", "det_distance", "det_spacing"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite (in pixels)")
        if self.source_distance <= self.resolution / 2:
            raise ValueError(
                "source_distance must exceed resolution/2 so the ray window stays in front of the source"
            )

    @property
    def source_detector_distance(self):
        return self.source_distance + self.det_distance

    @property
    def pixel_extent(self):
        """Coordinate of the outermost pixel centre, not the image edge."""
        return (self.resolution - 1) / 2

    @property
    def det_extent(self):
        return (self.det_count - 1) * self.det_spacing / 2

    @property
    def ray_sample_count(self):
        return self.ray_sampling * self.resolution + 1

    @property
    def ray_step(self):
        return 1 / self.ray_sampling


def compute_ray_coordinates(geometry: FanbeamGeometry, *, dtype=np.float32):
    """Return NumPy zero-angle rays [ray_sample_count, det_count].

    Coordinates are prepared in float64 before the requested output conversion.
    This initialization is shared by backends and runs outside differentiation.
    """
    if np.dtype(dtype) not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("dtype must be float32 or float64")
    u = np.arange(geometry.det_count, dtype=np.float64) - (geometry.det_count - 1) / 2
    u = u * geometry.det_spacing
    t = geometry.source_distance + np.linspace(
        -geometry.resolution / 2,
        geometry.resolution / 2,
        geometry.ray_sample_count,
        dtype=np.float64,
    )
    distance = np.sqrt(u**2 + geometry.source_detector_distance**2)
    # Outer products pair every ray distance with every detector direction.
    ray_x = np.outer(t, u / distance)
    ray_y = geometry.source_distance - np.outer(t, geometry.source_detector_distance / distance)
    return (ray_x / geometry.pixel_extent).astype(dtype), (ray_y / geometry.pixel_extent).astype(dtype)
