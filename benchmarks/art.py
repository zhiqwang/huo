"""Historical ART reference for reconstruction comparisons from a source checkout."""

from collections.abc import Sequence
from copy import deepcopy
import math
from numbers import Integral
from operator import index

import torch
from torch import Tensor
import torch.nn.functional as F

from huo import FanbeamGeometry
from huo.torch import FanbeamProjector
from huo.torch._validation import check_tensor


def _backfill(residual: Tensor, angle: float, centres: Tensor, geometry: FanbeamGeometry) -> Tensor:
    """Interpolate one detector row at the projected image pixel centres."""
    cosine, sine = math.cos(angle), math.sin(angle)
    cos_centres, sin_centres = cosine * centres, sine * centres
    x_rot = cos_centres[None, :] - sin_centres[:, None]
    y_rot = sin_centres[None, :] + cos_centres[:, None]
    distance = geometry.source_distance - y_rot
    if (distance <= 0).any():
        raise ValueError("All image pixel centres must be in front of the source at this angle")

    detector = geometry.source_detector_distance * x_rot / distance
    position = detector / geometry.det_spacing + (geometry.det_count - 1) / 2
    # One zero sample at each edge preserves the partial interpolation weights.
    position = position.clamp(-1, geometry.det_count) + 1
    left = position.long()
    right = (left + 1).clamp(max=geometry.det_count + 1)
    padded = F.pad(residual, (1, 1))
    return torch.lerp(padded[..., left], padded[..., right], position - left)


@torch.no_grad()
def reconstruct_art(
    projector: FanbeamProjector,
    measured: Tensor,
    *,
    sweeps: int = 1,
    angle_order: Sequence[int] | None = None,
) -> Tensor:
    """Run the historical residual-backfill updates with the current projector.

    Measurements have shape ``[..., views, detectors]``. Start from zero and
    carry the image across sweeps, clipping to nonnegative values at each view.
    Supply a permutation of view indices to repeat that order every sweep.
    By default, draw a fresh random permutation for each sweep.
    """
    if isinstance(sweeps, bool) or not isinstance(sweeps, Integral) or sweeps < 1:
        raise ValueError("sweeps must be a positive integer")

    geometry = projector.geometry
    views = projector.angles.numel()
    check_tensor(measured, (views, geometry.det_count), projector.angles, "measured")
    order = None if angle_order is None else [index(view) for view in angle_order]
    if order is not None and sorted(order) != list(range(views)):
        raise ValueError("angle_order must be a permutation of all view indices")

    # Copy the sampled rays so dtype conversions and loaded state are preserved.
    single_view = deepcopy(projector)
    angles = single_view.angles
    angle_values = angles.tolist()
    single_view.angles = angles[:1].clone()

    n = geometry.resolution
    centres = torch.arange(n, dtype=measured.dtype, device=measured.device) - geometry.pixel_extent
    image = measured.new_zeros((*measured.shape[:-2], n, n))
    for _ in range(sweeps):
        indices = order if order is not None else torch.randperm(views).tolist()
        for view in indices:
            single_view.angles.copy_(angles[view : view + 1])
            predicted = single_view(image)[..., 0, :]
            residual = (measured[..., view, :] - predicted) / n
            image.add_(_backfill(residual, angle_values[view], centres, geometry)).clamp_min_(0)
    return image
