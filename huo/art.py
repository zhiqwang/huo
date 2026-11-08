# Copyright (c) 2022, Zhiqiang Wang. All rights reserved.

"""Per-angle sampled operators in the shared fan-beam coordinates."""

import math

import torch
import torch.nn.functional as F


def forward_angle(image, ray_x, ray_y, angle, geometry):
    """Integrate one detector row at ``angle`` radians using zero padding."""
    # The public angle convention rotates scanner rays by R_{-angle}.
    c, s = math.cos(angle), math.sin(angle)
    x = c * ray_x + s * ray_y
    y = -s * ray_x + c * ray_y
    samples = torch.stack((x, y), dim=-1)
    # grid_sample requires [batch, channel, height, width], even for one image.
    values = F.grid_sample(
        image.reshape(1, 1, *image.shape),
        samples.unsqueeze(0),
        padding_mode="zeros",
        align_corners=True,
    )
    # Remove the singleton batch/channel axes while integrating the ray samples.
    return geometry.ray_step * values.sum(dim=(0, 1, 2))


def _linear_sample(projection, index):
    """Interpolate a detector row at fractional indices, with zero neighbours."""
    # One zero at each end retains partial edge weights. More distant samples
    # clamp to those zeros, not to the first/last measured detector value.
    padded = F.pad(projection, (1, 1))
    position = index.clamp(-1, projection.numel()) + 1
    left = position.long()  # position is nonnegative, so truncation equals floor.
    right = (left + 1).clamp(max=padded.numel() - 1)
    return torch.lerp(torch.take(padded, left), torch.take(padded, right), position - left)


def backprojection_angle(projection, angle, geometry):
    """Uniformly backfill one detector row; this is the B operator in the proof."""
    c, s = math.cos(angle), math.sin(angle)
    centres = torch.arange(geometry.resolution, dtype=projection.dtype, device=projection.device)
    centres = centres - geometry.pixel_extent
    y, x = torch.meshgrid(centres, centres, indexing="ij")
    # R_angle maps image pixels back into scanner coordinates, in pixel units.
    x_rot, y_rot = c * x - s * y, s * x + c * y
    denominator = geometry.source_distance - y_rot
    if (denominator <= 0).any():
        raise ValueError("All image pixel centres must be in front of the source at this angle")
    u = geometry.source_detector_distance * x_rot / denominator
    index = u / geometry.det_spacing + (geometry.det_count - 1) / 2
    return _linear_sample(projection, index)
