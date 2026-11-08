# Copyright (c) 2022, Zhiqiang Wang. All rights reserved.

"""Fan-beam projection and residual reconstruction in the shared geometry."""

import math

import torch
from torch import Tensor

from huo.art import backprojection_angle, forward_angle
from huo.geometry import FanbeamGeometry, compute_ray_coordinates


class RadonFanbeam:
    """A fixed fan-beam scanner with angle-by-detector sinograms.

    Python, TypeScript, and the mathematical documentation use these names.
    Angles are radians and all distances are in pixels. At zero angle the
    source is (0, source_distance), the detector plane is y=-det_distance,
    and the gantry rotates by R_{-angle}, following torch-radon's convention.

    Args:
        resolution: Number of pixels along each image axis.
        angles: Nonempty 1-D sequence of projection angles in radians.
        source_distance: Distance from the source to the image centre, in pixels.
        det_distance: Distance from the image centre to the detector plane, in pixels.
        det_count: Number of detector elements.
        det_spacing: Distance between detector centres, in pixels.
        ray_sampling: Integration intervals per pixel along a ray (default 2).
    """

    def __init__(
        self,
        resolution: int,
        angles,
        source_distance: float,
        det_distance: float,
        det_count: int,
        det_spacing: float,
        ray_sampling: int = 2,
    ):
        self._geometry = FanbeamGeometry(
            resolution, source_distance, det_distance, det_count, det_spacing, ray_sampling
        )
        self._angles = torch.as_tensor(angles, dtype=torch.float32, device="cpu").clone()
        if self._angles.ndim != 1 or self._angles.numel() == 0 or not torch.isfinite(self._angles).all():
            raise ValueError("angles must be a nonempty finite 1-D sequence in radians")
        self._ray_x, self._ray_y = compute_ray_coordinates(self.geometry)

    @property
    def geometry(self):
        """The immutable scanner configuration."""
        return self._geometry

    @property
    def angles(self):
        """Projection angles in radians, in sinogram row order."""
        return self._angles.clone()

    @staticmethod
    def _check_tensor(value, shape, name):
        if not isinstance(value, Tensor) or value.device.type != "cpu" or value.dtype != torch.float32:
            raise ValueError(f"{name} must be a CPU torch.float32 tensor")
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")

    def forward_angle(self, image: Tensor, angle: float) -> Tensor:
        """Project an image [resolution, resolution] at one angle in radians."""
        self._check_tensor(image, (self.geometry.resolution,) * 2, "image")
        if not math.isfinite(angle):
            raise ValueError("angle must be finite (in radians)")
        return forward_angle(image, self._ray_x, self._ray_y, angle, self.geometry)

    def backprojection_angle(self, projection: Tensor, angle: float) -> Tensor:
        """Uniformly backfill a detector row [det_count] at one angle in radians."""
        self._check_tensor(projection, (self.geometry.det_count,), "projection")
        if not math.isfinite(angle):
            raise ValueError("angle must be finite (in radians)")
        return backprojection_angle(projection, angle, self.geometry)

    def forward(self, image: Tensor) -> Tensor:
        """Project an image to a sinogram [len(angles), det_count]."""
        self._check_tensor(image, (self.geometry.resolution,) * 2, "image")
        return torch.stack(
            [
                forward_angle(image, self._ray_x, self._ray_y, angle.item(), self.geometry)
                for angle in self._angles
            ]
        )

    def backprojection(self, sinogram: Tensor) -> Tensor:
        """Sum uniform backfill over all sinogram rows [len(angles), det_count].

        This B operator has no fan-beam weighting and is not the discrete
        transpose of forward(). See the residual-backprojection proof.
        """
        self._check_tensor(sinogram, (len(self._angles), self.geometry.det_count), "sinogram")
        image = torch.zeros((self.geometry.resolution,) * 2, dtype=torch.float32)
        for angle, projection in zip(self._angles, sinogram):
            image += backprojection_angle(projection, angle.item(), self.geometry)
        return image

    def art(self, sinogram: Tensor) -> Tensor:
        """Run one residual-backprojection sweep from zero with nonnegative clipping.

        Input shape is [len(angles), det_count]. Each row is visited once in
        random order. The correction is divided by the ray-window length,
        which is ``resolution`` in pixel units.
        """
        self._check_tensor(sinogram, (len(self._angles), self.geometry.det_count), "sinogram")
        image = torch.zeros((self.geometry.resolution,) * 2, dtype=torch.float32)
        for index in torch.randperm(len(self._angles)):
            angle = self._angles[index].item()
            predicted = forward_angle(image, self._ray_x, self._ray_y, angle, self.geometry)
            residual = (sinogram[index] - predicted) / self.geometry.resolution
            image += backprojection_angle(residual, angle, self.geometry)
            image = image.clamp(min=0)
        return image
