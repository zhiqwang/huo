# Copyright (c) 2022, Zhiqiang Wang. All rights reserved.

"""
Fan-beam CT Radon transform following the `torch-radon`_ convention.

.. _torch-radon: https://torch-radon.readthedocs.io/en/latest/modules/radon.html
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


class _RadonForward(torch.autograd.Function):
    """Custom autograd Function pairing forward projection with backprojection.

    The *forward* pass performs the Radon transform (image → sinogram) and
    the *backward* pass performs the backprojection (sinogram → image),
    which is the adjoint of the forward projection.
    """

    @staticmethod
    def forward(ctx, img: Tensor, radon: "RadonFanbeam") -> Tensor:
        ctx.radon = radon
        num_angles = len(radon.angles)
        sinogram = torch.zeros(radon.det_count, num_angles, dtype=img.dtype)
        sinogram = sinogram.unsqueeze(0).unsqueeze(0)

        for i, angle in enumerate(radon.angles):
            sinogram[:, :, :, i] = radon._forward_angle(img, angle.item())

        return sinogram.squeeze(0).squeeze(0)

    @staticmethod
    def backward(ctx, grad_sinogram: Tensor):
        radon = ctx.radon
        P = radon.resolution
        grad_img = torch.zeros(P, P, dtype=grad_sinogram.dtype)
        sinogram_4d = grad_sinogram.unsqueeze(0).unsqueeze(0)

        for i, angle in enumerate(radon.angles):
            col = sinogram_4d[:, :, :, i]
            grad_img += radon._backprojection_angle(col, angle.item())

        return grad_img, None


class RadonFanbeam:
    """Fan-beam CT forward / back-projection.

    Follows the `torch-radon <https://torch-radon.readthedocs.io/en/latest/>`_
    convention where ``forward`` means Radon transform (image → sinogram) and
    ``backprojection`` means sinogram → image.

    The constructor pre-computes all fan-beam gantry coordinates so that
    :meth:`forward`, :meth:`backprojection`, and :meth:`art` only need the
    image or sinogram tensor.

    Args:
        resolution: Number of image pixels along each axis.
        angles: 1-D tensor (or list) of projection angles in degrees.
        source_distance: Source-to-object (centre of rotation) distance in mm.
        det_distance: Object (centre of rotation) to detector distance in mm.
        det_count: Number of detector elements.
        det_spacing: Spacing between detector elements in mm.
        volume_size: Diameter of the field of view (FOV) in mm.
        lat_sampling: Lateral sampling grid multiplier (default 2).
    """

    def __init__(
        self,
        resolution: int,
        angles,
        source_distance: float,
        det_distance: float,
        det_count: int,
        det_spacing: float,
        volume_size: float,
        lat_sampling: int = 2,
    ):
        self.resolution = resolution
        self.angles = torch.as_tensor(angles, dtype=torch.float32)
        self.source_distance = source_distance
        self.det_distance = det_distance
        self.det_count = det_count
        self.det_spacing = det_spacing
        self.volume_size = volume_size
        self.lat_sampling = lat_sampling

        # Derived geometry quantities (torch-radon naming)
        # SDD = Source-to-Detector Distance (standard CT abbreviation)
        self.sdd = source_distance + det_distance
        self.det_length = det_count * det_spacing

        img_step = volume_size / resolution
        self.img_end = (volume_size - img_step) / 2

        det_step = self.det_length / self.det_count
        self.det_end = (self.det_length - det_step) / 2

        # Pre-compute gantry coordinates
        self._gantry_coor_x, self._gantry_coor_y = self._compute_gantry_coordinates()

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _compute_gantry_coordinates(self):
        """Pre-compute fan-beam gantry ray-tracing coordinates.

        Returns a tuple ``(gantry_coor_x, gantry_coor_y)`` each of shape
        ``[lat_steps, det_count]``, normalised to [-1, 1].
        """
        src_y = -self.source_distance

        det_coor_x = torch.linspace(-self.det_end, self.det_end, steps=self.det_count)
        det_coor_y = torch.zeros_like(det_coor_x) + self.sdd - self.source_distance

        lat_end = self.volume_size / 2
        lat_steps = self.lat_sampling * self.resolution + 1
        lat_grid = torch.linspace(-lat_end, lat_end, steps=lat_steps).unsqueeze(1)

        # Shift origin to X-ray source position
        lat_grid = lat_grid - src_y
        det_coor_y = det_coor_y - src_y

        # Fan-beam distances
        fan_dist = (det_coor_x**2 + det_coor_y**2).sqrt()

        # Gantry coordinates [lat_steps, det_count]
        gantry_coor_x = lat_grid * det_coor_x / fan_dist
        gantry_coor_y = lat_grid * det_coor_y / fan_dist + src_y

        # Normalise to [-1, 1] for F.grid_sample
        gantry_coor_x = gantry_coor_x / self.img_end
        gantry_coor_y = gantry_coor_y / self.img_end

        return gantry_coor_x, gantry_coor_y

    # ------------------------------------------------------------------
    # Per-angle projections
    # ------------------------------------------------------------------

    def _forward_angle(self, img: Tensor, angle: float) -> Tensor:
        """Forward-project the image at a single angle.

        Args:
            img: 2-D image tensor ``[resolution, resolution]``.
            angle: Projection angle in degrees.

        Returns:
            Sinogram column ``[det_count]``.
        """
        img = img.unsqueeze(0).unsqueeze(0)
        rad = torch.tensor([angle * np.pi / 180.0])

        cos_a = torch.cos(rad)
        sin_a = torch.sin(rad)

        rot_x = self._gantry_coor_x * cos_a - self._gantry_coor_y * sin_a
        rot_y = self._gantry_coor_x * sin_a + self._gantry_coor_y * cos_a

        rot_x = rot_x.unsqueeze(0).unsqueeze(3)
        rot_y = rot_y.unsqueeze(0).unsqueeze(3)
        samples = torch.cat([rot_x, rot_y], 3)

        interp = F.grid_sample(img, samples, align_corners=True)
        interp = interp.squeeze(0).squeeze(0)

        lat_step = self.volume_size / self.resolution / self.lat_sampling
        return lat_step * torch.sum(interp, dim=0)

    def _backprojection_angle(self, sinogram_col: Tensor, angle: float) -> Tensor:
        """Back-project a sinogram column at a single angle.

        Args:
            sinogram_col: Tensor of shape ``[1, 1, det_count]``.
            angle: Projection angle in degrees.

        Returns:
            Image tensor ``[resolution, resolution]``.
        """
        sinogram_col = sinogram_col.unsqueeze(3)
        rad = torch.tensor([angle * np.pi / 180.0])

        theta = torch.tensor(
            [[torch.cos(rad), torch.sin(rad), 0], [-torch.sin(rad), torch.cos(rad), 0]]
        ).unsqueeze(0)

        P = self.resolution
        img_grid_rot = F.affine_grid(theta, (1, 1, P, P)).squeeze(0)
        grid_x = img_grid_rot[:, :, 0]
        grid_y = img_grid_rot[:, :, 1]

        grid_y = grid_y + self.source_distance / self.img_end
        det_positions = self.sdd * grid_x / grid_y
        det_positions /= self.det_end
        det_positions = det_positions.reshape(-1, 1)
        det_positions = det_positions.unsqueeze(2).unsqueeze(0)
        samples = torch.cat([torch.zeros_like(det_positions), det_positions], 3)

        img = F.grid_sample(sinogram_col, samples, align_corners=True)
        img = img.squeeze(0).squeeze(0)
        img = img.reshape(P, P)

        return img

    # ------------------------------------------------------------------
    # Public API (torch-radon convention)
    # ------------------------------------------------------------------

    def forward(self, img: Tensor) -> Tensor:
        """Forward projection (Radon transform) over all angles.

        Corresponds to ``RadonFanbeam.forward`` in torch-radon.

        This method is backed by :class:`_RadonForward`, a custom
        :class:`torch.autograd.Function` whose ``backward`` pass is the
        :meth:`backprojection` operation.  When *img* has
        ``requires_grad=True``, calling ``.backward()`` on the returned
        sinogram will compute the gradient via backprojection.

        Args:
            img: 2-D image tensor ``[resolution, resolution]``.

        Returns:
            Sinogram tensor ``[det_count, num_angles]``.
        """
        return _RadonForward.apply(img, self)

    def backprojection(self, sinogram: Tensor) -> Tensor:
        """Back-projection (sinogram → image) summed over all angles.

        Corresponds to ``RadonFanbeam.backprojection`` in torch-radon.

        Args:
            sinogram: Sinogram tensor ``[det_count, num_angles]``.

        Returns:
            Image tensor ``[resolution, resolution]``.
        """
        P = self.resolution
        img = torch.zeros(P, P)
        sinogram_4d = sinogram.unsqueeze(0).unsqueeze(0)

        for i, angle in enumerate(self.angles):
            col = sinogram_4d[:, :, :, i]
            img += self._backprojection_angle(col, angle.item())

        return img

    def art(self, sinogram: Tensor) -> Tensor:
        """Algebraic Reconstruction Technique (ART).

        Iteratively reconstructs the image from the sinogram using the
        Kaczmarz method (random angle ordering).

        Args:
            sinogram: Sinogram tensor ``[det_count, num_angles]``.

        Returns:
            Reconstructed image ``[resolution, resolution]``.
        """
        P = self.resolution
        sinogram_4d = sinogram.unsqueeze(0).unsqueeze(0)
        img = torch.zeros(P, P)

        indices = torch.randperm(len(self.angles))
        for i in indices:
            angle = self.angles[i].item()
            residual = sinogram_4d[:, :, :, i] - self._forward_angle(img, angle)
            residual /= self.volume_size
            img += self._backprojection_angle(residual, angle)
            img = torch.clamp(img, min=0)

        return img
