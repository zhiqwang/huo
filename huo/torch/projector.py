"""A PyTorch module mapping images to fan-beam line integrals."""

from dataclasses import asdict

import numpy as np
import torch
from torch import Tensor, nn

from huo.torch._autograd import project
from huo.torch._validation import check_tensor
from huo.geometry import FanbeamGeometry, compute_ray_coordinates


class FanbeamProjector(nn.Module):
    """Fixed-geometry forward projection: [..., N, N] -> [..., V, M].

    Angles are radians; distances are pixels. Only image values are
    differentiated. CPU float32 and float64 are supported. Angles and rays
    are buffers, so module dtype conversions also convert sampling constants.

    Args:
        geometry: Immutable scanner dimensions and sampling parameters.
        angles: Nonempty finite 1-D sequence, in sinogram row order.
        dtype: Floating dtype for images and scanner buffers (default float32).
    """

    def __init__(self, geometry: FanbeamGeometry, angles, *, dtype=torch.float32):
        super().__init__()
        if not isinstance(geometry, FanbeamGeometry):
            raise TypeError("geometry must be a FanbeamGeometry")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        self._geometry = geometry
        # Constants constructed during inference must remain saveable in a
        # later training graph. Never retain the caller's angle graph/storage.
        with torch.inference_mode(False), torch.no_grad():
            values = torch.as_tensor(angles, dtype=dtype, device="cpu").detach().clone()
            if values.ndim != 1 or values.numel() == 0 or not torch.isfinite(values).all():
                raise ValueError("angles must be a nonempty finite 1-D sequence in radians")
            coordinates = compute_ray_coordinates(
                geometry, dtype=np.float64 if dtype == torch.float64 else np.float32
            )
            ray_x, ray_y = (torch.from_numpy(value) for value in coordinates)
        self.register_buffer("angles", values)
        self.register_buffer("_ray_x", ray_x)
        self.register_buffer("_ray_y", ray_y)

    @property
    def geometry(self) -> FanbeamGeometry:
        """The immutable scanner configuration."""
        return self._geometry

    def forward(self, image: Tensor) -> Tensor:
        """Project images; loss.backward() automatically propagates through A^T."""
        check_tensor(image, (self.geometry.resolution,) * 2, self.angles, "image")
        return project(image, self._ray_x, self._ray_y, self.angles, self.geometry.ray_step)

    def get_extra_state(self):
        """Record geometry alongside angle buffers in state_dict."""
        return asdict(self.geometry)

    def set_extra_state(self, state):
        if state != asdict(self.geometry):
            raise ValueError("state_dict geometry must match the projector geometry")

    def extra_repr(self):
        return (
            f"resolution={self.geometry.resolution}, views={self.angles.numel()}, "
            f"det_count={self.geometry.det_count}, ray_sampling={self.geometry.ray_sampling}"
        )
