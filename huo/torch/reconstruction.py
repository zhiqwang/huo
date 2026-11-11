"""Image reconstruction with a standard PyTorch loss and optimizer."""

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from huo._settings import validate_optimization
from huo.torch._validation import check_tensor
from huo.torch.projector import FanbeamProjector


def reconstruct(
    projector: FanbeamProjector,
    measured: Tensor,
    *,
    steps: int = 300,
    lr: float = 1e-2,
    initial_image: Tensor | None = None,
    nonnegative: bool = True,
) -> Tensor:
    """Fit an image to fixed measurements with Adam and half mean squared error.

    ``measured`` has shape [..., views, det_count]. The result has shape
    [..., resolution, resolution]. Inputs are copied and never optimized in
    place. The returned image is detached; the optimizer trajectory is not
    differentiated. Use the projector directly for custom losses, optimizers,
    regularization, or training a neural network.

    Optimization enables gradients locally, including when called from
    no_grad or inference_mode. Nonnegativity is enforced after each step.
    """
    validate_optimization(steps, lr, nonnegative)
    check_tensor(
        measured, (projector.angles.numel(), projector.geometry.det_count), projector.angles, "measured"
    )
    image_shape = (*measured.shape[:-2], projector.geometry.resolution, projector.geometry.resolution)
    if initial_image is not None:
        check_tensor(initial_image, image_shape[-2:], projector.angles, "initial_image")
        if tuple(initial_image.shape) != image_shape:
            raise ValueError(f"initial_image must have shape {image_shape}")

    with torch.inference_mode(False), torch.enable_grad():
        target = measured.detach().clone()
        initial = target.new_zeros(image_shape) if initial_image is None else initial_image.detach().clone()
        if nonnegative:
            initial.clamp_min_(0)
        image = nn.Parameter(initial)
        optimizer = torch.optim.Adam([image], lr=lr)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = 0.5 * F.mse_loss(projector(image), target)
            loss.backward()
            optimizer.step()
            if nonnegative:
                with torch.no_grad():
                    image.clamp_min_(0)
        return image.detach()
