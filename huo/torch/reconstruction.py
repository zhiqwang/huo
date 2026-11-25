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
    tv_weight: float = 0.0,
) -> Tensor:
    """Fit fixed measurements with Adam and optional total variation.

    ``measured`` has shape [..., views, det_count]. The result has shape
    [..., resolution, resolution]. Inputs are copied and never optimized in
    place. The returned image is detached; the optimizer trajectory is not
    differentiated. Use the projector directly for custom losses, optimizers,
    regularization, or training a neural network.

    The objective is half the mean squared projection residual plus
    ``tv_weight`` times the sum of mean absolute differences along both image axes.
    Optimization enables gradients locally, including when called from
    no_grad or inference_mode. Nonnegativity is enforced after each step.
    """
    validate_optimization(steps, lr, nonnegative, tv_weight)
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
            if tv_weight:
                variation = image.diff(dim=-1).abs().mean() + image.diff(dim=-2).abs().mean()
                loss = loss + tv_weight * variation
            loss.backward()
            optimizer.step()
            if nonnegative:
                with torch.no_grad():
                    image.clamp_min_(0)
        return image.detach()
