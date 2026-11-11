"""Tensor validation that composes with torch.func transforms."""

import torch
from torch import Tensor


class _CheckFinite(torch.autograd.Function):
    @staticmethod
    def forward(value, name):
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def vmap(info, in_dims, value, name):
        # Reapply at outer levels until every mapped sample is visible.
        return _CheckFinite.apply(value, name), None

    @staticmethod
    def backward(ctx, *grad_outputs):
        return None, None

    @staticmethod
    def jvp(ctx, *tangents):
        return None


def check_tensor(value: Tensor, shape: tuple[int, int], reference: Tensor, name: str) -> None:
    """Check the trailing shape, floating dtype, device, and all batch values."""
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu" or value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must be a CPU float32 or float64 tensor")
    if value.dtype != reference.dtype or value.device != reference.device:
        raise ValueError(f"{name} must match the projector dtype and device")
    if value.ndim < 2 or tuple(value.shape[-2:]) != shape or value.numel() == 0:
        raise ValueError(f"{name} must have nonempty shape [..., {shape[0]}, {shape[1]}]")
    _CheckFinite.apply(value, name)
