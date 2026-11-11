"""Fixed-geometry projection and its private transpose for automatic gradients."""

import torch
from torch.autograd import Function
import torch.nn.functional as F


def _ray_grid(ray_x, ray_y, cosine, sine):
    # R_{-angle} maps scanner rays into image coordinates.
    return torch.stack((cosine * ray_x + sine * ray_y, -sine * ray_x + cosine * ray_y), dim=-1)


def _save_geometry(ctx, inputs):
    _, ray_x, ray_y, cosines, sines, ctx.ray_step, *shape = inputs
    ctx.height, ctx.width = shape if shape else inputs[0].shape[-2:]
    ctx.save_for_backward(ray_x, ray_y, cosines, sines)
    ctx.save_for_forward(ray_x, ray_y, cosines, sines)


def _backward_call(operator, *inputs):
    # Ordinary backward needs only the kernel; higher derivatives need its pair.
    return operator.apply(*inputs) if torch.is_grad_enabled() else operator.forward(*inputs)


class _Project(Function):
    """Apply all views of A with shared sampling constants and one graph node."""

    generate_vmap_rule = True

    @staticmethod
    def forward(image, ray_x, ray_y, cosines, sines, ray_step):
        images = image.reshape(-1, 1, *image.shape[-2:])
        rows = []
        for cosine, sine in zip(cosines.unbind(), sines.unbind()):
            grid = _ray_grid(ray_x, ray_y, cosine, sine)
            values = F.grid_sample(
                images,
                grid[None].expand(images.shape[0], -1, -1, -1),
                padding_mode="zeros",
                align_corners=True,
            )
            rows.append(ray_step * values.sum(dim=(1, 2)))
        return torch.stack(rows, dim=1).reshape(*image.shape[:-2], cosines.numel(), ray_x.shape[1])

    @staticmethod
    def setup_context(ctx, inputs, output):
        _save_geometry(ctx, inputs)

    @staticmethod
    def backward(ctx, gradient):
        image = _backward_call(
            _ProjectTranspose, gradient, *ctx.saved_tensors, ctx.ray_step, ctx.height, ctx.width
        )
        return image, None, None, None, None, None

    @staticmethod
    def jvp(ctx, tangent, *_):
        return _Project.apply(tangent, *ctx.saved_tensors, ctx.ray_step)


class _ProjectTranspose(Function):
    """Apply A^T inside backward; its own backward applies A."""

    generate_vmap_rule = True

    @staticmethod
    def forward(sinogram, ray_x, ray_y, cosines, sines, ray_step, height, width):
        rows = sinogram.reshape(-1, cosines.numel(), ray_x.shape[1])
        image = sinogram.new_zeros(rows.shape[0], 1, height, width)
        # Only the image VJP is requested. Image values are irrelevant, and no
        # zero-image forward or local autograd graph needs to be replayed.
        reference = torch.zeros_like(image)
        for row, cosine, sine in zip(rows.unbind(1), cosines.unbind(), sines.unbind()):
            grid = _ray_grid(ray_x, ray_y, cosine, sine)
            gradient = (row * ray_step)[:, None, None, :].expand(-1, 1, ray_x.shape[0], -1)
            # ATen schema from the pinned PyTorch: bilinear=0, zeros=0,
            # align_corners=True, compute only the input-image gradient.
            contribution, _ = torch.ops.aten.grid_sampler_2d_backward.default(
                gradient, reference, grid[None].expand(rows.shape[0], -1, -1, -1), 0, 0, True, [True, False]
            )
            image = image + contribution
        return image.reshape(*sinogram.shape[:-2], height, width)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _save_geometry(ctx, inputs)

    @staticmethod
    def backward(ctx, gradient):
        sinogram = _backward_call(_Project, gradient, *ctx.saved_tensors, ctx.ray_step)
        return sinogram, None, None, None, None, None, None, None

    @staticmethod
    def jvp(ctx, tangent, *_):
        return _ProjectTranspose.apply(tangent, *ctx.saved_tensors, ctx.ray_step, ctx.height, ctx.width)


def project(image, ray_x, ray_y, angles, ray_step):
    """Prepare fixed, saveable constants before entering the autograd Function."""
    with torch.inference_mode(False), torch.no_grad():
        ray_x, ray_y, angles = (
            value.clone() if value.is_inference() else value for value in (ray_x, ray_y, angles)
        )
        cosines, sines = angles.cos(), angles.sin()
    return _Project.apply(image, ray_x, ray_y, cosines, sines, ray_step)
