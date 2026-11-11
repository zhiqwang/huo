# Copyright (c) 2022, Zhiqiang Wang. All rights reserved.

"""Per-angle sampled operators and their fixed-geometry autograd transposes."""

import math

import torch
from torch import Tensor
from torch.autograd import Function
import torch.nn.functional as F

from huo.geometry import FanbeamGeometry


def _ray_grid(ray_x: Tensor, ray_y: Tensor, angle: float) -> Tensor:
    """Rotate normalized scanner rays into image coordinates [samples, M, 2]."""
    # The public angle convention rotates scanner rays by R_{-angle}.
    c, s = math.cos(angle), math.sin(angle)
    x = c * ray_x + s * ray_y
    y = -s * ray_x + c * ray_y
    return torch.stack((x, y), dim=-1)


def _linear_indices(index: Tensor, det_count: int) -> tuple[Tensor, Tensor, Tensor]:
    """Indices and weights in a detector row padded with one zero at each end."""
    # One zero at each end retains partial edge weights. More distant samples
    # clamp to those zeros, not to the first/last measured detector value.
    position = index.clamp(-1, det_count) + 1
    left = position.long()  # position is nonnegative, so truncation equals floor.
    right = (left + 1).clamp(max=det_count + 1)
    return left, right, position - left


def _linear_sample(projection: Tensor, index: Tensor) -> Tensor:
    """Interpolate a detector row at fractional indices, with zero neighbours."""
    padded = F.pad(projection, (1, 1))
    left, right, weight = _linear_indices(index, projection.numel())
    # index_select has a batching rule and avoids general advanced indexing.
    left_values = padded.index_select(0, left.reshape(-1)).reshape_as(index)
    right_values = padded.index_select(0, right.reshape(-1)).reshape_as(index)
    return torch.lerp(left_values, right_values, weight)


def _detector_index(reference: Tensor, angle: float, geometry: FanbeamGeometry) -> Tensor:
    """Map image pixel centres to fractional detector indices."""
    c, s = math.cos(angle), math.sin(angle)
    centres = torch.arange(geometry.resolution, dtype=reference.dtype, device=reference.device)
    centres = centres - geometry.pixel_extent
    # R_angle maps image pixels back into scanner coordinates, in pixel units.
    # Scale the 1-D centres once, then broadcast with rows as y and columns as x.
    cos_centres, sin_centres = c * centres, s * centres
    x_rot = cos_centres[None, :] - sin_centres[:, None]
    y_rot = sin_centres[None, :] + cos_centres[:, None]
    denominator = geometry.source_distance - y_rot
    if (denominator <= 0).any():
        raise ValueError("All image pixel centres must be in front of the source at this angle")
    u = geometry.source_detector_distance * x_rot / denominator
    return u / geometry.det_spacing + (geometry.det_count - 1) / 2


def _forward_transpose(
    gradient: Tensor,
    ray_x: Tensor,
    ray_y: Tensor,
    angle: float,
    image_shape: tuple[int, int],
    ray_step: float,
) -> Tensor:
    """Apply the native bilinear VJP without replaying a grid_sample forward."""
    grid = _ray_grid(ray_x, ray_y, angle)
    image = gradient.new_zeros(1, 1, *image_shape)
    samples = (gradient * ray_step).reshape(1, 1, 1, -1).expand(1, 1, *ray_x.shape)
    # PyTorch is pinned in pyproject.toml. This is its grid_sample backward
    # schema: bilinear=0, zero padding=0, align_corners=True. Only the image
    # gradient is requested, so its values do not affect the result.
    result, _ = torch.ops.aten.grid_sampler_2d_backward.default(
        samples, image, grid[None], 0, 0, True, [True, False]
    )
    return result[0, 0]


def _backprojection_transpose(gradient: Tensor, index: Tensor, det_count: int) -> Tensor:
    """Scatter image gradients with the backfill's two detector weights."""
    left, right, weight = _linear_indices(index, det_count)
    padded = gradient.new_zeros(det_count + 2)
    padded = padded.scatter_add(0, left.reshape(-1), (gradient * (1 - weight)).reshape(-1))
    padded = padded.scatter_add(0, right.reshape(-1), (gradient * weight).reshape(-1))
    return padded[1:-1]


def _autograd_rays(ray_x: Tensor, ray_y: Tensor) -> tuple[Tensor, Tensor]:
    """Make inference constants saveable before entering a custom Function."""
    with torch.inference_mode(False), torch.no_grad():
        return tuple(ray.clone() if ray.is_inference() else ray for ray in (ray_x, ray_y))


def _save_rays(ctx, inputs):
    _, ray_x, ray_y, ctx.angle, ctx.geometry, *shape = inputs
    ctx.image_shape = tuple(shape) if shape else tuple(inputs[0].shape)
    ctx.save_for_backward(ray_x, ray_y)
    ctx.save_for_forward(ray_x, ray_y)


def _apply_transpose(operator, *inputs):
    """Retain a paired Function only when backward is building another graph."""
    if torch.is_grad_enabled():
        return operator.apply(*inputs)
    return operator.forward(*inputs)


class RadonForward(Function):
    """Differentiate sampled projection with its own transpose, A_theta^T.

    Ray coordinates, angle, and geometry are fixed. Saving the shared rays
    lets backward recompute its weights without retaining the input image
    or a separate rotated sampling grid for every angle.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(image, ray_x, ray_y, angle, geometry):
        samples = _ray_grid(ray_x, ray_y, angle)
        # grid_sample requires [batch, channel, height, width], even for one image.
        values = F.grid_sample(
            image.reshape(1, 1, *image.shape),
            samples.unsqueeze(0),
            padding_mode="zeros",
            align_corners=True,
        )
        return geometry.ray_step * values.sum(dim=(0, 1, 2))

    @staticmethod
    def setup_context(ctx, inputs, output):
        _save_rays(ctx, inputs)

    @staticmethod
    def backward(ctx, grad_projection):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None
        ray_x, ray_y = ctx.saved_tensors
        grad_image = _apply_transpose(
            RadonAdjoint, grad_projection, ray_x, ray_y, ctx.angle, ctx.geometry, *ctx.image_shape
        )
        return grad_image, None, None, None, None

    @staticmethod
    def jvp(ctx, image_tangent, *_):
        ray_x, ray_y = ctx.saved_tensors
        return RadonForward.apply(image_tangent, ray_x, ray_y, ctx.angle, ctx.geometry)


class RadonAdjoint(Function):
    """Apply A_theta^T, with A_theta as its backward and A_theta^T as its JVP."""

    generate_vmap_rule = True

    @staticmethod
    def forward(projection, ray_x, ray_y, angle, geometry, height, width):
        # Scalar dimensions also compose with generated vmap/JVP rules.
        return _forward_transpose(projection, ray_x, ray_y, angle, (height, width), geometry.ray_step)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _save_rays(ctx, inputs)

    @staticmethod
    def backward(ctx, grad_image):
        ray_x, ray_y = ctx.saved_tensors
        projection = _apply_transpose(RadonForward, grad_image, ray_x, ray_y, ctx.angle, ctx.geometry)
        return projection, None, None, None, None, None, None

    @staticmethod
    def jvp(ctx, projection_tangent, *_):
        ray_x, ray_y = ctx.saved_tensors
        return RadonAdjoint.apply(projection_tangent, ray_x, ray_y, ctx.angle, ctx.geometry, *ctx.image_shape)


class RadonBackprojection(Function):
    """Differentiate uniform residual backfill with its own transpose, B_theta^T.

    B_theta is not A_theta^T, so its backward cannot reuse RadonForward.
    Its sampling weights depend only on fixed geometry, not projection values.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(projection, angle, geometry):
        index = _detector_index(projection, angle, geometry)
        return _linear_sample(projection, index)

    @staticmethod
    def setup_context(ctx, inputs, output):
        projection, ctx.angle, ctx.geometry = inputs
        ctx.det_count = projection.numel()

    @staticmethod
    def backward(ctx, grad_image):
        grad_projection = _apply_transpose(
            RadonBackprojectionTranspose, grad_image, ctx.angle, ctx.geometry, ctx.det_count
        )
        return grad_projection, None, None

    @staticmethod
    def jvp(ctx, projection_tangent, *_):
        return RadonBackprojection.apply(projection_tangent, ctx.angle, ctx.geometry)


class RadonBackprojectionTranspose(Function):
    """Apply B_theta^T without retaining detector indices in higher-order graphs."""

    generate_vmap_rule = True

    @staticmethod
    def forward(image, angle, geometry, det_count):
        index = _detector_index(image, angle, geometry)
        return _backprojection_transpose(image, index, det_count)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, ctx.angle, ctx.geometry, ctx.det_count = inputs

    @staticmethod
    def backward(ctx, grad_projection):
        image = _apply_transpose(RadonBackprojection, grad_projection, ctx.angle, ctx.geometry)
        return image, None, None, None

    @staticmethod
    def jvp(ctx, image_tangent, *_):
        return RadonBackprojectionTranspose.apply(image_tangent, ctx.angle, ctx.geometry, ctx.det_count)


def forward_angle(
    image: Tensor, ray_x: Tensor, ray_y: Tensor, angle: float, geometry: FanbeamGeometry
) -> Tensor:
    """Integrate an image [N, N] into a detector row [M] at ``angle`` radians.

    Ray coordinates have shape [ray_sample_count, M] and use the normalized
    image grid, with zero padding outside its support. Only image values are
    differentiated; ray coordinates and scanner geometry are fixed.
    """
    ray_x, ray_y = _autograd_rays(ray_x, ray_y)
    return RadonForward.apply(image, ray_x, ray_y, float(angle), geometry)


def adjoint_angle(
    projection: Tensor, ray_x: Tensor, ray_y: Tensor, angle: float, geometry: FanbeamGeometry
) -> Tensor:
    """Apply the Euclidean transpose of forward_angle to a detector row [M].

    This is A_theta^T, using exactly the forward sampler's interpolation and
    ray-step weights. The uniform backfill in backprojection_angle is B_theta.
    """
    ray_x, ray_y = _autograd_rays(ray_x, ray_y)
    return RadonAdjoint.apply(
        projection, ray_x, ray_y, float(angle), geometry, geometry.resolution, geometry.resolution
    )


def backprojection_angle(projection: Tensor, angle: float, geometry: FanbeamGeometry) -> Tensor:
    """Uniformly backfill a detector row [M] into an image [N, N].

    This is the B operator in the proof, with ``angle`` in radians. Only
    projection values are differentiated; angle and geometry are fixed.
    """
    return RadonBackprojection.apply(projection, float(angle), geometry)
