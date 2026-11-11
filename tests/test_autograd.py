"""Independent matrix, transform, and memory checks for the forward module."""

import json
import math
from pathlib import Path
import weakref

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from huo import FanbeamGeometry, FanbeamProjector
from huo._autograd import _Project


def sampled_matrix(projector):
    n = projector.geometry.resolution
    basis = torch.eye(n * n, dtype=projector.angles.dtype).reshape(n * n, n, n)
    return torch.stack([projector(vector).flatten() for vector in basis], dim=1)


@pytest.mark.parametrize("resolution", [5, 6])
@pytest.mark.parametrize("angles", [[0], [-0.37], [math.pi / 2, -0.37, 0]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("ray_sampling", [1, 2, 3])
def test_gradient_matches_independent_matrix(resolution, angles, dtype, ray_sampling):
    geometry = FanbeamGeometry(resolution, 4 * resolution, 2 * resolution, 7, 1.3, ray_sampling)
    projector = FanbeamProjector(geometry, angles, dtype=dtype)
    generator = torch.Generator().manual_seed(42)
    image = torch.randn(resolution, resolution, dtype=dtype, generator=generator).T.requires_grad_()
    output = projector(image)
    cotangent = torch.randn(2 * output.numel(), dtype=dtype, generator=generator)[::2].reshape_as(output)
    original = cotangent.clone()
    assert not image.is_contiguous() and not cotangent.is_contiguous()
    matrix = sampled_matrix(projector)
    expected = (matrix.T @ cotangent.flatten()).reshape_as(image)
    (actual,) = torch.autograd.grad(output, image, cotangent)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(cotangent, original, atol=0, rtol=0)
    torch.testing.assert_close(
        (output * cotangent).sum(), (image * actual).sum(), atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("batch", [(), (2,)])
@pytest.mark.parametrize("ray_sampling", [1, 2, 3])
def test_first_and_second_derivatives(batch, ray_sampling):
    geometry = FanbeamGeometry(3, 12, 6, 5, 1.3, ray_sampling)
    projector = FanbeamProjector(geometry, [0.37, -0.41], dtype=torch.float64)
    image = torch.randn(*batch, 3, 3, dtype=torch.float64, requires_grad=True)
    assert gradcheck(projector, (image,))
    assert gradgradcheck(projector, (image,), gen_non_contig_grad_outputs=True)


def test_partial_padding_weights_and_their_gradients():
    fixture = json.loads((Path(__file__).parent / "fixtures" / "geometry.json").read_text())["image_boundary"]
    rays = [torch.tensor(fixture[key], dtype=torch.float64) for key in ("ray_x", "ray_y")]
    image = torch.tensor(fixture["image"], dtype=torch.float64, requires_grad=True)
    cosines, sines = torch.ones(1, dtype=image.dtype), torch.zeros(1, dtype=image.dtype)

    def operator(value):
        return _Project.apply(value, *rays, cosines, sines, 0.5)

    torch.testing.assert_close(operator(image)[0], torch.tensor(fixture["expected"], dtype=image.dtype))
    assert gradcheck(operator, (image,))
    assert gradgradcheck(operator, (image,))


def test_inference_constants_support_later_training():
    geometry = FanbeamGeometry(5, 20, 10, 7, 1.3)
    reference = FanbeamProjector(geometry, [0.37, -0.41], dtype=torch.float64)
    with torch.inference_mode():
        projector = FanbeamProjector(geometry, [0.37, -0.41], dtype=torch.float64)
    image = torch.randn(5, 5, dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(projector(image), reference(image), atol=0, rtol=0)
    assert gradcheck(projector, (image,))
    assert gradgradcheck(projector, (image,))


def test_torch_func_transforms_and_hessian():
    projector = FanbeamProjector(FanbeamGeometry(3, 12, 6, 5, 1.3, 3), [0, -0.37], dtype=torch.float64)
    generator = torch.Generator().manual_seed(17)
    image = torch.randn(3, 3, dtype=torch.float64, generator=generator)
    direction = torch.randn(3, 3, dtype=image.dtype, generator=generator)
    matrix = sampled_matrix(projector)

    def loss(value):
        return 0.5 * projector(value).square().sum()

    primal, tangent = torch.func.jvp(projector, (image,), (direction,))
    torch.testing.assert_close(primal, projector(image))
    torch.testing.assert_close(tangent.flatten(), matrix @ direction.flatten())
    torch.testing.assert_close(torch.func.grad(loss)(image).flatten(), matrix.T @ matrix @ image.flatten())
    for jacobian in (torch.func.jacrev(projector)(image), torch.func.jacfwd(projector)(image)):
        torch.testing.assert_close(jacobian.reshape_as(matrix), matrix)
    torch.testing.assert_close(torch.func.hessian(loss)(image).reshape(9, 9), matrix.T @ matrix)
    _, hvp = torch.func.jvp(torch.func.grad(loss), (image,), (direction,))
    torch.testing.assert_close(hvp.flatten(), matrix.T @ matrix @ direction.flatten())

    batch = torch.stack((image, direction), dim=-1)
    expected = torch.stack((projector(image), projector(direction)), dim=-1)
    torch.testing.assert_close(torch.vmap(projector, in_dims=-1, out_dims=-1)(batch), expected)
    gradients = torch.vmap(torch.func.grad(loss), in_dims=-1, out_dims=-1)(batch)
    torch.testing.assert_close(gradients.reshape(9, 2), matrix.T @ matrix @ batch.reshape(9, 2))


def test_batch_gradients_and_nested_vmap():
    projector = FanbeamProjector(FanbeamGeometry(5, 20, 10, 7, 1.3), [0.37, -0.41])
    images = torch.randn(2, 3, 5, 5).transpose(0, 1).requires_grad_()
    actual = projector(images)
    expected = torch.stack([torch.stack([projector(image) for image in group]) for group in images])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.vmap(torch.vmap(projector))(images), expected)
    direction = torch.randn_like(actual)
    (gradient,) = torch.autograd.grad(actual, images, direction)
    (reference,) = torch.autograd.grad(expected, images, direction)
    torch.testing.assert_close(gradient, reference)


@pytest.mark.parametrize("views", [1, 16])
def test_higher_order_graph_keeps_only_shared_sampling_storage(views):
    projector = FanbeamProjector(FanbeamGeometry(16, 64, 32, 20, 1.3), torch.arange(views) * 0.37)
    image = torch.ones(16, 16, requires_grad=True)
    references = []

    def pack(tensor):
        saved = tensor.detach()
        references.append(weakref.ref(saved))
        return saved

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = projector(image)
        cotangent = torch.ones_like(output, requires_grad=True)
        (gradient,) = torch.autograd.grad(output, image, cotangent, create_graph=True)
    assert gradient.requires_grad
    storage = {}
    for reference in references:
        tensor = reference()
        if tensor is not None:
            buffer = tensor.untyped_storage()
            storage[buffer.data_ptr()] = buffer.nbytes()
    geometry = projector.geometry
    budget = (2 * geometry.ray_sample_count * geometry.det_count + 2 * views) * image.element_size()
    assert sum(storage.values()) <= budget


def test_gradient_reaches_a_neural_network(projector):
    network = torch.nn.Sequential(torch.nn.Linear(4, 256), torch.nn.Unflatten(-1, (16, 16)))
    prediction = projector(network(torch.ones(2, 4)))
    prediction.square().mean().backward()
    gradient = network[0].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.norm() > 0
