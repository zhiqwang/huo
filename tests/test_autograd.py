"""Independent transpose and numerical checks for fixed-geometry autograd."""

import json
import math
from pathlib import Path
import weakref

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from huo import FanbeamGeometry, RadonFanbeam
from huo.art import RadonBackprojectionTranspose, adjoint_angle, backprojection_angle, forward_angle
from huo.geometry import compute_ray_coordinates


def _operator(kind, geometry, angle, dtype):
    if kind in ("forward", "adjoint"):
        ray_x, ray_y = (ray.to(dtype) for ray in compute_ray_coordinates(geometry))
        operator = forward_angle if kind == "forward" else adjoint_angle
        return lambda data: operator(data, ray_x, ray_y, angle, geometry)
    if kind == "backprojection_transpose":
        return lambda image: RadonBackprojectionTranspose.apply(image, angle, geometry, geometry.det_count)
    return lambda projection: backprojection_angle(projection, angle, geometry)


def _input_shape(kind, geometry):
    return (
        (geometry.resolution,) * 2
        if kind in ("forward", "backprojection_transpose")
        else (geometry.det_count,)
    )


@pytest.mark.parametrize("kind", ["forward", "backprojection"])
@pytest.mark.parametrize("resolution", [5, 6])
@pytest.mark.parametrize("angle", [0, -0.37])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("ray_sampling", [1, 2, 3])
def test_gradient_matches_sampled_matrix_transpose(kind, resolution, angle, dtype, ray_sampling):
    geometry = FanbeamGeometry(resolution, 4 * resolution, 2 * resolution, 7, 1.3, ray_sampling)
    operator = _operator(kind, geometry, angle, dtype)
    generator = torch.Generator().manual_seed(42)
    if kind == "forward":
        data = torch.randn(resolution, resolution, dtype=dtype, generator=generator).T.requires_grad_()
    else:
        data = torch.randn(14, dtype=dtype, generator=generator)[::2].requires_grad_()
    output = operator(data)
    cotangent = torch.randn(2 * output.numel(), dtype=dtype, generator=generator)[::2].reshape_as(output)
    original_cotangent = cotangent.clone()
    assert not data.is_contiguous() and not cotangent.is_contiguous()

    # Columns obtained from basis inputs define the actual sampled linear map,
    # independently of either backward implementation or a proposed adjoint.
    basis = torch.eye(data.numel(), dtype=dtype).reshape(data.numel(), *data.shape)
    matrix = torch.stack([operator(vector).flatten() for vector in basis], dim=1)
    expected = (matrix.T @ cotangent.flatten()).reshape_as(data)
    (actual,) = torch.autograd.grad(output, data, cotangent)
    tolerance = 5e-6 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(cotangent, original_cotangent, atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["forward", "adjoint", "backprojection", "backprojection_transpose"])
@pytest.mark.parametrize("ray_sampling", [1, 2, 3])
def test_first_and_second_order_gradients(kind, ray_sampling):
    geometry = FanbeamGeometry(5, 20, 10, 7, 1.3, ray_sampling)
    operator = _operator(kind, geometry, 0.37, torch.float64)
    shape = _input_shape(kind, geometry)
    data = torch.randn(shape, dtype=torch.float64, generator=torch.Generator().manual_seed(42))
    data.requires_grad_()
    assert gradcheck(operator, (data,))
    assert gradgradcheck(operator, (data,), gen_non_contig_grad_outputs=True)


@pytest.mark.parametrize("kind", ["forward", "adjoint"])
def test_inference_coordinates_can_be_used_in_a_gradient_graph(kind):
    geometry = FanbeamGeometry(5, 20, 10, 7, 1.3)
    with torch.inference_mode():
        operator = _operator(kind, geometry, 0.37, torch.float64)
    data = torch.randn(
        _input_shape(kind, geometry), dtype=torch.float64, generator=torch.Generator().manual_seed(42)
    )
    data.requires_grad_()
    assert gradcheck(operator, (data,))
    assert gradgradcheck(operator, (data,))


@pytest.mark.parametrize("method", ["forward", "adjoint", "backprojection"])
def test_scanner_created_during_inference_supports_later_backward(method):
    parameters = (5, [0.37, -0.41], 20, 10, 7, 1.3)
    reference = RadonFanbeam(*parameters)
    with torch.inference_mode():
        scanner = RadonFanbeam(*parameters)
    shape = (5, 5) if method == "forward" else (2, 7)
    data = torch.randn(shape, generator=torch.Generator().manual_seed(42)).requires_grad_()
    expected = getattr(reference, method)(data)
    actual = getattr(scanner, method)(data)
    (expected_gradient,) = torch.autograd.grad(expected.square().sum(), data)
    (actual_gradient,) = torch.autograd.grad(actual.square().sum(), data)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["forward", "backprojection"])
def test_gradients_retain_partial_zero_padding_weights(kind):
    fixtures = json.loads((Path(__file__).parent / "fixtures" / "geometry.json").read_text())
    if kind == "forward":
        fixture = fixtures["image_boundary"]
        geometry = FanbeamGeometry(2, 2, 1, 2, 1)
        ray_x = torch.tensor(fixture["ray_x"], dtype=torch.float64)
        ray_y = torch.tensor(fixture["ray_y"], dtype=torch.float64)
        data = torch.tensor(fixture["image"], dtype=torch.float64, requires_grad=True)

        def operator(image):
            return forward_angle(image, ray_x, ray_y, 0, geometry)

    else:
        fixture = fixtures["detector_boundary"]
        geometry = FanbeamGeometry(3, 3, 1.5, 3, 1.5)
        data = torch.tensor(fixture["projection"], dtype=torch.float64, requires_grad=True)

        def operator(projection):
            return backprojection_angle(projection, 0, geometry)

    assert gradcheck(operator, (data,))
    assert gradgradcheck(operator, (data,), gen_non_contig_grad_outputs=True)


@pytest.mark.parametrize("kind", ["forward", "adjoint", "backprojection", "backprojection_transpose"])
def test_func_transforms_match_the_sampled_linear_map(kind):
    geometry = FanbeamGeometry(5, 20, 10, 7, 1.3, 3)
    operator = _operator(kind, geometry, -0.37, torch.float64)
    generator = torch.Generator().manual_seed(13)
    data = torch.randn(_input_shape(kind, geometry), dtype=torch.float64, generator=generator)
    tangent = torch.randn(data.shape, dtype=data.dtype, generator=generator)
    basis = torch.eye(data.numel(), dtype=data.dtype).reshape(data.numel(), *data.shape)
    matrix = torch.stack([operator(vector).flatten() for vector in basis], dim=1)

    def loss(value):
        return 0.5 * operator(value).square().sum()

    primal, actual_jvp = torch.func.jvp(operator, (data,), (tangent,))
    torch.testing.assert_close(primal, operator(data))
    torch.testing.assert_close(actual_jvp.flatten(), matrix @ tangent.flatten())
    torch.testing.assert_close(torch.func.grad(loss)(data).flatten(), matrix.T @ (matrix @ data.flatten()))
    for jacobian in (torch.func.jacrev(operator)(data), torch.func.jacfwd(operator)(data)):
        torch.testing.assert_close(jacobian.reshape_as(matrix), matrix)
    hessian = torch.func.hessian(loss)(data).reshape(data.numel(), data.numel())
    torch.testing.assert_close(hessian, matrix.T @ matrix)

    # Exercise a non-leading mapped axis and compositions through the backward.
    batch = torch.stack((data, tangent), dim=-1)
    expected = torch.stack((operator(data), operator(tangent)), dim=-1)
    torch.testing.assert_close(torch.vmap(operator, in_dims=-1, out_dims=-1)(batch), expected)
    gradients = torch.vmap(torch.func.grad(loss), in_dims=-1, out_dims=-1)(batch)
    torch.testing.assert_close(
        gradients.reshape(data.numel(), 2), matrix.T @ matrix @ batch.reshape(data.numel(), 2)
    )


@pytest.mark.parametrize("method", ["forward", "adjoint", "backprojection"])
def test_public_multi_angle_transforms_and_hessian_vector_product(method):
    scanner = RadonFanbeam(5, [math.pi / 2, -0.37, 0], 20, 10, 7, 1.3, ray_sampling=3)
    operator = getattr(scanner, method)
    shape = (5, 5) if method == "forward" else (3, 7)
    generator = torch.Generator().manual_seed(17)
    data = torch.randn(shape, generator=generator).requires_grad_()
    tangent = torch.randn(shape, generator=generator)
    basis = torch.eye(data.numel()).reshape(data.numel(), *data.shape)
    matrix = torch.stack([operator(vector).flatten() for vector in basis], dim=1)

    def loss(value):
        return 0.5 * operator(value).square().sum()

    gradient = torch.autograd.grad(loss(data), data, create_graph=True)[0]
    actual = torch.autograd.grad((gradient * tangent).sum(), data)[0]
    expected = (matrix.T @ (matrix @ tangent.flatten())).reshape_as(data)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(torch.func.grad(loss)(data), gradient)
    _, actual = torch.func.jvp(torch.func.grad(loss), (data,), (tangent,))
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    batch = torch.stack((data, tangent))
    torch.testing.assert_close(torch.vmap(operator)(batch), torch.stack([operator(value) for value in batch]))
    nested_batch = torch.stack((batch, batch))
    torch.testing.assert_close(
        torch.vmap(torch.vmap(operator))(nested_batch), torch.stack([torch.vmap(operator)(batch)] * 2)
    )


@pytest.mark.parametrize("method", ["forward", "adjoint", "backprojection"])
def test_higher_order_graph_retains_bounded_sampling_storage(method):
    scanner = RadonFanbeam(16, [index * math.pi / 8 for index in range(16)], 64, 32, 20, 1.3)
    shape = (16, 16) if method == "forward" else (16, 20)
    data = torch.ones(shape, requires_grad=True)
    references = []

    def pack(tensor):
        saved = tensor.detach()
        references.append(weakref.ref(saved))
        return saved

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = getattr(scanner, method)(data)
        cotangent = torch.ones_like(output, requires_grad=True)
        (gradient,) = torch.autograd.grad(output, data, cotangent, create_graph=True)
    assert gradient.requires_grad
    storage = {}
    for reference in references:
        tensor = reference()
        if tensor is not None:
            buffer = tensor.untyped_storage()
            storage[buffer.data_ptr()] = buffer.nbytes()
    # Budget for shared rays and small metadata, independent of per-view grids.
    shared_bytes = 2 * scanner.geometry.ray_sample_count * scanner.geometry.det_count * data.element_size()
    assert sum(storage.values()) <= 2 * shared_bytes
