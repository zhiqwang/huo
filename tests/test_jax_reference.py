"""Optional JAX experiments checked against the sampled PyTorch operator.

Run with ``uv run --group test --all-extras pytest tests/test_jax_reference.py``.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

jax = pytest.importorskip("jax")
pytest.importorskip("optax")
import jax.numpy as jnp
from jax.test_util import check_grads

from benchmarks.jax_projector import IMPLEMENTATIONS, make_projector, make_reconstructor
from huo import FanbeamGeometry
from huo.jax import FanbeamProjector as JaxProjector, reconstruct as jax_reconstruct
from huo.torch import FanbeamProjector, reconstruct


@pytest.fixture(autouse=True)
def double_precision():
    with jax.enable_x64():
        yield


def sampling_arrays(projector):
    return tuple(
        jnp.asarray(value.numpy()) for value in (projector._ray_x, projector._ray_y, projector.angles)
    )


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("resolution", [5, 6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward_gradient_and_batches_match_pytorch(implementation, resolution, dtype):
    geometry = FanbeamGeometry(resolution, 4 * resolution, 2 * resolution, 7, 1.3, 3)
    projector = FanbeamProjector(geometry, [math.pi / 2, -0.37, 0, -0.37], dtype=dtype)
    generator = torch.Generator().manual_seed(42)
    image = torch.randn(2, 3, resolution, resolution, dtype=dtype, generator=generator).transpose(0, 1)
    image.requires_grad_()
    expected = projector(image)
    weights = torch.randn(expected.shape, dtype=dtype, generator=generator)
    (gradient,) = torch.autograd.grad(expected, image, weights)
    project = make_projector(geometry.ray_step, implementation=implementation)
    rays = sampling_arrays(projector)
    values = jnp.asarray(image.detach().numpy())
    actual = jax.jit(project)(values, *rays)
    actual_gradient = jax.jit(jax.grad(lambda x: jnp.sum(project(x, *rays) * jnp.asarray(weights.numpy()))))(
        values
    )
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    np.testing.assert_allclose(actual, expected.detach(), atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(actual_gradient, gradient, atol=tolerance, rtol=tolerance)
    mapped = jax.vmap(jax.vmap(lambda x: project(x, *rays)))(values)
    np.testing.assert_allclose(mapped, actual, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_jvp_hessian_and_finite_differences(implementation):
    projector = FanbeamProjector(FanbeamGeometry(3, 12, 6, 5, 1.3), [0, -0.37], dtype=torch.float64)
    basis = torch.eye(9, dtype=torch.float64).reshape(9, 3, 3)
    matrix = projector(basis).reshape(9, -1).T.numpy()
    image = jnp.asarray(np.random.default_rng(17).normal(size=(3, 3)))
    direction = jnp.ones_like(image)
    project = make_projector(implementation=implementation)
    rays = sampling_arrays(projector)
    forward = lambda x: project(x, *rays)
    loss = lambda x: 0.5 * jnp.sum(forward(x) ** 2)
    _, tangent = jax.jvp(forward, (image,), (direction,))
    np.testing.assert_allclose(tangent.ravel(), matrix @ np.ones(9), atol=2e-12, rtol=2e-12)
    for jacobian in (jax.jacfwd(forward)(image), jax.jacrev(forward)(image)):
        np.testing.assert_allclose(jacobian.reshape(matrix.shape), matrix, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(
        jax.hessian(loss)(image).reshape(9, 9), matrix.T @ matrix, atol=1e-11, rtol=1e-11
    )
    check_grads(loss, (image,), order=2, modes=("fwd", "rev"), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_partial_zero_padding(implementation):
    fixture = json.loads((Path(__file__).parent / "fixtures" / "geometry.json").read_text())["image_boundary"]
    project = make_projector(implementation=implementation)
    rays = tuple(jnp.array(fixture[key], dtype=jnp.float64) for key in ("ray_x", "ray_y"))
    image = jnp.array(fixture["image"], dtype=jnp.float64)
    forward = lambda x: project(x, *rays, jnp.zeros(1))
    np.testing.assert_allclose(jax.jit(forward)(image)[0], fixture["expected"], atol=1e-12, rtol=1e-12)
    check_grads(forward, (image,), order=2, modes=("fwd", "rev"), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("nonnegative", [True, False])
def test_adam_reconstruction_matches_pytorch(implementation, nonnegative):
    projector = FanbeamProjector(FanbeamGeometry(8, 32, 16, 12, 1.3), [0, 0.5, 1, 1.5], dtype=torch.float64)
    image = torch.zeros(2, 8, 8, dtype=torch.float64)
    image[:, 2:6, 3:5] = 1 if nonnegative else -1
    measured = projector(image)
    project = make_projector(implementation=implementation)
    solver = jax.jit(make_reconstructor(project, steps=40, lr=0.03, nonnegative=nonnegative))
    rays = sampling_arrays(projector)
    actual = solver(jnp.asarray(measured.numpy()), jnp.zeros_like(jnp.asarray(image.numpy())), *rays)
    expected = reconstruct(projector, measured, steps=40, lr=0.03, nonnegative=nonnegative)
    np.testing.assert_allclose(actual, expected.numpy(), atol=2e-10, rtol=2e-10)
    residual = project(actual, *rays) - measured.numpy()
    assert float(jnp.mean(residual**2)) < 0.1 * float(measured.square().mean())


@pytest.mark.parametrize("method", ["scan", "vmap"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("nonnegative", [True, False])
def test_public_jax_projection_gradient_and_solver_match_pytorch(method, dtype, nonnegative):
    geometry = FanbeamGeometry(8, 32, 16, 12, 1.3)
    angles = [1.5, -0.37, 0, -0.37]
    projector = FanbeamProjector(geometry, angles, dtype=dtype)
    jax_dtype = jnp.float32 if dtype == torch.float32 else jnp.float64
    state = JaxProjector.create(geometry, angles, dtype=jax_dtype, method=method)
    image = torch.zeros((2, 8, 8), dtype=dtype)
    image[:, 2:6, 3:5] = 1 if nonnegative else -1
    image.requires_grad_()
    measured = projector(image)
    expected_gradient = torch.autograd.grad(0.5 * measured.square().mean(), image)[0]
    forward = jax.jit(lambda p, x: p(x))
    values = jnp.asarray(image.detach().numpy())
    actual = forward(state, values)
    actual_gradient = jax.jit(jax.grad(lambda x, p: 0.5 * jnp.mean(p(x) ** 2)))(values, state)
    tolerance = 3e-5 if dtype == torch.float32 else 2e-10
    np.testing.assert_allclose(actual, measured.detach(), atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(actual_gradient, expected_gradient, atol=tolerance, rtol=tolerance)
    initial = torch.full_like(image, -0.1)
    expected = reconstruct(
        projector, measured.detach(), steps=30, lr=0.03, initial_image=initial, nonnegative=nonnegative
    )
    result = jax_reconstruct(
        state,
        jnp.asarray(measured.detach().numpy()),
        steps=30,
        lr=0.03,
        initial_image=jnp.asarray(initial.numpy()),
        nonnegative=nonnegative,
    )
    np.testing.assert_allclose(result, expected.numpy(), atol=tolerance, rtol=tolerance)
