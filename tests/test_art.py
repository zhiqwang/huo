"""Public projection, reconstruction, and CLI regressions."""

from dataclasses import FrozenInstanceError
import math

import numpy as np
import pytest
import torch

from huo import RadonFanbeam
from huo.cli import main


@pytest.mark.parametrize("method", ["forward", "backprojection", "adjoint", "art"])
def test_zero_input_and_output_shape(radon, method):
    n, m, v = radon.geometry.resolution, radon.geometry.det_count, len(radon.angles)
    data = torch.zeros((n, n) if method == "forward" else (v, m))
    result = getattr(radon, method)(data)
    assert result.shape == ((v, m) if method == "forward" else (n, n))
    assert torch.count_nonzero(result) == 0


@pytest.mark.parametrize("method", ["forward", "backprojection", "adjoint"])
def test_signed_linearity(radon, method):
    shape = (radon.geometry.resolution,) * 2 if method == "forward" else (len(radon.angles), 20)
    generator = torch.Generator().manual_seed(42)
    a, b = [torch.randn(shape, generator=generator) for _ in range(2)]
    operator = getattr(radon, method)
    torch.testing.assert_close(operator(2 * a - b), 2 * operator(a) - operator(b), atol=2e-5, rtol=2e-5)


def test_sinogram_rows_follow_unsorted_radian_angles():
    radon = RadonFanbeam(8, [math.pi / 2, -0.37, 0], 20, 8, 13, 1)
    image = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64
    sinogram = radon.forward(image)
    assert sinogram.shape == (3, 13)
    for index, angle in enumerate(radon.angles):
        torch.testing.assert_close(sinogram[index], radon.forward_angle(image, angle.item()))


def test_backprojection_sums_angle_rows(radon):
    generator = torch.Generator().manual_seed(42)
    sinogram = torch.randn(len(radon.angles), 20, generator=generator)
    expected = sum(
        radon.backprojection_angle(row, angle.item()) for row, angle in zip(sinogram, radon.angles)
    )
    torch.testing.assert_close(radon.backprojection(sinogram), expected)


@pytest.mark.parametrize("method", ["forward", "backprojection", "adjoint"])
def test_nonnegative_weights(radon, method):
    shape = (16, 16) if method == "forward" else (12, 20)
    assert (getattr(radon, method)(torch.ones(shape)) >= 0).all()


def test_art_reduces_projection_error(radon):
    axis = torch.arange(16, dtype=torch.float32) - 7.5
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    image = (((x - 1) / 4) ** 2 + ((y + 0.5) / 3) ** 2 <= 1).float()
    sinogram = radon.forward(image)
    torch.manual_seed(0)
    reconstructed = radon.art(sinogram)
    assert torch.isfinite(reconstructed).all() and (reconstructed >= 0).all()
    assert (radon.forward(reconstructed) - sinogram).norm() < sinogram.norm()
    torch.manual_seed(0)
    torch.testing.assert_close(radon.art(sinogram), reconstructed)


@pytest.mark.parametrize("grad_mode", [torch.enable_grad, torch.no_grad, torch.inference_mode])
def test_art_preserves_input_and_agrees_across_grad_modes(radon, grad_mode):
    generator = torch.Generator().manual_seed(42)
    sinogram = torch.rand(12, 40, generator=generator)[:, ::2].requires_grad_()
    original = sinogram.detach().clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        expected = radon.art(sinogram.detach())
        with grad_mode():
            torch.manual_seed(0)
            actual = radon.art(sinogram)
    assert actual.requires_grad == (grad_mode is torch.enable_grad)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(sinogram, original, atol=0, rtol=0)


def test_art_gradient_matches_finite_difference():
    radon = RadonFanbeam(5, [0.37, -0.41, math.pi / 2], 20, 10, 13, 1.3)
    generator = torch.Generator().manual_seed(12)
    sinogram = (2 + torch.rand(3, 26, generator=generator))[:, ::2].requires_grad_()
    weights = torch.randn(5, 5, generator=generator)
    direction = torch.randn(3, 13, generator=generator)

    def loss(value):
        torch.manual_seed(4)
        return (radon.art(value) * weights).sum()

    with torch.random.fork_rng(devices=[]):
        (gradient,) = torch.autograd.grad(loss(sinogram), sinogram)
        epsilon = 1e-3
        with torch.no_grad():
            numerical = (loss(sinogram + epsilon * direction) - loss(sinogram - epsilon * direction)) / (
                2 * epsilon
            )
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    torch.testing.assert_close((gradient * direction).sum(), numerical, atol=2e-3, rtol=2e-3)


def test_angles_are_copied_and_normalized_before_projection():
    angles = torch.tensor([math.pi / 2, -0.37123456789, 0.123456789], dtype=torch.float64)
    expected_angles = angles.float()
    radon = RadonFanbeam(8, angles, 32, 16, 13, 1.3)
    image = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64
    angles.fill_(100)
    radon.angles.fill_(200)
    expected = torch.stack([radon.forward_angle(image, angle.item()) for angle in expected_angles])
    torch.testing.assert_close(radon.angles, expected_angles, atol=0, rtol=0)
    torch.testing.assert_close(radon.forward(image), expected, atol=0, rtol=0)


def test_geometry_cannot_diverge_from_cached_rays(radon):
    with pytest.raises(FrozenInstanceError):
        radon.geometry.source_distance = 10
    original = radon.angles
    radon.angles[0] = 100
    torch.testing.assert_close(radon.angles, original)


@pytest.mark.parametrize("method", ["backprojection", "adjoint", "art"])
def test_rejects_detector_by_angle_layout(radon, method):
    with pytest.raises(ValueError, match="shape"):
        getattr(radon, method)(torch.zeros(20, 12))


@pytest.mark.parametrize(
    "bad_image",
    [torch.zeros(16, 15), torch.zeros(16, 16, dtype=torch.float64), torch.full((16, 16), float("nan"))],
)
def test_rejects_invalid_image(radon, bad_image):
    with pytest.raises(ValueError):
        radon.forward(bad_image)


@pytest.mark.parametrize("resolution", [5, 6])
@pytest.mark.parametrize("ray_sampling", [1, 2, 3])
def test_adjoint_matches_forward_inner_product(resolution, ray_sampling):
    scanner = RadonFanbeam(resolution, [0.37, math.pi / 2, -0.41], 24, 12, 9, 1.3, ray_sampling)
    generator = torch.Generator().manual_seed(42)
    image = torch.randn(resolution, resolution, generator=generator).T
    sinogram = torch.randn(3, 18, generator=generator)[:, ::2]
    actual = scanner.adjoint(sinogram)
    expected = sum(scanner.adjoint_angle(row, angle.item()) for row, angle in zip(sinogram, scanner.angles))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        (scanner.forward(image) * sinogram).sum(), (image * actual).sum(), atol=2e-5, rtol=2e-5
    )
    variable = image.detach().requires_grad_()
    (gradient,) = torch.autograd.grad(scanner.forward(variable), variable, sinogram)
    torch.testing.assert_close(actual, gradient)


@pytest.mark.parametrize("method", ["forward", "backprojection", "adjoint"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_vmapped_public_operators_still_reject_nonfinite_values(radon, method, invalid):
    shape = (16, 16) if method == "forward" else (12, 20)
    data = torch.zeros(2, *shape)
    data[1, 0, 0] = invalid
    with pytest.raises(ValueError, match="finite"):
        torch.vmap(getattr(radon, method))(data)


def test_cli_accepts_explicit_radian_angle_file(tmp_path):
    angles = np.array([0.7, -0.3, 0], dtype=np.float32)
    radon = RadonFanbeam(8, angles, 20, 8, 13, 1)
    sinogram = radon.forward(torch.ones(8, 8)).numpy()
    np.save(tmp_path / "sinogram.npy", sinogram)
    np.save(tmp_path / "angles.npy", angles)
    main(
        [
            str(tmp_path / "sinogram.npy"),
            "--angles",
            str(tmp_path / "angles.npy"),
            "--resolution",
            "8",
            "--source-distance",
            "20",
            "--det-distance",
            "8",
            "--det-count",
            "13",
            "--det-spacing",
            "1",
            "-o",
            str(tmp_path / "image.npy"),
        ]
    )
    result = np.load(tmp_path / "image.npy")
    assert result.shape == (8, 8) and np.isfinite(result).all() and (result >= 0).all()
