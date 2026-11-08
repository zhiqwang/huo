"""Public projection, reconstruction, and CLI regressions."""

from dataclasses import FrozenInstanceError
import math

import numpy as np
import pytest
import torch

from huo import RadonFanbeam
from huo.cli import main


@pytest.mark.parametrize("method", ["forward", "backprojection", "art"])
def test_zero_input_and_output_shape(radon, method):
    n, m, v = radon.geometry.resolution, radon.geometry.det_count, len(radon.angles)
    data = torch.zeros((n, n) if method == "forward" else (v, m))
    result = getattr(radon, method)(data)
    assert result.shape == ((v, m) if method == "forward" else (n, n))
    assert torch.count_nonzero(result) == 0


@pytest.mark.parametrize("method", ["forward", "backprojection"])
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


@pytest.mark.parametrize("method", ["forward", "backprojection"])
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


def test_geometry_cannot_diverge_from_cached_rays(radon):
    with pytest.raises(FrozenInstanceError):
        radon.geometry.source_distance = 10
    original = radon.angles
    radon.angles[0] = 100
    torch.testing.assert_close(radon.angles, original)


@pytest.mark.parametrize("method", ["backprojection", "art"])
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
