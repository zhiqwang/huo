"""Analytic fan-beam geometry and physical-unit conversion checks."""

import math

import numpy as np
import pytest

from huo import FanbeamGeometry
from huo.geometry import compute_ray_coordinates


@pytest.mark.parametrize("resolution", [5, 6])
@pytest.mark.parametrize("dtype,tolerance", [(np.float32, 2e-6), (np.float64, 1e-12)])
def test_ray_samples_follow_source_to_detector(resolution, dtype, tolerance):
    geometry = FanbeamGeometry(resolution, 20, 16, 31, 1.3)
    ray_x, ray_y = compute_ray_coordinates(geometry, dtype=dtype)
    x, y = ray_x.astype(np.float64) * geometry.pixel_extent, ray_y.astype(np.float64) * geometry.pixel_extent
    u = (np.arange(31, dtype=np.float64) - 15) * 1.3
    cross = -36 * x - u * (y - 20)
    distance = cross / np.sqrt(u**2 + 36**2)
    np.testing.assert_allclose(distance, 0, atol=tolerance, rtol=0)
    expected_t = 20 - resolution / 2 + np.arange(2 * resolution + 1, dtype=np.float64) / 2
    np.testing.assert_allclose(
        np.hypot(x, y - 20), np.broadcast_to(expected_t[:, None], x.shape), atol=tolerance, rtol=0
    )


@pytest.mark.parametrize("resolution", [5, 6])
def test_forward_quarter_turn_has_the_public_radian_sign(resolution):
    torch = pytest.importorskip("torch")
    from huo.torch import FanbeamProjector

    projector = FanbeamProjector(FanbeamGeometry(resolution, 20, 16, 31, 1.3), [0, math.pi / 2])
    image = torch.arange(resolution**2, dtype=torch.float32).reshape(resolution, resolution) / 10 - 1
    torch.testing.assert_close(
        projector(image)[1], projector(torch.rot90(image, -1))[0], atol=3e-6, rtol=2e-6
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("resolution", 1),
        ("resolution", 2.5),
        ("det_count", 1),
        ("det_spacing", 0),
        ("source_distance", float("nan")),
        ("source_distance", 2),
        ("ray_sampling", 0),
    ],
)
def test_invalid_geometry(field, value):
    parameters = dict(resolution=5, source_distance=20, det_distance=16, det_count=31, det_spacing=1.3)
    parameters[field] = value
    with pytest.raises(ValueError, match=field):
        FanbeamGeometry(**parameters)


def test_physical_units_and_opposite_scanner_orientation():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    from huo.torch import FanbeamProjector

    # Independent integration in mm with the source on -y. The conversion
    # changes length units, angle convention, detector order, and line-integral scale.
    n, m, length, source, detector = 6, 11, 12.0, 40.0, 16.0
    pixel_size = length / n
    physical_angles = torch.tensor([0.0, 0.37, -0.41], dtype=torch.float64)
    image = torch.arange(n * n, dtype=torch.float32).reshape(n, n) / 10 - 1
    u = (torch.arange(m, dtype=torch.float64) - (m - 1) / 2) * 2
    t = source + torch.linspace(-length / 2, length / 2, 2 * n + 1, dtype=torch.float64)[:, None]
    ray_x = t * u / torch.sqrt(u.square() + (source + detector) ** 2)
    ray_y = -source + t * (source + detector) / torch.sqrt(u.square() + (source + detector) ** 2)
    physical_rows = []
    for angle in physical_angles:
        c, s = math.cos(angle), math.sin(angle)
        grid = torch.stack((c * ray_x - s * ray_y, s * ray_x + c * ray_y), -1)
        grid = (grid / ((length - pixel_size) / 2)).float()[None]
        values = F.grid_sample(image[None, None], grid, align_corners=True)[0, 0]
        physical_rows.append(values.sum(0) * pixel_size / 2)
    geometry = FanbeamGeometry(n, source / pixel_size, detector / pixel_size, m, 2 / pixel_size)
    projector = FanbeamProjector(geometry, math.pi - physical_angles)
    converted = torch.stack(physical_rows).flip(1) / pixel_size
    torch.testing.assert_close(projector(image), converted, atol=1e-5, rtol=3e-6)
