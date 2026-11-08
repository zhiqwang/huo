"""Analytic geometry checks in radians and pixel units, including migration."""

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from huo import FanbeamGeometry, RadonFanbeam
from huo.art import _linear_sample, forward_angle
from huo.geometry import compute_ray_coordinates

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "geometry.json").read_text())


@pytest.mark.parametrize("resolution", [5, 6])
def test_ray_samples_follow_source_to_detector(resolution):
    geometry = FanbeamGeometry(resolution, 20, 16, 31, 1.3)
    ray_x, ray_y = compute_ray_coordinates(geometry)
    x, y = ray_x.double() * geometry.pixel_extent, ray_y.double() * geometry.pixel_extent
    u = (torch.arange(31, dtype=torch.float64) - 15) * 1.3
    # Source (0,20), detector (u,-16): their cross product with each sample
    # offset vanishes. This independently measures line incidence in pixels.
    cross = -36 * x - u * (y - 20)
    distance = cross / torch.sqrt(u.square() + 36**2)
    torch.testing.assert_close(distance, torch.zeros_like(distance), atol=2e-6, rtol=0)
    expected_t = 20 - resolution / 2 + torch.arange(2 * resolution + 1, dtype=torch.float64) / 2
    torch.testing.assert_close(torch.hypot(x, y - 20), expected_t[:, None].expand_as(x), atol=2e-6, rtol=0)


@pytest.mark.parametrize("resolution", [5, 6])
@pytest.mark.parametrize("angle", [0, math.pi / 2, 0.37, -0.41])
def test_detector_ramp_at_pixel_centres(resolution, angle):
    radon = RadonFanbeam(resolution, [angle], 20, 16, 31, 1.3)
    centres = torch.arange(resolution, dtype=torch.float64) - (resolution - 1) / 2
    y, x = torch.meshgrid(centres, centres, indexing="ij")
    c, s = math.cos(angle), math.sin(angle)
    u = 36 * (c * x - s * y) / (20 - s * x - c * y)
    expected = (0.3 * u + 0.7).float()
    column = 0.3 * (torch.arange(31, dtype=torch.float32) - 15) * 1.3 + 0.7
    torch.testing.assert_close(radon.backprojection(column[None]), expected, atol=3e-6, rtol=2e-6)


@pytest.mark.parametrize("resolution", [5, 6])
def test_forward_quarter_turn_has_the_public_radian_sign(resolution):
    radon = RadonFanbeam(resolution, [0, math.pi / 2], 20, 16, 31, 1.3)
    image = torch.arange(resolution**2, dtype=torch.float32).reshape(resolution, resolution) / 10 - 1
    torch.testing.assert_close(
        radon.forward_angle(image, math.pi / 2),
        radon.forward_angle(torch.rot90(image, -1), 0),
        atol=3e-6,
        rtol=2e-6,
    )


def test_forward_zero_padding_with_partial_edge_weights():
    f = FIXTURES["image_boundary"]
    geometry = FanbeamGeometry(2, 2, 1, 2, 1)
    result = forward_angle(
        torch.tensor(f["image"], dtype=torch.float32),
        torch.tensor(f["ray_x"], dtype=torch.float32),
        torch.tensor(f["ray_y"], dtype=torch.float32),
        0,
        geometry,
    )
    torch.testing.assert_close(result, torch.tensor(f["expected"]))


def test_detector_zero_padding_with_partial_edge_weights():
    f = FIXTURES["detector_boundary"]
    radon = RadonFanbeam(3, [0], 3, 1.5, 3, 1.5)
    result = radon.backprojection(torch.tensor([f["projection"]], dtype=torch.float32))
    torch.testing.assert_close(result, torch.tensor(f["expected"], dtype=torch.float32))


def test_detector_interpolation_at_centres_edges_and_outside_support():
    projection = torch.tensor([2.0, -1.0, 4.0])
    index = torch.tensor([-100.0, -1, -0.75, -0.5, 0, 0.25, 1, 1.75, 2, 2.5, 2.75, 3, 100])
    expected = torch.tensor([0.0, 0, 0.5, 1, 2, 1.25, -1, 2.75, 4, 2, 1, 0, 0])
    torch.testing.assert_close(_linear_sample(projection, index), expected)


@pytest.mark.parametrize("det_count", [2, 7])
def test_detector_interpolation_preserves_values_and_gradients(det_count):
    # A strided input exercises both interpolation and autograd without assuming
    # contiguous storage. grid_sample is an independent reference for the weights.
    projection = torch.linspace(-2, 3, 2 * det_count)[::2].requires_grad_()
    index = torch.linspace(-det_count, 2 * det_count, 35).reshape(5, 7)
    normalized = 2 * index / (det_count - 1) - 1
    grid = torch.stack((torch.zeros_like(normalized), normalized), dim=-1).unsqueeze(0)
    reference = F.grid_sample(
        projection.reshape(1, 1, det_count, 1), grid, padding_mode="zeros", align_corners=True
    ).reshape_as(index)
    actual = _linear_sample(projection, index)
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-6)
    weights = index.sin()
    expected_gradient = torch.autograd.grad((reference * weights).sum(), projection)[0]
    actual_gradient = torch.autograd.grad((actual * weights).sum(), projection)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-6, rtol=1e-6)


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


def test_previous_physical_geometry_converts_without_reflecting_the_image():
    # Previous convention: mm, source on -y, R_angle rays, [detector, angle].
    # Independent old-convention sampling establishes all four conversions:
    # lengths / pixel_size, angle -> pi-angle, reversed detectors, projections / pixel_size.
    n, m, length, source, detector = 6, 11, 12.0, 40.0, 16.0
    pixel_size = length / n
    old_angles = torch.tensor([0.0, 0.37, -0.41], dtype=torch.float64)
    image = torch.arange(n * n, dtype=torch.float32).reshape(n, n) / 10 - 1
    u = (torch.arange(m, dtype=torch.float64) - (m - 1) / 2) * 2
    t = source + torch.linspace(-length / 2, length / 2, 2 * n + 1, dtype=torch.float64)[:, None]
    ray_x = t * u / torch.sqrt(u.square() + (source + detector) ** 2)
    ray_y = -source + t * (source + detector) / torch.sqrt(u.square() + (source + detector) ** 2)
    old_rows = []
    for angle in old_angles:
        c, s = math.cos(angle), math.sin(angle)
        grid = torch.stack((c * ray_x - s * ray_y, s * ray_x + c * ray_y), -1)
        grid = (grid / ((length - pixel_size) / 2)).float()[None]
        values = F.grid_sample(image[None, None], grid, align_corners=True)[0, 0]
        old_rows.append(values.sum(0) * pixel_size / 2)
    radon = RadonFanbeam(
        n, math.pi - old_angles, source / pixel_size, detector / pixel_size, m, 2 / pixel_size
    )
    converted = torch.stack(old_rows).flip(1) / pixel_size
    torch.testing.assert_close(radon.forward(image), converted, atol=1e-5, rtol=3e-6)
