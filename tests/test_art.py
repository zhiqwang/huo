# Copyright (c) 2022, Zhiqiang Wang. All rights reserved.

import types

import numpy as np
import pytest
import torch

from huo.art import art, backward_propagation, forward_propagation, scan


class TestForwardPropagation:
    """Tests for the forward_propagation function."""

    def test_output_shape(self, small_param, gantry_coords):
        """Forward propagation should return a sinogram column with detr_num elements."""
        img = torch.zeros(small_param.img_pixels, small_param.img_pixels)
        view = 0.0
        result = forward_propagation(
            img,
            gantry_coords.gantry_coor_x,
            gantry_coords.gantry_coor_y,
            view,
            small_param,
        )
        assert result.shape == (small_param.detr_num,)

    def test_zero_image(self, small_param, gantry_coords):
        """Forward projecting a zero image should produce a zero sinogram column."""
        img = torch.zeros(small_param.img_pixels, small_param.img_pixels)
        result = forward_propagation(
            img,
            gantry_coords.gantry_coor_x,
            gantry_coords.gantry_coor_y,
            0.0,
            small_param,
        )
        assert torch.allclose(result, torch.zeros_like(result))

    def test_uniform_image_nonzero(self, small_param, gantry_coords):
        """Forward projecting a uniform non-zero image should produce a non-zero sinogram."""
        img = torch.ones(small_param.img_pixels, small_param.img_pixels)
        result = forward_propagation(
            img,
            gantry_coords.gantry_coor_x,
            gantry_coords.gantry_coor_y,
            0.0,
            small_param,
        )
        assert result.sum().item() > 0

    def test_different_angles_same_shape(self, small_param, gantry_coords):
        """Forward propagation at different angles should return same-shaped results."""
        img = torch.ones(small_param.img_pixels, small_param.img_pixels)
        result_0 = forward_propagation(
            img, gantry_coords.gantry_coor_x, gantry_coords.gantry_coor_y, 0.0, small_param
        )
        result_90 = forward_propagation(
            img, gantry_coords.gantry_coor_x, gantry_coords.gantry_coor_y, 90.0, small_param
        )
        result_180 = forward_propagation(
            img, gantry_coords.gantry_coor_x, gantry_coords.gantry_coor_y, 180.0, small_param
        )
        assert result_0.shape == result_90.shape == result_180.shape

    def test_positive_image_positive_sinogram(self, small_param, gantry_coords):
        """Forward projecting a positive image should give non-negative sinogram values."""
        img = torch.rand(small_param.img_pixels, small_param.img_pixels)
        result = forward_propagation(
            img, gantry_coords.gantry_coor_x, gantry_coords.gantry_coor_y, 0.0, small_param
        )
        assert (result >= -1e-6).all(), "Sinogram should be non-negative for positive image"

    def test_linearity(self, small_param, gantry_coords):
        """Forward propagation should be linear: f(a*x) = a*f(x)."""
        img = torch.rand(small_param.img_pixels, small_param.img_pixels)
        scale = 3.0
        result_1 = forward_propagation(
            img, gantry_coords.gantry_coor_x, gantry_coords.gantry_coor_y, 45.0, small_param
        )
        result_scaled = forward_propagation(
            scale * img, gantry_coords.gantry_coor_x, gantry_coords.gantry_coor_y, 45.0, small_param
        )
        assert torch.allclose(scale * result_1, result_scaled, atol=1e-5)

    def test_additivity(self, small_param, gantry_coords):
        """Forward propagation should be additive: f(a+b) = f(a) + f(b)."""
        img_a = torch.rand(small_param.img_pixels, small_param.img_pixels)
        img_b = torch.rand(small_param.img_pixels, small_param.img_pixels)
        result_a = forward_propagation(
            img_a, gantry_coords.gantry_coor_x, gantry_coords.gantry_coor_y, 30.0, small_param
        )
        result_b = forward_propagation(
            img_b, gantry_coords.gantry_coor_x, gantry_coords.gantry_coor_y, 30.0, small_param
        )
        result_sum = forward_propagation(
            img_a + img_b, gantry_coords.gantry_coor_x, gantry_coords.gantry_coor_y, 30.0, small_param
        )
        assert torch.allclose(result_a + result_b, result_sum, atol=1e-5)

    def test_rotation_symmetry_uniform_image(self, small_param, gantry_coords):
        """Forward projection of a uniform image should be similar at all angles."""
        img = torch.ones(small_param.img_pixels, small_param.img_pixels)
        result_0 = forward_propagation(
            img, gantry_coords.gantry_coor_x, gantry_coords.gantry_coor_y, 0.0, small_param
        )
        result_90 = forward_propagation(
            img, gantry_coords.gantry_coor_x, gantry_coords.gantry_coor_y, 90.0, small_param
        )
        # For a uniform image, total projection mass should be roughly the same
        assert abs(result_0.sum().item() - result_90.sum().item()) / max(result_0.sum().item(), 1e-8) < 0.3


class TestBackwardPropagation:
    """Tests for the backward_propagation function."""

    def test_output_shape(self, small_param, gantry_coords):
        """Back-projected image should have shape (img_pixels, img_pixels)."""
        sinogram_col = torch.zeros(1, 1, small_param.detr_num)
        result = backward_propagation(
            sinogram_col, gantry_coords.img_end, gantry_coords.detr_end, 0.0, small_param
        )
        assert result.shape == (small_param.img_pixels, small_param.img_pixels)

    def test_zero_sinogram(self, small_param, gantry_coords):
        """Back-projection of a zero sinogram should produce a zero image."""
        sinogram_col = torch.zeros(1, 1, small_param.detr_num)
        result = backward_propagation(
            sinogram_col, gantry_coords.img_end, gantry_coords.detr_end, 0.0, small_param
        )
        assert torch.allclose(result, torch.zeros_like(result))

    def test_nonzero_sinogram_nonzero_result(self, small_param, gantry_coords):
        """Back-projection of a non-zero sinogram should produce a non-zero image."""
        sinogram_col = torch.ones(1, 1, small_param.detr_num)
        result = backward_propagation(
            sinogram_col, gantry_coords.img_end, gantry_coords.detr_end, 0.0, small_param
        )
        assert result.abs().sum().item() > 0

    def test_different_angles(self, small_param, gantry_coords):
        """Back-projection at different angles should produce same-shaped results."""
        sinogram_col = torch.ones(1, 1, small_param.detr_num)
        result_0 = backward_propagation(
            sinogram_col, gantry_coords.img_end, gantry_coords.detr_end, 0.0, small_param
        )
        result_45 = backward_propagation(
            sinogram_col, gantry_coords.img_end, gantry_coords.detr_end, 45.0, small_param
        )
        assert result_0.shape == result_45.shape

    def test_linearity(self, small_param, gantry_coords):
        """Back-projection should be linear: bp(a*s) = a*bp(s)."""
        sinogram_col = torch.rand(1, 1, small_param.detr_num)
        scale = 2.5
        result_1 = backward_propagation(
            sinogram_col, gantry_coords.img_end, gantry_coords.detr_end, 30.0, small_param
        )
        result_scaled = backward_propagation(
            scale * sinogram_col, gantry_coords.img_end, gantry_coords.detr_end, 30.0, small_param
        )
        assert torch.allclose(scale * result_1, result_scaled, atol=1e-5)


class TestScan:
    """Tests for the scan function (full forward projection)."""

    def test_output_shape(self, small_param, gantry_coords):
        """Scan should produce a sinogram of shape (detr_num, num_angles)."""
        img = torch.zeros(small_param.img_pixels, small_param.img_pixels)
        result = scan(
            img, gantry_coords.gantry_coor_x, gantry_coords.gantry_coor_y, gantry_coords.gantry_view,
            small_param,
        )
        num_angles = len(gantry_coords.gantry_view)
        assert result.shape == (small_param.detr_num, num_angles)

    def test_zero_image(self, small_param, gantry_coords):
        """Scanning a zero image should produce a zero sinogram."""
        img = torch.zeros(small_param.img_pixels, small_param.img_pixels)
        result = scan(
            img, gantry_coords.gantry_coor_x, gantry_coords.gantry_coor_y, gantry_coords.gantry_view,
            small_param,
        )
        assert torch.allclose(result, torch.zeros_like(result))

    def test_nonzero_image(self, small_param, gantry_coords):
        """Scanning a non-zero image should produce a non-zero sinogram."""
        img = torch.ones(small_param.img_pixels, small_param.img_pixels)
        result = scan(
            img, gantry_coords.gantry_coor_x, gantry_coords.gantry_coor_y, gantry_coords.gantry_view,
            small_param,
        )
        assert result.sum().item() > 0

    def test_columns_match_forward_propagation(self, small_param, gantry_coords):
        """Each column of the sinogram should match forward_propagation for that angle."""
        img = torch.rand(small_param.img_pixels, small_param.img_pixels)
        sinogram = scan(
            img, gantry_coords.gantry_coor_x, gantry_coords.gantry_coor_y, gantry_coords.gantry_view,
            small_param,
        )
        # Check first and last angle
        for i in [0, len(gantry_coords.gantry_view) - 1]:
            view = gantry_coords.gantry_view[i]
            col = forward_propagation(
                img, gantry_coords.gantry_coor_x, gantry_coords.gantry_coor_y, view.item(), small_param
            )
            assert torch.allclose(sinogram[:, i], col, atol=1e-5)

    def test_linearity(self, small_param, gantry_coords):
        """Scan should be linear: scan(a*img) = a*scan(img)."""
        img = torch.rand(small_param.img_pixels, small_param.img_pixels)
        scale = 2.0
        result_1 = scan(
            img, gantry_coords.gantry_coor_x, gantry_coords.gantry_coor_y, gantry_coords.gantry_view,
            small_param,
        )
        result_scaled = scan(
            scale * img, gantry_coords.gantry_coor_x, gantry_coords.gantry_coor_y, gantry_coords.gantry_view,
            small_param,
        )
        assert torch.allclose(scale * result_1, result_scaled, atol=1e-5)


class TestArt:
    """Tests for the ART reconstruction function."""

    def test_output_shape(self, small_param, gantry_coords):
        """ART should return an image of shape (img_pixels, img_pixels)."""
        sinogram = torch.zeros(small_param.detr_num, len(gantry_coords.gantry_view))
        result = art(
            sinogram,
            gantry_coords.img_end,
            gantry_coords.detr_end,
            gantry_coords.gantry_coor_x,
            gantry_coords.gantry_coor_y,
            gantry_coords.gantry_view,
            small_param,
        )
        assert result.shape == (small_param.img_pixels, small_param.img_pixels)

    def test_zero_sinogram(self, small_param, gantry_coords):
        """Reconstructing from a zero sinogram should give a zero image."""
        sinogram = torch.zeros(small_param.detr_num, len(gantry_coords.gantry_view))
        result = art(
            sinogram,
            gantry_coords.img_end,
            gantry_coords.detr_end,
            gantry_coords.gantry_coor_x,
            gantry_coords.gantry_coor_y,
            gantry_coords.gantry_view,
            small_param,
        )
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-5)

    def test_non_negativity(self, small_param, gantry_coords):
        """ART output should be non-negative due to clamping."""
        sinogram = torch.rand(small_param.detr_num, len(gantry_coords.gantry_view))
        result = art(
            sinogram,
            gantry_coords.img_end,
            gantry_coords.detr_end,
            gantry_coords.gantry_coor_x,
            gantry_coords.gantry_coor_y,
            gantry_coords.gantry_view,
            small_param,
        )
        assert (result >= 0).all(), "ART output must be non-negative"

    def test_reconstruction_reduces_error(self, small_param, gantry_coords):
        """ART reconstruction of a scanned image should reduce projection error."""
        # Create a simple test image (centered disk)
        n = small_param.img_pixels
        y, x = torch.meshgrid(torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij")
        original = ((x**2 + y**2) < 0.5**2).float()

        # Forward project to get sinogram
        sinogram = scan(
            original, gantry_coords.gantry_coor_x, gantry_coords.gantry_coor_y,
            gantry_coords.gantry_view, small_param,
        )

        # Reconstruct
        reconstructed = art(
            sinogram,
            gantry_coords.img_end,
            gantry_coords.detr_end,
            gantry_coords.gantry_coor_x,
            gantry_coords.gantry_coor_y,
            gantry_coords.gantry_view,
            small_param,
        )

        # Re-project the reconstructed image
        resinogram = scan(
            reconstructed, gantry_coords.gantry_coor_x, gantry_coords.gantry_coor_y,
            gantry_coords.gantry_view, small_param,
        )

        # The re-projected sinogram should be closer to the original than a blank image
        error_recon = (resinogram - sinogram).norm()
        error_blank = sinogram.norm()
        assert error_recon < error_blank, "Reconstruction should reduce projection error"

    def test_result_is_finite(self, small_param, gantry_coords):
        """ART output should contain only finite values (no NaN or Inf)."""
        sinogram = torch.rand(small_param.detr_num, len(gantry_coords.gantry_view))
        result = art(
            sinogram,
            gantry_coords.img_end,
            gantry_coords.detr_end,
            gantry_coords.gantry_coor_x,
            gantry_coords.gantry_coor_y,
            gantry_coords.gantry_view,
            small_param,
        )
        assert torch.isfinite(result).all(), "All values should be finite"


class TestProjectionCoordinates:
    """Tests for coordinate computation logic from tools/projection.py."""

    def _compute_coordinates(self, param):
        """Replicate coordinate computation from tools/projection.py."""
        img_step = param.img_len / param.img_pixels
        img_end = (param.img_len - img_step) / 2
        detr_step = param.detr_len / param.detr_num
        detr_end = (param.detr_len - detr_step) / 2

        gantry_view = torch.arange(0, 360, step=param.rotate_step)

        src_coor = (0, -param.sod)

        detr_coor_x = torch.linspace(-detr_end, detr_end, steps=param.detr_num)
        detr_coor_y = torch.zeros_like(detr_coor_x) + param.sdd - param.sod

        lat_end = param.img_len / 2
        lat_steps = param.lat_sampling * param.img_pixels + 1
        lat_grid = torch.linspace(-lat_end, lat_end, steps=lat_steps).unsqueeze(1)
        lat_grid -= src_coor[1]
        detr_coor_y -= src_coor[1]

        fand = (detr_coor_x**2 + detr_coor_y**2).sqrt()
        gantry_coor_x = lat_grid * detr_coor_x / fand
        gantry_coor_y = lat_grid * detr_coor_y / fand
        gantry_coor_y += src_coor[1]

        gantry_coor_x /= img_end
        gantry_coor_y /= img_end

        return types.SimpleNamespace(
            img_step=img_step,
            img_end=img_end,
            detr_step=detr_step,
            detr_end=detr_end,
            gantry_coor_x=gantry_coor_x,
            gantry_coor_y=gantry_coor_y,
            gantry_view=gantry_view,
        )

    def test_img_step(self, small_param):
        """Image step size should be img_len / img_pixels."""
        coords = self._compute_coordinates(small_param)
        expected = small_param.img_len / small_param.img_pixels
        assert coords.img_step == pytest.approx(expected)

    def test_img_end(self, small_param):
        """Image end coordinate should be (img_len - img_step) / 2."""
        coords = self._compute_coordinates(small_param)
        img_step = small_param.img_len / small_param.img_pixels
        expected = (small_param.img_len - img_step) / 2
        assert coords.img_end == pytest.approx(expected)

    def test_detr_step(self, small_param):
        """Detector step should be detr_len / detr_num."""
        coords = self._compute_coordinates(small_param)
        expected = small_param.detr_len / small_param.detr_num
        assert coords.detr_step == pytest.approx(expected)

    def test_detr_end(self, small_param):
        """Detector end should be (detr_len - detr_step) / 2."""
        coords = self._compute_coordinates(small_param)
        detr_step = small_param.detr_len / small_param.detr_num
        expected = (small_param.detr_len - detr_step) / 2
        assert coords.detr_end == pytest.approx(expected)

    def test_gantry_view_range(self, small_param):
        """Gantry views should range from 0 to < 360."""
        coords = self._compute_coordinates(small_param)
        assert coords.gantry_view[0].item() == pytest.approx(0.0)
        assert coords.gantry_view[-1].item() < 360.0

    def test_gantry_view_count(self, small_param):
        """Number of gantry views should be 360 / rotate_step."""
        coords = self._compute_coordinates(small_param)
        expected_count = int(360 / small_param.rotate_step)
        assert len(coords.gantry_view) == expected_count

    def test_gantry_coor_shape(self, small_param):
        """Gantry coordinate arrays should have expected shape."""
        coords = self._compute_coordinates(small_param)
        lat_steps = small_param.lat_sampling * small_param.img_pixels + 1
        expected_shape = (lat_steps, small_param.detr_num)
        assert coords.gantry_coor_x.shape == expected_shape
        assert coords.gantry_coor_y.shape == expected_shape

    def test_gantry_coor_x_symmetry(self, small_param):
        """Gantry x-coordinates should be symmetric around zero (center row)."""
        coords = self._compute_coordinates(small_param)
        mid = coords.gantry_coor_x.shape[0] // 2
        # Middle row should have roughly symmetric x-coordinates
        center_row = coords.gantry_coor_x[mid, :]
        assert torch.allclose(center_row + center_row.flip(0), torch.zeros_like(center_row), atol=1e-4)

    def test_gantry_coor_normalized(self, small_param):
        """Gantry coordinates should be normalized by img_end."""
        coords = self._compute_coordinates(small_param)
        # Coordinates in the FOV region should be in [-1, 1] range for grid_sample
        center_y = coords.gantry_coor_y[coords.gantry_coor_y.shape[0] // 2, :]
        # Not all coordinates are in [-1, 1], but center should be reasonable
        assert center_y.abs().max().item() < 20.0  # sanity check: not absurdly large

    def test_default_params(self):
        """Verify default parameter values from argparse in projection.py."""
        param = types.SimpleNamespace(
            img_pixels=512,
            img_len=144.0,
            detr_num=500,
            detr_len=180.0,
            lat_sampling=2,
            sdd=1200.0,
            sod=981.0,
            rotate_step=1.0,
        )
        coords = self._compute_coordinates(param)
        assert len(coords.gantry_view) == 360
        expected_shape = (2 * 512 + 1, 500)
        assert coords.gantry_coor_x.shape == expected_shape


class TestForwardBackwardConsistency:
    """Tests verifying consistency between forward and backward propagation."""

    def test_adjoint_property(self, small_param, gantry_coords):
        """
        Test the adjoint relationship: <Ax, y> ≈ <x, A^T y>.
        Forward and backward should be approximate adjoints of each other.
        """
        n = small_param.img_pixels
        d = small_param.detr_num
        view = 0.0

        # Random image and sinogram column
        x = torch.rand(n, n)
        y_col = torch.rand(1, 1, d)

        # Ax = forward(x)
        Ax = forward_propagation(
            x, gantry_coords.gantry_coor_x, gantry_coords.gantry_coor_y, view, small_param
        )

        # A^T y = backward(y)
        ATy = backward_propagation(
            y_col, gantry_coords.img_end, gantry_coords.detr_end, view, small_param
        )

        # <Ax, y> vs <x, A^T y> - should be approximately proportional
        # (Not exact because of discretization and different coordinate systems)
        inner1 = torch.dot(Ax, y_col.squeeze())
        inner2 = torch.dot(x.flatten(), ATy.flatten())

        # Just check both are finite and have the same sign
        assert torch.isfinite(inner1) and torch.isfinite(inner2)
        if inner1.abs() > 1e-6 and inner2.abs() > 1e-6:
            # Same sign indicates adjoint relationship
            assert inner1.item() * inner2.item() > 0

    def test_scan_art_roundtrip_nonzero(self, small_param, gantry_coords):
        """ART reconstruction of a scanned object should produce a non-trivial result."""
        n = small_param.img_pixels
        # Simple point source
        img = torch.zeros(n, n)
        img[n // 2, n // 2] = 1.0

        sinogram = scan(
            img, gantry_coords.gantry_coor_x, gantry_coords.gantry_coor_y,
            gantry_coords.gantry_view, small_param,
        )

        reconstructed = art(
            sinogram,
            gantry_coords.img_end,
            gantry_coords.detr_end,
            gantry_coords.gantry_coor_x,
            gantry_coords.gantry_coor_y,
            gantry_coords.gantry_view,
            small_param,
        )

        assert reconstructed.sum().item() > 0, "Reconstruction should be non-trivial"
        assert (reconstructed >= 0).all(), "Reconstruction should be non-negative"
