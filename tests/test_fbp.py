# Copyright (c) 2022, Zhiqiang Wang. All rights reserved.

import torch

from huo.fbp import fbp, parallel_backprojection, ramp_filter


class TestRampFilter:
    """Tests for the ramp_filter function."""

    def test_output_shape(self, small_param):
        """Ramp filter should preserve the sinogram shape."""
        num_angles = int(180 / small_param.rotate_step)
        sinogram = torch.rand(small_param.detr_num, num_angles)
        result = ramp_filter(sinogram, small_param)
        assert result.shape == sinogram.shape

    def test_zero_sinogram(self, small_param):
        """Filtering a zero sinogram should produce a zero result."""
        num_angles = int(180 / small_param.rotate_step)
        sinogram = torch.zeros(small_param.detr_num, num_angles)
        result = ramp_filter(sinogram, small_param)
        assert torch.allclose(result, torch.zeros_like(result))

    def test_result_is_finite(self, small_param):
        """Ramp filter output should contain only finite values."""
        num_angles = int(180 / small_param.rotate_step)
        sinogram = torch.rand(small_param.detr_num, num_angles)
        result = ramp_filter(sinogram, small_param)
        assert torch.isfinite(result).all(), "All values should be finite"

    def test_linearity(self, small_param):
        """Ramp filter should be linear: ramp(a*x) = a*ramp(x)."""
        num_angles = int(180 / small_param.rotate_step)
        sinogram = torch.rand(small_param.detr_num, num_angles)
        scale = 3.0
        result_1 = ramp_filter(sinogram, small_param)
        result_scaled = ramp_filter(scale * sinogram, small_param)
        assert torch.allclose(scale * result_1, result_scaled, atol=1e-5)

    def test_nonzero_input_changes_values(self, small_param):
        """Ramp filter should modify a non-constant sinogram."""
        num_angles = int(180 / small_param.rotate_step)
        sinogram = torch.rand(small_param.detr_num, num_angles)
        result = ramp_filter(sinogram, small_param)
        # The filtered result should differ from the original
        assert not torch.allclose(sinogram, result, atol=1e-6)


class TestParallelBackprojection:
    """Tests for the parallel_backprojection function."""

    def test_output_shape(self, small_param, parallel_coords):
        """Parallel back-projection should return an image of the correct shape."""
        sinogram_col = torch.zeros(1, 1, small_param.detr_num)
        result = parallel_backprojection(
            sinogram_col, parallel_coords.img_end, parallel_coords.detr_end, 0.0, small_param
        )
        assert result.shape == (small_param.img_pixels, small_param.img_pixels)

    def test_zero_sinogram(self, small_param, parallel_coords):
        """Back-projection of a zero sinogram should produce a zero image."""
        sinogram_col = torch.zeros(1, 1, small_param.detr_num)
        result = parallel_backprojection(
            sinogram_col, parallel_coords.img_end, parallel_coords.detr_end, 0.0, small_param
        )
        assert torch.allclose(result, torch.zeros_like(result))

    def test_nonzero_sinogram_nonzero_result(self, small_param, parallel_coords):
        """Back-projection of a non-zero sinogram should produce a non-zero image."""
        sinogram_col = torch.ones(1, 1, small_param.detr_num)
        result = parallel_backprojection(
            sinogram_col, parallel_coords.img_end, parallel_coords.detr_end, 0.0, small_param
        )
        assert result.abs().sum().item() > 0

    def test_different_angles(self, small_param, parallel_coords):
        """Parallel back-projection at different angles should produce same-shaped results."""
        sinogram_col = torch.ones(1, 1, small_param.detr_num)
        result_0 = parallel_backprojection(
            sinogram_col, parallel_coords.img_end, parallel_coords.detr_end, 0.0, small_param
        )
        result_45 = parallel_backprojection(
            sinogram_col, parallel_coords.img_end, parallel_coords.detr_end, 45.0, small_param
        )
        assert result_0.shape == result_45.shape

    def test_linearity(self, small_param, parallel_coords):
        """Parallel back-projection should be linear: bp(a*s) = a*bp(s)."""
        sinogram_col = torch.rand(1, 1, small_param.detr_num)
        scale = 2.5
        result_1 = parallel_backprojection(
            sinogram_col, parallel_coords.img_end, parallel_coords.detr_end, 30.0, small_param
        )
        result_scaled = parallel_backprojection(
            scale * sinogram_col, parallel_coords.img_end, parallel_coords.detr_end, 30.0, small_param
        )
        assert torch.allclose(scale * result_1, result_scaled, atol=1e-5)

    def test_result_is_finite(self, small_param, parallel_coords):
        """Parallel back-projection should produce only finite values."""
        sinogram_col = torch.rand(1, 1, small_param.detr_num)
        result = parallel_backprojection(
            sinogram_col, parallel_coords.img_end, parallel_coords.detr_end, 0.0, small_param
        )
        assert torch.isfinite(result).all(), "All values should be finite"


class TestFbp:
    """Tests for the full FBP reconstruction function."""

    def test_output_shape(self, small_param, parallel_coords):
        """FBP should return an image of shape (img_pixels, img_pixels)."""
        num_angles = len(parallel_coords.gantry_view)
        sinogram = torch.zeros(small_param.detr_num, num_angles)
        result = fbp(
            sinogram, parallel_coords.img_end, parallel_coords.detr_end,
            parallel_coords.gantry_view, small_param,
        )
        assert result.shape == (small_param.img_pixels, small_param.img_pixels)

    def test_zero_sinogram(self, small_param, parallel_coords):
        """Reconstructing from a zero sinogram should give a zero image."""
        num_angles = len(parallel_coords.gantry_view)
        sinogram = torch.zeros(small_param.detr_num, num_angles)
        result = fbp(
            sinogram, parallel_coords.img_end, parallel_coords.detr_end,
            parallel_coords.gantry_view, small_param,
        )
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-5)

    def test_result_is_finite(self, small_param, parallel_coords):
        """FBP output should contain only finite values (no NaN or Inf)."""
        num_angles = len(parallel_coords.gantry_view)
        sinogram = torch.rand(small_param.detr_num, num_angles)
        result = fbp(
            sinogram, parallel_coords.img_end, parallel_coords.detr_end,
            parallel_coords.gantry_view, small_param,
        )
        assert torch.isfinite(result).all(), "All values should be finite"

    def test_nonzero_sinogram_nonzero_result(self, small_param, parallel_coords):
        """FBP reconstruction of a non-zero sinogram should produce a non-trivial result."""
        num_angles = len(parallel_coords.gantry_view)
        sinogram = torch.rand(small_param.detr_num, num_angles)
        result = fbp(
            sinogram, parallel_coords.img_end, parallel_coords.detr_end,
            parallel_coords.gantry_view, small_param,
        )
        assert result.abs().sum().item() > 0

    def test_linearity(self, small_param, parallel_coords):
        """FBP should be linear: fbp(a*s) = a*fbp(s)."""
        num_angles = len(parallel_coords.gantry_view)
        sinogram = torch.rand(small_param.detr_num, num_angles)
        scale = 2.0
        result_1 = fbp(
            sinogram, parallel_coords.img_end, parallel_coords.detr_end,
            parallel_coords.gantry_view, small_param,
        )
        result_scaled = fbp(
            scale * sinogram, parallel_coords.img_end, parallel_coords.detr_end,
            parallel_coords.gantry_view, small_param,
        )
        assert torch.allclose(scale * result_1, result_scaled, atol=1e-4)
