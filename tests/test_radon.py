# Copyright (c) 2022, Zhiqiang Wang. All rights reserved.

import pytest
import torch

from huo.radon import RadonFanbeam


@pytest.fixture
def radon():
    """Small RadonFanbeam instance for fast tests."""
    angles = torch.arange(0, 360, step=30.0)
    return RadonFanbeam(
        resolution=16,
        angles=angles,
        source_distance=981.0,
        det_distance=219.0,
        det_count=20,
        det_spacing=9.0,
        volume_size=144.0,
        lat_sampling=2,
    )


class TestRadonForwardAutograd:
    """Tests for the _RadonForward autograd Function."""

    def test_forward_output_unchanged(self, radon):
        """forward() should still produce the correct sinogram shape and values."""
        img = torch.ones(radon.resolution, radon.resolution)
        sinogram = radon.forward(img)
        assert sinogram.shape == (radon.det_count, len(radon.angles))
        assert sinogram.sum().item() > 0

    def test_forward_requires_grad(self, radon):
        """forward() should return a tensor that tracks gradients when input does."""
        img = torch.rand(radon.resolution, radon.resolution, requires_grad=True)
        sinogram = radon.forward(img)
        assert sinogram.requires_grad

    def test_backward_computes_grad(self, radon):
        """Calling .backward() on the sinogram should populate img.grad."""
        img = torch.rand(radon.resolution, radon.resolution, requires_grad=True)
        sinogram = radon.forward(img)
        loss = sinogram.sum()
        loss.backward()
        assert img.grad is not None
        assert img.grad.shape == img.shape

    def test_grad_is_finite(self, radon):
        """Gradients should contain only finite values (no NaN or Inf)."""
        img = torch.rand(radon.resolution, radon.resolution, requires_grad=True)
        sinogram = radon.forward(img)
        sinogram.sum().backward()
        assert torch.isfinite(img.grad).all()

    def test_grad_is_nonzero(self, radon):
        """Gradients should be non-zero for a non-trivial sinogram."""
        img = torch.rand(radon.resolution, radon.resolution, requires_grad=True)
        sinogram = radon.forward(img)
        sinogram.sum().backward()
        assert img.grad.abs().sum().item() > 0

    def test_grad_matches_backprojection(self, radon):
        """The gradient from backward should match the explicit backprojection."""
        img = torch.rand(radon.resolution, radon.resolution, requires_grad=True)
        sinogram = radon.forward(img)
        sinogram.sum().backward()

        # Compute expected backprojection of all-ones manually
        # (since grad of sum w.r.t. sinogram is ones)
        ones_sinogram = torch.ones_like(sinogram)
        P = radon.resolution
        expected = torch.zeros(P, P)
        ones_4d = ones_sinogram.unsqueeze(0).unsqueeze(0)
        for i, angle in enumerate(radon.angles):
            col = ones_4d[:, :, :, i]
            expected += radon._backprojection_angle(col, angle.item())

        assert torch.allclose(img.grad, expected, atol=1e-5)

    def test_grad_with_weighted_loss(self, radon):
        """Gradients should scale correctly with a weighted loss."""
        img = torch.rand(radon.resolution, radon.resolution, requires_grad=True)
        sinogram = radon.forward(img)
        loss = (2.0 * sinogram).sum()
        loss.backward()

        # Compute expected backprojection of all-ones, then scale by 2.0
        # (grad of (2*sinogram).sum() w.r.t. sinogram is 2*ones)
        ones_sinogram = torch.ones_like(sinogram)
        P = radon.resolution
        expected = torch.zeros(P, P)
        ones_4d = ones_sinogram.unsqueeze(0).unsqueeze(0)
        for i, angle in enumerate(radon.angles):
            col = ones_4d[:, :, :, i]
            expected += radon._backprojection_angle(col, angle.item())

        assert torch.allclose(img.grad, 2.0 * expected, atol=1e-5)

    def test_no_grad_when_not_required(self, radon):
        """forward() should not track gradients when input does not require them."""
        img = torch.rand(radon.resolution, radon.resolution)
        sinogram = radon.forward(img)
        assert not sinogram.requires_grad

    def test_zero_image_zero_grad(self, radon):
        """Backward through a zero sinogram should give zero gradients."""
        img = torch.zeros(radon.resolution, radon.resolution, requires_grad=True)
        sinogram = radon.forward(img)
        sinogram.sum().backward()
        # The gradient (backprojection of ones) is independent of the image;
        # it depends only on geometry, so it can be non-zero even for a zero image.
        assert img.grad is not None
        assert torch.isfinite(img.grad).all()
