"""Public module, state, batch, and input contracts."""

from dataclasses import FrozenInstanceError
import io
import math

import pytest

torch = pytest.importorskip("torch")

from huo import FanbeamGeometry
from huo.torch import FanbeamProjector


def test_projection_shape_linearity_and_hooks(projector):
    assert isinstance(projector, torch.nn.Module)
    assert list(projector.parameters()) == []
    generator = torch.Generator().manual_seed(42)
    a, b = [torch.randn(16, 16, generator=generator) for _ in range(2)]
    observed = []
    hook = projector.register_forward_hook(lambda module, inputs, output: observed.append(output.shape))
    torch.testing.assert_close(projector(2 * a - b), 2 * projector(a) - projector(b), atol=3e-5, rtol=3e-5)
    hook.remove()
    assert observed == [(12, 20)] * 3
    assert torch.count_nonzero(projector(torch.zeros_like(a))) == 0
    assert (projector(torch.ones_like(a)) >= 0).all()


def test_projection_rows_follow_unsorted_angles():
    geometry = FanbeamGeometry(8, 32, 16, 13, 1.3)
    angles = torch.tensor([math.pi / 2, -0.37, 0.0])
    projector = FanbeamProjector(geometry, angles)
    image = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64
    expected = torch.stack([FanbeamProjector(geometry, angle[None])(image)[0] for angle in angles])
    angles.fill_(100)
    torch.testing.assert_close(projector(image), expected)


def test_module_dtype_and_state_dict(projector):
    projector.double()
    image = torch.randn(16, 16, dtype=torch.float64)
    expected = projector(image)
    assert expected.dtype == torch.float64
    assert all(buffer.dtype == torch.float64 for buffer in projector.buffers())
    stream = io.BytesIO()
    torch.save(projector.state_dict(), stream)
    stream.seek(0)
    restored = FanbeamProjector(projector.geometry, torch.zeros(12), dtype=torch.float64)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    torch.testing.assert_close(restored(image), expected, atol=0, rtol=0)
    with pytest.raises(FrozenInstanceError):
        restored.geometry.source_distance = 1
    incompatible = FanbeamProjector(
        FanbeamGeometry(16, 108, 73 / 3, 20, 1), torch.zeros(12), dtype=torch.float64
    )
    with pytest.raises(ValueError, match="geometry"):
        incompatible.load_state_dict(projector.state_dict())


@pytest.mark.parametrize("angles", [[], [[0]], [float("nan")], [float("inf")]])
def test_rejects_invalid_angles(angles):
    with pytest.raises(ValueError, match="angles"):
        FanbeamProjector(FanbeamGeometry(5, 20, 10, 7, 1.3), angles)


@pytest.mark.parametrize(
    "image",
    [
        torch.zeros(16, 15),
        torch.zeros(16, 16, dtype=torch.float64),
        torch.zeros(0, 16, 16),
        torch.zeros(16, 16, dtype=torch.int64),
    ],
)
def test_rejects_wrong_image_contract(projector, image):
    with pytest.raises(ValueError):
        projector(image)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_native_and_mapped_batches_reject_nonfinite_values(projector, invalid):
    images = torch.zeros(2, 16, 16)
    images[1, 0, 0] = invalid
    for operator in (projector, torch.vmap(projector)):
        with pytest.raises(ValueError, match="finite"):
            operator(images)
