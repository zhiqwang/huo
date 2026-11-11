"""Optimize leaf images against fixed measurements with PyTorch."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from huo import FanbeamGeometry
from huo.torch import FanbeamProjector, reconstruct
from huo.cli import main


@pytest.fixture
def problem():
    projector = FanbeamProjector(FanbeamGeometry(8, 32, 16, 12, 1.3), [0, 0.5, 1, 1.5])
    image = torch.zeros(8, 8)
    image[2:6, 3:5] = 1
    return projector, projector(image).detach()


def test_loss_decreases_and_inputs_are_preserved(problem):
    projector, measured = problem
    target = measured.clone().requires_grad_()
    initial = torch.zeros(8, 8, requires_grad=True)
    state = {name: buffer.clone() for name, buffer in projector.named_buffers()}
    random_state = torch.get_rng_state()
    result = reconstruct(projector, target, steps=100, lr=0.03, initial_image=initial)
    assert result.shape == (8, 8) and not result.requires_grad
    assert torch.isfinite(result).all() and (result >= 0).all()
    assert (projector(result) - measured).square().mean() < 0.05 * measured.square().mean()
    torch.testing.assert_close(target, measured, atol=0, rtol=0)
    assert target.grad is None and initial.grad is None and torch.count_nonzero(initial) == 0
    torch.testing.assert_close(torch.get_rng_state(), random_state, atol=0, rtol=0)
    for name, buffer in projector.named_buffers():
        torch.testing.assert_close(buffer, state[name], atol=0, rtol=0)


def test_reconstruct_matches_an_explicit_adam_loop(problem):
    projector, measured = problem
    expected = torch.nn.Parameter(torch.zeros(8, 8))
    optimizer = torch.optim.Adam([expected], lr=0.03)
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = 0.5 * (projector(expected) - measured).square().mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            expected.clamp_min_(0)
    actual = reconstruct(projector, measured, steps=20, lr=0.03)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("mode", [torch.enable_grad, torch.no_grad, torch.inference_mode])
def test_optimizer_works_under_outer_grad_modes(problem, mode):
    projector, measured = problem
    expected = reconstruct(projector, measured, steps=5)
    with mode():
        actual = reconstruct(projector, measured.clone(), steps=5)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_batch_reconstruction_and_unconstrained_values(problem):
    projector, measured = problem
    target = -torch.stack((measured, measured))
    result = reconstruct(projector, target, steps=50, lr=0.03, nonnegative=False)
    assert result.shape == (2, 8, 8) and result.min() < 0
    assert (projector(result) - target).square().mean() < target.square().mean()
    torch.testing.assert_close(result[0], result[1])


@pytest.mark.parametrize(
    "options",
    [
        {"steps": 0},
        {"steps": True},
        {"steps": 1.2},
        {"lr": 0},
        {"lr": float("nan")},
        {"initial_image": torch.zeros(2, 8, 8)},
    ],
)
def test_rejects_invalid_optimizer_settings(problem, options):
    with pytest.raises(ValueError):
        reconstruct(*problem, **options)


def test_cli_reconstructs_saved_measurements(problem, tmp_path, capsys):
    projector, measured = problem
    np.save(tmp_path / "measured.npy", measured.numpy())
    np.save(tmp_path / "angles.npy", projector.angles.numpy())
    output = tmp_path / "reconstructed.npy"
    main(
        [
            str(tmp_path / "measured.npy"),
            "--angles",
            str(tmp_path / "angles.npy"),
            "--resolution",
            "8",
            "--source-distance",
            "32",
            "--det-distance",
            "16",
            "--det-spacing",
            "1.3",
            "--steps",
            "40",
            "--lr",
            "0.03",
            "--output",
            str(output),
        ]
    )
    result = torch.from_numpy(np.load(output))
    assert result.shape == (8, 8) and (result >= 0).all()
    assert (projector(result) - measured).square().mean() < measured.square().mean()
    assert "loss:" in capsys.readouterr().out
