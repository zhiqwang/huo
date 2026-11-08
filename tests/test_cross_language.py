"""Compare full Python and TypeScript operators for the same scanner and data.

Run ``npm --prefix packages run build`` first. This integration check is skipped
when Node or the built JavaScript dependencies are unavailable.
"""

from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import subprocess
from unittest.mock import patch

import pytest
import torch

from huo import RadonFanbeam

ROOT = Path(__file__).resolve().parents[1]
READY = (
    shutil.which("node")
    and (ROOT / "packages/src/art.js").exists()
    and (ROOT / "packages/node_modules/@jax-js/jax").exists()
)


@pytest.mark.skipif(not READY, reason="Build the TypeScript package and install Node to check parity")
@pytest.mark.parametrize("resolution", [5, 6, 31, 32])
def test_same_geometry_and_signed_inputs(resolution):
    radon = RadonFanbeam(resolution, [0, math.pi / 2, 0.37, -0.41], 4 * resolution, 2 * resolution, 19, 1.7)
    generator = torch.Generator().manual_seed(42)
    image = torch.randn(resolution, resolution, generator=generator)
    sinogram = torch.randn(4, 19, generator=generator)
    expected = {"forward": radon.forward(image), "backprojection": radon.backprojection(sinogram)}
    with patch("torch.randperm", side_effect=lambda count: torch.arange(count)):
        expected["art"] = radon.art(sinogram)
    payload = dict(
        parameters={**asdict(radon.geometry), "angles": radon.angles.tolist()},
        image=image.flatten().tolist(),
        sinogram=sinogram.flatten().tolist(),
    )
    process = subprocess.run(
        ["node", str(ROOT / "packages/tests/parity.mjs")],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    result = json.loads(process.stdout)
    for method, reference in expected.items():
        actual = torch.tensor(result[method]).reshape(reference.shape)
        relative_error = (actual - reference).norm() / reference.norm().clamp(min=1e-12)
        assert relative_error < 2e-5, f"{method}, N={resolution}: relative L2 error {relative_error.item()}"
