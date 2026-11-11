"""Compare Python and browser forward projection in the same physical geometry.

Run ``npm --prefix packages run build`` first. This integration check is skipped
when Node or the built JavaScript dependencies are unavailable.
"""

from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import subprocess

import pytest
import torch

from huo import FanbeamGeometry, FanbeamProjector

ROOT = Path(__file__).resolve().parents[1]
READY = (
    shutil.which("node")
    and (ROOT / "packages/src/art.js").exists()
    and (ROOT / "packages/node_modules/@jax-js/jax").exists()
)


@pytest.mark.skipif(not READY, reason="Build the TypeScript package and install Node to check parity")
@pytest.mark.parametrize("resolution", [5, 6, 31, 32])
def test_same_geometry_and_signed_inputs(resolution):
    geometry = FanbeamGeometry(resolution, 4 * resolution, 2 * resolution, 19, 1.7)
    projector = FanbeamProjector(geometry, [0, math.pi / 2, 0.37, -0.41])
    generator = torch.Generator().manual_seed(42)
    image = torch.randn(resolution, resolution, generator=generator)
    sinogram = torch.randn(4, 19, generator=generator)
    expected = projector(image)
    payload = dict(
        parameters={**asdict(geometry), "angles": projector.angles.tolist()},
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
    actual = torch.tensor(result["forward"]).reshape(expected.shape)
    relative_error = (actual - expected).norm() / expected.norm().clamp(min=1e-12)
    assert relative_error < 2e-5, f"N={resolution}: relative L2 error {relative_error.item()}"
