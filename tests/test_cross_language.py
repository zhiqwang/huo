"""Compare browser WASM operators, derivatives and Adam with the Python backends.

Run ``npm --prefix packages run build`` first. Skip only when Node or built
JavaScript dependencies are unavailable. All backends use the same inputs.
"""

from dataclasses import asdict
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from huo import FanbeamGeometry
from huo.torch import FanbeamProjector, reconstruct

ROOT = Path(__file__).resolve().parents[1]
READY = (
    shutil.which("node")
    and (ROOT / "packages/dist/src/index.js").exists()
    and (ROOT / "packages/node_modules/@jax-js/optax").exists()
)


@pytest.mark.skipif(not READY, reason="Build the TypeScript package and install Node to check parity")
@pytest.mark.parametrize("resolution", [5, 6, 31, 32])
def test_same_geometry_derivatives_and_reconstruction(resolution):
    geometry = FanbeamGeometry(
        resolution, 4 * resolution, 2 * resolution, 19, 1.7, ray_sampling=3 if resolution % 2 else 2
    )
    projector = FanbeamProjector(geometry, [math.pi / 2, -0.37, 0, -0.37, 0.8])
    generator = torch.Generator().manual_seed(42)
    shape = (2, resolution, resolution)
    image = torch.randn(shape, generator=generator)
    direction = torch.randn(shape, generator=generator)
    measured = torch.randn(2, 5, 19, generator=generator)
    initial = torch.linspace(-0.1, 0.1, image.numel()).reshape(shape)
    objective = lambda x: 0.5 * (projector(x) - measured).square().mean()
    expected = {
        "forward": projector(image),
        "loss": objective(image),
        "gradient": torch.func.grad(objective)(image),
        "jvp": torch.func.jvp(projector, (image,), (direction,))[1],
        "hvp": torch.func.jvp(torch.func.grad(objective), (image,), (direction,))[1],
        "reconstructed": reconstruct(
            projector, measured, steps=20, lr=0.03, initial_image=initial, nonnegative=False
        ),
    }
    payload = dict(
        parameters={**asdict(geometry), "angles": projector.angles.tolist()},
        image=image.flatten().tolist(),
        direction=direction.flatten().tolist(),
        measured=measured.flatten().tolist(),
        initial=initial.flatten().tolist(),
        image_shape=list(shape),
        measured_shape=list(measured.shape),
    )
    process = subprocess.run(
        ["node", str(ROOT / "packages/tests/parity.mjs")],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    result = json.loads(process.stdout)
    for name, value in expected.items():
        actual = torch.tensor(result[name]).reshape(value.shape)
        error = (actual - value).norm() / value.norm().clamp(min=1e-12)
        assert error < 3e-5, f"N={resolution}, {name}: relative L2 error {error.item()}"

    if importlib.util.find_spec("jax") and importlib.util.find_spec("optax"):
        import jax
        import jax.numpy as jnp
        from huo.jax import FanbeamProjector as JaxProjector, reconstruct as jax_reconstruct

        p = JaxProjector.create(geometry, projector.angles.numpy())
        x, d, target = (jnp.asarray(value.numpy()) for value in (image, direction, measured))
        loss = lambda x: 0.5 * jnp.mean((p(x) - target) ** 2)
        references = {
            "forward": p(x),
            "loss": loss(x),
            "gradient": jax.grad(loss)(x),
            "jvp": jax.jvp(p, (x,), (d,))[1],
            "hvp": jax.jvp(jax.grad(loss), (x,), (d,))[1],
            "reconstructed": jax_reconstruct(
                p,
                target,
                steps=20,
                lr=0.03,
                initial_image=jnp.asarray(initial.numpy()),
                nonnegative=False,
            ),
        }
        for name, value in references.items():
            expected_array = np.asarray(value)
            actual = np.asarray(result[name]).reshape(expected_array.shape)
            error = np.linalg.norm(actual - expected_array) / max(np.linalg.norm(expected_array), 1e-12)
            assert error < 3e-5, f"JAX N={resolution}, {name}: relative L2 error {error}"
