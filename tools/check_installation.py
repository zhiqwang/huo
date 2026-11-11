"""Smoke-test a non-editable base/torch/jax installation in isolation.

Run with the selected environment's Python and -I, so imports use the installed
wheel rather than this checkout. CI runs this in three separate environments.
"""

import argparse
from contextlib import redirect_stdout, redirect_stderr
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("base", "torch", "jax"), required=True)
    args = parser.parse_args()
    for dependency in ("torch", "jax", "optax"):
        expected = dependency == args.backend or (args.backend == "jax" and dependency == "optax")
        assert (importlib.util.find_spec(dependency) is not None) == expected, dependency

    import huo
    from huo.cli import main as cli
    from huo.geometry import compute_ray_coordinates

    checkout = Path(__file__).resolve().parents[1]
    package_path = Path(huo.__file__).resolve()
    assert not package_path.is_relative_to(checkout / "huo"), package_path
    assert package_path.is_relative_to(Path(sys.prefix).resolve()), package_path
    assert not {"torch", "jax", "optax"}.intersection(sys.modules)
    geometry = huo.FanbeamGeometry(8, 32, 16, 12, 1.3)
    assert compute_ray_coordinates(geometry)[0].shape == (17, 12)
    angles = np.array([0, 0.5, 1, 1.5], np.float32)
    phantom = np.zeros((8, 8), np.float32)
    phantom[2:6, 3:5] = 1
    if args.backend == "torch":
        import torch
        from huo.torch import FanbeamProjector, reconstruct

        projector = FanbeamProjector(geometry, angles)
        measured = projector(torch.from_numpy(phantom))
    elif args.backend == "jax":
        import jax
        import jax.numpy as jnp
        from huo.jax import FanbeamProjector, reconstruct

        projector = FanbeamProjector.create(geometry, angles)
        measured = jax.jit(lambda state, x: state(x))(projector, jnp.asarray(phantom))
    else:
        measured = np.zeros((4, 12), np.float32)
        with redirect_stdout(io.StringIO()):
            try:
                cli(["--help"])
            except SystemExit as error:
                assert error.code == 0

    with tempfile.TemporaryDirectory(prefix="huo-install-") as directory:
        folder = Path(directory)
        np.save(folder / "measured.npy", np.asarray(measured))
        np.save(folder / "angles.npy", angles)
        command = [
            str(folder / "measured.npy"),
            "--angles",
            str(folder / "angles.npy"),
            "--resolution",
            "8",
            "--source-distance",
            "32",
            "--det-distance",
            "16",
            "--det-spacing",
            "1.3",
            "--steps",
            "10",
            "--lr",
            ".03",
            "-o",
            str(folder / "image.npy"),
        ]
        if args.backend == "base":
            for backend in ("torch", "jax"):
                errors = io.StringIO()
                with redirect_stderr(errors):
                    try:
                        cli([*command, "--backend", backend])
                    except SystemExit as error:
                        assert error.code == 2
                    else:
                        raise AssertionError("Missing backend did not produce a CLI error")
                assert f"--extra {backend}" in errors.getvalue()
        else:
            image = reconstruct(projector, measured, steps=10, lr=0.03)
            assert np.asarray(image).shape == (8, 8) and np.isfinite(np.asarray(image)).all()
            result = subprocess.run(
                [sys.executable, "-I", "-m", "huo.cli", *command, "--backend", args.backend],
                check=True,
                capture_output=True,
                text=True,
            )
            assert f"backend: {args.backend}" in result.stdout
            np.testing.assert_allclose(np.load(folder / "image.npy"), np.asarray(image), atol=2e-6, rtol=2e-6)
    forbidden = {"torch", "jax", "optax"} - ({"jax", "optax"} if args.backend == "jax" else {args.backend})
    assert not forbidden.intersection(sys.modules)
    checks = "CLI help and missing-extra errors" if args.backend == "base" else "reconstruction and CLI"
    print(f"{args.backend}: installed wheel, isolated imports, {checks} passed")


if __name__ == "__main__":
    main()
