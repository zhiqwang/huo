"""Public JAX contracts, PyTree transforms, compilation, and reconstruction."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("optax")
import jax.numpy as jnp
from jax.experimental import checkify
import optax

from huo import FanbeamGeometry
from huo.geometry import compute_ray_coordinates
from huo.jax import FanbeamProjector, reconstruct
from huo.cli import main


@pytest.fixture
def projector():
    return FanbeamProjector.create(FanbeamGeometry(8, 32, 16, 12, 1.3), [0, 0.5, 1, 1.5])


@pytest.fixture
def phantom():
    return jnp.zeros((8, 8)).at[2:6, 3:5].set(1)


def test_state_owns_angles_and_is_an_immutable_pytree():
    geometry = FanbeamGeometry(8, 32, 16, 12, 1.3)
    angles = np.array([0, 0.5], dtype=np.float32)
    projector = FanbeamProjector.create(geometry, angles)
    angles[:] = 99
    np.testing.assert_array_equal(projector.angles, [0, 0.5])
    for actual, expected in zip((projector.ray_x, projector.ray_y), compute_ray_coordinates(geometry)):
        np.testing.assert_array_equal(actual, expected)
    assert projector.views == 2 and projector.det_count == 12 and projector.dtype == jnp.float32
    leaves, definition = jax.tree.flatten(projector)
    assert len(leaves) == 3
    restored = jax.tree.unflatten(definition, leaves)
    image = jnp.arange(64, dtype=jnp.float32).reshape(8, 8)
    np.testing.assert_array_equal(restored(image), projector(image))
    with pytest.raises(FrozenInstanceError):
        projector.method = "vmap"


def test_new_geometry_values_reuse_the_same_compiled_function(phantom):
    first = FanbeamProjector.create(FanbeamGeometry(8, 32, 16, 12, 1.3), [0, 0.5])
    second = FanbeamProjector.create(FanbeamGeometry(8, 40, 16, 12, 1.3), [0.2, 0.8])
    traces = []

    @jax.jit
    def forward(state, image):
        traces.append(None)
        return state(image)

    a = forward(first, phantom)
    b = forward(second, phantom)
    assert len(traces) == 1
    assert not np.allclose(a, b)
    np.testing.assert_allclose(b, second(phantom), atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("method", ["scan", "vmap"])
def test_native_and_mapped_batches_gradients_and_jvp(method):
    with jax.enable_x64():
        p = FanbeamProjector.create(
            FanbeamGeometry(3, 12, 6, 5, 1.3), [0, -0.37], dtype=jnp.float64, method=method
        )
        basis = jnp.eye(9).reshape(9, 3, 3)
        matrix = np.asarray(p(basis)).reshape(9, -1).T
        image = jnp.arange(9, dtype=jnp.float64).reshape(3, 3) / 9
        direction = jnp.ones_like(image)
        loss = lambda x, state: 0.5 * jnp.sum(state(x) ** 2)
        gradient = jax.jit(jax.grad(loss))(image, p)
        np.testing.assert_allclose(
            gradient.ravel(), matrix.T @ matrix @ np.asarray(image).ravel(), atol=1e-11, rtol=1e-11
        )
        _, tangent = jax.jvp(lambda x: p(x), (image,), (direction,))
        np.testing.assert_allclose(tangent.ravel(), matrix @ np.ones(9), atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(
            jax.hessian(loss)(image, p).reshape(9, 9), matrix.T @ matrix, atol=1e-11, rtol=1e-11
        )
        images = jnp.stack((image, direction)).reshape(1, 2, 3, 3)
        np.testing.assert_allclose(
            jax.jit(lambda state, x: state(x))(p, images),
            jax.vmap(jax.vmap(p))(images),
            atol=1e-11,
            rtol=1e-11,
        )
        geometry_gradient = jax.grad(lambda state: jnp.sum(state(image)))(p)
        assert all(np.count_nonzero(leaf) == 0 for leaf in jax.tree.leaves(geometry_gradient))


def test_pytree_geometry_batch_device_and_dtype(projector, phantom):
    second = FanbeamProjector.create(FanbeamGeometry(8, 40, 16, 12, 1.3), [0, 0.5, 1, 1.5])
    states = jax.tree.map(lambda a, b: jnp.stack((a, b)), projector, second)
    mapped = jax.jit(jax.vmap(lambda state, x: state(x), in_axes=(0, None)))(states, phantom)
    np.testing.assert_allclose(mapped, jnp.stack((projector(phantom), second(phantom))), atol=2e-6, rtol=2e-6)
    placed = jax.device_put(projector, jax.devices("cpu")[0])
    assert all(leaf.device == jax.devices("cpu")[0] for leaf in jax.tree.leaves(placed))
    with jax.enable_x64():
        converted = jax.tree.map(lambda array: array.astype(jnp.float64), placed)
        assert converted(phantom.astype(jnp.float64)).dtype == jnp.float64


@pytest.mark.parametrize("angles", [[], [[0]], [float("nan")], [float("inf")]])
def test_rejects_invalid_angles(angles):
    with pytest.raises(ValueError, match="angles"):
        FanbeamProjector.create(FanbeamGeometry(8, 32, 16, 12, 1.3), angles)


def test_dtype_and_method_guards(projector):
    geometry = FanbeamGeometry(8, 32, 16, 12, 1.3)
    with jax.enable_x64(False), pytest.raises(ValueError, match="jax_enable_x64"):
        FanbeamProjector.create(geometry, [0], dtype=jnp.float64)
    for dtype in (jnp.int32, jnp.float16, jnp.complex64):
        with pytest.raises(ValueError, match="dtype"):
            FanbeamProjector.create(geometry, [0], dtype=dtype)
    with pytest.raises(ValueError, match="method"):
        FanbeamProjector.create(geometry, [0], method="automatic")
    with pytest.raises(ValueError, match="dtype"):
        projector(np.ones((8, 8), dtype=np.float64))


@pytest.mark.parametrize(
    "image", [np.zeros((8, 7), np.float32), np.zeros((0, 8, 8), np.float32), np.zeros((8, 8), np.int32)]
)
def test_shape_dtype_checks_survive_jit(projector, image):
    for forward in (lambda x: projector(x), jax.jit(lambda x: projector(x))):
        with pytest.raises(ValueError):
            forward(image)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_eager_and_checkified_nonfinite_inputs(projector, phantom, bad):
    image = phantom.at[0, 0].set(bad)
    with pytest.raises(ValueError, match="finite"):
        projector(image)
    checked = jax.jit(checkify.checkify(lambda state, x: state(x)))
    error, _ = checked(projector, image)
    assert "image must contain only finite" in error.get()
    mapped = jax.jit(checkify.checkify(jax.vmap(lambda x: projector(x))))
    error, _ = mapped(jnp.stack((phantom, image)))
    assert "finite" in error.get()
    # Captured constants also work under transformations: checking only the
    # input object's Python type would incorrectly try bool(traced_predicate).
    np.testing.assert_allclose(
        jax.jit(lambda: projector(phantom))(), projector(phantom), atol=2e-6, rtol=2e-6
    )


@pytest.mark.parametrize("method", ["scan", "vmap"])
def test_compiled_adam_matches_explicit_loop_and_stops_outer_gradients(method):
    with jax.enable_x64():
        p = FanbeamProjector.create(
            FanbeamGeometry(8, 32, 16, 12, 1.3), [0, 0.5, 1, 1.5], dtype=jnp.float64, method=method
        )
        phantom = jnp.zeros((2, 8, 8), dtype=jnp.float64).at[:, 2:6, 3:5].set(1)
        measured = p(phantom)
        initial = jnp.full_like(phantom, -0.1)
        expected = jnp.maximum(initial, 0)
        optimizer = optax.adam(0.03)
        state = optimizer.init(expected)
        for _ in range(12):
            gradient = jax.grad(lambda x: 0.5 * jnp.mean((p(x) - measured) ** 2))(expected)
            updates, state = optimizer.update(gradient, state, expected)
            expected = jnp.maximum(optax.apply_updates(expected, updates), 0)
        actual = reconstruct(p, measured, steps=12, lr=0.03, initial_image=initial)
        np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-11)
        np.testing.assert_array_equal(initial, -0.1)
        assert float(jnp.mean((p(actual) - measured) ** 2)) < float(jnp.mean(measured**2))
        outer = jax.jit(lambda state, y: reconstruct(state, y, steps=12, lr=0.03, initial_image=initial))
        np.testing.assert_allclose(outer(p, measured), actual, atol=1e-11, rtol=1e-11)
        derivative = jax.grad(lambda y: reconstruct(p, y, steps=2).sum())(measured)
        np.testing.assert_array_equal(derivative, 0)


def test_signed_reconstruction_and_parameter_gradients(projector, phantom):
    measured = -projector(phantom)
    result = reconstruct(projector, measured, steps=30, lr=0.03, nonnegative=False)
    assert float(result.min()) < 0
    assert float(jnp.mean((projector(result) - measured) ** 2)) < float(jnp.mean(measured**2))
    weights = jnp.ones((64, 4))
    gradient = jax.grad(lambda w: jnp.square(projector((w @ jnp.ones(4)).reshape(8, 8))).mean())(weights)
    assert bool(jnp.isfinite(gradient).all()) and float(jnp.linalg.norm(gradient)) > 0


@pytest.mark.parametrize(
    "options",
    [
        {"steps": 0},
        {"steps": True},
        {"steps": 1.2},
        {"lr": 0},
        {"lr": float("nan")},
        {"nonnegative": None},
        {"initial_image": np.zeros((2, 8, 8), np.float32)},
    ],
)
def test_rejects_invalid_solver_settings(projector, phantom, options):
    with pytest.raises(ValueError):
        reconstruct(projector, projector(phantom), **options)


def test_nonfinite_measurements_and_compiled_solver_checks(projector, phantom):
    measured = projector(phantom).at[0, 0].set(jnp.nan)
    with pytest.raises(ValueError, match="measured"):
        reconstruct(projector, measured, steps=2)
    checked = jax.jit(checkify.checkify(lambda p, y: reconstruct(p, y, steps=2)))
    error, _ = checked(projector, measured)
    assert "measured must contain only finite" in error.get()


@pytest.mark.parametrize("method", ["scan", "vmap"])
def test_cli_jax_reconstruction(projector, phantom, method, tmp_path, capsys):
    measured = projector(phantom)
    np.save(tmp_path / "measured.npy", measured)
    np.save(tmp_path / "angles.npy", projector.angles)
    output = tmp_path / "result.npy"
    main(
        [
            str(tmp_path / "measured.npy"),
            "--backend",
            "jax",
            "--method",
            method,
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
            "30",
            "--lr",
            ".03",
            "-o",
            str(output),
        ]
    )
    actual = jnp.asarray(np.load(output))
    assert float(jnp.mean((projector(actual) - measured) ** 2)) < float(jnp.mean(measured**2))
    assert "backend: jax" in capsys.readouterr().out
