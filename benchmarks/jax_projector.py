"""Experimental pure-JAX equivalent of Huo's fixed-geometry sampled operator.

This benchmark backend takes the same normalized zero-angle rays as PyTorch.
Geometry arrays are explicit runtime arguments, so JIT cannot precompute every
view's interpolation weights as closed-over constants. Only images are
differentiated. No hand-written transpose or VJP is needed.
"""

from functools import partial

import jax
import jax.numpy as jnp
import optax

from huo.jax._project import project as project_samples

IMPLEMENTATIONS = ("vmap", "scan", "custom_jvp")


def make_projector(ray_step=0.5, *, implementation="scan"):
    """Return project(image, ray_x, ray_y, angles): [..., N, N] -> [..., V, M]."""
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"implementation must be one of {IMPLEMENTATIONS}")

    native = partial(
        project_samples, ray_step=ray_step, method="vmap" if implementation == "vmap" else "scan"
    )

    if implementation != "custom_jvp":
        return native

    @jax.custom_jvp
    def project(image, ray_x, ray_y, angles):
        return native(image, ray_x, ray_y, angles)

    @project.defjvp
    def project_jvp(primals, tangents):
        image, ray_x, ray_y, angles = primals
        direction, _, _, _ = tangents
        # The linear JVP A dx is transposed automatically by JAX for reverse AD.
        return project(image, ray_x, ray_y, angles), native(direction, ray_x, ray_y, angles)

    return project


def projection_loss(project, image, measured, ray_x, ray_y, angles):
    """Half mean squared residual, including all leading batch dimensions."""
    return 0.5 * jnp.mean(jnp.square(project(image, ray_x, ray_y, angles) - measured))


def make_reconstructor(project, *, steps=100, lr=0.03, nonnegative=True):
    """Return a fixed-settings comparator using Huo's Adam objective and constraint.

    The returned function takes (measured, initial_image, ray_x, ray_y, angles).
    JIT the returned function to compile the whole fixed-step optimization loop.
    """
    optimizer = optax.adam(lr, b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0)

    def reconstruct(measured, initial_image, ray_x, ray_y, angles):
        measured, image = jax.tree.map(jax.lax.stop_gradient, (measured, initial_image))
        if nonnegative:
            image = jnp.maximum(image, 0)

        def loss(value):
            return projection_loss(project, value, measured, ray_x, ray_y, angles)

        def step(carry, _):
            value, state = carry
            gradient = jax.grad(loss)(value)
            updates, state = optimizer.update(gradient, state, value)
            value = optax.apply_updates(value, updates)
            if nonnegative:
                value = jnp.maximum(value, 0)
            return (value, state), None

        (image, _), _ = jax.lax.scan(step, (image, optimizer.init(image)), None, length=steps)
        return jax.lax.stop_gradient(image)

    return reconstruct
