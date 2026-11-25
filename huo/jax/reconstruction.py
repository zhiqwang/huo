"""Compile complete Adam reconstruction loops with native JAX gradients."""

from functools import partial

import jax
import jax.numpy as jnp
import optax

from huo._settings import validate_optimization
from huo.jax._validation import check_array, check_finite
from huo.jax.projector import FanbeamProjector


def _mean_absolute_difference(image, axis):
    difference = jnp.diff(image, axis=axis)
    # Match PyTorch's zero subgradient for equal neighbouring pixels.
    return jnp.mean(jnp.where(difference == 0, 0, jnp.abs(difference)))


@partial(jax.jit, static_argnames=("steps", "lr", "nonnegative", "tv_weight"))
def _solve(projector, measured, initial, *, steps, lr, nonnegative, tv_weight):
    target, image = jax.tree.map(jax.lax.stop_gradient, (measured, initial))
    if nonnegative:
        image = jnp.maximum(image, 0)
    optimizer = optax.adam(lr, b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0)

    def loss(value):
        residual = 0.5 * jnp.mean(jnp.square(projector(value) - target))
        if tv_weight:
            variation = _mean_absolute_difference(value, -1) + _mean_absolute_difference(value, -2)
            residual = residual + tv_weight * variation
        return residual

    def step(carry, _):
        value, state = carry
        updates, state = optimizer.update(jax.grad(loss)(value), state, value)
        value = optax.apply_updates(value, updates)
        if nonnegative:
            value = jnp.maximum(value, 0)
        return (value, state), None

    (image, _), _ = jax.lax.scan(step, (image, optimizer.init(image)), None, length=steps)
    return jax.lax.stop_gradient(image)


def reconstruct(
    projector: FanbeamProjector,
    measured,
    *,
    steps: int = 300,
    lr: float = 1e-2,
    initial_image=None,
    nonnegative: bool = True,
    tv_weight: float = 0.0,
):
    """Fit fixed measurements with Adam and optional total variation.

    Measurements have shape [..., V, M]; the result is [..., N, N]. Inputs
    remain unchanged and gradients through the optimizer trajectory are stopped.
    The full loop is JIT compiled and cached. If wrapping this function in JIT,
    steps, lr, nonnegative and tv_weight must be static arguments. Float32/64
    and arbitrary leading batch dimensions are supported.

    The objective is half the mean squared projection residual plus
    ``tv_weight`` times the sum of mean absolute differences along both image axes.
    Eager calls reject nonfinite inputs/results. Inside transformations these
    value checks are enabled by checkify.checkify, while shape/dtype checks
    remain active during tracing.
    """
    if not isinstance(projector, FanbeamProjector):
        raise TypeError("projector must be a huo.jax.FanbeamProjector")
    validate_optimization(steps, lr, nonnegative, tv_weight)
    target = check_array(measured, (projector.views, projector.det_count), projector.dtype, "measured")
    shape = (*target.shape[:-2], projector.resolution, projector.resolution)
    if initial_image is None:
        initial = jnp.zeros_like(target, shape=shape)
    else:
        initial = check_array(initial_image, shape[-2:], projector.dtype, "initial_image")
        if initial.shape != shape:
            raise ValueError(f"initial_image must have shape {shape}")
    image = _solve(
        projector,
        target,
        initial,
        steps=int(steps),
        lr=float(lr),
        nonnegative=nonnegative,
        tv_weight=float(tv_weight),
    )
    check_finite(image, "reconstruction")
    return image
