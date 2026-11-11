"""Shape/dtype checks and optional compiled value checks for JAX arrays."""

import jax
import jax.numpy as jnp
from jax.experimental import checkify
import numpy as np


def check_dtype(dtype):
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("dtype must be float32 or float64")
    if dtype == np.dtype("float64") and not jax.config.x64_enabled:
        raise ValueError("float64 requires jax.config.update('jax_enable_x64', True)")
    return dtype


def check_finite(value, name):
    finite = jnp.isfinite(value).all()
    if isinstance(finite, jax.core.Tracer):
        # JIT cannot raise a Python exception on a traced value. Users can
        # functionalize this check with checkify.checkify before jitting.
        checkify.debug_check(finite, f"{name} must contain only finite values")
    elif not bool(finite):
        raise ValueError(f"{name} must contain only finite values")


def check_array(value, shape, dtype, name):
    try:
        value_dtype = np.dtype(value.dtype)
    except (AttributeError, TypeError) as error:
        raise ValueError(f"{name} must be a JAX or NumPy floating array") from error
    if value_dtype != check_dtype(dtype):
        raise ValueError(f"{name} must match the projector dtype")
    if value.ndim < 2 or tuple(value.shape[-2:]) != shape or value.size == 0:
        raise ValueError(f"{name} must have nonempty shape [..., {shape[0]}, {shape[1]}]")
    value = jnp.asarray(value)
    check_finite(value, name)
    return value
