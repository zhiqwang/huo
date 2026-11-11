"""JAX projection and reconstruction; install the jax extra to use them."""

try:
    from huo.jax.projector import FanbeamProjector
    from huo.jax.reconstruction import reconstruct
except ModuleNotFoundError as error:
    if error.name in {"jax", "jaxlib", "optax"}:
        raise ModuleNotFoundError(
            "The JAX backend requires the 'jax' extra: uv sync --extra jax", name=error.name
        ) from error
    raise

__all__ = ["FanbeamProjector", "reconstruct"]
