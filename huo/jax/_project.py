"""Pure JAX sampling; native AD supplies the transpose and higher derivatives."""

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates


def project(image, ray_x, ray_y, angles, *, ray_step, method):
    ray_x, ray_y, angles = jax.tree.map(jax.lax.stop_gradient, (ray_x, ray_y, angles))
    height, width = image.shape[-2:]
    images = image.reshape((-1, height, width))

    def one_view(angle):
        cosine, sine = jnp.cos(angle), jnp.sin(angle)
        x = cosine * ray_x + sine * ray_y
        y = -sine * ray_x + cosine * ray_y
        coordinates = ((y + 1) * ((height - 1) / 2), (x + 1) * ((width - 1) / 2))
        values = jax.vmap(
            lambda sample: map_coordinates(sample, coordinates, order=1, mode="constant", cval=0)
        )(images)
        return values.sum(axis=1) * ray_step

    if method == "vmap":
        rows = jax.vmap(one_view)(angles)
    else:
        # Keep per-view interpolation weights out of the retained AD state.
        view = jax.checkpoint(one_view, prevent_cse=False)
        _, rows = jax.lax.scan(lambda carry, angle: (carry, view(angle)), None, angles)
    return jnp.moveaxis(rows, 0, 1).reshape((*image.shape[:-2], angles.size, ray_x.shape[1]))
