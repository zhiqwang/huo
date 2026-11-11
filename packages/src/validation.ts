import { numpy as np } from "@jax-js/jax";

/** Metadata-only validation: safe inside grad, JVP, vmap, and JIT. Borrows inputs. */
export function check_array(value: np.Array, shape: readonly number[], reference: np.Array, name: string) {
  if (value.dtype !== reference.dtype || value.dtype !== np.float32) {
    throw new TypeError(`${name} must match the projector float32 dtype`);
  }
  if (
    value.ndim < shape.length ||
    value.size === 0 ||
    shape.some((size, i) => value.shape[value.ndim - shape.length + i] !== size)
  ) {
    throw new RangeError(`${name} must have nonempty trailing shape [${shape}], got [${value.shape}]`);
  }
}

/** Value validation belongs outside transformed functions. Borrows the array. */
export async function check_finite(value: np.Array, name: string): Promise<void> {
  if (!(await np.isfinite(value.ref).all().data())[0]) {
    throw new RangeError(`${name} must contain only finite values`);
  }
}

export function positive_integer(value: number, name: string) {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${name} must be a positive integer`);
}
