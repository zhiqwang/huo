import { lax, numpy as np, tree, type Device } from "@jax-js/jax";
import { FanbeamGeometry, compute_ray_coordinates } from "./geometry.js";
import { check_array } from "./validation.js";

/** Dynamic JsTree leaves. Geometry/angle gradients are deliberately stopped. */
export type ProjectorState = { angles: np.Array; ray_x: np.Array; ray_y: np.Array };

/** Bilinear sampling with partial zero padding. Consumes all three arrays. */
export function sample_bilinear(image: np.Array, x: np.Array, y: np.Array): np.Array {
  const n = image.shape.at(-1)!;
  const leading = image.shape.slice(0, -2);
  const grid_shape = x.shape;
  const images = image.reshape([-1, n * n]);
  const col = x.add(1).mul((n - 1) / 2),
    row = y.add(1).mul((n - 1) / 2);
  const col0 = np.floor(col.ref),
    row0 = np.floor(row.ref);
  const wx = col.sub(col0.ref),
    wy = row.sub(row0.ref);
  let result: np.Array | undefined;
  for (let r = 0; r < 2; r++)
    for (let c = 0; c < 2; c++) {
      const ri = row0.ref.add(r),
        ci = col0.ref.add(c);
      const valid = np.logicalAnd(
        np.logicalAnd(ri.ref.greaterEqual(0), ri.ref.less(n)),
        np.logicalAnd(ci.ref.greaterEqual(0), ci.ref.less(n)),
      );
      const indices = np.clip(ri.mul(n).add(ci), 0, n * n - 1).astype(np.int32);
      const weight = (r ? wy.ref : np.subtract(1, wy.ref)).mul(c ? wx.ref : np.subtract(1, wx.ref));
      const contribution = np.take(images.ref, indices, 1).mul(np.where(valid, weight, 0));
      result = result ? result.add(contribution) : contribution;
    }
  tree.dispose([images, col0, row0, wx, wy]);
  return result!.reshape([...leading, ...grid_shape]);
}

/** Pure forward map [..., N, N] -> [..., V, M]. Consumes image and state.
 *
 * Pass state explicitly to jit/grad, using tree.ref(state) to retain ownership.
 * This function does no host reads. Call check_finite outside transformations
 * when runtime NaN/Inf validation is needed.
 */
export function project(image: np.Array, state: ProjectorState, ray_step = 0.5): np.Array {
  const window = (state.ray_x.shape[0] - 1) * ray_step;
  const n = Math.round(window);
  if (!Number.isInteger(n) || n < 2) throw new RangeError("image side length must be >= 2");
  if (Math.abs(window - n) > 1e-10 * n)
    throw new RangeError("ray grid and ray_step must span an integer image side");
  check_array(image, [n, n], state.angles, "image");
  if (
    state.angles.ndim !== 1 ||
    !state.angles.size ||
    state.ray_x.ndim !== 2 ||
    state.ray_x.size === 0 ||
    state.ray_x.shape.some((size, i) => state.ray_y.shape[i] !== size) ||
    state.ray_y.ndim !== 2 ||
    state.ray_x.dtype !== image.dtype ||
    state.ray_y.dtype !== image.dtype
  ) {
    throw new RangeError("state must have matching rays [samples, detectors] and nonempty 1-D angles");
  }
  if (!Number.isFinite(ray_step) || ray_step <= 0)
    throw new RangeError("ray_step must be positive and finite");
  const shape = [1, ...state.ray_x.shape];
  const angles = lax.stopGradient(state.angles);
  const ray_x = lax.stopGradient(state.ray_x).reshape(shape);
  const ray_y = lax.stopGradient(state.ray_y).reshape(shape);
  const cosine = np.cos(angles.ref).reshape([-1, 1, 1]);
  const sine = np.sin(angles).reshape([-1, 1, 1]);
  const x = cosine.ref.mul(ray_x.ref).add(sine.ref.mul(ray_y.ref));
  const y = cosine.mul(ray_y).sub(sine.mul(ray_x));
  return sample_bilinear(image, x, y).sum(-2).mul(ray_step);
}

/** Owned fixed scanner state. Construct outside transforms, dispose after use. */
export class FanbeamProjector {
  readonly geometry: FanbeamGeometry;
  readonly #state: ProjectorState;
  #disposed = false;

  constructor(
    geometry: FanbeamGeometry,
    angles: ArrayLike<number>,
    options: {
      dtype?: np.DType;
      device?: Device;
    } = {},
  ) {
    if (!(geometry instanceof FanbeamGeometry)) throw new TypeError("geometry must be a FanbeamGeometry");
    const dtype = options.dtype ?? np.float32;
    if (dtype !== np.float32) throw new TypeError("the jax-js backend supports float32");
    const input = Array.from(angles);
    if (!input.length || !input.every((value) => typeof value === "number" && Number.isFinite(value))) {
      throw new RangeError("angles must be a nonempty finite 1-D sequence in radians");
    }
    const values = new Float32Array(input);
    if (!values.length || !values.every(Number.isFinite)) {
      throw new RangeError("angles must be a nonempty finite 1-D sequence in radians");
    }
    const { ray_x, ray_y } = compute_ray_coordinates(geometry);
    const shape = [geometry.ray_sample_count, geometry.det_count];
    const allocation = { dtype, device: options.device };
    this.geometry = geometry;
    this.#state = Object.freeze({
      angles: np.array(values, allocation),
      ray_x: np.array(ray_x, allocation).reshape(shape),
      ray_y: np.array(ray_y, allocation).reshape(shape),
    });
    Object.freeze(this);
  }

  /** Borrowed state; use tree.ref(projector.state) before passing to a transform. */
  get state(): ProjectorState {
    if (this.#disposed) throw new Error("projector has been disposed");
    return this.#state;
  }
  get views(): number {
    return this.state.angles.size;
  }
  get dtype(): np.DType {
    return this.state.angles.dtype;
  }

  /** Differentiable, synchronous projection. Consumes image, borrows scanner state. */
  forward(image: np.Array): np.Array {
    check_array(image, [this.geometry.resolution, this.geometry.resolution], this.state.angles, "image");
    return project(image, tree.ref(this.state), this.geometry.ray_step);
  }

  dispose(): void {
    if (!this.#disposed) {
      tree.dispose(this.#state);
      this.#disposed = true;
    }
  }
}
