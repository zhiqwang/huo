import { grad, jit, lax, numpy as np, tree } from "@jax-js/jax";
import { adam, applyUpdates, type OptState } from "@jax-js/optax";
import { FanbeamProjector, project, type ProjectorState } from "./projector.js";
import { check_array, check_finite, positive_integer } from "./validation.js";

export interface ReconstructionProgress {
  step: number;
  steps: number;
  /** Half mean squared residual evaluated at this snapshot, after the update. */
  loss: number;
  image: Float32Array;
  shape: number[];
}

export interface ReconstructionOptions {
  steps?: number;
  lr?: number;
  initial_image?: np.Array;
  nonnegative?: boolean;
  view_chunk_size?: number;
  report_every?: number;
  on_progress?: (progress: ReconstructionProgress) => void | Promise<void>;
  signal?: AbortSignal;
}

type Chunk = { state: ProjectorState; target: np.Array; weight: number };

/** Adam on the complete mean projection loss, with memory bounded by angle chunks.
 *
 * Borrows projector, measured, and initial_image; returns a newly owned array.
 * Native AD supplies each chunk's gradient. Weight and accumulate ALL chunks
 * before one Adam update, including an uneven last chunk. This async helper
 * is not differentiable; use project() in your own transformed loss instead.
 * Value checks, progress callbacks, cancellation and event-loop yields live here,
 * outside the pure compiled functions. Compiled resources are released on exit.
 */
export async function reconstruct(
  projector: FanbeamProjector,
  measured: np.Array,
  options: ReconstructionOptions = {},
): Promise<np.Array> {
  const {
    steps = 300,
    lr = 0.01,
    initial_image,
    nonnegative = true,
    view_chunk_size = 8,
    report_every = 10,
    on_progress,
    signal,
  } = options;
  for (const [name, value] of Object.entries({ steps, view_chunk_size, report_every }))
    positive_integer(value, name);
  if (!Number.isFinite(lr) || lr <= 0) throw new RangeError("lr must be positive and finite");
  if (typeof nonnegative !== "boolean") throw new TypeError("nonnegative must be a boolean");
  if (on_progress !== undefined && typeof on_progress !== "function")
    throw new TypeError("on_progress must be a function");
  signal?.throwIfAborted();
  const { resolution: n, det_count: m, ray_step } = projector.geometry;
  const views = projector.views;
  check_array(measured, [views, m], projector.state.angles, "measured");
  const image_shape = [...measured.shape.slice(0, -2), n, n];
  if (initial_image) {
    check_array(initial_image, image_shape, projector.state.angles, "initial_image");
    if (initial_image.ndim !== image_shape.length)
      throw new RangeError(`initial_image must have shape [${image_shape}]`);
  }
  const scanner = tree.ref(projector.state);
  const target = lax.stopGradient(measured.ref);
  let image: np.Array | undefined = initial_image ? lax.stopGradient(initial_image.ref) : undefined;
  let optimizer_state: OptState | undefined;
  let gradient: np.Array | undefined;
  const chunks: Chunk[] = [];
  const objective = (x: np.Array, state: ProjectorState, y: np.Array) =>
    np
      .square(project(x, state, ray_step).sub(y))
      .mean()
      .mul(0.5);
  const evaluate = jit(objective);
  const differentiate = jit(grad(objective));
  const optimizer = adam(lr, { b1: 0.9, b2: 0.999, eps: 1e-8, epsRoot: 0 });
  const update = jit((x: np.Array, g: np.Array, state: OptState): [np.Array, OptState] => {
    const [updates, next_state] = optimizer.update(g, state);
    const next = applyUpdates(x, updates);
    return [nonnegative ? np.maximum(next, 0) : next, next_state];
  });

  const report = async (step: number) => {
    await check_finite(image!, "reconstruction");
    if (!on_progress) return;
    let loss = 0;
    for (const chunk of chunks) {
      signal?.throwIfAborted();
      loss += (await evaluate(image!.ref, tree.ref(chunk.state), chunk.target.ref).data())[0] * chunk.weight;
    }
    if (!Number.isFinite(loss)) throw new RangeError("projection loss must be finite");
    const snapshot = (await image!.ref.data()).slice() as Float32Array;
    await on_progress({ step, steps, loss, image: snapshot, shape: [...image_shape] });
  };

  try {
    await check_finite(target, "measured");
    if (image) await check_finite(image, "initial_image");
    else image = np.zeros(image_shape, { dtype: measured.dtype, device: measured.device });
    if (nonnegative) image = np.maximum(image, 0);
    for (let start = 0; start < views; start += view_chunk_size) {
      const end = Math.min(start + view_chunk_size, views);
      chunks.push({
        state: {
          angles: scanner.angles.ref.slice([start, end]),
          ray_x: scanner.ray_x.ref,
          ray_y: scanner.ray_y.ref,
        },
        target: target.ref.slice(...target.shape.slice(0, -2).map(() => [] as []), [start, end], []),
        weight: (end - start) / views,
      });
    }
    optimizer_state = optimizer.init(image.ref);
    await report(0);
    for (let step = 1; step <= steps; step++) {
      // A macrotask lets the browser paint and deliver cancellation events.
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
      signal?.throwIfAborted();
      gradient = np.zerosLike(image.ref);
      for (const chunk of chunks) {
        signal?.throwIfAborted();
        const part = differentiate(image.ref, tree.ref(chunk.state), chunk.target.ref).mul(chunk.weight);
        gradient = gradient.add(part);
        // Realize each accumulated gradient before building the next chunk.
        // blockUntilReady borrows its receiver and does not copy it to the host.
        await gradient.blockUntilReady();
      }
      [image, optimizer_state] = update(image, gradient, optimizer_state);
      gradient = undefined;
      if (step % report_every === 0 || step === steps) await report(step);
    }
    signal?.throwIfAborted();
    const result = image;
    image = undefined;
    return result;
  } finally {
    tree.dispose([image, gradient, optimizer_state]);
    for (const chunk of chunks) {
      tree.dispose(chunk.state);
      chunk.target.dispose();
    }
    tree.dispose(scanner);
    target.dispose();
    evaluate.dispose();
    differentiate.dispose();
    update.dispose();
  }
}
