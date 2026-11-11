# jax-js Guide

The TypeScript package uses the same forward model and half mean squared
projection loss as the Python backends. Native jax-js operations differentiate
the sampled image interpolation. Image reconstruction uses Adam, with an
optional nonnegative constraint. There is no separate backprojection or ART API.

## Build and run

Use Node.js 22 or newer for local WASM execution:

```bash
npm --prefix packages ci
npm --prefix packages run build
npm --prefix packages test
```

The public exports are in `packages/src/index.ts`. The examples below run in
sequence from `packages/`, using Node's ES module mode or an ES module file.
The [browser demo](demo/index.md) supplies an import map for browser use.

```javascript
import { init, jit, numpy as np, tree, valueAndGrad, jvp } from "@jax-js/jax";
import { FanbeamGeometry, FanbeamProjector, project, reconstruct } from "./dist/src/index.js";

await init("wasm");
const n = 32;
const geometry = new FanbeamGeometry({
  resolution: n, source_distance: 128, det_distance: 64,
  det_count: 40, det_spacing: 1.3,
});
const angles = Array.from({ length: 24 }, (_, i) => i * Math.PI / 12);
const projector = new FanbeamProjector(geometry, angles);
const pixels = Float32Array.from({ length: n * n }, (_, i) =>
  Math.floor(i / n) >= 10 && Math.floor(i / n) < 22 && i % n >= 12 && i % n < 20 ? 1 : 0);
const phantom = np.array(pixels).reshape([n, n]);
const forward = jit((image, state) => project(image, state, geometry.ray_step));
const measured = forward(phantom.ref, tree.ref(projector.state));
console.log(measured.shape); // [24, 40]
```

`FanbeamGeometry` uses the shared [pixel/radian conventions](geometry.md).
`new FanbeamProjector(geometry, angles, { device? })` owns its scanner arrays.
This backend supports **float32** on WASM and WebGPU; its `dtype` option accepts
only `np.float32`. Python's float64 support is separate. Scanner geometry and
angles are fixed when differentiating.

## Forward model and native gradients

`projector.forward(image)` is synchronous and differentiable. Images have shape
`[..., N, N]` and sinograms have shape `[..., V, M]`, including arbitrary leading
batch dimensions. Bilinear sampling preserves partial weights at image edges
and assigns zero to out-of-range neighbours.

The pure function `project(image, state, ray_step=0.5)` provides the same model
with explicit scanner inputs. `ProjectorState` is a plain `JsTree` containing
`angles`, `ray_x`, and `ray_y`. Image size is recovered from ray-window length;
pass `geometry.ray_step` when using a sampling multiplier other than two.
The caller retains physical geometry and static settings outside the tree.
`ray_step` is static under JIT: capture it as above or use `staticArgnums: [2]`.
Matching shapes and dtypes can reuse one JIT function across scanner values.

```javascript
const objective = (image, state, target) =>
  np.square(project(image, state, geometry.ray_step).sub(target)).mean().mul(0.5);
const lossAndGrad = jit(valueAndGrad(objective));
const initial = np.zeros([n, n]);
const [loss, gradient] = lossAndGrad(initial.ref, tree.ref(projector.state), measured.ref);
console.log("initial loss:", (await loss.data())[0]);
gradient.dispose();

const [prediction, tangent] = jvp(
  image => projector.forward(image), [initial.ref], [np.onesLike(initial.ref)],
);
prediction.dispose();
tangent.dispose();
```

JVP, VJP, gradients, mapped batches and forward-over-reverse Hessian-vector
products compose with the projector. Neural-network parameters can be the
inputs to a loss that constructs an image before projection. No custom JVP or
hand-written transpose is used.

The pinned jax-js 0.1.27 helper APIs differ from Python: `jacfwd` requires vector
inputs and returns input-major axes, while `jacrev` does not handle this
rectangular image-to-sinogram map correctly. For explicit Jacobians, map JVPs
over image basis vectors or VJPs over sinogram basis vectors. The regression
suite checks those compositions and HVPs against the sampled matrix.

## Array ownership

jax-js uses explicit reference counting. Pure functions consume one reference
to every array argument; use `.ref` or `tree.ref` to retain an input for later
use. `projector.forward` consumes its image and borrows the projector's state.
`projector.state` itself is borrowed; pass `tree.ref(projector.state)` into
`project`, `jit`, or another consuming function.

Reading with `.data()` also consumes a reference. Use `await image.ref.data()`
when keeping the array. Dispose unused results and compiled functions.
`blockUntilReady()` borrows its receiver and synchronizes execution without
copying its contents to JavaScript. See the upstream
[ownership rules](https://github.com/ekzhang/jax-js#reference-counting).

The asynchronous `reconstruct` helper **borrows** its projector, measurements
and optional initial image, and returns a newly owned image. Inputs remain
unchanged. It releases temporary arrays and JIT resources on success,
cancellation, and progress-callback failure. `projector.dispose()` releases
scanner-owned references and may be called more than once.

## Reconstruction and browser progress

```javascript
const image = await reconstruct(projector, measured, {
  steps: 200, lr: 0.03, view_chunk_size: 8,
  report_every: 20,
  on_progress: ({ step, loss }) => console.log(step, loss),
});
console.log(image.shape); // [32, 32]
image.dispose();
```

| Option | Default and meaning |
| --- | --- |
| `steps` | `300` positive integer Adam updates. |
| `lr` | `0.01` positive finite learning rate. |
| `initial_image` | Zeros; supplied arrays must exactly match measurement batch dimensions followed by `[N, N]`. |
| `nonnegative` | `true`; clamp both the initial image and each updated image. |
| `view_chunk_size` | `8`; positive integer maximum views per gradient chunk. |
| `report_every` | `10`; progress reports at step zero, every interval, and the final step. |
| `on_progress` | Optional asynchronous callback with `{step, steps, loss, image, shape}`. |
| `signal` | Optional `AbortSignal`; cancellation rejects with its abort reason. |

The solver computes each chunk's mean loss gradient, weights it by that chunk's
fraction of all views, and adds the gradients **before a single Adam update**.
An uneven last chunk is weighted correctly. This preserves the complete
mean-loss objective, including all batch dimensions; changing chunk size does
not change the intended algorithm. Float32 reduction order can change rounding.
Adam settings are `b1=0.9`, `b2=0.999`, `eps=1e-8`, `epsRoot=0`.

Each progress event contains a copied `Float32Array` snapshot and the loss
evaluated at that image, after the reported update. Mutating a snapshot does
not modify the solver. Canvas rendering, frame delays and cancellation controls
belong in the callback/UI. The helper yields to the event loop between updates.

The helper is asynchronous and does not differentiate through its optimization
trajectory. Use `project` inside your own pure loss and optimizer step for
custom training or unrolling. [`@jax-js/optax`](https://jax-js.com/docs/_jax-js/optax)
supplies Adam, SGD and optimizer-state operations.

## Validation, execution and limits

Shapes and float32 dtypes are checked without reading array values, including
inside transformations. The pure projector assumes finite images. To validate
untrusted inputs eagerly, call the exported `await check_finite(image, "image")`
outside JIT/AD. `reconstruct` validates measurements and initial values, and
checks its image at report intervals and at completion. No asynchronous data
reads or UI callbacks occur inside a differentiated function.

The public forward operation vectorizes the supplied views. jax-js currently
does not provide Python JAX's `lax.scan` or checkpointing facilities. The solver
therefore compiles chunk gradients and the Adam update separately, realizes
each accumulated gradient before the next chunk, and controls the iteration
from JavaScript. JIT wrappers are reused within one solve and disposed when
it finishes; the helper does not retain compiled executables across solves.

The demo selects WebGPU when available and otherwise uses WASM. Host geometry
preparation remains outside differentiation. Automated tests run on WASM. A
Chromium/SwiftShader software WebGPU smoke check also verified forward,
gradient and HVP against that device's sampled matrix (maximum absolute error
below `2e-7`), plus five Adam updates. This is not a hardware GPU benchmark.

Cross-device precision is different: in that small check, WebGPU and WASM
forward values differed by up to `2.44e-4`, while reconstructed pixels after
five updates differed by `1.61e-6`. The shader's sine/cosine results also differ;
[WGSL permits approximation error](https://www.w3.org/TR/WGSL/#floating-point-accuracy)
for these functions. Do not use the WASM/Python parity tolerance as a universal
WebGPU tolerance. These observed errors are not bounds for other geometries.

Browser/device performance must be measured on the intended hardware. The
[Python CPU benchmark](jax_benchmark.md) is not a browser speed comparison.
For timings, separate compilation and rendering, then synchronize each result.

Release the remaining resources from the examples:

```javascript
phantom.dispose(); measured.dispose(); initial.dispose();
forward.dispose(); lossAndGrad.dispose(); projector.dispose();
```

`npm test` checks geometry, boundaries, sampled-matrix derivatives, native and
mapped batches, chunk/full-loss Adam parity, snapshots, cancellation and
reference counts. Python's `tests/test_cross_language.py` also compares forward
projection, loss, gradients, JVP, HVP and reconstruction against PyTorch and
Python JAX, including signed batches, repeated angles, odd/even images and
different ray-sampling multipliers.
