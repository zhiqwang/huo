// Numerical bridge used by the Python tests; no browser or legacy ART required.
import { readFileSync } from "node:fs";
import { init, numpy as np, jit, valueAndGrad, grad, jvp, tree } from "@jax-js/jax";
import { FanbeamGeometry, FanbeamProjector, project, reconstruct } from "../dist/src/index.js";

const input = JSON.parse(readFileSync(0, "utf8"));
await init("wasm");
const p = new FanbeamProjector(new FanbeamGeometry(input.parameters), input.parameters.angles);
const array = (data, shape) => np.array(new Float32Array(data)).reshape(shape);
const image = array(input.image, input.image_shape);
const direction = array(input.direction, input.image_shape);
const measured = array(input.measured, input.measured_shape);
const initial = array(input.initial, input.image_shape);
const f = (x) => p.forward(x);
const loss = (x) => np.square(f(x).sub(measured.ref)).mean().mul(0.5);
const forward = jit((x, state) => project(x, state, p.geometry.ray_step));
const value_grad = jit(valueAndGrad(loss));
const tangent = jit((x, d) => jvp(f, [x], [d]));
const hessian_vector = jit((x, d) => jvp(grad(loss), [x], [d]));
const data = async (x) => Array.from(await x.data());
try {
  const prediction = await data(forward(image.ref, tree.ref(p.state)));
  const [value, gradient] = value_grad(image.ref);
  const [primal, jv] = tangent(image.ref, direction.ref);
  primal.dispose();
  const [g, hv] = hessian_vector(image.ref, direction.ref);
  g.dispose();
  const result = await reconstruct(p, measured, {
    steps: 20,
    lr: 0.03,
    initial_image: initial,
    nonnegative: false,
    view_chunk_size: 3,
  });
  process.stdout.write(
    JSON.stringify({
      forward: prediction,
      loss: (await data(value))[0],
      gradient: await data(gradient),
      jvp: await data(jv),
      hvp: await data(hv),
      reconstructed: await data(result),
    }),
  );
} finally {
  tree.dispose([image, direction, measured, initial]);
  forward.dispose();
  value_grad.dispose();
  tangent.dispose();
  hessian_vector.dispose();
  p.dispose();
}
