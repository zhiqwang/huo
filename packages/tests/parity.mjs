// Called by the Python integration test with a shared scanner and signed inputs.
import { readFileSync } from "node:fs";
import { init, numpy as np } from "@jax-js/jax";
import { RadonFanbeam } from "../src/art.js";

const input = JSON.parse(readFileSync(0, "utf8"));
await init("wasm");
const radon = new RadonFanbeam(input.parameters);
const { resolution, det_count, angles } = input.parameters;
const image = np.array(new Float32Array(input.image)).reshape([resolution, resolution]);
const sinogram = np.array(new Float32Array(input.sinogram)).reshape([angles.length, det_count]);
const previous_random = Math.random;
Math.random = () => 1 - Number.EPSILON; // Identity permutation for this comparison.
try {
  const forward = await (await radon.forward(image)).data();
  const backprojection = await (await radon.backprojection(sinogram)).data();
  const art = await (await radon.art(sinogram)).data();
  process.stdout.write(JSON.stringify({
    forward: Array.from(forward), backprojection: Array.from(backprojection), art: Array.from(art),
  }));
} finally {
  Math.random = previous_random;
  image.dispose();
  sinogram.dispose();
}
