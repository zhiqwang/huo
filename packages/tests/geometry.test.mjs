import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { init, jit, numpy as np, tree } from "@jax-js/jax";
import { FanbeamGeometry, FanbeamProjector } from "../dist/src/index.js";
import { compute_ray_coordinates } from "../dist/src/geometry.js";
import { sample_bilinear } from "../dist/src/projector.js";
import { close } from "./helpers.mjs";

await init("wasm");
const fixtures = JSON.parse(readFileSync(new URL("../../tests/fixtures/geometry.json", import.meta.url)));
const parameters = (resolution) => ({
  resolution,
  source_distance: 20,
  det_distance: 16,
  det_count: 31,
  det_spacing: 1.3,
});

for (const resolution of [5, 6])
  for (const ArrayType of [Float32Array, Float64Array]) {
    test(`source-to-detector rays, N=${resolution}, ${ArrayType.name}`, () => {
      const geometry = new FanbeamGeometry(parameters(resolution));
      const { ray_x, ray_y } = compute_ray_coordinates(geometry, ArrayType);
      const tolerance = ArrayType === Float64Array ? 1e-12 : 5e-6;
      for (let k = 0; k < geometry.ray_sample_count; k++)
        for (let i = 0; i < 31; i++) {
          const u = (i - 15) * 1.3;
          const x = ray_x[k * 31 + i] * geometry.pixel_extent;
          const y = ray_y[k * 31 + i] * geometry.pixel_extent;
          close([(-36 * x - u * (y - 20)) / Math.hypot(u, 36)], [0], tolerance);
          close([Math.hypot(x, y - 20)], [20 - resolution / 2 + k / 2], tolerance);
        }
    });
  }

for (const resolution of [5, 6]) {
  test(`quarter-turn radians and orientation, N=${resolution}`, async () => {
    const geometry = new FanbeamGeometry(parameters(resolution));
    const p = new FanbeamProjector(geometry, [Math.PI / 2]);
    const zero = new FanbeamProjector(geometry, [0]);
    const data = Float32Array.from({ length: resolution ** 2 }, (_, i) => i / 10 - 1);
    const rotated = new Float32Array(data.length);
    for (let row = 0; row < resolution; row++)
      for (let col = 0; col < resolution; col++) {
        rotated[row * resolution + col] = data[(resolution - 1 - col) * resolution + row];
      }
    try {
      close(
        await p.forward(np.array(data).reshape([resolution, resolution])).data(),
        await zero.forward(np.array(rotated).reshape([resolution, resolution])).data(),
        1e-5,
      );
    } finally {
      p.dispose();
      zero.dispose();
    }
  });
}

test("shared image boundary fixture retains partial zero padding under JIT", async () => {
  const f = fixtures.image_boundary;
  const sample = jit((image, x, y) => sample_bilinear(image, x, y).sum(0).mul(0.5));
  try {
    close(await sample(np.array(f.image), np.array(f.ray_x), np.array(f.ray_y)).data(), f.expected, 1e-7);
  } finally {
    sample.dispose();
  }
});

test("non-binary ray steps retain the configured resolution", async () => {
  const p = new FanbeamProjector(new FanbeamGeometry({ ...parameters(6), ray_sampling: 49 }), [0]);
  try {
    const image = p.forward(np.ones([6, 6]));
    assert.deepEqual(image.shape, [1, 31]);
    image.dispose();
  } finally {
    p.dispose();
  }
});

test("scanner owns angles; state refs and disposal are explicit", async () => {
  const angles = new Float32Array([0, 0.5]);
  const p = new FanbeamProjector(new FanbeamGeometry(parameters(5)), angles);
  angles.fill(99);
  close(await p.state.angles.ref.data(), [0, 0.5]);
  assert.throws(() => {
    p.geometry = null;
  }, TypeError);
  const owned = tree.ref(p.state);
  p.dispose();
  p.dispose();
  assert.throws(() => p.state, /disposed/);
  close(await owned.angles.ref.data(), [0, 0.5]);
  tree.dispose(owned);
});

test("geometry, angles, shape, and dtype validation", () => {
  const geometry = new FanbeamGeometry(parameters(5));
  assert.throws(() => {
    geometry.resolution = 8;
  }, TypeError);
  for (const [field, value] of [
    ["resolution", 1],
    ["det_count", 1],
    ["source_distance", NaN],
    ["source_distance", 2],
    ["det_spacing", 0],
    ["ray_sampling", 0],
  ]) {
    assert.throws(() => new FanbeamGeometry({ ...parameters(5), [field]: value }), RangeError);
  }
  for (const angles of [[], [NaN], [Infinity], [[0]], [true], [1e50]]) {
    assert.throws(() => new FanbeamProjector(geometry, angles), /angles/);
  }
  assert.throws(() => new FanbeamProjector(geometry, [0], { dtype: np.float64 }), /float32/);
  const p = new FanbeamProjector(geometry, [0]);
  for (const value of [np.zeros([5, 4]), np.zeros([5, 5], { dtype: np.int32 })]) {
    try {
      assert.throws(() => p.forward(value), /shape|dtype/);
    } finally {
      value.dispose();
    }
  }
  p.dispose();
});
