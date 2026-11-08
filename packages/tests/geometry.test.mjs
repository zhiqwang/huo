import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { init, numpy as np } from "@jax-js/jax";
import { RadonFanbeam, forward_angle } from "../src/art.js";
import { FanbeamGeometry, compute_ray_coordinates } from "../src/geometry.js";

await init("wasm");
const fixtures = JSON.parse(readFileSync(new URL("../../tests/fixtures/geometry.json", import.meta.url)));
const parameters = (resolution) => ({
  resolution, source_distance: 20, det_distance: 16, det_count: 31, det_spacing: 1.3,
});
function close(actual, expected, tolerance = 5e-6) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++) {
    assert.ok(Math.abs(actual[i] - expected[i]) <= tolerance,
      `sample ${i}: ${actual[i]} != ${expected[i]}`);
  }
}

for (const resolution of [5, 6]) {
  test(`ray samples follow source to detector, N=${resolution}`, () => {
    const geometry = new FanbeamGeometry(parameters(resolution));
    const { ray_x, ray_y } = compute_ray_coordinates(geometry);
    for (let k = 0; k < geometry.ray_sample_count; k++) for (let i = 0; i < 31; i++) {
      const u = (i - 15) * 1.3;
      const x = ray_x[k * 31 + i] * geometry.pixel_extent;
      const y = ray_y[k * 31 + i] * geometry.pixel_extent;
      close([(-36 * x - u * (y - 20)) / Math.hypot(u, 36)], [0]);
      close([Math.hypot(x, y - 20)], [20 - resolution / 2 + k / 2]);
    }
  });
  for (const angle of [0, Math.PI / 2, 0.37, -0.41]) {
    test(`detector ramp, N=${resolution}, angle=${angle}`, async () => {
      const radon = new RadonFanbeam({ ...parameters(resolution), angles: [angle] });
      const row = Float32Array.from({ length: 31 }, (_, i) => 0.3 * (i - 15) * 1.3 + 0.7);
      const sinogram = np.array(row).reshape([1, 31]);
      try {
        const result = await radon.backprojection(sinogram);
        const c = Math.cos(angle), s = Math.sin(angle), expected = [];
        for (let r = 0; r < resolution; r++) for (let col = 0; col < resolution; col++) {
          const x = col - (resolution - 1) / 2, y = r - (resolution - 1) / 2;
          expected.push(0.3 * 36 * (c * x - s * y) / (20 - s * x - c * y) + 0.7);
        }
        close(await result.data(), expected);
      } finally { sinogram.dispose(); }
    });
  }
  test(`quarter-turn uses radians and the public sign, N=${resolution}`, async () => {
    const radon = new RadonFanbeam({ ...parameters(resolution), angles: [0, Math.PI / 2] });
    const data = Float32Array.from({ length: resolution ** 2 }, (_, i) => i / 10 - 1);
    const rotated = new Float32Array(data.length);
    for (let row = 0; row < resolution; row++) for (let col = 0; col < resolution; col++) {
      rotated[row * resolution + col] = data[(resolution - 1 - col) * resolution + row];
    }
    const image = np.array(data).reshape([resolution, resolution]);
    const rotated_image = np.array(rotated).reshape([resolution, resolution]);
    try {
      close(await (await radon.forward_angle(image, Math.PI / 2)).data(),
        await (await radon.forward_angle(rotated_image, 0)).data());
    } finally { image.dispose(); rotated_image.dispose(); }
  });
}

test("zero padding retains partial image edge weights", () => {
  const f = fixtures.image_boundary;
  const geometry = new FanbeamGeometry({ resolution: 2, source_distance: 2, det_distance: 1,
    det_count: 2, det_spacing: 1 });
  close(forward_angle(new Float32Array(f.image.flat()), new Float32Array(f.ray_x.flat()),
    new Float32Array(f.ray_y.flat()), 0, geometry), f.expected);
});

test("zero padding retains partial detector edge weights", async () => {
  const f = fixtures.detector_boundary;
  const radon = new RadonFanbeam({ resolution: 3, angles: [0], source_distance: 3, det_distance: 1.5,
    det_count: 3, det_spacing: 1.5 });
  const sinogram = np.array(new Float32Array(f.projection)).reshape([1, 3]);
  try { close(await (await radon.backprojection(sinogram)).data(), f.expected.flat()); }
  finally { sinogram.dispose(); }
});

test("angle-major rows, progress callbacks, and full backprojection", async () => {
  const angles = [Math.PI / 2, -0.37, 0];
  const radon = new RadonFanbeam({ ...parameters(5), angles });
  const image = np.array(new Float32Array(25).fill(1)).reshape([5, 5]);
  const reported = [];
  const sinogram = await radon.forward(image, {
    on_progress: (data, angle_index) => reported.push({ data: data.slice(), angle_index }),
  });
  try {
    assert.deepEqual(sinogram.shape, [3, 31]);
    assert.deepEqual(reported.map((entry) => entry.angle_index), [0, 1, 2]);
    const data = await sinogram.ref.data(), expected = new Float32Array(25);
    for (let index = 0; index < 3; index++) {
      const row = data.slice(index * 31, (index + 1) * 31);
      close(row, await (await radon.forward_angle(image, angles[index])).data());
      close(reported[index].data.slice(index * 31, (index + 1) * 31), row);
      const projection = np.array(row);
      const contribution = await (await radon.backprojection_angle(projection, angles[index])).data();
      projection.dispose();
      for (let i = 0; i < expected.length; i++) expected[i] += contribution[i];
    }
    close(await (await radon.backprojection(sinogram)).data(), expected);
    const reconstructed = await (await radon.art(sinogram)).data();
    assert.ok(reconstructed.every((value) => Number.isFinite(value) && value >= 0));
  } finally { image.dispose(); sinogram.dispose(); }
});

test("rejects detector-by-angle layout", async () => {
  const radon = new RadonFanbeam({ ...parameters(5), angles: [0, 1] });
  const sinogram = np.array(new Float32Array(62)).reshape([31, 2]);
  try { await assert.rejects(radon.art(sinogram), /shape/); }
  finally { sinogram.dispose(); }
});

test("immutable geometry and invalid inputs", () => {
  const geometry = new FanbeamGeometry(parameters(5));
  assert.throws(() => { geometry.resolution = 8; }, TypeError);
  for (const [field, value] of [["resolution", 1], ["det_count", 1], ["source_distance", NaN],
    ["source_distance", 2], ["det_spacing", 0], ["ray_sampling", 0]]) {
    assert.throws(() => new FanbeamGeometry({ ...parameters(5), [field]: value }), RangeError);
  }
});
