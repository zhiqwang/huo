import assert from "node:assert/strict";
import test from "node:test";
import { createImageMetrics } from "../dist/demo/metrics.js";

test("image metrics use the same range and Gaussian window as scikit-image", () => {
  const size = 16;
  const phantom = Float32Array.from({ length: size ** 2 }, (_, i) => {
    const row = Math.floor(i / size);
    const col = i % size;
    if (row >= 6 && row < 10 && col >= 8 && col < 10) return 1;
    return row >= 2 && row < 14 && col >= 2 && col < 14 ? 0.5 : 0;
  });
  const measure = createImageMetrics(phantom, size);
  assert.deepEqual(measure(phantom), { relativeL2: 0, psnr: Infinity, ssim: 1, flatVariation: 0 });

  const estimate = phantom.map((pixel, i) => {
    const odd = (Math.floor(i / size) + i % size) % 2;
    return pixel * 0.94 + 0.01 + (odd ? 0.03 : -0.02);
  });
  const actual = measure(estimate);
  // Reference values use float64 arithmetic and a fixed intensity range of 1.
  assert.ok(Math.abs(actual.relativeL2 - 7.432232413490497) < 1e-10);
  assert.ok(Math.abs(actual.psnr - 30.427521108549843) < 1e-10);
  assert.ok(Math.abs(actual.ssim - 0.9843864119259452) < 1e-10);
  assert.ok(Math.abs(actual.flatVariation - 5) < 1e-5);
});
