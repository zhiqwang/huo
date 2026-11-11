import assert from "node:assert/strict";
import { FanbeamGeometry, FanbeamProjector } from "../dist/src/index.js";
import { numpy as np } from "@jax-js/jax";

export function close(actual, expected, tolerance = 2e-5) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++) {
    assert.ok(
      Number.isFinite(actual[i]) &&
        Math.abs(actual[i] - expected[i]) <= tolerance * (1 + Math.abs(expected[i])),
      `sample ${i}: ${actual[i]} != ${expected[i]}`,
    );
  }
}

export function scanner(dtype = np.float32) {
  return new FanbeamProjector(
    new FanbeamGeometry({
      resolution: 3,
      source_distance: 12,
      det_distance: 6,
      det_count: 5,
      det_spacing: 1.3,
    }),
    [0, -0.37, 0.5, 0, 1.5],
    { dtype },
  );
}

export function objective(image, projector, target) {
  return np.square(projector.forward(image).sub(target)).mean().mul(0.5);
}
