import assert from "node:assert/strict";
import test from "node:test";
import { init, numpy as np, jit, grad, tree } from "@jax-js/jax";
import { adam, applyUpdates } from "@jax-js/optax";
import { reconstruct, check_finite } from "../dist/src/index.js";
import { close, scanner, objective } from "./helpers.mjs";

await init("wasm");

function variation(image) {
  const measure = (axis) => {
    const difference = np.diff(image.ref, 1, axis);
    return np.maximum(difference.ref, 0).add(np.maximum(difference.neg(), 0)).mean();
  };
  const value = measure(-1).add(measure(-2));
  image.dispose();
  return value;
}

for (const [nonnegative, tv_weight] of [[true, 0], [false, 0], [true, 0.5]])
  for (const chunk of [1, 3, 8]) {
    test(`chunk=${chunk} matches full-loss Adam, nonnegative=${nonnegative}, TV=${tv_weight}`, async () => {
      const p = scanner();
      const target = p.forward(np.full([2, 1, 3, 3], nonnegative ? 1 : -1));
      // Distinct pixels keep the chunk comparison away from TV's subgradient kink.
      const values = Float32Array.from({ length: 18 }, (_, i) => tv_weight ? 0.04 + 0.01 * i : -0.1);
      const initial = np.array(values).reshape([2, 1, 3, 3]);
      const optimizer = adam(0.03);
      let expected = nonnegative ? np.maximum(initial.ref, 0) : initial.ref;
      let state = optimizer.init(expected.ref);
      const update = jit((x, s) => {
        const gradient = grad((value) => tv_weight
          ? objective(value.ref, p, target.ref).add(variation(value).mul(tv_weight))
          : objective(value, p, target.ref))(x.ref);
        const [updates, nextState] = optimizer.update(gradient, s);
        const value = applyUpdates(x, updates);
        return [nonnegative ? np.maximum(value, 0) : value, nextState];
      });
      try {
        for (let step = 0; step < 12; step++) [expected, state] = update(expected, state);
        const actual = await reconstruct(p, target, {
          steps: 12,
          lr: 0.03,
          initial_image: initial,
          nonnegative,
          tv_weight,
          view_chunk_size: chunk,
        });
        close(await actual.ref.data(), await expected.ref.data(), 3e-6);
        assert.ok(
          (await objective(actual, p, target.ref).data())[0] <
            (await np.square(target.ref).mean().mul(0.5).data())[0],
        );
        close(await initial.ref.data(), values, 1e-7);
        assert.equal(initial.refCount, 1);
      } finally {
        expected.dispose();
        tree.dispose(state);
        update.dispose();
        initial.dispose();
        target.dispose();
        p.dispose();
      }
    });
  }

test("progress snapshots match post-update loss and cannot mutate solver arrays", async () => {
  const p = scanner();
  const measured = p.forward(np.ones([3, 3]));
  const seen = [];
  try {
    const image = await reconstruct(p, measured, {
      steps: 5,
      tv_weight: 0.5,
      report_every: 2,
      on_progress: async (info) => {
        const snapshot = np.array(info.image).reshape(info.shape);
        const [loss] = await objective(snapshot.ref, p, measured.ref).data();
        const [tv] = await variation(snapshot).data();
        close([info.loss, info.objective], [loss, loss + 0.5 * tv], 2e-6);
        seen.push(info.step);
        info.image.fill(NaN);
      },
    });
    assert.deepEqual(seen, [0, 2, 4, 5]);
    await check_finite(image, "result");
    image.dispose();
    assert.equal(measured.refCount, 1);
    assert.ok(tree.leaves(p.state).every((value) => value.refCount === 1));
  } finally {
    measured.dispose();
    p.dispose();
  }
});

test("TV contributes zero gradient at a flat starting image", async () => {
  const p = scanner();
  const measured = p.forward(np.ones([3, 3]));
  try {
    const plain = await reconstruct(p, measured, { steps: 1 });
    const regularized = await reconstruct(p, measured, { steps: 1, tv_weight: 0.5 });
    assert.deepEqual(await regularized.data(), await plain.data());
  } finally {
    measured.dispose();
    p.dispose();
  }
});

test("repeated solves, cancellation, and callback failures release borrowed references", async () => {
  const p = scanner();
  const measured = p.forward(np.ones([3, 3]));
  const initial = np.zeros([3, 3]);
  const options = { tv_weight: 0.5, initial_image: initial };
  const baseline = () => {
    assert.equal(measured.refCount, 1);
    assert.equal(initial.refCount, 1);
    assert.ok(tree.leaves(p.state).every((value) => value.refCount === 1));
  };
  try {
    for (let i = 0; i < 3; i++) {
      (await reconstruct(p, measured, { ...options, steps: 2 })).dispose();
      baseline();
    }
    const controller = new AbortController();
    await assert.rejects(
      reconstruct(p, measured, {
        ...options,
        steps: 5,
        report_every: 1,
        signal: controller.signal,
        on_progress: ({ step }) => {
          if (step === 1) controller.abort();
        },
      }),
      { name: "AbortError" },
    );
    baseline();
    await assert.rejects(
      reconstruct(p, measured, {
        ...options,
        steps: 5,
        on_progress: () => {
          throw new Error("callback failed");
        },
      }),
      /callback failed/,
    );
    baseline();
    await assert.rejects(reconstruct(p, measured, { signal: controller.signal }), { name: "AbortError" });
    baseline();
  } finally {
    measured.dispose();
    initial.dispose();
    p.dispose();
  }
});

test("invalid measurements/settings fail without consuming caller inputs", async () => {
  const p = scanner();
  const measured = np.zeros([5, 5]);
  try {
    for (const options of [
      { steps: 0 },
      { steps: 1.5 },
      { lr: NaN },
      { lr: 0 },
      { nonnegative: 1 },
      { tv_weight: -1 },
      { tv_weight: Infinity },
      { tv_weight: "0.5" },
      { view_chunk_size: 0 },
      { report_every: 0 },
    ]) {
      await assert.rejects(reconstruct(p, measured, options), /must be/);
    }
    const invalid = np.full([5, 5], NaN);
    try {
      await assert.rejects(reconstruct(p, invalid, { steps: 1 }), /measured.*finite/);
      assert.equal(invalid.refCount, 1);
    } finally {
      invalid.dispose();
    }
    const wrong = np.zeros([3, 5]);
    try {
      await assert.rejects(reconstruct(p, wrong, { steps: 1 }), /shape/);
      assert.equal(wrong.refCount, 1);
    } finally {
      wrong.dispose();
    }
    assert.equal(measured.refCount, 1);
    assert.ok(tree.leaves(p.state).every((value) => value.refCount === 1));
  } finally {
    measured.dispose();
    p.dispose();
  }
});
