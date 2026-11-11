import assert from "node:assert/strict";
import test from "node:test";
import { init, numpy as np, jit, grad, jvp, vjp, vmap, tree } from "@jax-js/jax";
import { FanbeamGeometry, FanbeamProjector, project } from "../dist/src/index.js";
import { close, scanner } from "./helpers.mjs";

await init("wasm");
test("native AD agrees with the sampled matrix", async () => {
  const dtype = np.float32;
  const p = scanner(dtype);
  const columns = await p.forward(np.eye(9, 9, { dtype }).reshape([9, 3, 3])).data();
  const rows = p.views * p.geometry.det_count;
  const xData = Float64Array.from({ length: 9 }, (_, i) => Math.sin(i * 0.31));
  const dData = Float64Array.from({ length: 9 }, (_, i) => Math.cos(i * 0.43));
  const apply = (x) =>
    Array.from({ length: rows }, (_, r) =>
      x.reduce((sum, value, c) => sum + columns[c * rows + r] * value, 0),
    );
  const transpose = (y) =>
    Array.from({ length: 9 }, (_, c) => y.reduce((sum, value, r) => sum + columns[c * rows + r] * value, 0));
  const x = np.array(Array.from(xData), { dtype }).reshape([3, 3]);
  const d = np.array(Array.from(dData), { dtype }).reshape([3, 3]);
  const forward = (image) => p.forward(image);
  const loss = (image) => np.square(forward(image)).sum().mul(0.5);
  const compiled = jit(grad(loss));
  const tolerance = 3e-5;
  try {
    close(await compiled(x.ref).data(), transpose(apply(xData)), tolerance);
    const [y, tangent] = jvp(forward, [x.ref], [d.ref]);
    close(await y.data(), apply(xData), tolerance);
    close(await tangent.data(), apply(dData), tolerance);
    const [g, hvp] = jvp(grad(loss), [x.ref], [d.ref]);
    close(await g.data(), transpose(apply(xData)), tolerance);
    close(await hvp.data(), transpose(apply(dData)), tolerance);
    const matrix = Array.from({ length: rows * 9 }, (_, i) => columns[(i % 9) * rows + Math.floor(i / 9)]);
    const [value, pullback] = vjp(forward, [x.ref]);
    value.dispose();
    try {
      const jacobian = vmap((ct) => pullback(ct)[0])(
        np.eye(rows).reshape([rows, p.views, p.geometry.det_count]),
      );
      close(await jacobian.data(), matrix, tolerance);
    } finally {
      pullback.dispose();
    }
    const gram = Array.from({ length: 81 }, (_, i) =>
      Array.from(
        { length: rows },
        (_, r) => columns[Math.floor(i / 9) * rows + r] * columns[(i % 9) * rows + r],
      ).reduce((a, b) => a + b, 0),
    );
    const hessianRows = vmap((direction) => {
      const [g, product] = jvp(grad(loss), [x.ref], [direction]);
      g.dispose();
      return product;
    })(np.eye(9).reshape([9, 3, 3]));
    close(await hessianRows.data(), gram, tolerance);
    const epsilon = 0.001;
    const finiteDifference =
      ((await loss(x.ref.add(d.ref.mul(epsilon))).data())[0] -
        (await loss(x.ref.sub(d.ref.mul(epsilon))).data())[0]) /
      (2 * epsilon);
    const expected = transpose(apply(xData)).reduce((sum, value, i) => sum + value * dData[i], 0);
    close([finiteDifference], [expected], 1e-3);
  } finally {
    x.dispose();
    d.dispose();
    compiled.dispose();
    p.dispose();
  }
});

test("native leading batches, nested vmap, and parameter gradients", async () => {
  const p = scanner();
  const images = np.arange(54).astype(np.float32).reshape([2, 3, 3, 3]).mul(0.01);
  const mapped = jit(
    vmap(
      vmap((x, s) => project(x, s), [0, null]),
      [0, null],
    ),
  );
  const parameters = np.ones([9, 4]);
  try {
    close(await mapped(images.ref, tree.ref(p.state)).data(), await p.forward(images.ref).data());
    const g = grad((w) => np.square(p.forward(np.matmul(w, np.ones([4])).reshape([3, 3]))).mean())(
      parameters.ref,
    );
    const values = await g.data();
    assert.ok(values.every(Number.isFinite) && values.some((value) => Math.abs(value) > 0));
  } finally {
    mapped.dispose();
    images.dispose();
    parameters.dispose();
    p.dispose();
  }
});

test("explicit scanner state reuses compilation; scanner gradients are zero", async () => {
  const p = scanner();
  const other = new FanbeamProjector(
    new FanbeamGeometry({
      resolution: 3,
      source_distance: 20,
      det_distance: 6,
      det_count: 5,
      det_spacing: 1.3,
    }),
    [0.2, 0.3, 0.7, 1, 1.9],
  );
  let traces = 0;
  const forward = jit((image, state) => {
    traces++;
    return project(image, state);
  });
  const x = np.arange(9).astype(np.float32).reshape([3, 3]);
  try {
    const first = await forward(x.ref, tree.ref(p.state)).data();
    const second = await forward(x.ref, tree.ref(other.state)).data();
    assert.equal(traces, 1);
    assert.notDeepEqual(first, second);
    close(second, await other.forward(x.ref).data());
    const gradient = grad((state) => project(x.ref, state).sum())(tree.ref(p.state));
    for (const leaf of tree.leaves(gradient)) assert.ok((await leaf.data()).every((value) => value === 0));
  } finally {
    forward.dispose();
    x.dispose();
    p.dispose();
    other.dispose();
  }
});
