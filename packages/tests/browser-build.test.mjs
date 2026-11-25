import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { basename } from "node:path";
import test from "node:test";

test("the browser import map serves the installed ESM packages with their chunks", () => {
  const dist = new URL("../dist/", import.meta.url);
  const page = new URL("demo/index.html", dist);
  const html = readFileSync(page, "utf8");
  const importMap = html.match(/<script type="importmap">([\s\S]*?)<\/script>/);
  assert.ok(importMap, "the demo needs an import map");
  const { imports } = JSON.parse(importMap[1]);

  for (const name of ["@jax-js/jax", "@jax-js/optax"]) {
    const installed = new URL(import.meta.resolve(name));
    const built = new URL(imports[name], page);
    assert.ok(built.href.startsWith(dist.href), `${name} must be served from dist/`);
    assert.equal(basename(built.pathname), basename(installed.pathname));

    const modules = readdirSync(new URL(".", installed)).filter((file) => file.endsWith(".js"));
    for (const file of modules) {
      const expected = readFileSync(new URL(file, installed));
      const actual = readFileSync(new URL(file, built));
      assert.ok(actual.equals(expected), `${name}/${file} must match the installed module`);
    }
  }
});
