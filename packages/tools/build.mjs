// A clean output directory prevents removed operators from surviving in the site.
import { cpSync, mkdirSync, rmSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

process.chdir(fileURLToPath(new URL("../", import.meta.url)));
const docs = process.argv.includes("--docs");
rmSync("dist", { recursive: true, force: true });

// Keep jax-js's entry point and dynamically loaded backends on the same module graph.
for (const name of ["jax", "optax"]) {
  const moduleDir = dirname(fileURLToPath(import.meta.resolve(`@jax-js/${name}`)));
  const destination = join("dist", "vendor", name);
  cpSync(moduleDir, destination, {
    recursive: true,
    filter: (source) => source.endsWith(".js") || statSync(source).isDirectory(),
  });
  cpSync(join(moduleDir, "..", "LICENSE"), join(destination, "LICENSE"));
}

const compiler = fileURLToPath(import.meta.resolve("typescript/bin/tsc"));
const result = spawnSync(
  process.execPath,
  [
    compiler,
    ...(docs ? ["--sourceMap", "false", "--declaration", "false", "--declarationMap", "false"] : []),
    ...process.argv.slice(2).filter((argument) => argument !== "--docs"),
  ],
  { stdio: "inherit" },
);
if (result.status !== 0) process.exit(result.status ?? 1);
if (!docs) {
  mkdirSync("dist/demo/static", { recursive: true });
  cpSync("demo/index.html", "dist/demo/index.html");
  cpSync("demo/static/demo.css", "dist/demo/static/demo.css");
}
