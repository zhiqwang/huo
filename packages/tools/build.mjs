// A clean output directory prevents removed operators from surviving in the site.
import { cpSync, mkdirSync, rmSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

process.chdir(fileURLToPath(new URL("../", import.meta.url)));
const docs = process.argv.includes("--docs");
rmSync("dist", { recursive: true, force: true });
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
