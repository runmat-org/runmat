import { mkdirSync, copyFileSync, writeFileSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { hash, inputFile } from "./manifest.mjs";
export function requireGraphicsReplayBootstrap(bootstrap) {
  if (!bootstrap.includes("export function importGeometryScene(") || bootstrap.includes("export function initRunMat(") || bootstrap.includes("export class RunMatWasm")) {
    throw new Error("Documentation figures require the graphics-only WASM export mode");
  }
}
/** One immutable renderer closure, shared by all saved documentation figures. */
export function collectGraphics(root, output) {
  if (!root) throw new Error("Build the graphics runtime before exporting saved examples");
  const files = ["runmat_wasm_web.js", "runmat_wasm_web_bg.wasm"].map(path => {
    const bytes = readFileSync(inputFile(root, path));
    return { path, bytes: bytes.length, sha256: hash(bytes) };
  });
  const bootstrap = readFileSync(inputFile(root, files[0].path), "utf8");
  requireGraphicsReplayBootstrap(bootstrap);
  const digest = hash(JSON.stringify(files)); const base = `graphics/${digest}`;
  mkdirSync(join(output, base), { recursive: true });
  for (const file of files) copyFileSync(inputFile(root, file.path), join(output, base, file.path));
  const asset = file => ({ url: `/docs-assets/${base}/${file.path}`, sha256: file.sha256, bytes: file.bytes });
  writeFileSync(join(output, "graphics/runtime.json"), JSON.stringify({ bootstrap: asset(files[0]), wasm: asset(files[1]) }, null, 2) + "\n");
}
