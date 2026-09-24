import { spawnSync } from "node:child_process";

const FORMATTED = new Map();

export function formatRustModule(source) {
  const cached = FORMATTED.get(source);
  if (cached !== undefined) return cached;
  const result = spawnSync("rustfmt", [
    "--emit", "stdout", "--edition", "2021", "--config",
    "reorder_imports=true,reorder_modules=false",
  ], { input: source, encoding: "utf8", maxBuffer: 16 * 1024 * 1024 });
  if (result.error) throw new Error(`rustfmt could not run: ${result.error.message}`);
  if (result.status !== 0 || result.signal !== null) {
    throw new Error(`rustfmt failed: ${result.stderr.trim() || `status ${result.status ?? result.signal}`}`);
  }
  if (!result.stdout.endsWith("\n") || result.stdout.includes("\r") || result.stdout.includes("\0")) {
    throw new Error("rustfmt returned noncanonical module source");
  }
  FORMATTED.set(source, result.stdout);
  return result.stdout;
}
