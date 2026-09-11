#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const defaultRegistryPath = join(
  repoRoot,
  "crates",
  "runmat-runtime",
  "src",
  "builtins",
  "generated_wasm_registry.rs",
);
const registryPath = outputPath(process.argv.slice(2), defaultRegistryPath);
const tmpDir = mkdtempSync(join(dirname(registryPath), ".runmat-wasm-registry-"));
const tmpRegistry = join(tmpDir, "generated_wasm_registry.rs");

class GenerationFailure extends Error {
  constructor(status) {
    super(`wasm registry generation failed with status ${status}`);
    this.status = status;
  }
}

function outputPath(args, fallback) {
  if (args.length === 0) return fallback;
  if (args.length !== 2 || args[0] !== "--output" || !args[1]) {
    throw new Error("usage: regenerate-wasm-registry.mjs [--output PATH]");
  }
  return resolve(args[1]);
}

try {
  console.log("==> generating wasm builtin registry for runmat-runtime/plot-web");
  const result = spawnSync(
    "cargo",
    [
      "check",
      "-p",
      "runmat-runtime",
      "--target",
      "wasm32-unknown-unknown",
      "--no-default-features",
      "--features",
      "plot-web",
    ],
    {
      cwd: repoRoot,
      env: {
        ...process.env,
        RUNMAT_GENERATE_WASM_REGISTRY: "1",
        RUNMAT_WASM_REGISTRY_OUT: tmpRegistry,
      },
      stdio: ["ignore", "ignore", "inherit"],
    },
  );
  if (result.status !== 0) {
    throw new GenerationFailure(result.status ?? 1);
  }

  let contents = readFileSync(tmpRegistry, "utf8");
  const entryCount = (contents.match(/__runmat_wasm_register_/g) ?? []).length;
  const builtinCount = (contents.match(/__runmat_wasm_register_builtin_/g) ?? []).length;
  if (entryCount <= 0 || builtinCount <= 0) {
    console.error(
      `generated wasm registry is empty or incomplete (${entryCount} entries, ${builtinCount} builtins)`,
    );
    throw new GenerationFailure(1);
  }

  contents = contents
    .replace(
      "pub const REGISTRY_COMPLETE: bool = false;",
      "pub const REGISTRY_COMPLETE: bool = true;",
    )
    .replace(
      "pub const REGISTRY_ENTRY_COUNT: usize = 0;",
      `pub const REGISTRY_ENTRY_COUNT: usize = ${entryCount};`,
    );
  if (!contents.includes("pub const REGISTRY_COMPLETE: bool = true;")) {
    console.error("failed to mark generated wasm registry complete");
    throw new GenerationFailure(1);
  }
  if (!contents.includes(`pub const REGISTRY_ENTRY_COUNT: usize = ${entryCount};`)) {
    console.error("failed to stamp generated wasm registry entry count");
    throw new GenerationFailure(1);
  }

  writeFileSync(tmpRegistry, contents);
  renameSync(tmpRegistry, registryPath);
  console.log(
    `==> wrote ${registryPath} (${entryCount} registry entries, ${builtinCount} builtins)`,
  );
} catch (error) {
  if (error instanceof GenerationFailure) {
    process.exitCode = error.status;
  } else {
    throw error;
  }
} finally {
  rmSync(tmpDir, { recursive: true, force: true });
}
