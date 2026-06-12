#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const registryPath = join(
  repoRoot,
  "crates",
  "runmat-runtime",
  "src",
  "builtins",
  "generated_wasm_registry.rs",
);
const tmpDir = mkdtempSync(join(tmpdir(), "runmat-wasm-registry-"));
const tmpRegistry = join(tmpDir, "generated_wasm_registry.rs");

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
    process.exit(result.status ?? 1);
  }

  let contents = readFileSync(tmpRegistry, "utf8");
  const entryCount = (contents.match(/__runmat_wasm_register_/g) ?? []).length;
  const builtinCount = (contents.match(/__runmat_wasm_register_builtin_/g) ?? []).length;
  if (entryCount <= 0 || builtinCount <= 0) {
    console.error(
      `generated wasm registry is empty or incomplete (${entryCount} entries, ${builtinCount} builtins)`,
    );
    process.exit(1);
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
    process.exit(1);
  }
  if (!contents.includes(`pub const REGISTRY_ENTRY_COUNT: usize = ${entryCount};`)) {
    console.error("failed to stamp generated wasm registry entry count");
    process.exit(1);
  }

  writeFileSync(tmpRegistry, contents);
  renameSync(tmpRegistry, registryPath);
  console.log(
    `==> wrote ${registryPath} (${entryCount} registry entries, ${builtinCount} builtins)`,
  );
} finally {
  rmSync(tmpDir, { recursive: true, force: true });
}
