#!/usr/bin/env node
// Verify that one cached build-script executable reads the selected worktree,
// not the worktree where that executable happened to be compiled.
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, readdirSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const [buildScriptArg, firstRootArg, secondRootArg] = process.argv.slice(2);
if (!buildScriptArg || !firstRootArg || !secondRootArg || process.argv.length !== 5) {
  throw new Error(
    "usage: verify-wasm-build-script-source-root.mjs BUILD_SCRIPT ROOT_A ROOT_B",
  );
}

const buildScript = resolve(buildScriptArg);
const roots = [resolve(firstRootArg), resolve(secondRootArg)];
const scratch = mkdtempSync(join(tmpdir(), "runmat-wasm-build-root-"));

function sourceInputs(root) {
  const inputs = [];
  const manifest = join(root, "crates", "runmat-runtime");
  function collect(directory, labelDirectory) {
    const entries = readdirSync(directory, { withFileTypes: true });
    entries.sort((left, right) => left.name < right.name ? -1 : left.name > right.name ? 1 : 0);
    for (const entry of entries) {
      const path = join(directory, entry.name);
      const label = `${labelDirectory}/${entry.name}`;
      if (statSync(path).isDirectory()) {
        collect(path, label);
      } else if (entry.name.endsWith(".rs") && entry.name !== "generated_wasm_registry.rs") {
        inputs.push([label, path]);
      }
    }
  }
  collect(join(manifest, "src", "builtins"), "src/builtins");
  inputs.push(["../runmat-macros/src/lib.rs", join(root, "crates", "runmat-macros", "src", "lib.rs")]);
  inputs.sort((left, right) => left[0] < right[0] ? -1 : left[0] > right[0] ? 1 : 0);
  return inputs;
}

function expectedFingerprint(root) {
  const hash = createHash("sha256");
  for (const [label, path] of sourceInputs(root)) {
    const name = Buffer.from(label);
    const content = readFileSync(path);
    const length = Buffer.alloc(8);
    length.writeBigUInt64LE(BigInt(name.length));
    hash.update(length);
    hash.update(name);
    length.writeBigUInt64LE(BigInt(content.length));
    hash.update(length);
    hash.update(content);
  }
  return `sha256:${hash.digest("hex")}`;
}

function observedFingerprint(root, ordinal) {
  const output = join(scratch, `registry-${ordinal}.rs`);
  const environment = {
    ...process.env,
    CARGO_MANIFEST_DIR: join(root, "crates", "runmat-runtime"),
    CARGO_CFG_TARGET_ARCH: "wasm32",
    CARGO_CFG_TARGET_OS: "unknown",
    TARGET: "wasm32-unknown-unknown",
    CARGO_FEATURE_PLOT_WEB: "1",
    RUNMAT_GENERATE_WASM_REGISTRY: "1",
    RUNMAT_WASM_REGISTRY_OUT: output,
  };
  delete environment.CARGO_FEATURE_BLAS_LAPACK;
  const result = spawnSync(buildScript, [], { env: environment, encoding: "utf8" });
  if (result.error || result.status !== 0) {
    throw new Error(
      `build script failed for root ${root}: ${result.error?.message ?? (result.stderr || result.status)}`,
    );
  }
  const registry = readFileSync(output, "utf8");
  const match = registry.match(/pub const REGISTRY_SOURCE_FINGERPRINT: &str = "(sha256:[a-f0-9]{64})";/);
  if (!match) throw new Error(`build script did not stamp a source fingerprint for ${root}`);
  return match[1];
}

try {
  const expected = roots.map(expectedFingerprint);
  if (expected[0] === expected[1]) {
    throw new Error("source roots must have distinct builtin fingerprints for this regression test");
  }
  for (let ordinal = 0; ordinal < roots.length; ordinal += 1) {
    const observed = observedFingerprint(roots[ordinal], ordinal);
    if (observed !== expected[ordinal]) {
      throw new Error(
        `source root ${roots[ordinal]}: observed ${observed}, expected ${expected[ordinal]}`,
      );
    }
    console.log(`${roots[ordinal]}: ${observed}`);
  }
} finally {
  rmSync(scratch, { recursive: true, force: true });
}
