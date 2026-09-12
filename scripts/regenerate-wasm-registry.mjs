#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
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
  const manifestRows = contents.split("\n").map((line) => line.trim())
    .filter((line) => line.startsWith("// @runmat-wasm-registration-v2\t"));
  const uniqueManifestRows = new Set(manifestRows);
  const requiredKinds = ["builtin", "constant", "gpu_spec", "fusion_spec"];
  const observedKinds = new Set(manifestRows.map((row) => row.split("\t")[1]));
  if (entryCount <= 0 || builtinCount <= 0 || manifestRows.length !== entryCount
    || uniqueManifestRows.size !== manifestRows.length
    || [...observedKinds].some((kind) => !requiredKinds.includes(kind))) {
    console.error(
      `generated wasm registry is empty, incomplete, or lacks exact typed manifest coverage (${entryCount} entries, ${builtinCount} builtins, ${manifestRows.length} manifest rows)`,
    );
    throw new GenerationFailure(1);
  }

  const canonicalRows = [...manifestRows].sort();
  const manifestEntries = canonicalRows.map((row) => {
    const fields = row.split("\t");
    if (fields.length !== 6) {
      throw new Error("generated WASM registration row has invalid field cardinality");
    }
    if (!((fields[3] === "none" && fields[4] === "") || fields[3] === "some")) {
      throw new Error("generated WASM registration row has invalid variant presence");
    }
    return {
      kind: fields[1],
      declaration: fields[2],
      variant: fields[3] === "none" ? null : fields[4],
      builtin_path: fields[5],
    };
  });
  const manifestDigest = createHash("sha256").update(JSON.stringify(manifestEntries)).digest("hex");
  const kindCounts = Object.fromEntries(requiredKinds.map((kind) => [
    kind,
    manifestRows.filter((row) => row.split("\t")[1] === kind).length,
  ]));

  contents = contents
    .replace(
      "pub const REGISTRY_COMPLETE: bool = false;",
      "pub const REGISTRY_COMPLETE: bool = true;",
    )
    .replace(
      "pub const REGISTRY_ENTRY_COUNT: usize = 0;",
      `pub const REGISTRY_ENTRY_COUNT: usize = ${entryCount};`,
    )
    .replace(
      'pub const REGISTRY_MANIFEST_DIGEST: &str = "0000000000000000000000000000000000000000000000000000000000000000";',
      `pub const REGISTRY_MANIFEST_DIGEST: &str = "${manifestDigest}";`,
    );
  for (const [kind, metadata] of [
    ["builtin", "REGISTRY_BUILTIN_COUNT"],
    ["constant", "REGISTRY_CONSTANT_COUNT"],
    ["gpu_spec", "REGISTRY_GPU_SPEC_COUNT"],
    ["fusion_spec", "REGISTRY_FUSION_SPEC_COUNT"],
  ]) {
    contents = contents.replace(
      `pub const ${metadata}: usize = 0;`,
      `pub const ${metadata}: usize = ${kindCounts[kind]};`,
    );
  }
  if (!contents.includes("pub const REGISTRY_COMPLETE: bool = true;")) {
    console.error("failed to mark generated wasm registry complete");
    throw new GenerationFailure(1);
  }
  if (!contents.includes(`pub const REGISTRY_ENTRY_COUNT: usize = ${entryCount};`)) {
    console.error("failed to stamp generated wasm registry entry count");
    throw new GenerationFailure(1);
  }
  if (!contents.includes(`pub const REGISTRY_MANIFEST_DIGEST: &str = "${manifestDigest}";`)) {
    console.error("failed to stamp generated wasm registry manifest digest");
    throw new GenerationFailure(1);
  }
  for (const [kind, metadata] of [
    ["builtin", "REGISTRY_BUILTIN_COUNT"],
    ["constant", "REGISTRY_CONSTANT_COUNT"],
    ["gpu_spec", "REGISTRY_GPU_SPEC_COUNT"],
    ["fusion_spec", "REGISTRY_FUSION_SPEC_COUNT"],
  ]) {
    if (!contents.includes(`pub const ${metadata}: usize = ${kindCounts[kind]};`)) {
      console.error(`failed to stamp generated wasm registry ${kind} count`);
      throw new GenerationFailure(1);
    }
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
