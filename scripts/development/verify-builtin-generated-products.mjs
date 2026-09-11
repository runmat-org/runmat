#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { contentDigest } from "./builtin-migration/evidence.mjs";

const repository = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const product = Object.freeze({
  product_id: "wasm-registry",
  path: "crates/runmat-runtime/src/builtins/generated_wasm_registry.rs",
  generator: "scripts/regenerate-wasm-registry.mjs",
});

const directory = mkdtempSync(join(tmpdir(), "runmat-generated-product-proof-"));
try {
  const first = generate(join(directory, "first.rs"));
  const second = generate(join(directory, "second.rs"));
  const checkedIn = observation(resolve(repository, product.path));
  const generatorPath = resolve(repository, product.generator);
  const record = {
    product_id: product.product_id,
    path: product.path,
    generator: {
      path: product.generator,
      content_digest: contentDigest(readFileSync(generatorPath)),
    },
    checked_in: checkedIn,
    first,
    second,
    deterministic: first.content_digest === second.content_digest,
    synchronized: checkedIn.content_digest === first.content_digest,
  };
  const value = {
    schema_version: 1,
    kind: "runmat-builtin-generated-products-proof",
    authority: "machine-derived-integration-evidence",
    products: [record],
    result: record.deterministic && record.synchronized ? "pass" : "fail",
  };
  process.stdout.write(`${JSON.stringify(value)}\n`);
  if (value.result !== "pass") process.exitCode = 1;
} finally {
  rmSync(directory, { recursive: true, force: true });
}

function generate(output) {
  const generatorPath = resolve(repository, product.generator);
  const completed = spawnSync(process.execPath, [generatorPath, "--output", output], {
    cwd: repository,
    env: process.env,
    encoding: "utf8",
    maxBuffer: 16 * 1024 * 1024,
  });
  if (completed.error) throw completed.error;
  if (completed.status !== 0) {
    const detail = completed.stderr.trim() || completed.stdout.trim();
    throw new Error(`generated product run failed with status ${completed.status}: ${detail}`);
  }
  return observation(output);
}

function observation(path) {
  const bytes = readFileSync(path);
  return { byte_length: bytes.length, content_digest: contentDigest(bytes) };
}
