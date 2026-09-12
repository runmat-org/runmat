#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { contentDigest } from "./builtin-migration/evidence.mjs";

const repository = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const PRODUCTS = Object.freeze([Object.freeze({
  product_id: "wasm-registry",
  path: "crates/runmat-runtime/src/builtins/generated_wasm_registry.rs",
  generator: "scripts/regenerate-wasm-registry.mjs",
})]);

const requestedProducts = selectProducts(process.argv.slice(2));

const directory = mkdtempSync(join(tmpdir(), "runmat-generated-product-proof-"));
try {
  const records = requestedProducts.map((product) => {
    const first = generate(product, join(directory, `${product.product_id}-first.rs`));
    const second = generate(product, join(directory, `${product.product_id}-second.rs`));
    const checkedIn = observation(resolve(repository, product.path));
    const generatorPath = resolve(repository, product.generator);
    return {
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
  });
  const value = {
    schema_version: 1,
    kind: "runmat-builtin-generated-products-proof",
    authority: "machine-derived-integration-evidence",
    products: records,
    result: records.every((record) => record.deterministic && record.synchronized) ? "pass" : "fail",
  };
  process.stdout.write(`${JSON.stringify(value)}\n`);
  if (value.result !== "pass") process.exitCode = 1;
} finally {
  rmSync(directory, { recursive: true, force: true });
}

function selectProducts(argumentsList) {
  if (argumentsList.length === 1 && argumentsList[0] === "--no-products") return [];
  if (argumentsList.length === 0 || argumentsList.length % 2 !== 0) {
    throw new Error("usage: verify-builtin-generated-products.mjs --no-products | --product ID [--product ID ...]");
  }
  const requested = [];
  for (let index = 0; index < argumentsList.length; index += 2) {
    if (argumentsList[index] !== "--product" || !argumentsList[index + 1]) {
      throw new Error("usage: verify-builtin-generated-products.mjs --no-products | --product ID [--product ID ...]");
    }
    requested.push(argumentsList[index + 1]);
  }
  if (new Set(requested).size !== requested.length || JSON.stringify(requested) !== JSON.stringify([...requested].sort())) {
    throw new Error("generated product selections must be unique and canonically ordered");
  }
  return requested.map((productId) => {
    const product = PRODUCTS.find((entry) => entry.product_id === productId);
    if (!product) throw new Error(`unknown generated product ${productId}`);
    return product;
  });
}

function generate(product, output) {
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
