#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { lstatSync, mkdtempSync, readFileSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

import { contentDigest } from "./builtin-migration/evidence.mjs";
import { readGeneratedProductsInput } from "./builtin-migration/generated-products-input.mjs";

const repository = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const productInput = readGeneratedProductsInput(readFileSync(0, "utf8"));
const productRegistry = productInput.products;
const requestedProducts = selectProducts(process.argv.slice(2), productRegistry);

const directory = mkdtempSync(join(tmpdir(), "runmat-generated-product-proof-"));
try {
  const records = requestedProducts.map((product) => {
    const firstPath = join(directory, `${product.product_id}-first.rs`);
    const secondPath = join(directory, `${product.product_id}-second.rs`);
    const first = generate(product, firstPath);
    const second = generate(product, secondPath);
    const checkedIn = observation(repositoryFile(product.path, `${product.product_id} checked-in product`));
    const generatorPath = repositoryFile(product.generator.path, `${product.product_id} generator`);
    return {
      product_id: product.product_id,
      path: product.path,
      generator: {
        path: product.generator.path,
        content_digest: contentDigest(readFileSync(generatorPath)),
      },
      checked_in: checkedIn,
      first,
      second,
      deterministic: first.content_digest === second.content_digest,
      synchronized: checkedIn.content_digest === first.content_digest,
      verification: semanticVerification(product, firstPath, productInput.native_registration_manifest),
    };
  });
  const value = {
    schema_version: 1,
    kind: "runmat-builtin-generated-products-proof",
    authority: "machine-derived-integration-evidence",
    products: records,
    result: records.every((record) => record.deterministic && record.synchronized && record.verification.result === "pass") ? "pass" : "fail",
  };
  process.stdout.write(`${JSON.stringify(value)}\n`);
  if (value.result !== "pass") process.exitCode = 1;
} finally {
  rmSync(directory, { recursive: true, force: true });
}

function semanticVerification(product, generatedPath, nativeManifest) {
  if (product.verification.kind === "content_identity") {
    return { kind: "content_identity", result: "pass" };
  }
  const generatedManifest = registryManifestIdentity(readFileSync(generatedPath, "utf8"));
  const matches = generatedManifest.schema_version === nativeManifest.schema_version
    && generatedManifest.digest === nativeManifest.digest
    && ["builtin", "constant", "gpu_spec", "fusion_spec"]
      .every((kind) => generatedManifest.counts[kind] === nativeManifest.counts[kind]);
  return {
    kind: "native_wasm_registration_manifest",
    generated_manifest: generatedManifest,
    native_manifest: nativeManifest,
    result: matches ? "pass" : "fail",
  };
}

function registryManifestIdentity(source) {
  const constant = (name, pattern) => {
    const matches = [...source.matchAll(pattern)];
    if (matches.length !== 1) throw new Error(`generated WASM registry must define ${name} exactly once`);
    return matches[0][1];
  };
  return {
    schema_version: 1,
    digest: constant("REGISTRY_MANIFEST_DIGEST", /pub const REGISTRY_MANIFEST_DIGEST: &str = "([a-f0-9]{64})";/g),
    counts: {
      builtin: Number(constant("REGISTRY_BUILTIN_COUNT", /pub const REGISTRY_BUILTIN_COUNT: usize = ([0-9]+);/g)),
      constant: Number(constant("REGISTRY_CONSTANT_COUNT", /pub const REGISTRY_CONSTANT_COUNT: usize = ([0-9]+);/g)),
      gpu_spec: Number(constant("REGISTRY_GPU_SPEC_COUNT", /pub const REGISTRY_GPU_SPEC_COUNT: usize = ([0-9]+);/g)),
      fusion_spec: Number(constant("REGISTRY_FUSION_SPEC_COUNT", /pub const REGISTRY_FUSION_SPEC_COUNT: usize = ([0-9]+);/g)),
    },
  };
}

function selectProducts(argumentsList, registry) {
  if (argumentsList.length === 1 && argumentsList[0] === "--no-products") {
    if (registry.length !== 0) throw new Error("reviewed generated products cannot be omitted");
    return [];
  }
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
  for (const productId of requested) {
    if (!registry.some((entry) => entry.product_id === productId)) {
      throw new Error(`unknown generated product ${productId}`);
    }
  }
  if (JSON.stringify(requested) !== JSON.stringify(registry.map((entry) => entry.product_id))) {
    throw new Error("generated product selection must exactly match the globally reviewed bundle products");
  }
  return requested.map((productId) => registry.find((entry) => entry.product_id === productId));
}

function generate(product, output) {
  const generatorPath = repositoryFile(product.generator.path, `${product.product_id} generator`);
  if (contentDigest(readFileSync(generatorPath)) !== product.generator.baseline_digest) {
    throw new Error(`${product.product_id}: generator bytes differ from the globally reviewed product registry`);
  }
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

function repositoryFile(relativePath, label) {
  const candidate = resolve(repository, relativePath);
  const stat = lstatSync(candidate);
  if (!stat.isFile() || realpathSync(candidate) !== candidate) {
    throw new Error(`${label} must be a canonical regular repository file`);
  }
  const fromRepository = relative(repository, candidate);
  if (fromRepository === ".." || fromRepository.startsWith(`..${sep}`) || isAbsolute(fromRepository)) {
    throw new Error(`${label} is outside the canonical repository`);
  }
  return candidate;
}
