import assert from "node:assert/strict";
import test from "node:test";

import {
  parseGeneratedProductDefinitions,
  parseGeneratedProductsProof,
} from "../generated-products.mjs";
import { stageGeneratedProductsInput, readGeneratedProductsInput } from "../generated-products-input.mjs";
import { evidenceDigest } from "../evidence.mjs";

const SOURCE_DIGEST = `sha256:${"a".repeat(64)}`;
const CONTENT_DIGEST = `sha256:${"b".repeat(64)}`;
const MANIFEST = {
  schema_version: 1,
  digest: "c".repeat(64),
  counts: { builtin: 4, constant: 0, gpu_spec: 1, fusion_spec: 2 },
};

function definition() {
  return {
    product_id: "wasm-registry",
    path: "generated/wasm.rs",
    producer: "integration",
    generator: { path: "scripts/generate.mjs", baseline_digest: SOURCE_DIGEST },
    baseline_digest: CONTENT_DIGEST,
    verification: { kind: "native_wasm_registration_manifest" },
  };
}

function proof() {
  const observation = { byte_length: 10, content_digest: CONTENT_DIGEST };
  return {
    schema_version: 2,
    kind: "runmat-builtin-generated-products-proof",
    authority: "machine-derived-integration-evidence",
    products: [{
      product_id: "wasm-registry",
      path: "generated/wasm.rs",
      generator: { path: "scripts/generate.mjs", content_digest: SOURCE_DIGEST },
      checked_in: observation,
      first: observation,
      second: observation,
      deterministic: true,
      synchronized: true,
      verification: {
        kind: "native_wasm_registration_manifest",
        generated_manifest: structuredClone(MANIFEST),
        native_manifest: structuredClone(MANIFEST),
        result: "pass",
      },
    }],
    result: "pass",
  };
}

function expected() {
  return {
    integration_products: [definition()],
    source_files: [{ path: "scripts/generate.mjs", content_digest: SOURCE_DIGEST }],
    native_registration_manifest: structuredClone(MANIFEST),
  };
}

test("generated-product proof binds exact native and WASM registration identities", () => {
  assert.equal(parseGeneratedProductsProof(proof(), expected()).result, "pass");

  const observedDrift = proof();
  observedDrift.products[0].verification.generated_manifest.digest = "d".repeat(64);
  observedDrift.products[0].verification.result = "fail";
  observedDrift.result = "fail";
  assert.equal(parseGeneratedProductsProof(observedDrift, expected()).result, "fail");

  for (const mutate of [
    (value) => { value.products[0].verification.generated_manifest.digest = "d".repeat(64); },
    (value) => { value.products[0].verification.generated_manifest.counts.constant = 1; },
    (value) => { value.products[0].verification.native_manifest.digest = "d".repeat(64); },
  ]) {
    const value = proof();
    mutate(value);
    assert.throws(() => parseGeneratedProductsProof(value, expected()), /manifest|parity result/);
  }
});

test("integration-product verification contracts are a closed typed vocabulary", () => {
  const value = definition();
  value.verification.kind = "arbitrary-script-assertion";
  assert.throws(() => parseGeneratedProductDefinitions([value]), /unsupported integration product verification contract/);

  const composition = definition();
  composition.product_id = "catalog-math-composition";
  composition.path = "crates/runmat-builtins/src/catalog/entries/math/mod.rs";
  composition.verification = {
    kind: "rust_module_composition",
    crate_role: "catalog",
    module_path: "crate::catalog::entries::math",
  };
  assert.equal(parseGeneratedProductDefinitions([composition])[0].verification.crate_role, "catalog");
  const wrongRole = structuredClone(composition);
  wrongRole.verification.crate_role = "runtime";
  assert.throws(() => parseGeneratedProductDefinitions([wrongRole]), /does not match its logical parent|outside its crate role/);
  const extraField = structuredClone(composition);
  extraField.verification.command = "arbitrary";
  assert.throws(() => parseGeneratedProductDefinitions([extraField]), /fields must be exactly/);
});

test("module composition input and proof bind the exact reviewed parent projection", () => {
  const composition = definition();
  composition.product_id = "catalog-math-composition";
  composition.path = "crates/runmat-builtins/src/catalog/entries/math/mod.rs";
  composition.verification = {
    kind: "rust_module_composition",
    crate_role: "catalog",
    module_path: "crate::catalog::entries::math",
  };
  const child = {
    module: "arithmetic",
    source_kind: "directory",
    source_path: "crates/runmat-builtins/src/catalog/entries/math/arithmetic/mod.rs",
    role: "group",
    visibility: "private",
    feature_policy: { kind: "always" },
    macro_use: false,
    reexport: { kind: "glob", visibility: "public" },
    aggregation_sources: [{ role: "entries", kind: "slice" }],
  };
  const projection = {
    schema_version: 2,
    kind: "runmat-builtin-module-composition-projection",
    products: [{
      product_id: composition.product_id,
      crate_role: "catalog",
      path: composition.path,
      module_path: "crate::catalog::entries::math",
      aggregations: ["entries"],
      children: [child],
    }],
  };
  const staged = stageGeneratedProductsInput([composition], null, projection);
  assert.deepEqual(readGeneratedProductsInput(staged.stdin).module_composition_projection, projection);
  const observation = { byte_length: 10, content_digest: CONTENT_DIGEST };
  const value = {
    schema_version: 2,
    kind: "runmat-builtin-generated-products-proof",
    authority: "machine-derived-integration-evidence",
    products: [{
      product_id: composition.product_id,
      path: composition.path,
      generator: { path: composition.generator.path, content_digest: SOURCE_DIGEST },
      checked_in: observation,
      first: observation,
      second: observation,
      deterministic: true,
      synchronized: true,
      verification: {
        kind: "rust_module_composition",
        projection_digest: evidenceDigest(projection.products[0]),
        result: "pass",
      },
    }],
    result: "pass",
  };
  const expectedValue = {
    integration_products: [composition],
    source_files: [{ path: composition.generator.path, content_digest: SOURCE_DIGEST }],
    native_registration_manifest: null,
    module_composition_projection: projection,
  };
  assert.equal(parseGeneratedProductsProof(value, expectedValue).result, "pass");
  const emptyProjection = structuredClone(projection);
  emptyProjection.products[0].children = [];
  const emptyValue = structuredClone(value);
  emptyValue.products[0].verification.projection_digest = evidenceDigest(emptyProjection.products[0]);
  assert.equal(parseGeneratedProductsProof(emptyValue, {
    ...expectedValue,
    module_composition_projection: emptyProjection,
  }).result, "pass");
  const stale = structuredClone(value);
  stale.products[0].verification.projection_digest = SOURCE_DIGEST;
  assert.throws(() => parseGeneratedProductsProof(stale, expectedValue), /exact staged projection/);
  assert.throws(() => stageGeneratedProductsInput([composition], null, null), /require their exact projection/);
});

test("empty runtime composition remains a verified baseline product", () => {
  const composition = definition();
  composition.product_id = "runtime-math-composition";
  composition.path = "crates/runmat-runtime/src/builtins/math/mod.rs";
  composition.verification = {
    kind: "rust_module_composition",
    crate_role: "runtime",
    module_path: "crate::builtins::math",
  };
  const product = {
    product_id: composition.product_id,
    crate_role: "runtime",
    path: composition.path,
    module_path: "crate::builtins::math",
    aggregations: [],
    children: [],
  };
  const projection = {
    schema_version: 2,
    kind: "runmat-builtin-module-composition-projection",
    products: [product],
  };
  const observation = { byte_length: 10, content_digest: CONTENT_DIGEST };
  const proofValue = {
    schema_version: 2,
    kind: "runmat-builtin-generated-products-proof",
    authority: "machine-derived-integration-evidence",
    products: [{
      product_id: composition.product_id,
      path: composition.path,
      generator: { path: composition.generator.path, content_digest: SOURCE_DIGEST },
      checked_in: observation,
      first: observation,
      second: observation,
      deterministic: true,
      synchronized: true,
      verification: {
        kind: "rust_module_composition",
        projection_digest: evidenceDigest(product),
        result: "pass",
      },
    }],
    result: "pass",
  };
  assert.equal(parseGeneratedProductsProof(proofValue, {
    integration_products: [composition],
    source_files: [{ path: composition.generator.path, content_digest: SOURCE_DIGEST }],
    native_registration_manifest: null,
    module_composition_projection: projection,
  }).result, "pass");
});
