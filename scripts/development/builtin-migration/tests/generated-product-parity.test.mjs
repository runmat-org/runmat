import assert from "node:assert/strict";
import test from "node:test";

import {
  parseGeneratedProductDefinitions,
  parseGeneratedProductsProof,
} from "../generated-products.mjs";

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
    schema_version: 1,
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
});
