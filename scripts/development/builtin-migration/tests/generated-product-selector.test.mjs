import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { parseGeneratedProductsProof } from "../generated-products.mjs";

const repository = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../..");
const verifier = path.join(repository, "scripts/development/verify-builtin-generated-products.mjs");

function run(argumentsList, products = []) {
  const needsManifest = products.some((entry) => entry.verification?.kind === "native_wasm_registration_manifest");
  const input = JSON.stringify({
    schema_version: 2,
    kind: "runmat-builtin-generated-products-input",
    products,
    native_registration_manifest: needsManifest ? MANIFEST : null,
  });
  return spawnSync(process.execPath, [verifier, ...argumentsList], { cwd: repository, encoding: "utf8", input });
}

test("generated product verification requires an explicit reviewed product selection", () => {
  const none = run(["--no-products"]);
  assert.equal(none.status, 0, none.stderr);
  const proof = JSON.parse(none.stdout);
  assert.deepEqual(proof.products, []);
  assert.equal(proof.result, "pass");

  for (const argumentsList of [[], ["--product"], ["--product", "unknown"], ["--no-products", "--product", "wasm-registry"], ["--product", "wasm-registry", "--product", "wasm-registry"]]) {
    assert.notEqual(run(argumentsList).status, 0, `unexpectedly accepted ${JSON.stringify(argumentsList)}`);
  }
});

test("generated product execution is selected only from the adapter-supplied global registry", () => {
  const reviewed = [{
    product_id: "wasm-registry",
    path: "crates/runmat-runtime/src/builtins/generated_wasm_registry.rs",
    producer: "integration",
    generator: {
      path: "scripts/regenerate-wasm-registry.mjs",
      baseline_digest: `sha256:${"0".repeat(64)}`,
    },
    baseline_digest: null,
    verification: { kind: "native_wasm_registration_manifest" },
  }];
  assert.notEqual(run(["--no-products"], reviewed).status, 0);
  const forged = run(["--product", "wasm-registry"], reviewed);
  assert.notEqual(forged.status, 0);
  assert.match(forged.stderr, /generator bytes differ from the globally reviewed product registry/);
});

test("generated product proof must name the exact globally reviewed generator", () => {
  const digest = `sha256:${"a".repeat(64)}`;
  const proof = {
    schema_version: 1,
    kind: "runmat-builtin-generated-products-proof",
    authority: "machine-derived-integration-evidence",
    products: [{
      product_id: "registry",
      path: "generated/registry.rs",
      generator: { path: "scripts/forged.mjs", content_digest: digest },
      checked_in: { byte_length: 1, content_digest: digest },
      first: { byte_length: 1, content_digest: digest },
      second: { byte_length: 1, content_digest: digest },
      deterministic: true,
      synchronized: true,
      verification: { kind: "content_identity", result: "pass" },
    }],
    result: "pass",
  };
  const expected = {
    integration_products: [{
      product_id: "registry",
      path: "generated/registry.rs",
      producer: "integration",
      generator: { path: "scripts/reviewed.mjs", baseline_digest: digest },
      baseline_digest: null,
      verification: { kind: "content_identity" },
    }],
    source_files: [{ path: "scripts/reviewed.mjs", content_digest: digest }],
    native_registration_manifest: null,
  };
  assert.throws(() => parseGeneratedProductsProof(proof, expected), /globally reviewed product registry/);
});

const MANIFEST = {
  schema_version: 1,
  digest: "0".repeat(64),
  counts: { builtin: 1, constant: 0, gpu_spec: 0, fusion_spec: 0 },
};
