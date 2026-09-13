import fs from "node:fs";
import path from "node:path";

import { contentDigest, evidenceDigest } from "../evidence.mjs";
import { parseGeneratedProductsProof } from "../generated-products.mjs";
import {
  readGeneratedProductsInput, stageGeneratedProductsInput,
} from "../generated-products-input.mjs";
import { reviewedIntegrationProducts } from "../integration-products.mjs";

export function compiledInventoryWithManifest(compiledInventory, identities) {
  const value = structuredClone(compiledInventory);
  const manifest = value.snapshot.observed.registration_manifest;
  manifest.entries = manifest.entries.filter((entry) =>
    entry.kind !== "builtin" || identities.includes(entry.declaration));
  manifest.counts.builtin = manifest.entries.filter((entry) => entry.kind === "builtin").length;
  manifest.digest = contentDigest(Buffer.from(JSON.stringify(manifest.entries))).slice("sha256:".length);
  value.digest.value = contentDigest(Buffer.from(JSON.stringify(value.snapshot))).slice("sha256:".length);
  return value;
}

export function writeWasmRegistry(repository, manifest) {
  const entries = manifest.entries.flatMap((entry) => [
    [
      "// @runmat-wasm-registration-v2",
      entry.kind,
      entry.declaration,
      entry.variant === null ? "none" : "some",
      entry.variant ?? "",
      entry.builtin_path,
    ].join("\t"),
    `pub fn __runmat_wasm_register_builtin_${entry.declaration}_builtin() {}`,
  ]);
  const source = [
    ...entries,
    `pub const REGISTRY_MANIFEST_DIGEST: &str = "${manifest.digest}";`,
    `pub const REGISTRY_BUILTIN_COUNT: usize = ${manifest.counts.builtin};`,
    `pub const REGISTRY_CONSTANT_COUNT: usize = ${manifest.counts.constant};`,
    `pub const REGISTRY_GPU_SPEC_COUNT: usize = ${manifest.counts.gpu_spec};`,
    `pub const REGISTRY_FUSION_SPEC_COUNT: usize = ${manifest.counts.fusion_spec};`,
    "",
  ].join("\n");
  fs.writeFileSync(
    path.join(repository, "crates/runmat-runtime/src/builtins/generated_wasm_registry.rs"),
    source,
  );
  return source;
}

export function verifyGeneratedProducts({ fixture, subjectInventory, projection, compiledInventory = fixture.compiledInventory }) {
  const bundle = fixture.control.bundles.get(fixture.bundleIds[1]);
  const products = reviewedIntegrationProducts(
    bundle.integration_product_refs, fixture.control.integrationProducts, bundle.id,
  );
  const nativeManifest = compiledInventory.snapshot.observed.registration_manifest;
  const staged = stageGeneratedProductsInput(products, nativeManifest, projection);
  const decoded = readGeneratedProductsInput(staged.stdin);
  const projectionById = new Map(projection.products.map((product) => [product.product_id, product]));
  const records = products.map((product) => {
    const composition = projectionById.get(product.product_id) ?? null;
    const checkedIn = composition?.state === "absent"
      ? absentObservation()
      : fileObservation(path.join(fixture.repository, product.path));
    const verification = composition === null
      ? manifestVerification(nativeManifest, readWasmManifest(fixture.repository))
      : {
        kind: "rust_module_composition",
        projection_digest: evidenceDigest(composition),
        result: "pass",
      };
    return {
      product_id: product.product_id,
      path: product.path,
      generator: {
        path: product.generator.path,
        content_digest: product.generator.baseline_digest,
      },
      checked_in: checkedIn,
      first: structuredClone(checkedIn),
      second: structuredClone(checkedIn),
      deterministic: true,
      synchronized: true,
      verification,
    };
  });
  const passed = records.every((record) => record.verification.result === "pass");
  const proofValue = {
    schema_version: 3,
    kind: "runmat-builtin-generated-products-proof",
    authority: "machine-derived-integration-evidence",
    products: records,
    result: passed ? "pass" : "fail",
  };
  return {
    decoded,
    proofValue,
    proof: parseGeneratedProductsProof(proofValue, {
      integration_products: products,
      source_files: fixture.inventory.source.files,
      native_registration_manifest: nativeManifest,
      module_composition_projection: projection,
    }),
  };
}

function manifestVerification(nativeManifest, generatedManifest) {
  const nativeIdentity = {
    schema_version: nativeManifest.schema_version,
    digest: nativeManifest.digest,
    counts: structuredClone(nativeManifest.counts),
  };
  const matches = JSON.stringify(generatedManifest) === JSON.stringify(nativeIdentity);
  return {
    kind: "native_wasm_registration_manifest",
    generated_manifest: generatedManifest,
    native_manifest: nativeIdentity,
    result: matches ? "pass" : "fail",
  };
}

function readWasmManifest(repository) {
  const source = fs.readFileSync(
    path.join(repository, "crates/runmat-runtime/src/builtins/generated_wasm_registry.rs"),
    "utf8",
  );
  const capture = (pattern, label) => {
    const match = source.match(pattern);
    if (!match) throw new Error(`fixture WASM registry is missing ${label}`);
    return match[1];
  };
  return {
    schema_version: 1,
    digest: capture(/REGISTRY_MANIFEST_DIGEST: &str = "([a-f0-9]{64})";/, "manifest digest"),
    counts: {
      builtin: Number(capture(/REGISTRY_BUILTIN_COUNT: usize = ([0-9]+);/, "builtin count")),
      constant: Number(capture(/REGISTRY_CONSTANT_COUNT: usize = ([0-9]+);/, "constant count")),
      gpu_spec: Number(capture(/REGISTRY_GPU_SPEC_COUNT: usize = ([0-9]+);/, "GPU count")),
      fusion_spec: Number(capture(/REGISTRY_FUSION_SPEC_COUNT: usize = ([0-9]+);/, "fusion count")),
    },
  };
}

function absentObservation() {
  return { state: "absent", byte_length: null, content_digest: null };
}

function fileObservation(target) {
  const contents = fs.readFileSync(target);
  return {
    state: "present", byte_length: contents.length, content_digest: contentDigest(contents),
  };
}
