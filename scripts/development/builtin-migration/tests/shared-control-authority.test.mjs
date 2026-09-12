import assert from "node:assert/strict";
import test from "node:test";

import {
  subjectAuthorityPathFailures, validateAuthorityDependencyPolicy,
} from "../authority-paths.mjs";
import {
  bundleBaselineEvidence, parseBundleRemovals, validateCompleteBundleBaselineEvidence,
} from "../baseline-evidence.mjs";
import {
  parseIntegrationProductRegistry, validateGeneratedRegistryCoverage,
  validateIntegrationProductCoverage,
} from "../integration-products.mjs";

const D1 = `sha256:${"1".repeat(64)}`;
const D2 = `sha256:${"2".repeat(64)}`;
const D3 = `sha256:${"3".repeat(64)}`;

test("global integration products bind generator and product baselines exactly", () => {
  const inventory = fixtureInventory();
  const products = parseIntegrationProductRegistry(productRegistry(), inventory);
  assert.deepEqual([...products.keys()], ["wasm-registry"]);

  const badGenerator = productRegistry();
  badGenerator["wasm-registry"].generator.baseline_digest = D3;
  assert.throws(
    () => parseIntegrationProductRegistry(badGenerator, inventory),
    /generator does not match/,
  );

  const duplicatePath = productRegistry();
  duplicatePath["z-other"] = {
    path: "generated/registry.rs",
    producer: "integration",
    generator: { path: "scripts/generate.mjs", baseline_digest: D1 },
    baseline_digest: D2,
    verification: { kind: "content_identity" },
  };
  assert.throws(
    () => parseIntegrationProductRegistry(duplicatePath, inventory),
    /already owned/,
  );
});

test("bundle baseline evidence preserves type, absence, digest, and shared ownership", () => {
  const inventory = fixtureInventory();
  const expected = bundleBaselineEvidence(inventory, ["alpha", "beta"]);
  assert.deepEqual(expected.find((entry) => entry.path === "shared/runtime.rs"), {
    kind: "runtime-owner",
    path: "shared/runtime.rs",
    source_snapshot: "present",
    content_digest: D3,
    affected_identities: ["alpha", "beta"],
  });
  assert.deepEqual(expected.find((entry) => entry.path === "planned/provider.rs"), {
    kind: "provider",
    path: "planned/provider.rs",
    source_snapshot: "absent",
    content_digest: null,
    affected_identities: ["alpha"],
  });
  assert.deepEqual(
    validateCompleteBundleBaselineEvidence(expected, inventory, ["alpha", "beta"], "bundle"),
    expected,
  );

  const coarsened = structuredClone(expected);
  coarsened[0].kind = "runtime-owner";
  assert.throws(
    () => validateCompleteBundleBaselineEvidence(
      coarsened,
      inventory,
      ["alpha", "beta"],
      "bundle",
    ),
    /typed path evidence|canonically ordered/,
  );

  const removal = [{
    kind: "file",
    path: "shared/runtime.rs",
    baseline_digest: D3,
    affected_identities: ["alpha", "beta"],
  }];
  assert.deepEqual(
    parseBundleRemovals(removal, "bundle", ["alpha", "beta"], expected),
    removal,
  );
  removal[0].affected_identities = ["alpha"];
  assert.throws(
    () => parseBundleRemovals(removal, "bundle", ["alpha", "beta"], expected),
    /complete typed baseline evidence ownership/,
  );
});

test("integration references and dependency ownership resolve without worker ownership", () => {
  const inventory = fixtureInventory();
  const products = parseIntegrationProductRegistry(productRegistry(), inventory);
  const bundles = fixtureBundles();
  const identities = fixtureControls();
  assert.doesNotThrow(() => validateIntegrationProductCoverage(bundles, products));
  assert.doesNotThrow(() => validateGeneratedRegistryCoverage(
    inventory,
    bundles,
    identities,
    products,
  ));
  assert.doesNotThrow(() => validateAuthorityDependencyPolicy(bundles, identities, products));

  const forged = fixtureControls();
  forged.get("alpha").shared_dependencies[0].owner_id = "alpha-bundle";
  assert.throws(
    () => validateAuthorityDependencyPolicy(bundles, forged, products),
    /not a direct prerequisite/,
  );

  const wrongProductPath = fixtureControls();
  wrongProductPath.get("alpha").shared_dependencies[1].path = "generated/other.rs";
  assert.throws(
    () => validateAuthorityDependencyPolicy(bundles, wrongProductPath, products),
    /differs from wasm-registry/,
  );
});

test("subject authority validation requires every reviewed path and removal state", () => {
  const inventory = fixtureInventory();
  const control = {
    bundles: fixtureBundles(),
    identities: fixtureControls(),
  };
  control.bundles.get("alpha-bundle").expected_removals = [{ path: "legacy/alpha.json" }];
  control.bundles.get("alpha-bundle").integration_outputs = [{ path: "generated/registry.rs" }];
  inventory.source.files.push({ path: "legacy/alpha.json", content_digest: D1 });
  inventory.identities[0].semantic_authority.catalog_provenance = [{
    provenance: { source_file: "catalog/alpha.rs" },
  }];
  inventory.identities[0].semantic_authority.implementation_provenance = [{
    authority: "canonical_binding",
    source_file: "shared/runtime.rs",
    function: "alpha_builtin",
    binding_variant: "default",
    builtin_path: "builtins::alpha",
  }];
  assert.deepEqual(subjectAuthorityPathFailures(control, inventory), [
    "alpha-bundle: expected removal remains at legacy/alpha.json",
  ]);
  inventory.source.files = inventory.source.files.filter((entry) => entry.path !== "legacy/alpha.json");
  assert.deepEqual(subjectAuthorityPathFailures(control, inventory), []);
  inventory.source.files = inventory.source.files.filter((entry) => entry.path !== "catalog/alpha.rs");
  assert.match(subjectAuthorityPathFailures(control, inventory).join("\n"), /catalog package is absent/);
});

function productRegistry() {
  return {
    "wasm-registry": {
      path: "generated/registry.rs",
      producer: "integration",
      generator: { path: "scripts/generate.mjs", baseline_digest: D1 },
      baseline_digest: D2,
      verification: { kind: "content_identity" },
    },
  };
}

function fixtureInventory() {
  return {
    source: { files: [
      { path: "scripts/generate.mjs", content_digest: D1 },
      { path: "generated/registry.rs", content_digest: D2 },
      { path: "shared/runtime.rs", content_digest: D3 },
      { path: "catalog/alpha.rs", content_digest: D1 },
    ] },
    identities: [{
      identity: "alpha",
      ownership: { catalog: ["catalog/alpha.rs"], runtime: ["shared/runtime.rs"] },
      semantic_authority: { catalog_provenance: [], implementation_provenance: [] },
      registrations: { runtime: [], native_link: {} },
      documentation: {}, tests: {},
      dependencies: { generated_registry: ["generated/registry.rs"] },
      provider: { gpu_or_wgpu_paths: ["planned/provider.rs"], fusion_paths: [] },
    }, {
      identity: "beta",
      ownership: { catalog: [], runtime: ["shared/runtime.rs"] },
      semantic_authority: { catalog_provenance: [], implementation_provenance: [] },
      registrations: { runtime: [], native_link: {} },
      documentation: {}, tests: {}, dependencies: {}, provider: {},
    }],
  };
}

function fixtureBundles() {
  return new Map([
    ["foundation", {
      id: "foundation",
      prerequisites: [],
      authored_write_set: [{ kind: "file", path: "shared/runtime.rs" }],
      integration_product_refs: [],
      expected_removals: [],
      integration_outputs: [],
    }],
    ["alpha-bundle", {
      id: "alpha-bundle",
      prerequisites: [{ bundle_id: "foundation", kind: "infrastructure" }],
      authored_write_set: [{ kind: "file", path: "catalog/alpha.rs" }],
      integration_product_refs: ["wasm-registry"],
      expected_removals: [],
      integration_outputs: [{ path: "generated/registry.rs" }],
    }],
  ]);
}

function fixtureControls() {
  return new Map([["alpha", {
    bundle_id: "alpha-bundle",
    public_identity: {
      kind: "primary",
      primary_spelling: { identity: "alpha", spelling: "alpha" },
    },
    forms: {
      kind: "callable",
      callable_spellings: ["alpha"],
      constant_spellings: [],
    },
    implementation: {
      callable: {
        kind: "owned",
        owner_path: "shared/runtime.rs",
        bindings: [{
          kind: "canonical_binding",
          function: "alpha_builtin",
          variant: "default",
          builtin_path: "builtins::alpha",
          native_symbol: "runmat_builtin_binding_v1_616c706861_64656661756c74",
        }],
      },
      constant: { kind: "none", reason: "no-constant-form" },
    },
    shared_dependencies: [{
      kind: "runtime-owner",
      path: "shared/runtime.rs",
      ownership: "prerequisite",
      owner_id: "foundation",
    }, {
      kind: "registry",
      path: "generated/registry.rs",
      ownership: "integration",
      owner_id: "wasm-registry",
    }],
    expected_authorities: {
      catalog_package: "catalog/alpha.rs",
      catalog_entry_count: 1,
      catalog_constant_count: 0,
      documentation: "catalog",
      native_link: "not-applicable",
      wasm_registry: "not-applicable",
    },
  }]]);
}
