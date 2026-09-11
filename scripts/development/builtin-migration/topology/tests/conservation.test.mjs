import assert from "node:assert/strict";
import test from "node:test";

import { buildAuthorityComponentGraph, buildComponentIndex, validateTopologyClaims } from "../components.mjs";

test("authority graph uses transitive catalog/runtime ownership and ignores non-authority paths", () => {
  const inventory = {
    digest: `sha256:${"a".repeat(64)}`,
    identities: [
      inventoryRow("alpha", { runtime: ["runtime/shared-a.rs"], sidecar: "docs/shared.json", provider: "provider/shared.rs", category: "math/basic" }),
      inventoryRow("beta", { runtime: ["runtime/shared-a.rs", "runtime/shared-b.rs"], sidecar: "docs/beta.json", provider: "provider/beta.rs", category: "math/basic" }),
      inventoryRow("gamma", { runtime: ["runtime/shared-b.rs"], sidecar: "docs/gamma.json", provider: "provider/gamma.rs", category: "math/other" }),
      inventoryRow("sidecar_only", { runtime: ["runtime/sidecar.rs"], sidecar: "docs/shared.json", provider: "provider/sidecar.rs", category: "io/data" }),
      inventoryRow("provider_only", { runtime: ["runtime/provider.rs"], sidecar: "docs/provider.json", provider: "provider/shared.rs", category: "acceleration/gpu" }),
    ],
  };
  const graph = buildAuthorityComponentGraph(inventory);
  assert.deepEqual(graph.summary, { identities: 5, components: 3, singleton_components: 2, shared_components: 1, maximum_component_size: 3 });
  assert.deepEqual(graph.candidates.map((candidate) => [candidate.candidate_id, candidate.identities]), [
    ["component-alpha", ["alpha", "beta", "gamma"]],
    ["component-provider_only", ["provider_only"]],
    ["component-sidecar_only", ["sidecar_only"]],
  ]);
  assert.deepEqual(graph.candidates[0].shared_sources, ["runtime/shared-a.rs", "runtime/shared-b.rs"]);
  assert.deepEqual(graph.candidates[0].category_values, ["math/basic", "math/other"]);
});

test("component claims conserve exact frozen sets and permit reviewed family packing", () => {
  const components = buildComponentIndex([
    { id: "component-beta", identities: ["b"] },
    { id: "component-alpha", identities: ["a"] },
  ]);
  const claims = new Map([["c01-math-basic", {
    bundle: "c01-math-basic", cohort: "C01",
    components: ["component-beta", "component-alpha"], identities: ["b", "a"],
    identity_targets: [target("b", "math", "basic"), target("a", "math", "basic")],
  }]]);
  const compositions = new Map([["c01-math-basic", composition("shared-target-package", ["math/basic"], [
    { kind: "tree", path: "crates/runmat-builtins/src/catalog/definitions/math/basic" },
  ])]]);

  const result = validateTopologyClaims(components, claims, { compositions });
  assert.deepEqual(result.claims[0].components, ["component-alpha", "component-beta"]);
  assert.deepEqual(result.claims[0].identities, ["a", "b"]);
  assert.deepEqual(result.summary, { components: 2, claimed_components: 2, identities: 2, claimed_identities: 2, bundles: 1 });
});

test("one indivisible component may retain identity-local target packages", () => {
  const components = buildComponentIndex([{ id: "component-table-owner", identities: ["arraydatastore", "table", "uitable"] }]);
  const claims = new Map([["c04-table-owner", {
    bundle: "c04-table-owner", cohort: "C04", components: ["component-table-owner"],
    identities: ["arraydatastore", "table", "uitable"],
    identity_targets: [
      target("arraydatastore", "io", "datastore"),
      target("table", "table", "core"),
      target("uitable", "plotting", "ui"),
    ],
  }]]);
  const compositions = new Map([["c04-table-owner", {
    kind: "mixed-family-component",
    target_packages: [
      { domain: "io", family: "datastore" },
      { domain: "plotting", family: "ui" },
      { domain: "table", family: "core" },
    ],
    authored_write_set: [{ kind: "file", path: "crates/runmat-runtime/src/builtins/table_owner.rs" }],
    shared_authority_sources: ["crates/runmat-runtime/src/builtins/table_owner.rs"],
    evidence: ["reviewed test composition"],
  }]]);
  assert.doesNotThrow(() => validateTopologyClaims(components, claims, { compositions }));
});

test("component conservation rejects partial moves, missing targets, and duplicate ownership", () => {
  const components = buildComponentIndex([
    { id: "component-alpha", identities: ["a", "aa"] },
    { id: "component-beta", identities: ["b"] },
  ]);
  const partial = new Map([["c01-alpha", {
    bundle: "c01-alpha", cohort: "C01", components: ["component-alpha"], identities: ["a"],
    identity_targets: [target("a", "math", "basic")],
  }]]);
  assert.throws(() => validateTopologyClaims(components, partial, { requireComplete: false }), /exact union/);

  const missingTarget = singleClaim("component-alpha", ["a", "aa"], [target("a", "math", "basic")]);
  assert.throws(() => validateTopologyClaims(components, missingTarget, { requireComplete: false }), /enumerate every claimed identity/);

  const duplicated = new Map([
    ...singleClaim("component-alpha", ["a", "aa"], [target("a", "math", "basic"), target("aa", "math", "basic")]),
    ["c01-alpha-copy", { bundle: "c01-alpha-copy", cohort: "C01", components: ["component-alpha"], identities: ["a", "aa"], identity_targets: [target("a", "math", "basic"), target("aa", "math", "basic")] }],
  ]);
  assert.throws(() => validateTopologyClaims(components, duplicated, {
    requireComplete: false,
    requireCompositions: false,
  }), /claimed by multiple bundles/);
});

test("multi-component grouping requires exact package and write-set evidence, never category or size", () => {
  const components = buildComponentIndex([
    { id: "component-alpha", identities: ["a"] },
    { id: "component-beta", identities: ["b"] },
  ]);
  const claims = new Map([["c01-packed", {
    bundle: "c01-packed", cohort: "C01", components: ["component-alpha", "component-beta"], identities: ["a", "b"],
    identity_targets: [target("a", "math", "basic"), target("b", "math", "basic")],
  }]]);
  assert.throws(() => validateTopologyClaims(components, claims), /requires its reviewed composition/);
  const categoryEvidence = new Map([["c01-packed", {
    kind: "shared-target-package", target_packages: [{ domain: "math", family: "basic" }],
    authored_write_set: [{ kind: "tree", path: "crates/runmat-builtins/src/catalog/definitions/math/basic" }],
    shared_authority_sources: [], evidence: ["category-and-size"], category: "math/basic",
  }]]);
  assert.throws(() => validateTopologyClaims(components, claims, { compositions: categoryEvidence }), /fields must be exactly/);

  const crossPackage = structuredClone(claims.get("c01-packed"));
  crossPackage.identity_targets[1] = target("b", "math", "other");
  const crossPackageClaims = new Map([["c01-packed", crossPackage]]);
  const crossPackageEvidence = new Map([["c01-packed", composition("shared-target-package", ["math/basic", "math/other"], [
    { kind: "tree", path: "crates/runmat-builtins/src/catalog/definitions/math" },
  ])]]);
  assert.throws(() => validateTopologyClaims(components, crossPackageClaims, { compositions: crossPackageEvidence }), /cannot be combined across target packages/);
});

function target(name, domain, family) { return { identity: name, domain, family }; }

function composition(kind, packages, authoredWriteSet) {
  return {
    kind,
    target_packages: packages.map((entry) => {
      const [domain, family] = entry.split("/");
      return { domain, family };
    }),
    authored_write_set: authoredWriteSet,
    shared_authority_sources: [],
    evidence: ["reviewed test composition"],
  };
}

function singleClaim(component, identities, identityTargets) {
  return new Map([["c01-alpha", { bundle: "c01-alpha", cohort: "C01", components: [component], identities, identity_targets: identityTargets }]]);
}

function inventoryRow(name, { runtime, sidecar, provider, category }) {
  const [domain, family] = category.split("/");
  return {
    identity: name,
    domain,
    family,
    ownership: { catalog: [], runtime, sidecars: [sidecar], catalog_documentation: [], runtime_documentation_shadows: [] },
    registrations: { runtime: [{ declared_category: category }] },
    provider: { gpu_or_wgpu_paths: [provider], fusion_paths: [] },
    semantic_authority: { catalog_entries: [], legacy_functions: [] },
  };
}
