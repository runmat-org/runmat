import { bundleBaselineEvidence } from "../baseline-evidence.mjs";
import { MATURITY_GATES } from "../control.mjs";
import { fixtureGatePlans } from "./helpers.mjs";

export const SEQUENTIAL_IDENTITIES = ["alpha", "beta"];
export const SEQUENTIAL_BUNDLES = ["math-reduction-alpha", "math-reduction-beta"];
export const SEQUENTIAL_FAMILY = "reduction";

export function sequentialDispositionInput() {
  return {
    schema_version: 1,
    kind: "runmat-builtin-dispositions",
    authority: "review-input-only",
    identities: Object.fromEntries(SEQUENTIAL_IDENTITIES.map((identity) => [identity, {
      disposition: "canonical",
      canonical: null,
      domain: "math",
      family: SEQUENTIAL_FAMILY,
      reason: null,
      review: { status: "reviewed", evidence: ["sequential fixture review"] },
    }])),
  };
}

export function compositionChild(identity) {
  return {
    module: `${identity}_support`,
    source_kind: "directory",
    source_path: `crates/runmat-runtime/src/builtins/math/${SEQUENTIAL_FAMILY}/${identity}_support/mod.rs`,
    role: "group",
    visibility: "public",
    declaration_condition: { kind: "always" },
    declaration_order: identity === "alpha" ? 2 : 3,
    macro_use: false,
    reexports: [],
    aggregation_sources: [],
  };
}

export function sequentialBundleControls(inventory, { parallelBundles = false } = {}) {
  const gatePlans = sequentialGatePlans(inventory);
  return new Map(SEQUENTIAL_BUNDLES.map((bundleId, index) => {
    const identity = SEQUENTIAL_IDENTITIES[index];
    const child = compositionChild(identity);
    return [bundleId, {
      prerequisites: index === 0 || parallelBundles
        ? [] : [{ bundle_id: SEQUENTIAL_BUNDLES[0], kind: "semantic" }],
      additional_authored_write_set: [
        { kind: "tree", path: `crates/runmat-builtins/src/catalog/entries/math/${SEQUENTIAL_FAMILY}/${identity}` },
        { kind: "file", path: `crates/runmat-runtime/src/builtins/math/${SEQUENTIAL_FAMILY}/${identity}.rs` },
        { kind: "tree", path: child.source_path.slice(0, -"/mod.rs".length) },
      ],
      integration_product_refs: ["runtime-math-reduction", "wasm-registry"],
      module_composition_transition: {
        schema_version: 5,
        kind: "runmat-builtin-module-composition-transition",
        transition_id: bundleId,
        product_states: [{
          product_id: "runtime-math-reduction", before_state: "present", after_state: "present",
        }],
        changes: [{
          product_id: "runtime-math-reduction", operation: "add", before: null, after: child,
        }],
      },
      source_migrations: [],
      expected_removals: [],
      baseline_evidence: bundleBaselineEvidence(inventory, [identity]),
      gate_plans: structuredClone(gatePlans),
      owner_role: "builtin-migrator",
      complexity: { class: "low", weight: 1, basis: ["sequential shared-parent fixture"] },
      review: { status: "reviewed", evidence: ["sequential fixture review"] },
    }];
  }));
}

function sequentialGatePlans(inventory) {
  const plans = fixtureGatePlans(inventory);
  const architecture = plans.find((plan) => plan.gate === "architecture");
  plans.push({ ...structuredClone(architecture), gate: "wasm-registry" });
  return plans.sort((left, right) => left.gate.localeCompare(right.gate));
}

export function sequentialIdentityControls() {
  return new Map(SEQUENTIAL_IDENTITIES.map((identity, index) => [
    identity, identityControl(identity, SEQUENTIAL_BUNDLES[index]),
  ]));
}

function identityControl(identity, bundleId) {
  const required = new Set([
    "identity", "disposition", "catalog-contract", "runtime-binding", "documentation",
    "link-reachability", "wasm-registry",
  ]);
  const maturity = Object.fromEntries(MATURITY_GATES.map((gate) => [
    gate,
    required.has(gate)
      ? { applicability: "required", reason: null, evidence: [] }
      : { applicability: "not-applicable", reason: "Outside the sequential fixture", evidence: ["sequential fixture review"] },
  ]));
  return {
    public_identity: { kind: "primary", primary_spelling: { identity, spelling: identity } },
    forms: { kind: "callable", callable_spellings: [identity], constant_spellings: [] },
    implementation: {
      callable: {
        kind: "owned",
        owner_path: `crates/runmat-runtime/src/builtins/math/${SEQUENTIAL_FAMILY}/${identity}.rs`,
        bindings: [{
          kind: "canonical_binding", function: `${identity}_builtin`, variant: "default",
          builtin_path: `builtins::${identity}`, native_symbol: nativeSymbol(identity),
        }],
      },
      constant: { kind: "none", reason: "no-constant-form" },
    },
    shared_dependencies: [],
    complexity: { class: "low", weight: 1, basis: ["single identity"] },
    maturity,
    expected_authorities: {
      catalog_package: `crates/runmat-builtins/src/catalog/entries/math/${SEQUENTIAL_FAMILY}/${identity}/mod.rs`,
      catalog_alias_package: null,
      catalog_constant_package: null,
      catalog_entry_count: 1,
      catalog_constant_count: 0,
      documentation: "catalog",
      native_link: "required",
      wasm_registry: "required",
    },
    owner: "sequential-fixture",
    review: { status: "reviewed", evidence: ["sequential fixture review"] },
  };
}

function nativeSymbol(identity) {
  return `runmat_builtin_binding_v1_${Buffer.from(identity).toString("hex")}_${Buffer.from("default").toString("hex")}`;
}
