import assert from "node:assert/strict";
import test from "node:test";

import { evidenceDigest } from "../../evidence.mjs";
import { buildComponentIndex } from "../components.mjs";
import { validateTopologyClaims } from "../components.mjs";
import { applyTopologyCorrections, parseStabilityCorrectionArtifact } from "../corrections.mjs";

const BASELINE = {
  revision: `git:${"1".repeat(40)}`,
  inventory_digest: `sha256:${"a".repeat(64)}`,
  component_graph_digest: `sha256:${"b".repeat(64)}`,
  control_draft_digest: `sha256:${"c".repeat(64)}`,
};
const REVIEW_DIGESTS = {
  c01_c03: `sha256:${"d".repeat(64)}`,
  c04_c05: `sha256:${"e".repeat(64)}`,
  c06_c07: `sha256:${"f".repeat(64)}`,
};
const RECONCILIATION_DIGEST = `sha256:${"9".repeat(64)}`;

test("corrections preserve components while assigning exact identity-local targets", () => {
  const { components, claims } = fixture();
  const corrections = [{
    component: "component-mixed",
    from: endpoint("c04-mixed", "C04", [target("arraydatastore", "table", "core_and_selectors"), target("table", "table", "core_and_selectors")]),
    to: endpoint("c04-mixed", "C04", [target("arraydatastore", "io", "datastore"), target("table", "table", "core")]),
  }];

  const compositions = new Map([
    ["c04-mixed", {
      kind: "mixed-family-component",
      target_packages: [{ domain: "io", family: "datastore" }, { domain: "table", family: "core" }],
      authored_write_set: [{ kind: "file", path: "crates/runmat-runtime/src/builtins/table/mixed.rs" }],
      shared_authority_sources: ["crates/runmat-runtime/src/builtins/table/mixed.rs"],
      evidence: ["reviewed test composition"],
    }],
    ["c04-table", {
      kind: "single-component",
      target_packages: [{ domain: "table", family: "core" }],
      authored_write_set: [{ kind: "file", path: "crates/runmat-runtime/src/builtins/table/core/uitable.rs" }],
      shared_authority_sources: [],
      evidence: ["reviewed test composition"],
    }],
  ]);
  const result = applyTopologyCorrections(components, claims, corrections, { compositions });
  assert.equal(result.validation.conserved, true);
  assert.deepEqual(result.validation.corrected_components, ["component-mixed"]);
  assert.deepEqual(result.claims.map((claim) => claim.bundle), ["c04-mixed", "c04-table"]);
  assert.deepEqual(result.targets, [
    { identity: "arraydatastore", component: "component-mixed", bundle: "c04-mixed", cohort: "C04", domain: "io", family: "datastore" },
    { identity: "table", component: "component-mixed", bundle: "c04-mixed", cohort: "C04", domain: "table", family: "core" },
    { identity: "uitable", component: "component-table", bundle: "c04-table", cohort: "C04", domain: "table", family: "core" },
  ]);
});

test("corrections reject stale family state and partial component destinations", () => {
  const { components, claims } = fixture();
  const stale = [{
    component: "component-mixed",
    from: endpoint("c04-mixed", "C04", [target("arraydatastore", "table", "core"), target("table", "table", "core")]),
    to: endpoint("c04-mixed", "C04", [target("arraydatastore", "io", "datastore"), target("table", "table", "core")]),
  }];
  assert.throws(() => applyTopologyCorrections(components, claims, stale), /stale from identity target state/);

  const partial = structuredClone(stale);
  partial[0].from = endpoint("c04-mixed", "C04", [target("arraydatastore", "table", "core_and_selectors"), target("table", "table", "core_and_selectors")]);
  partial[0].to = endpoint("c04-mixed", "C04", [target("table", "table", "core")]);
  assert.throws(() => applyTopologyCorrections(components, claims, partial), /exact frozen component identities/);
});

test("corrections reject cohort-crossing merges and default policy fields", () => {
  const { components, claims } = fixture();
  const crossing = [{
    component: "component-mixed",
    from: endpoint("c04-mixed", "C04", [target("arraydatastore", "table", "core_and_selectors"), target("table", "table", "core_and_selectors")]),
    to: endpoint("c04-table", "C05", [target("arraydatastore", "io", "datastore"), target("table", "table", "core")]),
  }];
  assert.throws(() => applyTopologyCorrections(components, claims, crossing), /has cohort C04, not C05/);

  const withDefault = structuredClone(crossing);
  withDefault[0].to.default_family = "core";
  assert.throws(() => applyTopologyCorrections(components, claims, withDefault), /to fields must be exactly/);
});

test("corrections reject rows that do not change topology", () => {
  const { components, claims } = fixture();
  const currentTargets = [target("arraydatastore", "table", "core_and_selectors"), target("table", "table", "core_and_selectors")];
  const noop = [{ component: "component-mixed", from: endpoint("c04-mixed", "C04", currentTargets), to: endpoint("c04-mixed", "C04", currentTargets) }];
  assert.throws(() => applyTopologyCorrections(components, claims, noop), /must change bundle, cohort, or identity-local target/);
});

test("V2 correction artifact binds reconciliation and exact pre/from state", () => {
  const { components, claims } = fixture();
  const value = correctionArtifact(components, claims);
  const parsed = parseStabilityCorrectionArtifact(value, correctionContext(components, claims));
  assert.equal(parsed.digest, evidenceDigest(value));
  assert.equal(parsed.result.validation.conserved, true);
  assert.deepEqual(parsed.corrections.map((entry) => entry.id), ["correct-mixed-family"]);
});

test("V2 correction artifact rejects unknown fields and stale evidence bindings", () => {
  const { components, claims } = fixture();
  const context = correctionContext(components, claims);

  const unknown = correctionArtifact(components, claims);
  unknown.defaults = { family: "core" };
  assert.throws(() => parseStabilityCorrectionArtifact(unknown, context), /fields must be exactly/);

  const reconciliation = correctionArtifact(components, claims);
  reconciliation.reconciliation_digest = `sha256:${"0".repeat(64)}`;
  assert.throws(() => parseStabilityCorrectionArtifact(reconciliation, context), /bind the expected reconciliation/);

  const preState = correctionArtifact(components, claims);
  preState.pre_state_digest = `sha256:${"0".repeat(64)}`;
  assert.throws(() => parseStabilityCorrectionArtifact(preState, context), /pre-state digest does not match/);
});

test("V2 correction artifact rejects stale from state and unreviewed correction shapes", () => {
  const { components, claims } = fixture();
  const context = correctionContext(components, claims);

  const stale = correctionArtifact(components, claims);
  stale.corrections[0].from.identity_targets[0].family = "changed";
  assert.throws(() => parseStabilityCorrectionArtifact(stale, context), /stale from identity target state/);

  const selector = correctionArtifact(components, claims);
  selector.corrections[0].selector = "table.*";
  assert.throws(() => parseStabilityCorrectionArtifact(selector, context), /fields must be exactly/);

  const classification = correctionArtifact(components, claims);
  classification.corrections[0].classification = "implicit";
  assert.throws(() => parseStabilityCorrectionArtifact(classification, context), /classification must be one of/);
});

test("V2 correction artifact requires exact reviewed resulting bundle metadata", () => {
  const { components, claims } = fixture();
  const context = correctionContext(components, claims);

  const missing = correctionArtifact(components, claims);
  missing.bundle_updates = [];
  assert.throws(() => parseStabilityCorrectionArtifact(missing, context), /exactly enumerate every affected surviving bundle/);

  const missingOwner = correctionArtifact(components, claims);
  missingOwner.bundle_updates[0].composition.shared_authority_sources = [];
  assert.throws(() => parseStabilityCorrectionArtifact(missingOwner, context), /requires exact shared authority source evidence/);

  const staleMembership = correctionArtifact(components, claims);
  staleMembership.bundle_updates[0].authority_components = ["component-table"];
  assert.throws(() => parseStabilityCorrectionArtifact(staleMembership, context), /identities must equal the exact union|full bundle update differs/);
});

test("V2 identity target amendments preserve topology and bind complete component metadata", () => {
  const { components, claims } = fixture();
  const context = correctionContext(components, claims);
  const value = correctionArtifact(components, claims);
  value.corrections = [];
  value.identity_target_amendments = [targetAmendment()];
  value.bundle_updates = [amendedTableBundle()];

  const parsed = parseStabilityCorrectionArtifact(value, context);
  assert.deepEqual(parsed.identityTargetAmendments.map((entry) => entry.component), ["component-table"]);
  assert.deepEqual(parsed.result.claims, validateTopologyClaims(components, claims, {
    requireComplete: true,
    requireCompositions: false,
  }).claims);

  const partial = structuredClone(value);
  partial.identity_target_amendments[0].to_identity_targets[0].identity = "table";
  assert.throws(() => parseStabilityCorrectionArtifact(partial, context), /exact frozen component identities/);

  const stale = structuredClone(value);
  stale.identity_target_amendments[0].from_identity_targets[0].classification = "preserved";
  assert.throws(() => parseStabilityCorrectionArtifact(stale, context), /stale identity target metadata pre-state/);

  const packageDrift = structuredClone(value);
  packageDrift.identity_target_amendments[0].to_identity_targets[0].family = "selectors";
  assert.throws(() => parseStabilityCorrectionArtifact(packageDrift, context), /cannot change identity, domain, or family/);

  const undeclaredDrift = structuredClone(value);
  undeclaredDrift.bundle_updates[0].atomic_reason = "Undeclared metadata rewrite";
  assert.throws(() => parseStabilityCorrectionArtifact(undeclaredDrift, context), /exact amended projection/);
});

function fixture() {
  const components = buildComponentIndex([
    { id: "component-mixed", identities: ["arraydatastore", "table"] },
    { id: "component-table", identities: ["uitable"] },
  ]);
  const claims = new Map([
    ["c04-mixed", claim("c04-mixed", ["component-mixed"], ["arraydatastore", "table"], [target("arraydatastore", "table", "core_and_selectors"), target("table", "table", "core_and_selectors")])],
    ["c04-table", claim("c04-table", ["component-table"], ["uitable"], [target("uitable", "table", "core")])],
  ]);
  return { components, claims };
}

function claim(bundle, components, identities, identityTargets) { return { bundle, cohort: "C04", components, identities, identity_targets: identityTargets }; }
function endpoint(bundle, cohort, identityTargets) { return { bundle, cohort, identity_targets: identityTargets }; }
function target(name, domain, family) { return { identity: name, domain, family }; }

function correctionArtifact(components, claims) {
  const before = validateTopologyClaims(components, claims, { requireComplete: true, requireCompositions: false });
  return {
    schema_version: 2,
    kind: "runmat-builtin-topology-stability-corrections",
    authority: "reviewer-authored-development-input",
    program: "RM-1064/C00-C07",
    baseline: structuredClone(BASELINE),
    review_digests: structuredClone(REVIEW_DIGESTS),
    reconciliation_digest: RECONCILIATION_DIGEST,
    pre_state_digest: evidenceDigest(before.claims),
    corrections: [{
      id: "correct-mixed-family",
      component: "component-mixed",
      from: endpoint("c04-mixed", "C04", [target("arraydatastore", "table", "core_and_selectors"), target("table", "table", "core_and_selectors")]),
      to: endpoint("c04-mixed", "C04", [target("arraydatastore", "io", "datastore"), target("table", "table", "core")]),
      classification: "semantic-correction",
      reason: "Restore exact identity-local target packages inside the shared owner",
      evidence: ["fixture stability review"],
    }],
    identity_target_amendments: [],
    bundle_updates: [{
      id: "c04-mixed",
      cohort: "C04",
      authority_components: ["component-mixed"],
      identities: ["arraydatastore", "table"],
      atomic_reason: "One shared frozen owner spans exact identity-local target families",
      composition: {
        kind: "mixed-family-component",
        target_packages: [{ domain: "io", family: "datastore" }, { domain: "table", family: "core" }],
        authored_write_set: [
          { kind: "tree", path: "crates/runmat-builtins/src/catalog/entries/io/datastore" },
          { kind: "tree", path: "crates/runmat-builtins/src/catalog/entries/table/core" },
        ],
        shared_authority_sources: ["crates/runmat-runtime/src/builtins/table/shared.rs"],
        evidence: ["fixture composition review"],
      },
      identity_targets: [
        { ...target("arraydatastore", "io", "datastore"), classification: "semantic-correction", evidence: ["fixture target review"] },
        { ...target("table", "table", "core"), classification: "semantic-correction", evidence: ["fixture target review"] },
      ],
      review: { status: "reviewed", evidence: ["fixture bundle review"] },
    }],
    deleted_bundles: [],
    review: { status: "reviewed", evidence: ["fixture root review"] },
  };
}

function correctionContext(components, claims) {
  return {
    baseline: BASELINE,
    reviewDigests: REVIEW_DIGESTS,
    reconciliationDigest: RECONCILIATION_DIGEST,
    componentIndex: components,
    claims,
    bundles: priorBundles(),
  };
}

function priorBundles() {
  return new Map([
    ["c04-mixed", {
      id: "c04-mixed",
      cohort: "C04",
      authority_components: ["component-mixed"],
      identities: ["arraydatastore", "table"],
      atomic_reason: "One shared frozen owner spans exact identity-local target families",
      composition: {
        kind: "single-component",
        target_packages: [{ domain: "table", family: "core_and_selectors" }],
        authored_write_set: [],
        shared_authority_sources: [],
        evidence: ["fixture prior composition review"],
      },
      identity_targets: [
        reviewedTarget("arraydatastore", "table", "core_and_selectors"),
        reviewedTarget("table", "table", "core_and_selectors"),
      ],
      review: { status: "reviewed", evidence: ["fixture prior bundle review"] },
    }],
    ["c04-table", tableBundle()],
  ]);
}

function tableBundle() {
  return {
    id: "c04-table",
    cohort: "C04",
    authority_components: ["component-table"],
    identities: ["uitable"],
    atomic_reason: "One complete frozen authority component owns this target package",
    composition: {
      kind: "single-component",
      target_packages: [{ domain: "table", family: "core" }],
      authored_write_set: [],
      shared_authority_sources: [],
      evidence: ["fixture table composition review"],
    },
    identity_targets: [reviewedTarget("uitable", "table", "core")],
    review: { status: "reviewed", evidence: ["fixture table bundle review"] },
  };
}

function amendedTableBundle() {
  const value = tableBundle();
  value.identity_targets[0].classification = "normalization";
  value.identity_targets[0].evidence = ["fixture explicit normalization review"];
  return value;
}

function targetAmendment() {
  return {
    id: "classify-uitable-normalization",
    component: "component-table",
    bundle: "c04-table",
    cohort: "C04",
    from_identity_targets: [reviewedTarget("uitable", "table", "core")],
    to_identity_targets: [{
      ...target("uitable", "table", "core"),
      classification: "normalization",
      evidence: ["fixture explicit normalization review"],
    }],
    reason: "Record the reviewed normalization without changing topology",
    evidence: ["fixture identity target amendment review"],
  };
}

function reviewedTarget(name, domain, family) {
  return {
    ...target(name, domain, family),
    classification: "newly-classified",
    evidence: ["fixture unresolved baseline review"],
  };
}
