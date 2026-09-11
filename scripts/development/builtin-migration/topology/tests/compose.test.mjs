import assert from "node:assert/strict";
import test from "node:test";

import { buildControlDraft } from "../../control-draft.mjs";
import { evidenceDigest } from "../../evidence.mjs";
import { buildAuthorityComponentGraph } from "../components.mjs";
import { composeTopologyCandidate } from "../compose.mjs";
import { claimDiscrepancies, reconcileTopologyClaims } from "../reconciliation.mjs";
import { parseCohortReview } from "../reviews.mjs";
import { topologyStateFromReviews } from "../state.mjs";

test("compose runs the exact reviewed topology pipeline deterministically", () => {
  const input = fixture();
  const first = composeTopologyCandidate(input);
  const second = composeTopologyCandidate(input);

  assert.deepEqual(first, second);
  assert.equal(first.summary.components, 8);
  assert.equal(first.summary.identities, 9);
  assert.equal(first.identities.alpha.bundle_id, "c01-math-core");
  assert.equal(first.identities.delta.bundle_id, "c01-math-core");
  assert.deepEqual(Object.keys(first.identities), ["alpha", "beta", "betaaux", "datetime", "delta", "gamma", "linalg", "parallel", "shape"]);
  assert.deepEqual(
    { domain: first.identities.beta.domain, family: first.identities.beta.family },
    { domain: "cells", family: "queries" },
  );
  assert.equal(first.inputs.reconciliation, evidenceDigest(input.reconciliationValue));
  assert.equal(first.inputs.stability_corrections, evidenceDigest(input.stabilityCorrectionsValue));
});

test("compose rejects stale bindings and stale full-bundle metadata", () => {
  const staleBinding = fixture();
  staleBinding.reconciliationValue.review_digests.c01_c03 = digest("0");
  assert.throws(() => composeTopologyCandidate(staleBinding), /review digests do not match/);

  const swappedRoles = fixture();
  const firstReview = swappedRoles.reviewValues.get("c01_c03");
  swappedRoles.reviewValues.set("c01_c03", swappedRoles.reviewValues.get("c06_c07"));
  swappedRoles.reviewValues.set("c06_c07", firstReview);
  assert.throws(() => composeTopologyCandidate(swappedRoles), /declares the wrong cohort set/);

  const staleReconciliationBundle = fixture();
  staleReconciliationBundle.reconciliationValue.bundle_updates[0].authority_components = ["component-alpha"];
  staleReconciliationBundle.reconciliationValue.bundle_updates[0].identities = ["alpha"];
  staleReconciliationBundle.reconciliationValue.bundle_updates[0].identity_targets = [reviewTarget("alpha", "math", "core")];
  assert.throws(
    () => composeTopologyCandidate(staleReconciliationBundle),
    /composition|full bundle update differs|exact union/,
  );

  const staleCorrectionState = fixture();
  staleCorrectionState.stabilityCorrectionsValue.pre_state_digest = digest("0");
  assert.throws(() => composeTopologyCandidate(staleCorrectionState), /pre-state digest does not match/);

  const fabricatedSharedOwner = fixture();
  fabricatedSharedOwner.stabilityCorrectionsValue.bundle_updates[0].composition.shared_authority_sources = ["crates/runmat-runtime/src/builtins/fixture/fabricated.rs"];
  assert.throws(() => composeTopologyCandidate(fabricatedSharedOwner), /differ from the frozen component graph/);
});

function fixture() {
  const baselineInventory = inventory();
  const componentGraph = buildAuthorityComponentGraph(baselineInventory);
  const controlDraft = buildControlDraft(baselineInventory);
  const baseline = {
    revision: baselineInventory.source.revision,
    inventory_digest: baselineInventory.digest,
    component_graph_digest: evidenceDigest(componentGraph),
    control_draft_digest: controlDraft.digest,
  };
  const reviewValues = new Map([
    ["c01_c03", cohortReview(baseline, ["C01", "C02", "C03"], [
      singleBundle("c01-math-core", "C01", "component-alpha", "alpha", "math", "core"),
      singleBundle("c02-shape-core", "C02", "component-shape", "shape", "arrays", "shape"),
      singleBundle("c03-linalg-core", "C03", "component-linalg", "linalg", "math", "linalg"),
    ])],
    ["c04_c05", cohortReview(baseline, ["C04", "C05"], [
      singleOwnerBundle("c04-cells-core", "C04", "component-beta", ["beta", "betaaux"], "cells", "core"),
      singleBundle("c05-parallel-core", "C05", "component-parallel", "parallel", "parallel", "core"),
    ])],
    ["c06_c07", cohortReview(baseline, ["C06", "C07"], [
      singleBundle("c06-io-core", "C06", "component-gamma", "gamma", "io", "core"),
      singleBundle("c07-datetime-core", "C07", "component-datetime", "datetime", "datetime", "core"),
    ])],
  ]);
  const componentIndex = new Map(componentGraph.candidates.map((entry) => [entry.candidate_id, entry.identities]));
  const parsedReviews = new Map([...reviewValues].map(([role, value]) => [role, parseCohortReview(value, componentIndex)]));
  const initial = topologyStateFromReviews(parsedReviews);
  const reviewDigests = Object.fromEntries([...parsedReviews].map(([role, review]) => [role, review.digest]));

  const destination = endpoint("c01-math-core", "C01", [target("delta", "math", "core")]);
  const reconciliationDecision = {
    component: "component-delta",
    issue: "missing",
    from: [],
    to: destination,
    reason: "The missing scalar identity shares the reviewed math/core package",
    evidence: ["fixture exact missing-component review"],
  };
  const reconciledBundle = sharedBundle(
    "c01-math-core",
    "C01",
    ["component-alpha", "component-delta"],
    ["alpha", "delta"],
    "math",
    "core",
  );
  const reconciliationValue = {
    schema_version: 2,
    kind: "runmat-builtin-topology-reconciliation",
    authority: "reviewer-authored-development-input",
    program: "RM-1064/C00-C07",
    baseline: structuredClone(baseline),
    review_digests: structuredClone(reviewDigests),
    discrepancy_digest: evidenceDigest(claimDiscrepancies(componentIndex, initial.claims)),
    decisions: [reconciliationDecision],
    bundle_updates: [reconciledBundle],
    deleted_bundles: [],
    review: reviewed("fixture reconciliation review"),
  };
  const reconciledClaims = reconcileTopologyClaims(componentIndex, initial.claims, [{
    component: reconciliationDecision.component,
    from: reconciliationDecision.from,
    to: reconciliationDecision.to,
  }], { requireCompositions: false });
  const reconciledClaimsMap = new Map(reconciledClaims.map((claim) => [claim.bundle, claim]));

  const correctionFrom = endpoint("c04-cells-core", "C04", [target("beta", "cells", "core"), target("betaaux", "cells", "core")]);
  const correctionTo = endpoint("c04-cells-core", "C04", [target("beta", "cells", "queries"), target("betaaux", "cells", "core")]);
  const stabilityCorrectionsValue = {
    schema_version: 2,
    kind: "runmat-builtin-topology-stability-corrections",
    authority: "reviewer-authored-development-input",
    program: "RM-1064/C00-C07",
    baseline: structuredClone(baseline),
    review_digests: structuredClone(reviewDigests),
    reconciliation_digest: evidenceDigest(reconciliationValue),
    pre_state_digest: evidenceDigest(reconciledClaims),
    corrections: [{
      id: "correct-beta-family",
      component: "component-beta",
      from: correctionFrom,
      to: correctionTo,
      classification: "semantic-correction",
      reason: "The predicate belongs to the precise cells query package",
      evidence: ["fixture semantic stability review"],
    }],
    identity_target_amendments: [],
    bundle_updates: [mixedOwnerBundle()],
    deleted_bundles: [],
    review: reviewed("fixture stability correction review"),
  };

  assert.equal(reconciledClaimsMap.get("c01-math-core").components.length, 2);
  return {
    baselineInventory,
    componentGraph,
    controlDraft,
    reviewValues,
    reconciliationValue,
    stabilityCorrectionsValue,
  };
}

function inventory() {
  const identities = ["alpha", "beta", "betaaux", "datetime", "delta", "gamma", "linalg", "parallel", "shape"].map((identity) => ({
    identity,
    disposition: { kind: "canonical", canonical: identity },
    ownership: { catalog: [], runtime: [`crates/runmat-runtime/src/builtins/fixture/${identity === "betaaux" ? "beta" : identity}.rs`] },
    semantic_authority: { catalog_entries: [], legacy_functions: [] },
    registrations: { runtime: [] },
    lexical_observations: { authority: "discovery_only", ownership: {}, registrations: {}, dependencies: {}, provider: {} },
  }));
  const payload = {
    schema_version: 2,
    kind: "runmat-builtin-migration-inventory",
    authority: "development-evidence-only",
    generated_from: ["fixture"],
    source: { revision: `git:${"1".repeat(40)}`, digest: digest("2") },
    compiled_inventory: { schema_version: 1, kind: "fixture", digest: digest("3"), build: { operating_system: "fixture-os", architecture: "fixture-arch" } },
    migration_findings: [],
    migration_findings_digest: evidenceDigest([]),
    scanned_source_coverage: { paths: [], digest: digest("4") },
    dispositions_digest: digest("5"),
    summary: { identities: identities.length },
    diagnostics: [],
    identities,
  };
  return { ...payload, digest: evidenceDigest(payload) };
}

function cohortReview(baseline, cohorts, bundles) {
  return {
    schema_version: 2,
    kind: "runmat-builtin-topology-cohort-review",
    authority: "reviewer-authored-development-input",
    program: "RM-1064/C00-C07",
    baseline: structuredClone(baseline),
    cohorts,
    bundles,
    ambiguities: [],
    review: reviewed(`fixture ${cohorts.join("-")} review`),
  };
}

function singleBundle(id, cohort, component, identity, domain, family, classification = "preserved") {
  return {
    id,
    cohort,
    authority_components: [component],
    identities: [identity],
    atomic_reason: "One complete frozen authority component owns this reviewed target package.",
    composition: {
      kind: "single-component",
      target_packages: [{ domain, family }],
      authored_write_set: [],
      shared_authority_sources: [],
      evidence: [`fixture ${id} composition`],
    },
    identity_targets: [reviewTarget(identity, domain, family, classification)],
    review: reviewed(`fixture ${id} bundle review`),
  };
}

function sharedBundle(id, cohort, components, identities, domain, family) {
  return {
    id,
    cohort,
    authority_components: components,
    identities,
    atomic_reason: "Independent complete components share one reviewed package and authored tree scope.",
    composition: {
      kind: "shared-target-package",
      target_packages: [{ domain, family }],
      authored_write_set: [{ kind: "tree", path: `crates/runmat-builtins/src/catalog/entries/${domain}/${family}` }],
      shared_authority_sources: [],
      evidence: [`fixture ${id} shared composition`],
    },
    identity_targets: identities.map((identity) => reviewTarget(identity, domain, family)),
    review: reviewed(`fixture ${id} bundle review`),
  };
}

function singleOwnerBundle(id, cohort, component, identities, domain, family) {
  return {
    id,
    cohort,
    authority_components: [component],
    identities,
    atomic_reason: "One complete frozen authority component owns this reviewed target package.",
    composition: {
      kind: "single-component",
      target_packages: [{ domain, family }],
      authored_write_set: [],
      shared_authority_sources: [],
      evidence: [`fixture ${id} composition`],
    },
    identity_targets: identities.map((identity) => reviewTarget(identity, domain, family)),
    review: reviewed(`fixture ${id} bundle review`),
  };
}

function mixedOwnerBundle() {
  return {
    id: "c04-cells-core",
    cohort: "C04",
    authority_components: ["component-beta"],
    identities: ["beta", "betaaux"],
    atomic_reason: "One complete frozen authority component spans two identity-local target packages.",
    composition: {
      kind: "mixed-family-component",
      target_packages: [{ domain: "cells", family: "core" }, { domain: "cells", family: "queries" }],
      authored_write_set: [],
      shared_authority_sources: ["crates/runmat-runtime/src/builtins/fixture/beta.rs"],
      evidence: ["fixture mixed-family composition"],
    },
    identity_targets: [
      reviewTarget("beta", "cells", "queries", "semantic-correction"),
      reviewTarget("betaaux", "cells", "core"),
    ],
    review: reviewed("fixture mixed-family bundle review"),
  };
}

function endpoint(bundle, cohort, identityTargets) {
  return { bundle, cohort, identity_targets: identityTargets };
}

function target(identity, domain, family) {
  return { identity, domain, family };
}

function reviewTarget(identity, domain, family, classification = "preserved") {
  return { ...target(identity, domain, family), classification, evidence: [`fixture ${identity} target review`] };
}

function reviewed(evidence) {
  return { status: "reviewed", evidence: [evidence] };
}

function digest(character) {
  return `sha256:${character.repeat(64)}`;
}
