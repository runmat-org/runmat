import assert from "node:assert/strict";
import test from "node:test";

import { evidenceDigest } from "../../evidence.mjs";
import { buildComponentIndex } from "../components.mjs";
import { claimDiscrepancies, parseReconciliationArtifact, reconcileTopologyClaims } from "../reconciliation.mjs";

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

test("reconciliation replaces exactly the missing and multiply-claimed component set", () => {
  const { components, claims } = fixture();
  assert.deepEqual(claimDiscrepancies(components, claims), [
    { component: "component-beta", issue: "multiply-claimed", from: [
      endpoint("c01-beta", "C01", [target("b", "math", "basic"), target("bb", "math", "basic")]),
      endpoint("c04-beta", "C04", [target("b", "cells", "queries"), target("bb", "cells", "queries")]),
    ] },
    { component: "component-gamma", issue: "missing", from: [] },
  ]);
  const decisions = [
    decision("component-gamma", [], endpoint("c04-gamma", "C04", [target("g", "cells", "queries")])),
    decision("component-beta", [
      endpoint("c01-beta", "C01", [target("b", "math", "basic"), target("bb", "math", "basic")]),
      endpoint("c04-beta", "C04", [target("b", "cells", "queries"), target("bb", "cells", "queries")]),
    ], endpoint("c01-alpha", "C01", [target("b", "math", "basic"), target("bb", "math", "basic")])),
  ];
  const compositions = new Map([["c01-alpha", {
    kind: "shared-target-package",
    target_packages: [{ domain: "math", family: "basic" }],
    authored_write_set: [{ kind: "tree", path: "crates/runmat-builtins/src/catalog/definitions/math/basic" }],
    shared_authority_sources: [],
    evidence: ["reviewed test composition"],
  }], ["c04-gamma", {
    kind: "single-component",
    target_packages: [{ domain: "cells", family: "queries" }],
    authored_write_set: [{ kind: "file", path: "crates/runmat-runtime/src/builtins/cells/queries/gamma.rs" }],
    shared_authority_sources: [],
    evidence: ["reviewed test composition"],
  }]]);

  const result = reconcileTopologyClaims(components, claims, decisions, { compositions });
  assert.deepEqual(result.map((claim) => claim.bundle), ["c01-alpha", "c04-gamma"]);
  assert.deepEqual(result[0].components, ["component-alpha", "component-beta"]);
  assert.deepEqual(result[0].identities, ["a", "b", "bb"]);
});

test("reconciliation rejects incomplete decisions and stale source state", () => {
  const { components, claims } = fixture();
  const beta = decision("component-beta", [
    endpoint("c01-beta", "C01", [target("b", "math", "basic"), target("bb", "math", "basic")]),
    endpoint("c04-beta", "C04", [target("b", "cells", "queries"), target("bb", "cells", "queries")]),
  ], endpoint("c01-alpha", "C01", [target("b", "math", "basic"), target("bb", "math", "basic")]));
  assert.throws(() => reconcileTopologyClaims(components, claims, [beta]), /exact current discrepancy set/);

  const stale = [
    { ...beta, from: [
      endpoint("c01-beta", "C01", [target("b", "math", "changed"), target("bb", "math", "basic")]),
      endpoint("c04-beta", "C04", [target("b", "cells", "queries"), target("bb", "cells", "queries")]),
    ] },
    decision("component-gamma", [], endpoint("c04-gamma", "C04", [target("g", "cells", "queries")])),
  ];
  assert.throws(() => reconcileTopologyClaims(components, claims, stale), /stale reconciliation from state/);
});

test("reconciliation rejects partial components and selector-shaped policy", () => {
  const { components, claims } = fixture();
  const partial = [
    decision("component-beta", [
      endpoint("c01-beta", "C01", [target("b", "math", "basic"), target("bb", "math", "basic")]),
      endpoint("c04-beta", "C04", [target("b", "cells", "queries"), target("bb", "cells", "queries")]),
    ], endpoint("c01-alpha", "C01", [target("b", "math", "basic")])),
    decision("component-gamma", [], endpoint("c04-gamma", "C04", [target("g", "cells", "queries")])),
  ];
  assert.throws(() => reconcileTopologyClaims(components, claims, partial), /exact frozen component identities/);

  const selected = structuredClone(partial);
  selected[0].to.selector = "category:math";
  assert.throws(() => reconcileTopologyClaims(components, claims, selected), /destination fields must be exactly/);
});

test("V2 reconciliation artifact binds reviews and the exact discrepancy state", () => {
  const { components, claims } = fixture();
  const decisions = artifactDecisions();
  const value = reconciliationArtifact(components, claims, decisions);
  const parsed = parseReconciliationArtifact(value, {
    baseline: BASELINE,
    reviewDigests: REVIEW_DIGESTS,
    componentIndex: components,
    claims,
  });
  assert.equal(parsed.digest, evidenceDigest(value));
  assert.deepEqual(parsed.claims.map((claim) => claim.bundle), ["c01-alpha", "c04-gamma"]);
  assert.deepEqual(parsed.decisions.map((entry) => entry.component), ["component-beta", "component-gamma"]);
});

test("V2 reconciliation artifact rejects unknown fields and stale bindings", () => {
  const { components, claims } = fixture();
  const context = { baseline: BASELINE, reviewDigests: REVIEW_DIGESTS, componentIndex: components, claims };

  const unknown = reconciliationArtifact(components, claims, artifactDecisions());
  unknown.selector = "category:math";
  assert.throws(() => parseReconciliationArtifact(unknown, context), /fields must be exactly/);

  const reviews = reconciliationArtifact(components, claims, artifactDecisions());
  reviews.review_digests.c04_c05 = `sha256:${"0".repeat(64)}`;
  assert.throws(() => parseReconciliationArtifact(reviews, context), /review digests do not match/);

  const discrepancy = reconciliationArtifact(components, claims, artifactDecisions());
  discrepancy.discrepancy_digest = `sha256:${"0".repeat(64)}`;
  assert.throws(() => parseReconciliationArtifact(discrepancy, context), /discrepancy digest does not match/);
});

test("V2 reconciliation artifact rejects issue, from-state, and decision policy drift", () => {
  const { components, claims } = fixture();
  const context = { baseline: BASELINE, reviewDigests: REVIEW_DIGESTS, componentIndex: components, claims };

  const issue = reconciliationArtifact(components, claims, artifactDecisions());
  issue.decisions[0].issue = "missing";
  assert.throws(() => parseReconciliationArtifact(issue, context), /exact discrepancy\/from state/);

  const from = reconciliationArtifact(components, claims, artifactDecisions());
  from.decisions[0].from[0].identity_targets[0].family = "changed";
  assert.throws(() => parseReconciliationArtifact(from, context), /exact discrepancy\/from state/);

  const policy = reconciliationArtifact(components, claims, artifactDecisions());
  policy.decisions[0].default_family = "basic";
  assert.throws(() => parseReconciliationArtifact(policy, context), /fields must be exactly/);
});

test("V2 reconciliation artifact requires exact reviewed post-application bundle metadata", () => {
  const { components, claims } = fixture();
  const context = { baseline: BASELINE, reviewDigests: REVIEW_DIGESTS, componentIndex: components, claims };

  const missing = reconciliationArtifact(components, claims, artifactDecisions());
  missing.bundle_updates.pop();
  assert.throws(() => parseReconciliationArtifact(missing, context), /exactly enumerate every affected surviving bundle/);

  const inferredScope = reconciliationArtifact(components, claims, artifactDecisions());
  inferredScope.bundle_updates[0].composition.authored_write_set = [];
  assert.throws(() => parseReconciliationArtifact(inferredScope, context), /requires an explicit authored write set/);

  const metadata = reconciliationArtifact(components, claims, artifactDecisions());
  metadata.bundle_updates[0].identity_targets[0].family = "changed";
  assert.throws(() => parseReconciliationArtifact(metadata, context), /composition target packages|full bundle update differs/);
});

function fixture() {
  const components = buildComponentIndex([
    { id: "component-alpha", identities: ["a"] },
    { id: "component-beta", identities: ["b", "bb"] },
    { id: "component-gamma", identities: ["g"] },
  ]);
  const claims = new Map([
    ["c01-alpha", claim("c01-alpha", "C01", ["component-alpha"], ["a"], [target("a", "math", "basic")])],
    ["c01-beta", claim("c01-beta", "C01", ["component-beta"], ["b", "bb"], [target("b", "math", "basic"), target("bb", "math", "basic")])],
    ["c04-beta", claim("c04-beta", "C04", ["component-beta"], ["b", "bb"], [target("b", "cells", "queries"), target("bb", "cells", "queries")])],
  ]);
  return { components, claims };
}

function decision(component, from, to) { return { component, from, to }; }
function claim(bundle, cohort, components, identities, identityTargets) { return { bundle, cohort, components, identities, identity_targets: identityTargets }; }
function endpoint(bundle, cohort, identityTargets) { return { bundle, cohort, identity_targets: identityTargets }; }
function target(name, domain, family) { return { identity: name, domain, family }; }

function artifactDecisions() {
  return [
    {
      component: "component-beta",
      issue: "multiply-claimed",
      from: [
        endpoint("c01-beta", "C01", [target("b", "math", "basic"), target("bb", "math", "basic")]),
        endpoint("c04-beta", "C04", [target("b", "cells", "queries"), target("bb", "cells", "queries")]),
      ],
      to: endpoint("c01-alpha", "C01", [target("b", "math", "basic"), target("bb", "math", "basic")]),
      reason: "The reviewed scalar family owns the shared component",
      evidence: ["fixture cross-review evidence"],
    },
    {
      component: "component-gamma",
      issue: "missing",
      from: [],
      to: endpoint("c04-gamma", "C04", [target("g", "cells", "queries")]),
      reason: "The component was omitted from both source reviews",
      evidence: ["fixture boundary evidence"],
    },
  ];
}

function reconciliationArtifact(components, claims, decisions) {
  return {
    schema_version: 2,
    kind: "runmat-builtin-topology-reconciliation",
    authority: "reviewer-authored-development-input",
    program: "RM-1064/C00-C07",
    baseline: structuredClone(BASELINE),
    review_digests: structuredClone(REVIEW_DIGESTS),
    discrepancy_digest: evidenceDigest(claimDiscrepancies(components, claims)),
    decisions,
    bundle_updates: [
      reviewedBundle({
        id: "c01-alpha",
        cohort: "C01",
        components: ["component-alpha", "component-beta"],
        identities: ["a", "b", "bb"],
        targets: [target("a", "math", "basic"), target("b", "math", "basic"), target("bb", "math", "basic")],
        kind: "shared-target-package",
        authoredWriteSet: [{ kind: "tree", path: "crates/runmat-builtins/src/catalog/entries/math/basic" }],
      }),
      reviewedBundle({
        id: "c04-gamma",
        cohort: "C04",
        components: ["component-gamma"],
        identities: ["g"],
        targets: [target("g", "cells", "queries")],
        kind: "single-component",
      }),
    ],
    deleted_bundles: ["c01-beta", "c04-beta"],
    review: { status: "reviewed", evidence: ["fixture root review"] },
  };
}

function reviewedBundle({ id, cohort, components, identities, targets, kind, authoredWriteSet = [] }) {
  const packages = [...new Map(targets.map((entry) => [`${entry.domain}/${entry.family}`, { domain: entry.domain, family: entry.family }])).values()];
  return {
    id,
    cohort,
    authority_components: components,
    identities,
    atomic_reason: "Reviewed fixture atomicity",
    composition: {
      kind,
      target_packages: packages,
      authored_write_set: authoredWriteSet,
      shared_authority_sources: [],
      evidence: ["fixture composition review"],
    },
    identity_targets: targets.map((entry) => ({ ...entry, classification: "newly-classified", evidence: ["fixture target review"] })),
    review: { status: "reviewed", evidence: ["fixture bundle review"] },
  };
}
