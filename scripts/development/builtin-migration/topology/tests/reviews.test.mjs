import assert from "node:assert/strict";
import test from "node:test";

import { parseCohortReview } from "../reviews.mjs";

const SHA = `sha256:${"a".repeat(64)}`;
const COMPONENTS = new Map([
  ["component-alpha", ["alpha"]],
  ["component-beta", ["beta"]],
  ["component-mixed", ["mixed.alpha", "mixed.beta"]],
]);

test("cohort review v2 parses a complete single-component family bundle", () => {
  const parsed = parseCohortReview(review([singleBundle()]), COMPONENTS);
  assert.equal(parsed.bundles.size, 1);
  assert.equal(parsed.identityTargets.get("alpha").family, "basic");
  assert.match(parsed.digest, /^sha256:[a-f0-9]{64}$/);
});

test("independent components may share a family only with explicit composition and write-set evidence", () => {
  const bundle = {
    id: "c01-math-basic",
    cohort: "C01",
    authority_components: ["component-alpha", "component-beta"],
    identities: ["alpha", "beta"],
    atomic_reason: "The independent contracts compose into one authored math/basic target package.",
    composition: {
      kind: "shared-target-package",
      target_packages: [{ domain: "math", family: "basic" }],
      authored_write_set: [
        { kind: "file", path: "crates/runmat-runtime/src/builtins/math/basic.rs" },
        { kind: "tree", path: "crates/runmat-builtins/src/catalog/entries/math/basic" },
      ],
      shared_authority_sources: [],
      evidence: ["review:math-basic-package-composition"],
    },
    identity_targets: [target("alpha", "math", "basic"), target("beta", "math", "basic")],
    review: reviewed("review:c01-math-basic"),
  };
  assert.doesNotThrow(() => parseCohortReview(review([bundle]), COMPONENTS));

  const noWriteSet = clone(bundle);
  noWriteSet.composition.authored_write_set = [];
  assert.throws(() => parseCohortReview(review([noWriteSet]), COMPONENTS), /requires an explicit authored write set/);
});

test("one indivisible authority component may retain identity-local mixed-family targets", () => {
  const bundle = {
    id: "c04-mixed-owner",
    cohort: "C04",
    authority_components: ["component-mixed"],
    identities: ["mixed.alpha", "mixed.beta"],
    atomic_reason: "One frozen source owns both identities and cannot be split across leases.",
    composition: {
      kind: "mixed-family-component",
      target_packages: [
        { domain: "cells", family: "conversion" },
        { domain: "structs", family: "conversion" },
      ],
      authored_write_set: [],
      shared_authority_sources: ["crates/runmat-runtime/src/builtins/aggregate/shared.rs"],
      evidence: ["review:frozen-shared-owner"],
    },
    identity_targets: [
      target("mixed.alpha", "cells", "conversion"),
      target("mixed.beta", "structs", "conversion"),
    ],
    review: reviewed("review:c04-mixed-owner"),
  };
  assert.doesNotThrow(() => parseCohortReview(review([bundle], ["C04"]), COMPONENTS));

  const independentCrossFamily = clone(bundle);
  independentCrossFamily.authority_components = ["component-alpha", "component-beta"];
  independentCrossFamily.identities = ["alpha", "beta"];
  independentCrossFamily.identity_targets = [target("alpha", "cells", "conversion"), target("beta", "structs", "conversion")];
  assert.throws(() => parseCohortReview(review([independentCrossFamily], ["C04"]), COMPONENTS), /multiple independent authority components cannot be combined/);
});

test("review rejects partial components and nonreciprocal identity targets", () => {
  const partial = singleBundle();
  partial.authority_components = ["component-mixed"];
  assert.throws(() => parseCohortReview(review([partial]), COMPONENTS), /exact union of complete authority components/);

  const missingTarget = singleBundle();
  missingTarget.identity_targets = [];
  assert.throws(() => parseCohortReview(review([missingTarget]), COMPONENTS), /nonempty array/);

  const wrongCase = singleBundle();
  wrongCase.identity_targets[0].identity = "Alpha";
  assert.throws(() => parseCohortReview(review([wrongCase]), COMPONENTS), /reciprocally and exactly cover/);
});

test("review rejects legacy versions, unknown selector or wave fields, and unresolved ambiguities", () => {
  const legacy = review([singleBundle()]);
  legacy.schema_version = 1;
  assert.throws(() => parseCohortReview(legacy, COMPONENTS), /schema_version 2/);

  const selector = review([singleBundle()]);
  selector.bundles[0].selector = { category: "math/*" };
  assert.throws(() => parseCohortReview(selector, COMPONENTS), /fields must be exactly/);

  const wave = review([singleBundle()]);
  wave.operational_waves = ["wave-1"];
  assert.throws(() => parseCohortReview(wave, COMPONENTS), /fields must be exactly/);

  const ambiguous = review([singleBundle()]);
  ambiguous.ambiguities = [{ identity: "alpha" }];
  assert.throws(() => parseCohortReview(ambiguous, COMPONENTS), /cannot retain ambiguities/);
});

test("review enforces canonical ordering and unique component ownership", () => {
  const noncanonicalComponents = {
    ...singleBundle(),
    id: "c01-math-basic",
    authority_components: ["component-beta", "component-alpha"],
    identities: ["alpha", "beta"],
    composition: {
      kind: "shared-target-package",
      target_packages: [{ domain: "math", family: "basic" }],
      authored_write_set: [{ kind: "tree", path: "crates/runmat-builtins/src/catalog/entries/math/basic" }],
      shared_authority_sources: [],
      evidence: ["review:composition"],
    },
    identity_targets: [target("alpha", "math", "basic"), target("beta", "math", "basic")],
  };
  assert.throws(() => parseCohortReview(review([noncanonicalComponents]), COMPONENTS), /canonical order/);

  const duplicate = singleBundle();
  const second = clone(singleBundle());
  second.id = "c01-second";
  assert.throws(() => parseCohortReview(review([duplicate, second]), COMPONENTS), /authority component appears in more than one bundle/);

  const undeclared = singleBundle();
  undeclared.cohort = "C02";
  assert.throws(() => parseCohortReview(review([undeclared]), COMPONENTS), /not declared/);
});

function review(bundles, cohorts = ["C01"]) {
  return {
    schema_version: 2,
    kind: "runmat-builtin-topology-cohort-review",
    authority: "reviewer-authored-development-input",
    program: "RM-1064/C00-C07",
    baseline: {
      revision: `git:${"b".repeat(40)}`,
      inventory_digest: SHA,
      component_graph_digest: SHA,
      control_draft_digest: SHA,
    },
    cohorts,
    bundles,
    ambiguities: [],
    review: reviewed("review:cohort-root"),
  };
}

function singleBundle() {
  return {
    id: "c01-alpha",
    cohort: "C01",
    authority_components: ["component-alpha"],
    identities: ["alpha"],
    atomic_reason: "The complete frozen authority component is the indivisible migration unit.",
    composition: {
      kind: "single-component",
      target_packages: [{ domain: "math", family: "basic" }],
      authored_write_set: [],
      shared_authority_sources: [],
      evidence: ["review:alpha-package"],
    },
    identity_targets: [target("alpha", "math", "basic")],
    review: reviewed("review:c01-alpha"),
  };
}

function target(identity, domain, family) {
  return { identity, domain, family, classification: "preserved", evidence: [`review:${identity}`] };
}

function reviewed(evidence) {
  return { status: "reviewed", evidence: [evidence] };
}

function clone(value) {
  return structuredClone(value);
}
