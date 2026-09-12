import assert from "node:assert/strict";
import test from "node:test";

import { evidenceDigest } from "../../evidence.mjs";
import { composeTopologyCandidate } from "../compose.mjs";
import { fixtureDigest as digest, fullTopologyChainFixture as fixture, reviewTarget } from "./full-chain-fixture.mjs";

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
