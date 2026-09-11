import assert from "node:assert/strict";
import test from "node:test";

import { applyReviewedBundleUpdates, claimsFromBundles, requireClaimsEqual, topologyStateFromReviews } from "../state.mjs";

test("review state preserves complete bundle metadata while exposing exact claims", () => {
  const alpha = bundle("c01-alpha", "component-alpha", "alpha");
  const beta = bundle("c04-beta", "component-beta", "beta");
  const state = topologyStateFromReviews(new Map([
    ["c04_c05", { bundles: new Map([[beta.id, beta]]) }],
    ["c01_c03", { bundles: new Map([[alpha.id, alpha]]) }],
  ]));
  assert.deepEqual([...state.bundles.keys()], ["c01-alpha", "c04-beta"]);
  assert.equal(state.bundles.get("c01-alpha").atomic_reason, "reviewed c01-alpha");
  assert.equal(state.claims.get("c04-beta").identity_targets[0].family, "core");
});

test("reviewed updates replace surviving metadata and delete only prior bundles", () => {
  const prior = new Map([
    ["c01-alpha", bundle("c01-alpha", "component-alpha", "alpha")],
    ["c04-beta", bundle("c04-beta", "component-beta", "beta")],
  ]);
  const replacement = bundle("c01-alpha", "component-alpha", "alpha");
  replacement.atomic_reason = "reviewed replacement";
  const next = applyReviewedBundleUpdates(prior, new Map([[replacement.id, replacement]]), ["c04-beta"]);
  assert.deepEqual([...next.keys()], ["c01-alpha"]);
  assert.equal(next.get("c01-alpha").atomic_reason, "reviewed replacement");
  assert.throws(() => applyReviewedBundleUpdates(next, new Map(), ["c04-beta"]), /does not exist/);
  assert.throws(() => applyReviewedBundleUpdates(prior, new Map([[replacement.id, replacement]]), [replacement.id]), /both updated and deleted/);
});

test("claim equality detects metadata-to-semantic projection drift", () => {
  const bundles = new Map([["c01-alpha", bundle("c01-alpha", "component-alpha", "alpha")]]);
  const claims = [...claimsFromBundles(bundles).values()];
  assert.doesNotThrow(() => requireClaimsEqual(claimsFromBundles(bundles), claims, "fixture"));
  claims[0].identity_targets[0].family = "changed";
  assert.throws(() => requireClaimsEqual(claimsFromBundles(bundles), claims, "fixture"), /differs/);
});

function bundle(id, component, identity) {
  return {
    id,
    cohort: id.startsWith("c01") ? "C01" : "C04",
    authority_components: [component],
    identities: [identity],
    atomic_reason: `reviewed ${id}`,
    composition: {},
    identity_targets: [{ identity, domain: "test", family: "core", classification: "preserved", evidence: ["test"] }],
    review: { status: "reviewed", evidence: ["test"] },
  };
}
