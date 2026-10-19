import { compareCodePoint } from "../constants.mjs";

export function topologyStateFromReviews(reviews) {
  if (!(reviews instanceof Map)) throw new Error("topology reviews must be a role-keyed Map");
  const bundles = new Map();
  const claims = new Map();
  for (const [role, review] of [...reviews].sort(([left], [right]) => compareCodePoint(left, right))) {
    if (!(review?.bundles instanceof Map)) throw new Error(`${role}: topology review must be parsed`);
    for (const [bundleId, bundle] of review.bundles) {
      if (bundles.has(bundleId)) throw new Error(`${bundleId}: bundle id occurs in multiple topology reviews`);
      bundles.set(bundleId, structuredClone(bundle));
      claims.set(bundleId, claimFromBundle(bundle));
    }
  }
  return { bundles: orderedMap(bundles), claims: orderedMap(claims) };
}

export function applyReviewedBundleUpdates(bundles, updates, deletedBundles) {
  if (!(bundles instanceof Map)) throw new Error("topology bundles must be a Map");
  if (!(updates instanceof Map)) throw new Error("topology bundle updates must be a Map");
  if (!Array.isArray(deletedBundles)) throw new Error("deleted topology bundles must be an array");
  for (const id of deletedBundles) {
    if (updates.has(id)) throw new Error(`${id}: topology bundle cannot be both updated and deleted`);
  }
  const next = new Map([...bundles].map(([id, bundle]) => [id, structuredClone(bundle)]));
  for (const id of deletedBundles) {
    if (!next.delete(id)) throw new Error(`${id}: deleted topology bundle does not exist in prior state`);
  }
  for (const [id, bundle] of updates) {
    if (bundle.id !== id) throw new Error(`${id}: topology bundle update key and id differ`);
    next.set(id, structuredClone(bundle));
  }
  return orderedMap(next);
}

export function claimsFromBundles(bundles) {
  if (!(bundles instanceof Map)) throw new Error("topology bundles must be a Map");
  return orderedMap(new Map([...bundles].map(([id, bundle]) => [id, claimFromBundle(bundle)])));
}

export function requireClaimsEqual(actual, expected, label) {
  if (!same([...actual], expected.map((claim) => [claim.bundle, claim]))) {
    throw new Error(`${label} reviewed bundle state differs from the exact post-application claims`);
  }
}

function claimFromBundle(bundle) {
  return {
    bundle: bundle.id,
    cohort: bundle.cohort,
    components: [...bundle.authority_components],
    identities: [...bundle.identities],
    identity_targets: bundle.identity_targets.map(({ identity, domain, family }) => ({ identity, domain, family })),
  };
}

function orderedMap(value) {
  return new Map([...value].sort(([left], [right]) => compareCodePoint(left, right)));
}

function same(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}
