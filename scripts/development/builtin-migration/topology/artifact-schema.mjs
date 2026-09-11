import { compareCodePoint } from "../constants.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { array, digest, exact, stableId } from "../schema.mjs";
import { parseReviewedBundle } from "./reviews.mjs";
import { canonicalUnique, parseReviewBaseline } from "./schema.mjs";

const REVIEW_DIGEST_FIELDS = Object.freeze(["c01_c03", "c04_c05", "c06_c07"]);

export function parseBoundBaseline(value, expected, label) {
  parseReviewBaseline(value);
  if (!expected || evidenceDigest(value) !== evidenceDigest(expected)) {
    throw new Error(`${label} baseline does not match the expected reviewed baseline`);
  }
  return value;
}

export function parseBoundReviewDigests(value, expected, label) {
  exact(value, REVIEW_DIGEST_FIELDS, `${label} review digests`);
  for (const field of REVIEW_DIGEST_FIELDS) digest(value[field], `${label} ${field} review digest`);
  if (!expected || evidenceDigest(value) !== evidenceDigest(expected)) {
    throw new Error(`${label} review digests do not match the expected cohort reviews`);
  }
  return value;
}

export function validateAffectedBundleUpdates(value, deletedValue, affectedBundleIds, componentIndex, claims, label) {
  const updates = array(value, `${label} bundle updates`, { empty: true })
    .map((entry, position) => parseReviewedBundle(entry, position, componentIndex));
  canonicalUnique(updates, (entry) => entry.id, `${label} bundle updates`);
  const deleted = array(deletedValue, `${label} deleted bundles`, { empty: true });
  canonicalUnique(deleted, (entry) => stableId(entry, `${label} deleted bundle`), `${label} deleted bundles`);

  const finalClaims = new Map(claims.map((claim) => [claim.bundle, claim]));
  const affected = [...new Set(affectedBundleIds)].sort(compareCodePoint);
  const expectedUpdates = affected.filter((bundle) => finalClaims.has(bundle));
  const expectedDeleted = affected.filter((bundle) => !finalClaims.has(bundle));
  if (!same(updates.map((entry) => entry.id), expectedUpdates)) {
    throw new Error(`${label} bundle updates must exactly enumerate every affected surviving bundle`);
  }
  if (!same(deleted, expectedDeleted)) {
    throw new Error(`${label} deleted bundles must exactly enumerate every affected removed bundle`);
  }

  for (const update of updates) {
    const claim = finalClaims.get(update.id);
    const targets = update.identity_targets.map(({ identity, domain, family }) => ({ identity, domain, family }));
    if (!claim
      || update.cohort !== claim.cohort
      || !same(update.authority_components, claim.components)
      || !same(update.identities, claim.identities)
      || !same(targets, claim.identity_targets)) {
      throw new Error(`${label} ${update.id}: full bundle update differs from the exact post-application claim`);
    }
  }
  return { updates: new Map(updates.map((entry) => [entry.id, entry])), deleted };
}

function same(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}
