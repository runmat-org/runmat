import { compareCodePoint } from "../constants.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { digest, enumValue, exact, identity, kind, nonempty, stableId } from "../schema.mjs";
import { parseBoundBaseline, parseBoundReviewDigests, validateAffectedBundleUpdates } from "./artifact-schema.mjs";
import { claimsMap, validateTopologyClaims } from "./components.mjs";
import {
  canonicalStrings,
  canonicalUnique,
  parseCohort,
  parseComponentId,
  parseIdentityTarget,
  parseReviewedEvidence,
  parseTargetPackage,
  TOPOLOGY_PROGRAM,
} from "./schema.mjs";

const CORRECTIONS_KIND = "runmat-builtin-topology-stability-corrections";
const CORRECTIONS_AUTHORITY = "reviewer-authored-development-input";
const CORRECTION_CLASSIFICATIONS = Object.freeze([
  "preservation",
  "normalization",
  "semantic-correction",
  "cohort-correction",
  "bundle-correction",
]);

export function parseStabilityCorrectionArtifact(value, context) {
  kind(value, 2, CORRECTIONS_KIND, "topology stability corrections");
  exact(value, ["schema_version", "kind", "authority", "program", "baseline", "review_digests", "reconciliation_digest", "pre_state_digest", "corrections", "identity_target_amendments", "bundle_updates", "deleted_bundles", "review"], "topology stability corrections");
  if (value.authority !== CORRECTIONS_AUTHORITY) throw new Error("topology stability corrections have invalid authority");
  if (value.program !== TOPOLOGY_PROGRAM) throw new Error("topology stability corrections have invalid program");
  parseBoundBaseline(value.baseline, context?.baseline, "topology stability corrections");
  parseBoundReviewDigests(value.review_digests, context?.reviewDigests, "topology stability corrections");
  digest(value.reconciliation_digest, "topology stability corrections reconciliation digest");
  if (value.reconciliation_digest !== context?.reconciliationDigest) {
    throw new Error("topology stability corrections do not bind the expected reconciliation");
  }
  digest(value.pre_state_digest, "topology stability corrections pre-state digest");
  parseReviewedEvidence(value.review, "topology stability corrections review");

  const before = validateTopologyClaims(context?.componentIndex, context?.claims, {
    requireComplete: true,
    requireCompositions: false,
  });
  const observedPreStateDigest = evidenceDigest(before.claims);
  if (value.pre_state_digest !== observedPreStateDigest) {
    throw new Error("topology stability corrections pre-state digest does not match the exact reconciled claims");
  }

  if (!Array.isArray(value.corrections)) throw new Error("topology stability corrections must be an array");
  const corrections = value.corrections.map((entry, position) => normalizeArtifactCorrection(context.componentIndex, entry, position));
  canonicalUnique(corrections, (entry) => entry.id, "topology stability correction records");
  const components = corrections.map((entry) => entry.component);
  if (new Set(components).size !== components.length) throw new Error("topology stability corrections must name each component at most once");

  if (!Array.isArray(value.identity_target_amendments)) {
    throw new Error("topology stability identity target amendments must be an array");
  }
  const identityTargetAmendments = value.identity_target_amendments.map((entry, position) => (
    normalizeIdentityTargetAmendment(context, entry, position)
  ));
  canonicalUnique(identityTargetAmendments, (entry) => entry.id, "topology stability identity target amendments");
  canonicalUnique(identityTargetAmendments, (entry) => entry.component, "topology stability amended components");
  const correctedComponents = new Set(components);
  for (const amendment of identityTargetAmendments) {
    if (correctedComponents.has(amendment.component)) {
      throw new Error(`${amendment.id}: component cannot have both a topology correction and an identity target amendment`);
    }
  }

  const projected = corrections.map(({ component, from, to }) => ({ component, from, to }));
  const result = applyTopologyCorrections(context.componentIndex, context.claims, projected, {
    requireCompositions: false,
  });
  const correctedBundleIds = new Set(corrections.flatMap((correction) => [correction.from.bundle, correction.to.bundle]));
  for (const amendment of identityTargetAmendments) {
    if (correctedBundleIds.has(amendment.bundle)) {
      throw new Error(`${amendment.id}: target metadata amendment bundle must be disjoint from topology corrections`);
    }
  }
  const affectedBundleIds = [
    ...correctedBundleIds,
    ...identityTargetAmendments.map((amendment) => amendment.bundle),
  ];
  const bundleState = validateAffectedBundleUpdates(
    value.bundle_updates,
    value.deleted_bundles,
    affectedBundleIds,
    context.componentIndex,
    result.claims,
    "topology stability corrections",
  );
  validateIdentityTargetBundleUpdates(context.bundles, bundleState.updates, identityTargetAmendments);
  return {
    value,
    digest: evidenceDigest(value),
    baseline: value.baseline,
    reviewDigests: value.review_digests,
    reconciliationDigest: value.reconciliation_digest,
    preStateDigest: value.pre_state_digest,
    corrections,
    identityTargetAmendments,
    result,
    bundleUpdates: bundleState.updates,
    deletedBundles: bundleState.deleted,
  };
}

function normalizeIdentityTargetAmendment(context, raw, position) {
  exact(raw, ["id", "component", "bundle", "cohort", "from_identity_targets", "to_identity_targets", "reason", "evidence"], `topology identity target amendment ${position}`);
  const id = stableId(raw.id, `topology identity target amendment ${position} id`);
  const component = parseComponentId(raw.component, `${id} component`);
  const members = context.componentIndex.get(component);
  if (!members) throw new Error(`${id}: unknown authority component ${component}`);
  const bundle = stableId(raw.bundle, `${id} bundle`);
  const cohort = parseCohort(raw.cohort, `${id} cohort`);
  const priorBundle = context.bundles?.get(bundle);
  if (!priorBundle || priorBundle.cohort !== cohort || !priorBundle.authority_components.includes(component)) {
    throw new Error(`${id}: bundle does not contain ${component} in the exact reconciled state`);
  }
  const fromTargets = parseAmendmentTargets(raw.from_identity_targets, members, `${id} from identity targets`);
  const toTargets = parseAmendmentTargets(raw.to_identity_targets, members, `${id} to identity targets`);
  const memberSet = new Set(members);
  const observed = priorBundle.identity_targets.filter((target) => memberSet.has(target.identity));
  if (!same(fromTargets, observed)) throw new Error(`${id}: stale identity target metadata pre-state`);
  for (const [index, from] of fromTargets.entries()) {
    const to = toTargets[index];
    if (!same(
      { identity: from.identity, domain: from.domain, family: from.family },
      { identity: to.identity, domain: to.domain, family: to.family },
    )) {
      throw new Error(`${id}: identity target amendment cannot change identity, domain, or family`);
    }
  }
  if (same(fromTargets, toTargets)) throw new Error(`${id}: identity target amendment must change classification or evidence`);
  nonempty(raw.reason, `${id} reason`);
  canonicalStrings(raw.evidence, `${id} evidence`);
  return {
    id,
    component,
    bundle,
    cohort,
    fromIdentityTargets: fromTargets,
    toIdentityTargets: toTargets,
    reason: raw.reason,
    evidence: raw.evidence,
  };
}

function parseAmendmentTargets(value, members, label) {
  if (!Array.isArray(value) || value.length === 0) throw new Error(`${label} must be a nonempty array`);
  const targets = value.map((target, position) => parseIdentityTarget(target, `${label} ${position}`));
  canonicalUnique(targets, (target) => target.identity.toLowerCase(), label);
  if (!same(targets.map((target) => target.identity), members)) {
    throw new Error(`${label} must enumerate the exact frozen component identities`);
  }
  return targets;
}

function validateIdentityTargetBundleUpdates(priorBundles, updates, amendments) {
  const byBundle = new Map();
  for (const amendment of amendments) {
    const entries = byBundle.get(amendment.bundle) ?? [];
    entries.push(amendment);
    byBundle.set(amendment.bundle, entries);
  }
  for (const [bundleId, bundleAmendments] of byBundle) {
    const expected = structuredClone(priorBundles.get(bundleId));
    for (const amendment of bundleAmendments) {
      const replacements = new Map(amendment.toIdentityTargets.map((target) => [target.identity, target]));
      expected.identity_targets = expected.identity_targets.map((target) => (
        replacements.has(target.identity) ? structuredClone(replacements.get(target.identity)) : target
      ));
    }
    if (!same(updates.get(bundleId), expected)) {
      throw new Error(`${bundleId}: identity target bundle update differs from the exact amended projection`);
    }
  }
}

export function applyTopologyCorrections(componentIndex, claims, corrections, options = {}) {
  if (!Array.isArray(corrections)) throw new Error("topology corrections must be an array");
  const before = validateTopologyClaims(componentIndex, claims, {
    requireComplete: true,
    requireCompositions: false,
  });
  const original = claimsMap(before.claims);
  const normalized = corrections.map((correction, position) => normalizeCorrection(componentIndex, correction, position));
  const componentIds = normalized.map((correction) => correction.component);
  if (new Set(componentIds).size !== componentIds.length) throw new Error("topology corrections must name each component at most once");

  for (const correction of normalized) validateFromState(componentIndex, original, correction);
  const next = claimsMap(before.claims);
  for (const correction of normalized) removeComponent(componentIndex, next, correction.component, correction.from.bundle);
  for (const correction of normalized) addComponent(next, correction);

  const after = validateTopologyClaims(componentIndex, next, {
    requireComplete: true,
    compositions: options.compositions ?? new Map(),
    requireCompositions: options.requireCompositions ?? true,
  });
  return {
    claims: after.claims,
    targets: flattenTargets(componentIndex, after.claims),
    validation: {
      corrected_components: [...componentIds].sort(compareCodePoint),
      component_count_before: before.summary.claimed_components,
      component_count_after: after.summary.claimed_components,
      identity_count_before: before.summary.claimed_identities,
      identity_count_after: after.summary.claimed_identities,
      conserved: before.summary.components === after.summary.components
        && before.summary.identities === after.summary.identities
        && after.summary.components === after.summary.claimed_components
        && after.summary.identities === after.summary.claimed_identities,
    },
  };
}

function normalizeCorrection(componentIndex, raw, position) {
  exact(raw, ["component", "from", "to"], `topology correction ${position}`);
  const component = stableId(raw.component, `topology correction ${position} component`);
  const members = componentIndex.get(component);
  if (!members) throw new Error(`unknown authority component ${component}`);
  const from = normalizeEndpoint(raw.from, component, "from");
  const to = normalizeEndpoint(raw.to, component, "to");
  if (!same(from.identity_targets.map((target) => target.identity), members)) {
    throw new Error(`${component}: from identity_targets must enumerate the exact frozen component identities`);
  }
  if (!same(to.identity_targets.map((target) => target.identity), members)) {
    throw new Error(`${component}: to identity_targets must enumerate the exact frozen component identities`);
  }
  if (same(from, to)) throw new Error(`${component}: correction must change bundle, cohort, or identity-local target`);
  return { component, from, to };
}

function normalizeArtifactCorrection(componentIndex, raw, position) {
  exact(raw, ["id", "component", "from", "to", "classification", "reason", "evidence"], `topology stability correction ${position}`);
  const id = stableId(raw.id, `topology stability correction ${position} id`);
  const normalized = normalizeCorrection(componentIndex, { component: raw.component, from: raw.from, to: raw.to }, position);
  const classification = enumValue(raw.classification, CORRECTION_CLASSIFICATIONS, `${id} classification`);
  nonempty(raw.reason, `${id} reason`);
  canonicalStrings(raw.evidence, `${id} evidence`);
  return { id, ...normalized, classification, reason: raw.reason, evidence: raw.evidence };
}

function normalizeEndpoint(raw, component, side) {
  exact(raw, ["bundle", "cohort", "identity_targets"], `${component} ${side}`);
  const bundle = stableId(raw.bundle, `${component} ${side} bundle`);
  if (!Array.isArray(raw.identity_targets) || raw.identity_targets.length === 0) {
    throw new Error(`${component}: ${side} identity_targets must be a nonempty array`);
  }
  const identityTargets = raw.identity_targets.map((target, position) => {
    exact(target, ["identity", "domain", "family"], `${component} ${side} target ${position}`);
    parseTargetPackage({ domain: target.domain, family: target.family }, `${component} ${side} target ${position} package`);
    return {
      identity: identity(target.identity, `${component} ${side} target identity`),
      domain: target.domain,
      family: target.family,
    };
  });
  canonicalUnique(identityTargets, (target) => target.identity.toLowerCase(), `${component} ${side} identity targets`);
  return { bundle, cohort: parseCohort(raw.cohort, `${component} ${side} cohort`), identity_targets: identityTargets };
}

function validateFromState(componentIndex, claims, correction) {
  const claim = claims.get(correction.from.bundle);
  if (!claim || !claim.components.includes(correction.component)) {
    throw new Error(`${correction.component}: stale from bundle ${correction.from.bundle}`);
  }
  if (claim.cohort !== correction.from.cohort) {
    throw new Error(`${correction.component}: stale from cohort ${correction.from.cohort}; observed ${claim.cohort}`);
  }
  const members = new Set(componentIndex.get(correction.component));
  const observedTargets = claim.identity_targets.filter((target) => members.has(target.identity));
  if (!same(observedTargets, correction.from.identity_targets)) {
    throw new Error(`${correction.component}: stale from identity target state`);
  }
}

function removeComponent(componentIndex, claims, component, bundle) {
  const claim = claims.get(bundle);
  const members = new Set(componentIndex.get(component));
  claim.components = claim.components.filter((candidate) => candidate !== component);
  if (claim.components.length === 0) {
    claims.delete(bundle);
    return;
  }
  claim.identities = claim.identities.filter((member) => !members.has(member));
  claim.identity_targets = claim.identity_targets.filter((target) => !members.has(target.identity));
}

function addComponent(claims, correction) {
  const { component, to } = correction;
  const members = to.identity_targets.map((target) => target.identity);
  const claim = claims.get(to.bundle);
  if (!claim) {
    claims.set(to.bundle, {
      bundle: to.bundle,
      cohort: to.cohort,
      components: [component],
      identities: members,
      identity_targets: structuredClone(to.identity_targets),
    });
    return;
  }
  if (claim.cohort !== to.cohort) throw new Error(`${component}: destination bundle ${to.bundle} has cohort ${claim.cohort}, not ${to.cohort}`);
  claim.components.push(component);
  claim.components.sort(compareCodePoint);
  claim.identities.push(...members);
  claim.identities.sort(compareCodePoint);
  claim.identity_targets.push(...structuredClone(to.identity_targets));
  claim.identity_targets.sort((left, right) => compareCodePoint(left.identity, right.identity));
}

function flattenTargets(componentIndex, claims) {
  const componentByIdentity = new Map();
  for (const [component, members] of componentIndex) for (const member of members) componentByIdentity.set(member, component);
  return claims.flatMap((claim) => claim.identity_targets.map((target) => ({
    identity: target.identity,
    component: componentByIdentity.get(target.identity),
    bundle: claim.bundle,
    cohort: claim.cohort,
    domain: target.domain,
    family: target.family,
  }))).sort((left, right) => compareCodePoint(left.identity, right.identity));
}

function same(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}
