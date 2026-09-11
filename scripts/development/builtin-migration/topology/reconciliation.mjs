import { compareCodePoint } from "../constants.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { digest, enumValue, exact, identity, kind, nonempty, stableId } from "../schema.mjs";
import { parseBoundBaseline, parseBoundReviewDigests, validateAffectedBundleUpdates } from "./artifact-schema.mjs";
import { claimsMap, exactComponentIdentityUnion, validateTopologyClaims } from "./components.mjs";
import {
  canonicalStrings,
  canonicalUnique,
  parseCohort,
  parseReviewedEvidence,
  parseTargetPackage,
  TOPOLOGY_PROGRAM,
} from "./schema.mjs";

const RECONCILIATION_KIND = "runmat-builtin-topology-reconciliation";
const RECONCILIATION_AUTHORITY = "reviewer-authored-development-input";

export function parseReconciliationArtifact(value, context) {
  kind(value, 2, RECONCILIATION_KIND, "topology reconciliation");
  exact(value, ["schema_version", "kind", "authority", "program", "baseline", "review_digests", "discrepancy_digest", "decisions", "bundle_updates", "deleted_bundles", "review"], "topology reconciliation");
  if (value.authority !== RECONCILIATION_AUTHORITY) throw new Error("topology reconciliation has invalid authority");
  if (value.program !== TOPOLOGY_PROGRAM) throw new Error("topology reconciliation has invalid program");
  parseBoundBaseline(value.baseline, context?.baseline, "topology reconciliation");
  parseBoundReviewDigests(value.review_digests, context?.reviewDigests, "topology reconciliation");
  digest(value.discrepancy_digest, "topology reconciliation discrepancy digest");
  parseReviewedEvidence(value.review, "topology reconciliation review");

  const discrepancies = claimDiscrepancies(context?.componentIndex, context?.claims);
  const observedDiscrepancyDigest = evidenceDigest(discrepancies);
  if (value.discrepancy_digest !== observedDiscrepancyDigest) {
    throw new Error("topology reconciliation discrepancy digest does not match the exact current discrepancy set");
  }

  if (!Array.isArray(value.decisions)) throw new Error("topology reconciliation decisions must be an array");
  const decisions = value.decisions.map(normalizeArtifactDecision);
  canonicalUnique(decisions, (entry) => entry.component, "topology reconciliation decisions");
  const observedByComponent = new Map(discrepancies.map((entry) => [entry.component, entry]));
  for (const decision of decisions) {
    const observed = observedByComponent.get(decision.component);
    if (!observed || observed.issue !== decision.issue || !same(observed.from, decision.from)) {
      throw new Error(`${decision.component}: reconciliation decision does not match the exact discrepancy/from state`);
    }
  }

  const projected = decisions.map(({ component, from, to }) => ({ component, from, to }));
  const claims = reconcileTopologyClaims(context.componentIndex, context.claims, projected, {
    requireCompositions: false,
  });
  const affectedBundleIds = decisions.flatMap((decision) => [
    ...decision.from.map((endpoint) => endpoint.bundle),
    decision.to.bundle,
  ]);
  const bundleState = validateAffectedBundleUpdates(
    value.bundle_updates,
    value.deleted_bundles,
    affectedBundleIds,
    context.componentIndex,
    claims,
    "topology reconciliation",
  );
  return {
    value,
    digest: evidenceDigest(value),
    baseline: value.baseline,
    reviewDigests: value.review_digests,
    discrepancyDigest: value.discrepancy_digest,
    decisions,
    claims,
    bundleUpdates: bundleState.updates,
    deletedBundles: bundleState.deleted,
  };
}

export function claimDiscrepancies(componentIndex, claims) {
  const validation = validateTopologyClaims(componentIndex, claims, {
    requireComplete: false,
    allowOverlaps: true,
    requireCompositions: false,
  });
  const owners = new Map(validation.component_claims.map((row) => [row.id, row.bundles]));
  return [...componentIndex.keys()].map((component) => {
    const from_bundles = owners.get(component) ?? [];
    const members = new Set(componentIndex.get(component));
    const from = from_bundles.map((bundle) => {
      const claim = claims.get(bundle);
      return {
        bundle,
        cohort: claim.cohort,
        identity_targets: claim.identity_targets.filter((target) => members.has(target.identity)).sort((left, right) => compareCodePoint(left.identity, right.identity)),
      };
    });
    return from_bundles.length === 1 ? null : {
      component,
      issue: from_bundles.length === 0 ? "missing" : "multiply-claimed",
      from,
    };
  }).filter(Boolean);
}

export function reconcileTopologyClaims(componentIndex, claims, decisions, options = {}) {
  if (!Array.isArray(decisions)) throw new Error("reconciliation decisions must be an array");
  const discrepancies = claimDiscrepancies(componentIndex, claims);
  const expected = discrepancies.map((row) => row.component);
  const normalized = decisions.map(normalizeDecision).sort((left, right) => compareCodePoint(left.component, right.component));
  const actual = normalized.map((row) => row.component);
  if (new Set(actual).size !== actual.length) throw new Error("reconciliation decisions must contain unique components");
  if (!same(expected, actual)) throw new Error("reconciliation decisions must equal the exact current discrepancy set");

  const discrepancyIndex = new Map(discrepancies.map((row) => [row.component, row]));
  const next = claimsMap(validateTopologyClaims(componentIndex, claims, {
    requireComplete: false,
    allowOverlaps: true,
    requireCompositions: false,
  }).claims);

  for (const decision of normalized) {
    const observed = discrepancyIndex.get(decision.component);
    if (!same(decision.from, observed.from)) {
      throw new Error(`${decision.component}: stale reconciliation from state`);
    }
    removeComponent(next, componentIndex, decision.component, observed.from.map((endpoint) => endpoint.bundle));
    addComponent(next, componentIndex, decision);
  }

  return validateTopologyClaims(componentIndex, next, {
    requireComplete: true,
    compositions: options.compositions ?? new Map(),
    requireCompositions: options.requireCompositions ?? true,
  }).claims;
}

function normalizeDecision(raw, position) {
  exact(raw, ["component", "from", "to"], `reconciliation decision ${position}`);
  const component = stableId(raw.component, `reconciliation decision ${position} component`);
  if (!Array.isArray(raw.from)) throw new Error(`${component}: reconciliation from state must be an array`);
  const from = raw.from.map((endpoint, endpointPosition) => normalizeEndpoint(endpoint, component, `from ${endpointPosition}`))
    .sort((left, right) => compareCodePoint(left.bundle, right.bundle));
  if (new Set(from.map((endpoint) => endpoint.bundle)).size !== from.length) throw new Error(`${component}: reconciliation from bundles must be unique`);
  return { component, from, to: normalizeEndpoint(raw.to, component, "destination") };
}

function normalizeArtifactDecision(raw, position) {
  exact(raw, ["component", "issue", "from", "to", "reason", "evidence"], `topology reconciliation decision ${position}`);
  const normalized = normalizeDecision({ component: raw.component, from: raw.from, to: raw.to }, position);
  const issue = enumValue(raw.issue, ["missing", "multiply-claimed"], `${normalized.component} reconciliation issue`);
  nonempty(raw.reason, `${normalized.component} reconciliation reason`);
  canonicalStrings(raw.evidence, `${normalized.component} reconciliation evidence`);
  return { ...normalized, issue, reason: raw.reason, evidence: raw.evidence };
}

function normalizeEndpoint(raw, component, side) {
  exact(raw, ["bundle", "cohort", "identity_targets"], `${component} ${side}`);
  const bundle = stableId(raw.bundle, `${component} ${side} bundle`);
  if (!Array.isArray(raw.identity_targets) || raw.identity_targets.length === 0) {
    throw new Error(`${component}: ${side} identity_targets must be a nonempty array`);
  }
  const identityTargets = raw.identity_targets.map((target, targetPosition) => {
    exact(target, ["identity", "domain", "family"], `${component} ${side} target ${targetPosition}`);
    parseTargetPackage({ domain: target.domain, family: target.family }, `${component} ${side} target ${targetPosition} package`);
    return { identity: identity(target.identity, `${component} ${side} identity`), domain: target.domain, family: target.family };
  });
  canonicalUnique(identityTargets, (target) => target.identity.toLowerCase(), `${component} ${side} identity targets`);
  return { bundle, cohort: parseCohort(raw.cohort, `${component} ${side} cohort`), identity_targets: identityTargets };
}

function removeComponent(claims, componentIndex, component, bundles) {
  const members = new Set(componentIndex.get(component));
  for (const bundle of bundles) {
    const claim = claims.get(bundle);
    if (!claim || !claim.components.includes(component)) throw new Error(`${component}: stale source bundle ${bundle}`);
    claim.components = claim.components.filter((entry) => entry !== component);
    if (claim.components.length === 0) {
      claims.delete(bundle);
      continue;
    }
    claim.identities = claim.identities.filter((member) => !members.has(member));
    claim.identity_targets = claim.identity_targets.filter((target) => !members.has(target.identity));
  }
}

function addComponent(claims, componentIndex, decision) {
  const { component, to } = decision;
  if (!componentIndex.has(component)) throw new Error(`unknown authority component ${component}`);
  const members = componentIndex.get(component);
  const targetMembers = to.identity_targets.map((target) => target.identity);
  if (!same(members, targetMembers)) throw new Error(`${component}: destination targets must enumerate the exact frozen component identities`);
  const claim = claims.get(to.bundle);
  if (!claim) {
    claims.set(to.bundle, {
      bundle: to.bundle,
      cohort: to.cohort,
      components: [component],
      identities: [...members],
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

function same(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}
