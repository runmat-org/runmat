import { compareCodePoint } from "../constants.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { array, exact, kind, nonempty, stableId } from "../schema.mjs";
import {
  canonicalCohorts,
  canonicalComponentIds,
  canonicalIdentities,
  canonicalUnique,
  COHORT_REVIEW_AUTHORITY,
  COHORT_REVIEW_KIND,
  COHORT_REVIEW_VERSION,
  parseCohort,
  parseIdentityTarget,
  parseComponentId,
  parseReviewBaseline,
  parseReviewedEvidence,
  TOPOLOGY_PROGRAM,
} from "./schema.mjs";
import { validateBundleComposition } from "./composition.mjs";

export function parseCohortReview(value, componentIndex) {
  kind(value, COHORT_REVIEW_VERSION, COHORT_REVIEW_KIND, "cohort review");
  exact(value, ["schema_version", "kind", "authority", "program", "baseline", "cohorts", "bundles", "ambiguities", "review"], "cohort review");
  if (value.authority !== COHORT_REVIEW_AUTHORITY) throw new Error("cohort review has invalid authority");
  if (value.program !== TOPOLOGY_PROGRAM) throw new Error("cohort review has invalid program");
  parseReviewBaseline(value.baseline);
  const cohorts = canonicalCohorts(value.cohorts, "cohort review cohorts");
  if (array(value.ambiguities, "cohort review ambiguities", { empty: true }).length !== 0) throw new Error("reviewed cohort review cannot retain ambiguities");
  parseReviewedEvidence(value.review, "cohort review review");

  const components = parseComponentIndex(componentIndex);
  const bundles = array(value.bundles, "cohort review bundles").map((entry, index) => parseReviewedBundle(entry, index, components, new Set(cohorts)));
  canonicalUnique(bundles, (entry) => entry.id, "cohort review bundles");
  validateReviewCoverage(bundles, cohorts);

  const identityTargets = new Map();
  const componentClaims = new Map();
  for (const bundle of bundles) {
    for (const target of bundle.identity_targets) {
      const key = target.identity.toLowerCase();
      if (identityTargets.has(key)) throw new Error(`${target.identity}: identity is claimed by more than one cohort review bundle`);
      identityTargets.set(key, { ...target, bundle_id: bundle.id, cohort: bundle.cohort });
    }
    for (const component of bundle.authority_components) {
      if (componentClaims.has(component)) throw new Error(`${component}: authority component is claimed by more than one cohort review bundle`);
      componentClaims.set(component, bundle.id);
    }
  }

  return {
    value,
    digest: evidenceDigest(value),
    baseline: value.baseline,
    cohorts: new Set(cohorts),
    bundles: new Map(bundles.map((entry) => [entry.id, entry])),
    identityTargets,
    componentClaims,
  };
}

export function parseReviewedBundle(value, index, componentIndex, declaredCohorts = null) {
  const label = `cohort review bundle ${index}`;
  exact(value, ["id", "cohort", "authority_components", "identities", "atomic_reason", "composition", "identity_targets", "review"], label);
  const id = stableId(value.id, `${label} id`);
  const cohort = parseCohort(value.cohort, `${id} cohort`);
  if (declaredCohorts && !declaredCohorts.has(cohort)) throw new Error(`${id}: bundle cohort is not declared by the review`);
  const authorityComponents = canonicalComponentIds(value.authority_components, `${id} authority components`);
  const identities = canonicalIdentities(value.identities, `${id} identities`);
  nonempty(value.atomic_reason, `${id} atomic reason`);
  const identityTargets = array(value.identity_targets, `${id} identity targets`).map((entry, targetIndex) => parseIdentityTarget(entry, `${id} identity target ${targetIndex}`));
  canonicalUnique(identityTargets, (entry) => entry.identity.toLowerCase(), `${id} identity targets`);
  parseReviewedEvidence(value.review, `${id} review`);

  const componentIdentities = [];
  for (const component of authorityComponents) {
    const owned = componentIndex.get(component);
    if (!owned) throw new Error(`${id}: unknown authority component ${component}`);
    componentIdentities.push(...owned);
  }
  const componentUnion = [...new Set(componentIdentities)].sort(compareCodePoint);
  const reviewedIdentities = identities;
  if (JSON.stringify(componentUnion) !== JSON.stringify(reviewedIdentities)) throw new Error(`${id}: identities must equal the exact union of complete authority components`);
  if (JSON.stringify(reviewedIdentities) !== JSON.stringify(identityTargets.map((entry) => entry.identity))) throw new Error(`${id}: identity targets must reciprocally and exactly cover bundle identities`);

  validateBundleComposition({
    bundle: id,
    components: authorityComponents,
    identity_targets: identityTargets,
  }, value.composition);
  return value;
}

function validateReviewCoverage(bundles, declaredCohorts) {
  const usedCohorts = [...new Set(bundles.map((entry) => entry.cohort))].sort(compareCodePoint);
  if (JSON.stringify(usedCohorts) !== JSON.stringify(declaredCohorts)) throw new Error("cohort review declared cohorts must exactly equal bundle cohorts");
  const identities = new Map();
  const components = new Set();
  for (const bundle of bundles) {
    for (const component of bundle.authority_components) {
      if (components.has(component)) throw new Error(`${component}: authority component appears in more than one bundle`);
      components.add(component);
    }
    for (const identity of bundle.identities) {
      const key = identity.toLowerCase();
      if (identities.has(key)) throw new Error(`${identity}: identity appears in both ${identities.get(key)} and ${bundle.id}`);
      identities.set(key, bundle.id);
    }
  }
}

function parseComponentIndex(value) {
  if (!(value instanceof Map)) throw new Error("cohort review component index must be a Map");
  const parsed = new Map();
  const componentIds = [];
  for (const [rawComponent, rawIdentities] of value) {
    const component = parseComponentId(rawComponent, "component index id");
    const identities = canonicalIdentities(rawIdentities, `${component} component identities`);
    if (parsed.has(component)) throw new Error(`${component}: duplicate component index entry`);
    parsed.set(component, identities);
    componentIds.push(component);
  }
  canonicalUnique(componentIds, (entry) => entry, "component index");
  return parsed;
}
