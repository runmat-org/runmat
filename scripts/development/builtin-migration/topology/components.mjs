import { compareCodePoint } from "../constants.mjs";
import { array, digest, exact, identity, nonempty, object, stableId } from "../schema.mjs";
import { validateBundleComposition } from "./composition.mjs";

const COHORT = /^C0[0-7]$/;
const PACKAGE_SEGMENT = /^[a-z][a-z0-9_]*(?:\/[a-z][a-z0-9_]*)*$/;

export function buildAuthorityComponentGraph(inventory) {
  object(inventory, "reviewed baseline inventory");
  const inventoryDigest = digest(inventory.digest, "reviewed baseline inventory digest");
  const rows = array(inventory.identities, "reviewed baseline identities");
  const byIdentity = new Map();
  const parent = new Map();
  for (const [position, row] of rows.entries()) {
    object(row, `reviewed baseline identity ${position}`);
    const name = identity(row.identity, `reviewed baseline identity ${position}`);
    if (name !== name.toLowerCase()) throw new Error(`${name}: reviewed baseline identity must be normalized lowercase`);
    if (byIdentity.has(name)) throw new Error(`duplicate reviewed baseline identity ${name}`);
    validateAuthorityOwnership(row, name);
    byIdentity.set(name, row);
    parent.set(name, name);
  }

  const find = (name) => {
    let root = name;
    while (parent.get(root) !== root) root = parent.get(root);
    while (parent.get(name) !== name) {
      const next = parent.get(name);
      parent.set(name, root);
      name = next;
    }
    return root;
  };
  const union = (left, right) => {
    const leftRoot = find(left);
    const rightRoot = find(right);
    if (leftRoot !== rightRoot) parent.set(rightRoot, leftRoot);
  };
  const owner = new Map();
  for (const [name, row] of byIdentity) {
    for (const source of [...row.ownership.catalog, ...row.ownership.runtime]) {
      const prior = owner.get(source);
      if (prior) union(name, prior);
      else owner.set(source, name);
    }
  }

  const groups = new Map();
  for (const name of byIdentity.keys()) {
    const root = find(name);
    const members = groups.get(root) ?? [];
    members.push(name);
    groups.set(root, members);
  }
  const candidates = [...groups.values()].map((members) => authorityCandidate(members.sort(compareCodePoint), byIdentity));
  candidates.sort((left, right) => compareCodePoint(left.candidate_id, right.candidate_id));
  return {
    schema_version: 1,
    kind: "runmat-builtin-topology-candidate",
    authority: "discovery-only-not-control-input",
    baseline_inventory_digest: inventoryDigest,
    summary: {
      identities: rows.length,
      components: candidates.length,
      singleton_components: candidates.filter((candidate) => candidate.identities.length === 1).length,
      shared_components: candidates.filter((candidate) => candidate.identities.length > 1).length,
      maximum_component_size: Math.max(...candidates.map((candidate) => candidate.identities.length)),
    },
    candidates,
  };
}

export function buildComponentIndex(components) {
  if (!Array.isArray(components)) throw new Error("authority components must be an array");
  const index = new Map();
  for (const [position, component] of components.entries()) {
    exact(component, ["id", "identities"], `authority component ${position}`);
    const id = stableId(component.id, `authority component ${position} id`);
    const identities = canonicalIdentities(component.identities, `${id} identities`);
    if (index.has(id)) throw new Error(`duplicate authority component ${id}`);
    index.set(id, identities);
  }
  return new Map([...index].sort(([left], [right]) => compareCodePoint(left, right)));
}

export function validateComponentIndex(componentIndex) {
  if (!(componentIndex instanceof Map)) throw new Error("authority component index must be a Map");
  const identities = new Map();
  for (const [componentId, members] of componentIndex) {
    stableId(componentId, "authority component id");
    const canonical = canonicalIdentities(members, `${componentId} identities`);
    if (!same(canonical, members)) throw new Error(`${componentId} identities must be canonical`);
    for (const member of members) {
      const prior = identities.get(member);
      if (prior) throw new Error(`${member}: authority identity belongs to both ${prior} and ${componentId}`);
      identities.set(member, componentId);
    }
  }
  return componentIndex;
}

export function validateTopologyClaims(componentIndex, claims, options = {}) {
  validateComponentIndex(componentIndex);
  if (!(claims instanceof Map)) throw new Error("topology claims must be a Map");
  const requireComplete = options.requireComplete ?? true;
  if (typeof requireComplete !== "boolean") throw new Error("requireComplete must be boolean");
  const allowOverlaps = options.allowOverlaps ?? false;
  if (typeof allowOverlaps !== "boolean") throw new Error("allowOverlaps must be boolean");
  const compositions = options.compositions ?? new Map();
  if (!(compositions instanceof Map)) throw new Error("bundle compositions must be a Map");
  const requireCompositions = options.requireCompositions ?? true;
  if (typeof requireCompositions !== "boolean") throw new Error("requireCompositions must be boolean");

  const componentClaims = new Map();
  const identityClaims = new Map();
  const canonicalClaims = [];
  for (const [bundleId, raw] of claims) {
    stableId(bundleId, "claim bundle id");
    const claim = canonicalClaim(raw, componentIndex, bundleId);
    canonicalClaims.push(claim);
    for (const componentId of claim.components) append(componentClaims, componentId, bundleId);
    for (const member of claim.identities) append(identityClaims, member, bundleId);
    if (requireCompositions) validateComposition(claim, compositions.get(bundleId));
  }

  if (!allowOverlaps) for (const [member, owners] of identityClaims) {
    if (owners.length > 1) throw new Error(`${member}: identity is claimed by multiple bundles: ${owners.join(", ")}`);
  }
  if (requireComplete) {
    for (const componentId of componentIndex.keys()) {
      const owners = componentClaims.get(componentId) ?? [];
      if (owners.length !== 1) throw new Error(`${componentId}: authority component must be claimed exactly once; observed ${owners.length}`);
    }
  }

  return {
    claims: canonicalClaims.sort((left, right) => compareCodePoint(left.bundle, right.bundle)),
    component_claims: canonicalOwnerRows(componentClaims),
    identity_claims: canonicalOwnerRows(identityClaims),
    summary: {
      components: componentIndex.size,
      claimed_components: componentClaims.size,
      identities: [...componentIndex.values()].reduce((sum, members) => sum + members.length, 0),
      claimed_identities: identityClaims.size,
      bundles: claims.size,
    },
  };
}

export function claimsMap(canonicalClaims) {
  if (!Array.isArray(canonicalClaims)) throw new Error("canonical claims must be an array");
  const result = new Map();
  for (const claim of canonicalClaims) {
    const bundle = nonempty(claim?.bundle, "claim bundle");
    if (result.has(bundle)) throw new Error(`duplicate claim bundle ${bundle}`);
    result.set(bundle, structuredClone(claim));
  }
  return result;
}

export function exactComponentIdentityUnion(componentIndex, componentIds) {
  validateComponentIndex(componentIndex);
  const seen = new Set();
  const members = [];
  for (const componentId of canonicalStrings(componentIds, "claim components", stableId)) {
    const componentMembers = componentIndex.get(componentId);
    if (!componentMembers) throw new Error(`unknown authority component ${componentId}`);
    for (const member of componentMembers) {
      if (seen.has(member)) throw new Error(`${member}: duplicated across authority components in one claim`);
      seen.add(member);
      members.push(member);
    }
  }
  return members.sort(compareCodePoint);
}

function canonicalClaim(raw, componentIndex, mapBundle) {
  exact(raw, ["bundle", "cohort", "components", "identities", "identity_targets"], `${mapBundle} claim`);
  const bundle = stableId(raw.bundle, `${mapBundle} bundle`);
  if (bundle !== mapBundle) throw new Error(`${mapBundle}: claim bundle must equal its Map key`);
  const cohort = nonempty(raw.cohort, `${bundle} cohort`);
  if (!COHORT.test(cohort)) throw new Error(`${bundle}: cohort must be C00 through C07`);
  const components = canonicalStrings(raw.components, `${bundle} components`, stableId);
  const expectedIdentities = exactComponentIdentityUnion(componentIndex, components);
  const identities = canonicalIdentities(raw.identities, `${bundle} identities`);
  if (!same(expectedIdentities, identities)) throw new Error(`${bundle}: identities must equal the exact union of its authority components`);
  if (!Array.isArray(raw.identity_targets)) throw new Error(`${bundle}: identity_targets must be an array`);
  const targets = raw.identity_targets.map((target, position) => canonicalTarget(target, `${bundle} identity target ${position}`));
  targets.sort((left, right) => compareCodePoint(left.identity, right.identity));
  const targetIdentities = targets.map((target) => target.identity);
  if (new Set(targetIdentities).size !== targetIdentities.length) throw new Error(`${bundle}: identity_targets must be unique`);
  if (!same(targetIdentities, identities)) throw new Error(`${bundle}: identity_targets must enumerate every claimed identity exactly once`);
  return { bundle, cohort, components, identities, identity_targets: targets };
}

function canonicalTarget(raw, label) {
  exact(raw, ["identity", "domain", "family"], label);
  const member = identity(raw.identity, `${label} identity`);
  const domain = packagePart(raw.domain, `${label} domain`);
  if (domain.includes("/")) throw new Error(`${label} domain must be one package segment`);
  const family = packagePart(raw.family, `${label} family`);
  return { identity: member, domain, family };
}

function validateComposition(claim, composition) {
  if (composition === undefined) {
    throw new Error(`${claim.bundle}: bundle requires its reviewed composition`);
  }
  validateBundleComposition(claim, composition);
}

function canonicalIdentities(values, label) {
  const result = canonicalStrings(values, label, identity);
  if (result.some((entry) => entry !== entry.toLowerCase())) throw new Error(`${label} must use normalized lowercase identities`);
  return result;
}

function canonicalStrings(values, label, validator) {
  if (!Array.isArray(values) || values.length === 0) throw new Error(`${label} must be a nonempty array`);
  const result = values.map((value, position) => validator(value, `${label} ${position}`));
  if (new Set(result).size !== result.length) throw new Error(`${label} must be unique`);
  return result.sort(compareCodePoint);
}

function packagePart(value, label) {
  const result = nonempty(value, label);
  if (!PACKAGE_SEGMENT.test(result)) throw new Error(`${label} must be a lowercase package path`);
  return result;
}

function append(map, key, value) {
  const entries = map.get(key) ?? [];
  entries.push(value);
  entries.sort(compareCodePoint);
  map.set(key, entries);
}

function canonicalOwnerRows(owners) {
  return [...owners].sort(([left], [right]) => compareCodePoint(left, right)).map(([id, bundles]) => ({ id, bundles }));
}

function same(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function validateAuthorityOwnership(row, name) {
  object(row.ownership, `${name} ownership`);
  canonicalSourcePaths(row.ownership.catalog, `${name} catalog ownership`);
  canonicalSourcePaths(row.ownership.runtime, `${name} runtime ownership`);
}

function authorityCandidate(members, byIdentity) {
  const rows = members.map((member) => byIdentity.get(member));
  const allSources = [...new Set(rows.flatMap((row) => [...row.ownership.catalog, ...row.ownership.runtime]))];
  const sharedSources = allSources.filter((source) => rows.filter((row) => row.ownership.catalog.includes(source) || row.ownership.runtime.includes(source)).length > 1).sort(compareCodePoint);
  const categoryEvidence = Object.fromEntries(rows.map((row) => [row.identity, observedCategories(row)]));
  const categoryValues = [...new Set(Object.values(categoryEvidence).flatMap((entries) => entries.map((entry) => entry.value)))].sort(compareCodePoint);
  return {
    candidate_id: `component-${members[0].replaceAll(".", "-")}`,
    identities: members,
    shared_sources: sharedSources,
    category_values: categoryValues,
    category_evidence: categoryEvidence,
    review: { status: "unreviewed", domain: null, family: null, cohort: null, atomic_bundle: null, evidence: [] },
  };
}

function observedCategories(row) {
  const result = [];
  for (const entry of row.semantic_authority?.catalog_entries ?? []) {
    if (typeof entry.category === "string" && entry.category) result.push({ authority: "catalog", value: entry.category });
  }
  for (const entry of row.semantic_authority?.legacy_functions ?? []) {
    if (typeof entry.category === "string" && entry.category) result.push({ authority: "legacy-function", value: entry.category });
  }
  for (const entry of row.registrations?.runtime ?? []) {
    if (typeof entry.declared_category === "string" && entry.declared_category) result.push({ authority: "runtime-registration", value: entry.declared_category });
  }
  if (typeof row.domain === "string") {
    result.push({ authority: "ownership-observation", value: typeof row.family === "string" ? `${row.domain}/${row.family}` : row.domain });
  }
  return result.sort((left, right) => compareCodePoint(`${left.authority}\0${left.value}`, `${right.authority}\0${right.value}`));
}

function canonicalSourcePaths(values, label) {
  if (!Array.isArray(values)) throw new Error(`${label} must be an array`);
  for (const [position, value] of values.entries()) nonempty(value, `${label} ${position}`);
  if (new Set(values).size !== values.length) throw new Error(`${label} must be unique`);
}
