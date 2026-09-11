import { compareCodePoint } from "../constants.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { digest, identity, nonempty, stableId } from "../schema.mjs";

/**
 * Proves that a parsed migration control preserves the complete topology-owned
 * projection. Parsing and review authority remain with the topology and
 * control schemas; this boundary only compares their already-validated maps.
 *
 * Expected topology view:
 *   { digest, bundles: Map, identities: Map }
 * Expected control view:
 *   { topology_digest, bundles: Map, identities: Map }
 */
export function validateTopologyControlProjection(topology, control) {
  const topologyDigest = digest(topology?.digest, "reviewed topology digest");
  const controlTopologyDigest = digest(control?.topology_digest, "control topology digest");
  if (controlTopologyDigest !== topologyDigest) {
    throw new Error("control topology digest does not match the reviewed topology");
  }

  const topologyBundles = requireMap(topology?.bundles, "reviewed topology bundles");
  const controlBundles = requireMap(control?.bundles, "control bundles");
  const topologyIdentities = requireMap(topology?.identities, "reviewed topology identities");
  const controlIdentities = requireMap(control?.identities, "control identities");

  requireExactKeys(topologyBundles, controlBundles, "bundle");
  for (const bundleId of canonicalKeys(topologyBundles, "reviewed topology bundle")) {
    compareBundle(bundleId, topologyBundles.get(bundleId), controlBundles.get(bundleId));
  }

  requireExactKeys(topologyIdentities, controlIdentities, "identity");
  for (const identityId of canonicalIdentityKeys(topologyIdentities, "reviewed topology identity")) {
    compareIdentity(identityId, topologyIdentities.get(identityId), controlIdentities.get(identityId));
  }

  return control;
}

function compareBundle(bundleId, topology, control) {
  stableId(bundleId, "topology bundle id");
  if (!topology || !control) throw new Error(`${bundleId}: missing topology or control bundle`);
  if (topology.id !== bundleId || control.id !== bundleId) {
    throw new Error(`${bundleId}: bundle key and id must agree in topology and control`);
  }

  const topologyComponents = canonicalStableIds(topology.authority_components, `${bundleId} topology authority components`);
  const controlComponents = canonicalStableIds(control.authority_components, `${bundleId} control authority components`);
  requireEqual(topologyComponents, controlComponents, `${bundleId}: control authority components differ from reviewed topology`);

  const topologyIdentities = canonicalIdentities(topology.identities, `${bundleId} topology identities`);
  const controlIdentities = canonicalIdentities(control.identities, `${bundleId} control identities`);
  requireEqual(topologyIdentities, controlIdentities, `${bundleId}: control identities differ from reviewed topology`);

  requireStringEqual(topology.cohort, control.cohort, `${bundleId}: control cohort differs from reviewed topology`);
  requireStringEqual(topology.atomic_reason, control.atomic_reason, `${bundleId}: control atomic reason differs from reviewed topology`);
}

function compareIdentity(identityId, topology, control) {
  identity(identityId, "topology identity id");
  if (!topology || !control) throw new Error(`${identityId}: missing topology or control identity`);
  if (topology.identity !== identityId || control.identity !== identityId) {
    throw new Error(`${identityId}: identity key and value must agree in topology and control`);
  }

  requireStringEqual(topology.bundle_id, control.bundle_id, `${identityId}: control bundle differs from reviewed topology`);
  requireStringEqual(topology.cohort, control.cohort, `${identityId}: control cohort differs from reviewed topology`);
  requireStringEqual(topology.domain, control.domain, `${identityId}: control domain differs from reviewed topology`);
  requireStringEqual(topology.family, control.family, `${identityId}: control family differs from reviewed topology`);

  if (evidenceDigest(topology.disposition) !== evidenceDigest(control.disposition)) {
    throw new Error(`${identityId}: control disposition differs from reviewed topology`);
  }
}

function requireMap(value, label) {
  if (!(value instanceof Map)) throw new Error(`${label} must be a Map`);
  return value;
}

function requireExactKeys(expected, actual, label) {
  const expectedKeys = canonicalKeys(expected, `reviewed topology ${label}`);
  const actualKeys = canonicalKeys(actual, `control ${label}`);
  requireEqual(expectedKeys, actualKeys, `control ${label} set differs from reviewed topology`);
}

function canonicalKeys(map, label) {
  const keys = [...map.keys()];
  for (const key of keys) nonempty(key, `${label} key`);
  return [...keys].sort(compareCodePoint);
}

function canonicalIdentityKeys(map, label) {
  const keys = canonicalKeys(map, label);
  for (const key of keys) identity(key, `${label} key`);
  return keys;
}

function canonicalStableIds(value, label) {
  return canonicalList(value, label, (entry) => stableId(entry, label));
}

function canonicalIdentities(value, label) {
  return canonicalList(value, label, (entry) => {
    const validated = identity(entry, label);
    if (validated !== validated.toLowerCase()) throw new Error(`${label} must use normalized lowercase identities`);
    return validated;
  });
}

function canonicalList(value, label, validate) {
  if (!Array.isArray(value) || value.length === 0) throw new Error(`${label} must be a nonempty array`);
  const entries = value.map(validate);
  if (new Set(entries).size !== entries.length) throw new Error(`${label} must be unique`);
  const sorted = [...entries].sort(compareCodePoint);
  requireEqual(sorted, entries, `${label} must use canonical order`);
  return entries;
}

function requireStringEqual(expected, actual, message) {
  nonempty(expected, `${message} expected value`);
  nonempty(actual, `${message} actual value`);
  if (actual !== expected) throw new Error(message);
}

function requireEqual(expected, actual, message) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) throw new Error(message);
}
