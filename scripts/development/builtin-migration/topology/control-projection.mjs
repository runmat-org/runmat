import { compareCodePoint } from "../constants.mjs";
import { scopesOverlap } from "../control-graph.mjs";
import { digest, identity, stableId } from "../schema.mjs";

export function materializeTopologyControl(topology, overlay) {
  const topologyDigest = digest(topology?.digest, "reviewed topology digest");
  const controlTopologyDigest = digest(overlay?.topology_digest, "control topology digest");
  if (controlTopologyDigest !== topologyDigest) {
    throw new Error("control topology digest does not match the reviewed topology");
  }

  const topologyBundles = requireMap(topology?.bundles, "reviewed topology bundles");
  const bundleControls = requireMap(overlay?.bundleControls, "control bundle policies");
  const topologyIdentities = requireMap(topology?.identities, "reviewed topology identities");
  const identityControls = requireMap(overlay?.identityControls, "control identity policies");
  requireExactKeys(topologyBundles, bundleControls, "bundle policy");
  requireExactIdentityKeys(topologyIdentities, identityControls, "identity policy");

  const bundles = new Map();
  for (const id of canonicalKeys(topologyBundles)) {
    const topologyBundle = topologyBundles.get(id);
    const { additional_authored_write_set: additionalScopes, ...executionPolicy } = bundleControls.get(id);
    stableId(id, "topology bundle id");
    if (topologyBundle.id !== id) throw new Error(`${id}: topology bundle key and id differ`);
    bundles.set(id, {
      id,
      identities: [...topologyBundle.identities],
      atomic_reason: topologyBundle.atomic_reason,
      ...structuredClone(executionPolicy),
      authored_write_set: effectiveAuthoredWriteSet(
        topologyBundle.composition.authored_write_set,
        additionalScopes,
      ),
    });
  }

  const identities = new Map();
  for (const id of canonicalKeys(topologyIdentities)) {
    const topologyIdentity = topologyIdentities.get(id);
    identity(id, "topology identity id");
    if (topologyIdentity.identity !== id) throw new Error(`${id}: topology identity key and value differ`);
    identities.set(id, {
      identity: id,
      disposition: operationalDisposition(id, topologyIdentity.disposition),
      cohort: topologyIdentity.cohort,
      bundle_id: topologyIdentity.bundle_id,
      domain: topologyIdentity.domain,
      family: topologyIdentity.family,
      ...structuredClone(identityControls.get(id)),
    });
  }
  return { bundles, identities };
}

function operationalDisposition(id, value) {
  if (value?.kind === "canonical" && value.canonical === null) {
    return { kind: "canonical", target: id };
  }
  if (value?.kind === "alias" && value.canonical) {
    return { kind: "alias", target: value.canonical };
  }
  if (value?.kind === "internal" && value.canonical === null && String(value.reason ?? "").trim()) {
    return { kind: "internal", reason: value.reason, evidence: [`reviewed-topology:${id}`] };
  }
  throw new Error(`${id}: reviewed topology has an invalid disposition`);
}

function effectiveAuthoredWriteSet(topologyScopes, additionalScopes) {
  if (!Array.isArray(topologyScopes) || !Array.isArray(additionalScopes)) {
    throw new Error("topology and additional authored write sets must be arrays");
  }
  for (let index = 0; index < additionalScopes.length; index += 1) {
    for (const topologyScope of topologyScopes) {
      if (scopesOverlap(additionalScopes[index], topologyScope)) {
        throw new Error(`${additionalScopes[index].path}: additional authored scope overlaps topology-owned scope ${topologyScope.path}`);
      }
    }
    for (let prior = 0; prior < index; prior += 1) {
      if (scopesOverlap(additionalScopes[index], additionalScopes[prior])) {
        throw new Error(`${additionalScopes[index].path}: additional authored scopes overlap`);
      }
    }
  }
  return [...structuredClone(topologyScopes), ...structuredClone(additionalScopes)]
    .sort((left, right) => compareCodePoint(`${left.kind}:${left.path}`, `${right.kind}:${right.path}`));
}

function requireExactKeys(expected, actual, label) {
  const expectedKeys = canonicalKeys(expected);
  const actualKeys = canonicalKeys(actual);
  if (JSON.stringify(actualKeys) !== JSON.stringify(expectedKeys)) {
    throw new Error(`control ${label} set differs from reviewed topology`);
  }
}

function requireExactIdentityKeys(expected, actual, label) {
  requireExactKeys(expected, actual, label);
  const folded = new Set();
  for (const key of actual.keys()) {
    const normalized = identity(key, `control ${label} key`).toLowerCase();
    if (folded.has(normalized)) throw new Error(`control ${label} keys collide case-insensitively at ${key}`);
    folded.add(normalized);
  }
}

function requireMap(value, label) {
  if (!(value instanceof Map)) throw new Error(`${label} must be a Map`);
  return value;
}

function canonicalKeys(map) {
  return [...map.keys()].sort(compareCodePoint);
}
