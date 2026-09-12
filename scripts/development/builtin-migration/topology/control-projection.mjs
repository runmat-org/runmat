import { compareCodePoint } from "../constants.mjs";
import { deriveIntegrationProductExclusions, scopesOverlap } from "../path-scope.mjs";
import { resolveIntegrationProducts } from "../integration-products.mjs";
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
  const integrationProducts = requireMap(
    overlay?.integrationProducts,
    "globally reviewed integration products",
  );
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
      integration_outputs: resolveIntegrationProducts(
        executionPolicy.integration_product_refs,
        integrationProducts,
        id,
      ),
      authored_write_set: effectiveAuthoredWriteSet(
        topologyBundle.composition.authored_write_set,
        additionalScopes,
        integrationProducts,
      ),
    });
  }

  const identities = new Map();
  for (const id of canonicalKeys(topologyIdentities)) {
    const topologyIdentity = topologyIdentities.get(id);
    identity(id, "topology identity id");
    if (topologyIdentity.identity !== id) throw new Error(`${id}: topology identity key and value differ`);
    const control = structuredClone(identityControls.get(id));
    validatePublicIdentityProjection(id, topologyIdentity.disposition, control.public_identity);
    identities.set(id, {
      identity: id,
      cohort: topologyIdentity.cohort,
      bundle_id: topologyIdentity.bundle_id,
      domain: topologyIdentity.domain,
      family: topologyIdentity.family,
      ...control,
    });
  }
  return { bundles, identities };
}

function validatePublicIdentityProjection(id, disposition, publicIdentity) {
  if (disposition?.kind === "canonical" && disposition.canonical === null
    && publicIdentity?.kind === "primary") return;
  if (disposition?.kind === "alias" && disposition.canonical
    && publicIdentity?.kind === "alias"
    && publicIdentity.canonical_identity.toLowerCase() === disposition.canonical.toLowerCase()) return;
  if (disposition?.kind === "internal" && disposition.canonical === null
    && String(disposition.reason ?? "").trim()
    && publicIdentity?.kind === "internal"
    && publicIdentity.reason === disposition.reason) return;
  throw new Error(`${id}: reviewed public identity differs from topology disposition`);
}

function effectiveAuthoredWriteSet(topologyScopes, additionalScopes, integrationProducts) {
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
  const scopes = [...structuredClone(topologyScopes), ...structuredClone(additionalScopes)]
    .sort((left, right) => compareCodePoint(`${left.kind}:${left.path}`, `${right.kind}:${right.path}`));
  return deriveIntegrationProductExclusions(scopes, integrationProducts);
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
