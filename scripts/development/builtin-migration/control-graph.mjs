import path from "node:path";

import { compareCodePoint } from "./constants.mjs";

export function validateBundleGraph(bundles, identities) {
  for (const [bundleId, bundle] of bundles) {
    const members = [...identities.values()].filter((entry) => entry.bundle_id === bundleId).map((entry) => entry.identity).sort(compareCodePoint);
    if (JSON.stringify(members) !== JSON.stringify(bundle.identities)) {
      throw new Error(`${bundleId}: bundle identities must exactly match reciprocal identity rows`);
    }
    const cohorts = new Set(bundle.identities.map((identity) => identities.get(identity)?.cohort));
    if (cohorts.size !== 1) throw new Error(`${bundleId}: atomic bundle identities must belong to one cohort`);
    for (const prerequisite of bundle.prerequisites) {
      if (!bundles.has(prerequisite.bundle_id)) throw new Error(`${bundleId}: dangling prerequisite ${prerequisite.bundle_id}`);
      if (prerequisite.bundle_id === bundleId) throw new Error(`${bundleId}: bundle cannot depend on itself`);
    }
    for (const authored of bundle.authored_write_set) {
      for (const generated of bundle.integration_outputs) {
        if (scopesOverlap(authored, generated)) {
          throw new Error(`${bundleId}: integration output ${generated.path} overlaps authored write scope ${authored.path}`);
        }
      }
    }
  }
  const outputsByPath = new Map();
  const outputsById = new Map();
  for (const [ownerId, owner] of bundles) for (const output of owner.integration_outputs) {
    const declaration = { product_id: output.product_id, path: output.path, producer: output.producer };
    const priorPath = outputsByPath.get(output.path);
    const priorId = outputsById.get(output.product_id);
    if (priorPath && JSON.stringify(priorPath.declaration) !== JSON.stringify(declaration)) {
      throw new Error(`integration output ${output.path} has conflicting declarations in ${priorPath.owner} and ${ownerId}`);
    }
    if (priorId && JSON.stringify(priorId.declaration) !== JSON.stringify(declaration)) {
      throw new Error(`integration product ${output.product_id} has conflicting declarations in ${priorId.owner} and ${ownerId}`);
    }
    outputsByPath.set(output.path, { owner: ownerId, declaration });
    outputsById.set(output.product_id, { owner: ownerId, declaration });
  }
  for (const { owner, declaration } of outputsByPath.values()) {
    for (const [authorId, author] of bundles) for (const scope of author.authored_write_set) {
      if (scopesOverlap(scope, declaration)) {
        throw new Error(`${authorId}: authored scope ${scope.path} overlaps ${owner} integration output ${declaration.path}`);
      }
    }
  }
  validateAliasPrerequisites(bundles, identities);
  detectCycles(bundles);
}

function validateAliasPrerequisites(bundles, identities) {
  for (const [identity, row] of identities) {
    if (row.public_identity?.kind !== "alias") continue;
    const canonical = identities.get(row.public_identity.canonical_identity.toLowerCase());
    if (!canonical || canonical.bundle_id === row.bundle_id) continue;
    const prerequisite = bundles.get(row.bundle_id).prerequisites
      .find((entry) => entry.bundle_id === canonical.bundle_id);
    if (!prerequisite || prerequisite.kind !== "semantic") {
      throw new Error(`${identity}: alias bundle must declare its canonical bundle as a semantic prerequisite`);
    }
  }
}

export function findAuthoredCollisions(bundles) {
  const collisions = [];
  const entries = [...bundles.values()].sort((a, b) => compareCodePoint(a.id, b.id));
  for (let left = 0; left < entries.length; left += 1) {
    for (let right = left + 1; right < entries.length; right += 1) {
      const overlaps = [];
      for (const a of entries[left].authored_write_set) {
        for (const b of entries[right].authored_write_set) {
          if (scopesOverlap(a, b)) overlaps.push({ left: a, right: b });
        }
      }
      if (overlaps.length) collisions.push({ bundles: [entries[left].id, entries[right].id], overlaps });
    }
  }
  return collisions;
}

export function pathAllowed(scopes, candidate) {
  return scopes.some((scope) => contains(scope, candidate));
}

export function scopesOverlap(left, right) {
  return contains(left, right.path) || contains(right, left.path);
}

function contains(scope, candidate) {
  const normalized = path.posix.normalize(candidate);
  if (scope.kind === "file") return normalized === scope.path;
  return normalized === scope.path || normalized.startsWith(`${scope.path}/`);
}

function detectCycles(bundles) {
  const visiting = new Set();
  const visited = new Set();
  const visit = (id, trail) => {
    if (visiting.has(id)) throw new Error(`Bundle prerequisite cycle: ${[...trail, id].join(" -> ")}`);
    if (visited.has(id)) return;
    visiting.add(id);
    for (const edge of bundles.get(id).prerequisites) visit(edge.bundle_id, [...trail, id]);
    visiting.delete(id);
    visited.add(id);
  };
  for (const id of bundles.keys()) visit(id, []);
}
