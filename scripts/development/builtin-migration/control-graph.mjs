import path from "node:path";

import { compareCodePoint } from "./constants.mjs";

export function validateBundleGraph(bundles, identities) {
  for (const [bundleId, bundle] of bundles) {
    const members = [...identities.values()].filter((entry) => entry.bundle_id === bundleId).map((entry) => entry.identity).sort(compareCodePoint);
    if (JSON.stringify(members) !== JSON.stringify(bundle.identities)) {
      throw new Error(`${bundleId}: bundle identities must exactly match reciprocal identity rows`);
    }
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
  const outputs = new Map();
  for (const [ownerId, owner] of bundles) for (const output of owner.integration_outputs) {
    const prior = outputs.get(output.path);
    if (prior) throw new Error(`integration output ${output.path} is declared by both ${prior} and ${ownerId}`);
    outputs.set(output.path, ownerId);
    for (const [authorId, author] of bundles) for (const scope of author.authored_write_set) if (scopesOverlap(scope, output)) throw new Error(`${authorId}: authored scope ${scope.path} overlaps ${ownerId} integration output ${output.path}`);
  }
  detectCycles(bundles);
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
