import { compareCodePoint } from "./constants.mjs";
import { evidenceDigest } from "./evidence.mjs";
import { parseInventoryEvidence } from "./inventory.mjs";
import {
  array, digest, enumValue, exact, identity, kind, nonempty, stableId, uniqueStrings,
} from "./schema.mjs";

const DOMAIN_PATH = /^[a-z][a-z0-9_]*(?:\/[a-z][a-z0-9_]*)*$/;

export function compileDispositionReview(value, inventoryValue) {
  const review = parseDispositionReview(value, inventoryValue);
  const rows = new Map();

  for (const group of review.groups) {
    for (const builtin of group.identities) {
      const canonical = group.disposition === "alias"
        ? group.alias_targets[builtin]
        : null;
      rows.set(builtin, {
        disposition: group.disposition,
        canonical,
        domain: group.domain,
        family: group.family,
        reason: group.disposition === "internal" ? group.reason : null,
        review: {
          status: "reviewed",
          evidence: [...new Set([...group.evidence, ...group.review.evidence])].sort(compareCodePoint),
        },
      });
    }
  }

  return {
    schema_version: 1,
    kind: "runmat-builtin-dispositions",
    authority: "review-input-only",
    identities: Object.fromEntries([...rows].sort(([left], [right]) => compareCodePoint(left, right))),
  };
}

export function parseDispositionReview(value, inventoryValue) {
  const inventory = parseInventoryEvidence(inventoryValue);
  kind(value, 1, "runmat-builtin-disposition-review", "disposition review");
  exact(value, [
    "schema_version", "kind", "authority", "baseline_inventory_digest", "groups", "review",
  ], "disposition review");
  if (value.authority !== "reviewer-authored-development-input") {
    throw new Error("disposition review has invalid authority");
  }
  if (digest(value.baseline_inventory_digest, "disposition review baseline digest") !== inventory.digest) {
    throw new Error("disposition review does not cite the exact baseline inventory");
  }
  const groups = array(value.groups, "disposition review groups").map(parseGroup);
  assertCanonicalOrder(groups.map((group) => group.id), "disposition review groups");
  if (new Set(groups.map((group) => group.id)).size !== groups.length) {
    throw new Error("disposition review group ids must be unique");
  }
  parseReview(value.review, "disposition review");

  const known = new Set(inventory.identities.map((entry) => entry.identity));
  const memberships = new Map();
  for (const group of groups) {
    for (const builtin of group.identities) {
      if (!known.has(builtin)) throw new Error(`${group.id}: unknown inventory identity ${builtin}`);
      if (memberships.has(builtin)) {
        throw new Error(`${builtin}: assigned to both ${memberships.get(builtin)} and ${group.id}`);
      }
      memberships.set(builtin, group.id);
    }
  }
  const missing = [...known].filter((builtin) => !memberships.has(builtin)).sort(compareCodePoint);
  if (missing.length) throw new Error(`disposition review omits identities: ${missing.join(", ")}`);

  const dispositionByIdentity = new Map(groups.flatMap((group) => (
    group.identities.map((builtin) => [builtin, group.disposition])
  )));
  for (const group of groups) {
    if (group.disposition !== "alias") continue;
    for (const [builtin, target] of Object.entries(group.alias_targets)) {
      if (!known.has(target)) throw new Error(`${builtin}: alias target ${target} is absent from the inventory`);
      if (dispositionByIdentity.get(target) !== "canonical") {
        throw new Error(`${builtin}: alias target ${target} must be reviewed as canonical`);
      }
    }
  }

  return { value, inventory, groups, digest: evidenceDigest(value) };
}

function parseGroup(value) {
  exact(value, [
    "id", "disposition", "identities", "alias_targets", "domain", "family", "reason",
    "evidence", "review",
  ], "disposition review group");
  const id = stableId(value.id, "disposition review group id");
  const disposition = enumValue(value.disposition, ["canonical", "alias", "internal"], `${id} disposition`);
  const identities = uniqueStrings(value.identities, `${id} identities`, { lower: true });
  assertCanonicalOrder(identities, `${id} identities`);
  const domain = pathComponent(value.domain, `${id} domain`);
  const family = pathComponent(value.family, `${id} family`);
  const evidence = uniqueStrings(value.evidence, `${id} evidence`);
  parseReview(value.review, `${id} review`);

  if (!value.alias_targets || typeof value.alias_targets !== "object" || Array.isArray(value.alias_targets)) {
    throw new Error(`${id} alias targets must be an object`);
  }
  const aliasEntries = Object.entries(value.alias_targets).map(([builtin, target]) => [
    identity(builtin, `${id} alias identity`).toLowerCase(),
    identity(target, `${id} alias target`).toLowerCase(),
  ]);
  const aliases = Object.fromEntries(aliasEntries);
  if (Object.keys(aliases).length !== aliasEntries.length) {
    throw new Error(`${id}: alias target identities collide case-insensitively`);
  }
  const aliasKeys = Object.keys(aliases).sort(compareCodePoint);
  if (disposition === "alias") {
    if (JSON.stringify(aliasKeys) !== JSON.stringify(identities)) {
      throw new Error(`${id}: alias targets must exactly cover the group identities`);
    }
    for (const [builtin, target] of Object.entries(aliases)) {
      if (builtin === target) throw new Error(`${id}: ${builtin} cannot alias itself`);
    }
  } else if (aliasKeys.length) {
    throw new Error(`${id}: only alias groups may declare alias targets`);
  }

  if (disposition === "internal") nonempty(value.reason, `${id} internal reason`);
  else if (value.reason !== null) throw new Error(`${id}: only internal groups may declare a reason`);
  return { id, disposition, identities, alias_targets: aliases, domain, family, reason: value.reason, evidence, review: value.review };
}

function parseReview(value, label) {
  exact(value, ["status", "evidence"], `${label} review`);
  if (value.status !== "reviewed") throw new Error(`${label} must be reviewed`);
  uniqueStrings(value.evidence, `${label} review evidence`);
}

function pathComponent(value, label) {
  const result = nonempty(value, label);
  if (!DOMAIN_PATH.test(result)) throw new Error(`${label} must be a lowercase path of Rust identifiers`);
  return result;
}

function assertCanonicalOrder(values, label) {
  if (JSON.stringify(values) !== JSON.stringify([...values].sort(compareCodePoint))) {
    throw new Error(`${label} must use canonical order`);
  }
}
