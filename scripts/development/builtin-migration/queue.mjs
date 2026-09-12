import { compareCodePoint } from "./constants.mjs";
import { assertControlBaseline } from "./control.mjs";
import { findAuthoredCollisions } from "./control-graph.mjs";
import { exact, kind, nonempty, object, stableId } from "./schema.mjs";

const CONTROL_RESOLVED_INVENTORY_FIELDS = new Set([
  "disposition", "domain", "domain-conflict", "family", "family-conflict",
]);

export function buildQueue(inventory, control, state = emptyQueueState()) {
  assertControlBaseline(control, inventory);
  validateQueueState(state, control);
  const inventoryByIdentity = new Map(inventory.identities.map((entry) => [entry.identity, entry]));
  const rows = [...control.bundles.values()].map((bundle) => queueRow(bundle, control, inventoryByIdentity, state));
  rows.sort((left, right) => {
    const order = control.cohorts.get(left.cohort).order - control.cohorts.get(right.cohort).order;
    return order || right.complexity.weight - left.complexity.weight || compareCodePoint(left.bundle_id, right.bundle_id);
  });
  return {
    schema_version: 2,
    kind: "runmat-builtin-migration-work-queue",
    authority: "development-scheduling-evidence-only",
    control_manifest_digest: control.digest,
    source: inventory.source,
    inventory_digest: inventory.digest,
    migration_findings_digest: inventory.migration_findings_digest,
    ordering: "cohort-then-complexity-weight-descending-then-bundle",
    authored_write_collisions: findAuthoredCollisions(control.bundles),
    summary: summarize(rows),
    rows,
  };
}

export function emptyQueueState() {
  return { schema_version: 1, kind: "runmat-builtin-migration-queue-state", bundles: {} };
}

function queueRow(bundle, control, inventory, state) {
  const controlled = bundle.identities.map((id) => control.identities.get(id));
  const observed = bundle.identities.map((id) => inventory.get(id) ?? null);
  const blockers = [];
  const migrationFindings = control.migrationFindings.filter((entry) => entry.bundle_id === bundle.id);
  for (const finding of migrationFindings) blockers.push(`migration-finding:${finding.finding_digest}`);
  for (const prerequisite of bundle.prerequisites) {
    if (state.bundles[prerequisite.bundle_id]?.state !== "sealed") blockers.push(`prerequisite:${prerequisite.bundle_id}`);
  }
  for (let index = 0; index < observed.length; index += 1) {
    if (!observed[index]) blockers.push(`missing-inventory:${bundle.identities[index]}`);
    else if (unresolvedOutsideControl(observed[index]).length) {
      blockers.push(`unresolved-inventory:${bundle.identities[index]}`);
    }
  }
  const recorded = state.bundles[bundle.id]?.state ?? null;
  const cohorts = [...new Set(controlled.map((entry) => entry.cohort))];
  if (cohorts.length !== 1) blockers.push("mixed-cohort-bundle");
  const migrationState = recorded ?? (blockers.length ? "blocked" : "ready");
  return {
    bundle_id: bundle.id,
    identities: bundle.identities,
    target_families: [...new Set(controlled.map((entry) => `${entry.domain}/${entry.family}`))].sort(compareCodePoint),
    cohort: cohorts[0],
    owner_role: bundle.owner_role,
    migration_state: blockers.length && !["sealed", "verified"].includes(migrationState) ? "blocked" : migrationState,
    blockers: [...new Set(blockers)].sort(compareCodePoint),
    prerequisites: bundle.prerequisites,
    authored_write_set: bundle.authored_write_set,
    integration_outputs: bundle.integration_outputs,
    complexity: bundle.complexity,
    migration_findings: migrationFindings,
    applicable_maturity: Object.fromEntries(controlled.map((entry) => [entry.identity, requiredMaturity(entry.maturity)])),
    inventory_observations: observed.map((entry, index) => ({
      identity: bundle.identities[index],
      present: Boolean(entry),
      discovery_only: true,
      unresolved: entry?.unresolved ?? ["identity-not-in-inventory"],
      source_metrics: entry?.source_metrics ?? null,
    })),
  };
}

function unresolvedOutsideControl(entry) {
  return entry.unresolved.filter((field) => !CONTROL_RESOLVED_INVENTORY_FIELDS.has(field));
}

function requiredMaturity(maturity) {
  return Object.entries(maturity).filter(([, value]) => value.applicability === "required").map(([gate]) => gate).sort(compareCodePoint);
}

function validateQueueState(value, control) {
  kind(value, 1, "runmat-builtin-migration-queue-state", "queue state");
  exact(value, ["schema_version", "kind", "bundles"], "queue state");
  object(value.bundles, "queue state bundles");
  for (const [bundleId, entry] of Object.entries(value.bundles)) {
    stableId(bundleId, "queue state bundle id");
    if (!control.bundles.has(bundleId)) throw new Error(`queue state references unknown bundle ${bundleId}`);
    exact(entry, ["artifact", "state"], `${bundleId} queue state entry`);
    nonempty(entry.artifact, `${bundleId} queue artifact`);
    if (!["leased", "submitted", "integrated", "verified", "sealed"].includes(entry.state)) throw new Error(`${bundleId}: invalid queue state entry`);
  }
}

function summarize(rows) {
  const byState = {};
  for (const row of rows) byState[row.migration_state] = (byState[row.migration_state] ?? 0) + 1;
  return {
    bundles: rows.length,
    identities: rows.reduce((sum, row) => sum + row.identities.length, 0),
    complexity_weight: rows.reduce((sum, row) => sum + row.complexity.weight, 0),
    by_state: Object.fromEntries(Object.entries(byState).sort(([a], [b]) => compareCodePoint(a, b))),
    blocked: rows.filter((row) => row.blockers.length).length,
  };
}
