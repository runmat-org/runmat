import { compareCodePoint } from "./constants.mjs";
import { assertControlBaseline, assertValidatedControl } from "./control.mjs";
import { findAuthoredCollisions } from "./control-graph.mjs";
import { deepImmutable } from "./immutable.mjs";
import { evidenceDigest } from "./evidence.mjs";
import { requiredBarrierBundleIds } from "./queue-barriers.mjs";
import {
  parseQueuePredecessor, validateMonotonicQueueTransition,
  validateSerializedSealTransition,
} from "./queue-history.mjs";
import { validateQueueSeal } from "./queue-seal.mjs";
import {
  buildAcceptedSealSet, buildBarrierSealSet, validateAcceptedSealSet,
  validateBarrierSealSet,
} from "./seal-set-schema.mjs";
import {
  array, digest, exact, kind, nonempty, object, repositoryPath, stableId,
} from "./schema.mjs";

const CONTROL_RESOLVED_INVENTORY_FIELDS = new Set([
  "disposition", "domain", "domain-conflict", "family", "family-conflict",
]);
const VALIDATED_QUEUE_STATES = new WeakSet();

export function buildQueue(inventory, control, state = null) {
  assertControlBaseline(control, inventory);
  const queueState = state === null
    ? validateQueueState(emptyQueueState(control), control, () => null)
    : assertValidatedQueueState(state, control);
  const inventoryByIdentity = new Map(inventory.identities.map((entry) => [entry.identity, entry]));
  const rows = [...control.bundles.values()].map((bundle) => queueRow(bundle, control, inventoryByIdentity, queueState));
  rows.sort((left, right) => {
    const order = control.cohorts.get(left.cohort).order - control.cohorts.get(right.cohort).order;
    return order || right.complexity.weight - left.complexity.weight || compareCodePoint(left.bundle_id, right.bundle_id);
  });
  return {
    schema_version: 3,
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

export function emptyQueueState(control) {
  assertValidatedControl(control);
  const payload = {
    schema_version: 3,
    kind: "runmat-builtin-migration-queue-state",
    authority: "reviewed-monotonic-scheduling-input",
    control_manifest_digest: control.digest,
    predecessor: null,
    bundles: {},
    seals: [],
  };
  return { ...payload, digest: evidenceDigest(payload) };
}

function queueRow(bundle, control, inventory, state) {
  const sealedBundleIds = new Set(state.sealedBundleIds);
  const controlled = bundle.identities.map((id) => control.identities.get(id));
  const observed = bundle.identities.map((id) => inventory.get(id) ?? null);
  const blockers = [];
  const migrationFindings = control.migrationFindings.filter((entry) => entry.bundle_id === bundle.id);
  blockers.push(...cohortBarrierBlockers(bundle, control, sealedBundleIds));
  for (const prerequisite of bundle.prerequisites) {
    if (!sealedBundleIds.has(prerequisite.bundle_id)) blockers.push(`prerequisite:${prerequisite.bundle_id}`);
  }
  for (let index = 0; index < observed.length; index += 1) {
    if (!observed[index]) blockers.push(`missing-inventory:${bundle.identities[index]}`);
    else if (unresolvedOutsideControl(observed[index]).length) {
      blockers.push(`unresolved-inventory:${bundle.identities[index]}`);
    }
  }
  const recorded = sealedBundleIds.has(bundle.id)
    ? "sealed"
    : state.value.bundles[bundle.id]?.state ?? null;
  const cohorts = [...new Set(controlled.map((entry) => entry.cohort))];
  if (cohorts.length !== 1) blockers.push("mixed-cohort-bundle");
  const migrationState = recorded ?? "ready";
  return {
    bundle_id: bundle.id,
    identities: bundle.identities,
    target_families: [...new Set(controlled.map((entry) => `${entry.domain}/${entry.family}`))].sort(compareCodePoint),
    cohort: cohorts[0],
    owner_role: bundle.owner_role,
    migration_state: blockers.length ? "blocked" : migrationState,
    blockers: [...new Set(blockers)].sort(compareCodePoint),
    prerequisites: bundle.prerequisites,
    authored_write_set: bundle.authored_write_set,
    integration_outputs: bundle.integration_outputs,
    complexity: bundle.complexity,
    finding_work_items: migrationFindings
      .filter((entry) => entry.disposition !== "reviewed-no-action")
      .map((entry) => entry.finding_digest),
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

export function cohortBarrierBlockers(bundle, control, sealedBundleIds) {
  const cohorts = [...new Set(bundle.identities.map((id) => control.identities.get(id)?.cohort))]
    .filter(Boolean);
  if (cohorts.length !== 1) return [];
  const active = control.cohorts.get(cohorts[0]);
  if (!active) return [];
  const blockers = [];
  for (const candidate of control.bundles.values()) {
    const candidateCohorts = [...new Set(candidate.identities
      .map((id) => control.identities.get(id)?.cohort))].filter(Boolean);
    if (candidateCohorts.length !== 1) continue;
    const candidateCohort = control.cohorts.get(candidateCohorts[0]);
    if (candidateCohort?.order >= active.order) continue;
    if (!sealedBundleIds.has(candidate.id)) {
      blockers.push(`cohort-barrier:${candidateCohort.id}:${candidate.id}`);
    }
  }
  return [...new Set(blockers)].sort(compareCodePoint);
}

function unresolvedOutsideControl(entry) {
  return entry.unresolved.filter((field) => !CONTROL_RESOLVED_INVENTORY_FIELDS.has(field));
}

function requiredMaturity(maturity) {
  return Object.entries(maturity).filter(([, value]) => value.applicability === "required").map(([gate]) => gate).sort(compareCodePoint);
}

export function validateQueueState(value, control, loadSeal, loadPredecessor = () => null) {
  assertValidatedControl(control);
  kind(value, 3, "runmat-builtin-migration-queue-state", "queue state");
  exact(value, [
    "schema_version", "kind", "authority", "control_manifest_digest", "predecessor",
    "bundles", "seals", "digest",
  ], "queue state");
  if (value.authority !== "reviewed-monotonic-scheduling-input") {
    throw new Error("queue state has invalid authority");
  }
  if (value.control_manifest_digest !== control.digest) {
    throw new Error("queue state belongs to another control manifest");
  }
  const predecessor = parseQueuePredecessor(value.predecessor);
  const predecessorState = predecessor === null ? null : assertValidatedQueueState(
    loadPredecessor(predecessor), control,
  );
  if (predecessorState !== null && predecessorState.stateDigest !== predecessor.state_digest) {
    throw new Error("queue state predecessor digest mismatch");
  }
  object(value.bundles, "queue state bundles");
  for (const [bundleId, entry] of Object.entries(value.bundles)) {
    stableId(bundleId, "queue state bundle id");
    if (!control.bundles.has(bundleId)) throw new Error(`queue state references unknown bundle ${bundleId}`);
    exact(entry, ["artifact", "state"], `${bundleId} queue state entry`);
    nonempty(entry.artifact, `${bundleId} queue artifact`);
    if (!["leased", "submitted", "integrated", "verified"].includes(entry.state)) throw new Error(`${bundleId}: invalid queue state entry`);
  }
  const sealedBundleIds = new Set();
  const acceptedSeals = [];
  const sealedBundles = [];
  const acceptedSealDependencies = [];
  const referenceKeys = [];
  for (const reference of array(value.seals, "queue seal references", { empty: true })) {
    exact(reference, ["path", "artifact_id", "digest", "bundle_id"], "queue seal reference");
    const referencePath = repositoryPath(reference.path, "queue seal reference path");
    const artifactId = stableId(reference.artifact_id, "queue seal reference artifact id");
    const referenceDigest = digest(reference.digest, "queue seal reference digest");
    const bundleId = stableId(reference.bundle_id, "queue seal reference bundle id");
    if (!control.bundles.has(bundleId)) throw new Error(`queue seal references unknown bundle ${bundleId}`);
    if (sealedBundleIds.has(bundleId)) throw new Error(`queue seal references duplicate bundle ${bundleId}`);
    const acceptedReference = {
      path: referencePath, artifact_id: artifactId, digest: referenceDigest, bundle_id: bundleId,
    };
    const seal = validateQueueSeal(loadSeal(acceptedReference), acceptedReference, control);
    referenceKeys.push(`${bundleId}\0${artifactId}`);
    acceptedSeals.push(acceptedReference);
    sealedBundles.push({
      reference: acceptedReference,
      integrated_revision: seal.integratedRevision,
      source_digest: seal.sourceDigest,
    });
    acceptedSealDependencies.push({
      bundleId, accepted: seal.acceptedSeals, barriers: seal.barrierSeals,
    });
    sealedBundleIds.add(bundleId);
  }
  if (JSON.stringify(referenceKeys) !== JSON.stringify([...referenceKeys].sort(compareCodePoint))) {
    throw new Error("queue seal references must use canonical bundle/artifact ordering");
  }
  const acceptedByBundle = new Map(acceptedSeals.map((entry) => [entry.bundle_id, entry]));
  for (const seal of acceptedSealDependencies) {
    for (const [role, references] of [["accepted", seal.accepted], ["barrier", seal.barriers]]) {
      for (const dependency of references) {
        if (dependency.bundle_id === seal.bundleId
          || JSON.stringify(acceptedByBundle.get(dependency.bundle_id)) !== JSON.stringify(dependency)) {
          throw new Error(`queue seal for ${seal.bundleId} references an absent, self, or different ${role} seal`);
        }
      }
    }
  }
  validateMonotonicQueueTransition(value, predecessorState, acceptedSeals, acceptedSealDependencies);
  digest(value.digest, "queue state digest");
  const { digest: _ignored, ...payload } = value;
  if (evidenceDigest(payload) !== value.digest) throw new Error("queue state digest mismatch");
  const parsed = deepImmutable({
    value,
    controlDigest: control.digest,
    stateDigest: value.digest,
    predecessor,
    predecessorState,
    sealedBundleIds: [...sealedBundleIds].sort(compareCodePoint),
    acceptedSeals,
    sealedBundles,
  });
  VALIDATED_QUEUE_STATES.add(parsed);
  return parsed;
}

export { validateSerializedSealTransition };

export function assertValidatedQueueState(value, control) {
  if (!VALIDATED_QUEUE_STATES.has(value)) throw new Error("queue requires an exact validated queue state");
  if (value.controlDigest !== control.digest) throw new Error("queue state was validated for another control manifest");
  return value;
}

export function acceptedSealSet(value, control) {
  const state = assertValidatedQueueState(value, control);
  return deepImmutable({
    value: buildAcceptedSealSet(control.digest, state.acceptedSeals),
    sealedBundles: state.sealedBundles,
  });
}

export function barrierSealSet(value, control, bundleId) {
  const state = assertValidatedQueueState(value, control);
  const bundle = control.bundles.get(stableId(bundleId, "barrier seal-set bundle id"));
  if (!bundle) throw new Error(`barrier seal set references unknown bundle ${bundleId}`);
  const required = new Set(requiredBarrierBundleIds(control, bundle.id));
  const accepted = new Map(state.sealedBundles.map((entry) => [entry.reference.bundle_id, entry]));
  const missing = [...required].filter((id) => !accepted.has(id)).sort(compareCodePoint);
  if (missing.length) throw new Error(`${bundle.id}: barrier seal set is missing ${missing.join(", ")}`);
  const seals = state.acceptedSeals.filter((reference) => required.has(reference.bundle_id));
  const sealedBundles = state.sealedBundles.filter((entry) => required.has(entry.reference.bundle_id));
  return deepImmutable({
    value: buildBarrierSealSet(control.digest, bundle.id, seals),
    sealedBundles,
  });
}

export { requiredBarrierBundleIds } from "./queue-barriers.mjs";

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
