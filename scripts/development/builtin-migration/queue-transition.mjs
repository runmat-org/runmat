import {
  requiredPilotBarrierBundleIds, requiredProductionBarrierBundleIds,
} from "./queue-barriers.mjs";
import { validateMonotonicQueueTransition } from "./queue-history.mjs";
import { digest, exact, repositoryPath, stableId } from "./schema.mjs";

export function parsePilotTransition(value) {
  if (value === null) return null;
  exact(value, ["evaluation"], "queue pilot transition");
  exact(value.evaluation, ["path", "artifact_id", "digest"], "queue pilot transition evaluation");
  return {
    evaluation: {
      path: repositoryPath(value.evaluation.path, "queue pilot transition evaluation path"),
      artifact_id: stableId(
        value.evaluation.artifact_id, "queue pilot transition evaluation artifact id",
      ),
      digest: digest(value.evaluation.digest, "queue pilot transition evaluation digest"),
    },
  };
}

export function validateQueuePhaseTransition({
  value, predecessorState, acceptedSeals, acceptedSealDependencies,
  sealedBundles, pilotTransition, control,
}) {
  if (predecessorState === null) {
    if (value.phase !== "pilot" || pilotTransition !== null) {
      throw new Error("initial queue state must begin in pilot phase without a transition");
    }
    validateMonotonicQueueTransition(value, null, acceptedSeals, acceptedSealDependencies);
    return null;
  }
  const previousPhase = predecessorState.value.phase;
  if (previousPhase === value.phase) {
    if (JSON.stringify(pilotTransition) !== JSON.stringify(predecessorState.pilotTransition)) {
      throw new Error("queue state changed its pilot transition without changing phase");
    }
    // This establishes exactly one addition before either helper dereferences it.
    validateMonotonicQueueTransition(
      value, predecessorState, acceptedSeals, acceptedSealDependencies,
    );
    validateSealEdgeWorkflow(value, predecessorState, acceptedSeals);
    validateAddedSealAdmission(control, predecessorState, sealedBundles, value.phase);
    return predecessorState.pilotEvaluation;
  }
  if (previousPhase !== "pilot" || value.phase !== "production") {
    throw new Error("queue phase may transition only once from pilot to production");
  }
  if (predecessorState.pilotTransition !== null || pilotTransition === null) {
    throw new Error("pilot-to-production transition must introduce one permanent evaluation reference");
  }
  if (JSON.stringify(acceptedSeals) !== JSON.stringify(predecessorState.acceptedSeals)) {
    throw new Error("pilot-to-production transition cannot add, remove, or replace accepted seals");
  }
  if (JSON.stringify(value.bundles) !== JSON.stringify(predecessorState.value.bundles)) {
    throw new Error("pilot-to-production transition cannot change queue workflow state");
  }
  throw new Error("pilot-to-production transition requires branded evaluation authority");
}

function validateAddedSealAdmission(control, predecessorState, sealedBundles, phase) {
  const previous = new Set(predecessorState.acceptedSeals.map((entry) => entry.bundle_id));
  const added = sealedBundles.find((entry) => !previous.has(entry.reference.bundle_id));
  if (added.queue_phase !== phase) {
    throw new Error(`${added.reference.bundle_id}: appended seal phase differs from queue phase`);
  }
  const required = phase === "pilot"
    ? requiredPilotBarrierBundleIds(control, added.reference.bundle_id)
    : requiredProductionBarrierBundleIds(control, added.reference.bundle_id);
  const missing = required.filter((bundleId) => !previous.has(bundleId));
  if (missing.length) {
    throw new Error(
      `${added.reference.bundle_id}: serialized seal is missing ${phase} barriers ${missing.join(", ")}`,
    );
  }
}

function validateSealEdgeWorkflow(value, predecessorState, acceptedSeals) {
  const previousSealIds = new Set(
    predecessorState.acceptedSeals.map((entry) => entry.bundle_id),
  );
  const added = acceptedSeals.find((entry) => !previousSealIds.has(entry.bundle_id));
  for (const [bundleId, previous] of Object.entries(predecessorState.value.bundles)) {
    if (bundleId !== added.bundle_id
      && JSON.stringify(value.bundles[bundleId]) !== JSON.stringify(previous)) {
      throw new Error(`${bundleId}: seal edge cannot change unrelated workflow state`);
    }
  }
  const allowedIds = new Set([
    ...Object.keys(predecessorState.value.bundles), added.bundle_id,
  ]);
  const unexpected = Object.keys(value.bundles).filter((bundleId) => !allowedIds.has(bundleId));
  if (unexpected.length) throw new Error("seal edge cannot introduce unrelated workflow entries");
  const expected = { artifact: added.artifact_id, state: "verified" };
  if (JSON.stringify(value.bundles[added.bundle_id]) !== JSON.stringify(expected)) {
    throw new Error(`${added.bundle_id}: seal edge must record the exact verified seal artifact`);
  }
}
