import { compareCodePoint } from "../constants.mjs";
import { assertValidatedControl } from "../control.mjs";
import {
  assertLoadedPilotWorkSessionCompletion, loadPilotWorkSessionCompletion,
} from "../pilot-work-session/index.mjs";
import { loadQueueAuthority } from "../queue-authority/index.mjs";
import { assertValidatedQueueSeal } from "../queue-seal.mjs";
import { assertLoadedPilotMeasurementReview } from "./review-loader.mjs";
import { artifactBinding, queueBinding } from "./schema.mjs";
import { derivePilotTiming } from "./timing.mjs";

export function reconstructPilotMeasurement({ review, session, control: controlValue, repository }) {
  const control = assertValidatedControl(controlValue);
  const reviewed = assertLoadedPilotMeasurementReview(review, { session, control });
  const initialQueue = loadBoundQueue(reviewed.value.initial_queue, session, control, "initial");
  const finalQueue = loadBoundQueue(reviewed.value.final_queue, session, control, "final");
  assertInitialQueue(initialQueue, control);
  if (finalQueue.state.value.phase !== "pilot" || finalQueue.checkpoint.value.phase !== "pilot") {
    throw new Error("pilot measurement final queue must remain in pilot phase");
  }
  const completions = reviewed.value.completed_sessions.map((reference) =>
    loadPilotWorkSessionCompletion({ session, reference, control, repository }));
  for (const completion of completions) {
    assertLoadedPilotWorkSessionCompletion(completion, { session, control });
  }
  const history = orderedHistory(initialQueue, finalQueue, completions, control);
  const counts = deriveCounts(control, history);
  const timing = derivePilotTiming(history);
  const source = {
    revision: finalQueue.checkpoint.value.source_revision,
    source_digest: finalQueue.checkpoint.value.source_digest,
    inventory_digest: finalQueue.checkpoint.value.inventory_digest,
  };
  const last = history.at(-1);
  if (last.value.source.revision !== source.revision
    || last.value.source.source_digest !== source.source_digest
    || last.value.source.inventory_digest !== source.inventory_digest) {
    throw new Error("pilot measurement final source differs from its last completed session");
  }
  return Object.freeze({ initialQueue, finalQueue, history, counts, timing, source });
}

export function derivedResultPayload(review, reconstructed, control) {
  return {
    schema_version: 1,
    kind: "runmat-builtin-migration-pilot-measurement-result",
    authority: "derived-from-reviewed-pilot-work-sessions",
    control_manifest_digest: control.digest,
    pilot_policy_digest: control.pilotPolicyDigest,
    review_manifest: artifactBinding(review.observation),
    initial_queue: queueBinding(reconstructed.initialQueue),
    final_queue: queueBinding(reconstructed.finalQueue),
    completed_sessions: reconstructed.history.map((entry) =>
      artifactBinding(entry.observation)),
    seals: reconstructed.history.map((entry) => ({ ...entry.seal.reference })),
    bundle_ids: reconstructed.history.map((entry) => entry.bundleId),
    source: reconstructed.source,
    counts: reconstructed.counts,
    timing: reconstructed.timing,
  };
}

function loadBoundQueue(binding, session, control, role) {
  const queue = loadQueueAuthority({
    session,
    statePath: binding.state.path,
    checkpointPath: binding.checkpoint.path,
    trustedCheckpointDigest: binding.checkpoint.semantic_digest,
    control,
  });
  if (JSON.stringify(queueBinding(queue)) !== JSON.stringify(binding)) {
    throw new Error(`pilot measurement ${role} queue observation mismatch`);
  }
  return queue;
}

function assertInitialQueue(queue, control) {
  const state = queue.state;
  const checkpoint = queue.checkpoint.value;
  if (state.value.phase !== "pilot" || checkpoint.phase !== "pilot"
    || state.predecessor !== null || state.acceptedSeals.length !== 0
    || checkpoint.head_event !== null
    || checkpoint.source_revision !== control.baseline.revision
    || checkpoint.source_digest !== control.baseline.source_digest
    || checkpoint.inventory_digest !== control.baseline.inventory_digest) {
    throw new Error("pilot measurement initial queue must be the empty reviewed pilot root");
  }
}

function orderedHistory(initialQueue, finalQueue, completions, control) {
  const byPredecessor = new Map();
  const bundleIds = new Set();
  const sessionIds = new Set();
  const leaseIds = new Map();
  const leaseDigests = new Map();
  const startReferences = new Set();
  const sealReferences = new Set();
  for (const completion of completions) {
    const seal = assertValidatedQueueSeal(completion.seal, control);
    if (seal.queuePhase !== "pilot" || !seal.ordinaryGateProof
      || seal.ordinaryGateProof.gates.size !== seal.ordinaryGateProof.requiredGateNames.length) {
      throw new Error(`${completion.bundleId}: pilot measurement requires every ordinary gate`);
    }
    assertUnique(bundleIds, completion.bundleId, "bundle");
    assertUnique(sessionIds, completion.sessionId, "session");
    assertSessionLeasesUnique(completion, leaseIds, leaseDigests);
    assertUnique(startReferences, referenceKey(completion.start.reference), "start session");
    assertUnique(sealReferences, referenceKey(seal.reference), "seal");
    const predecessor = queueKey(completion.preIntegrationQueue);
    if (byPredecessor.has(predecessor)) {
      throw new Error("pilot measurement queue history forks from one predecessor");
    }
    byPredecessor.set(predecessor, completion);
  }
  const history = [];
  let current = initialQueue;
  while (byPredecessor.has(queueKey(current))) {
    const completion = byPredecessor.get(queueKey(current));
    byPredecessor.delete(queueKey(current));
    history.push(completion);
    current = completion.successorQueue;
  }
  if (byPredecessor.size !== 0 || queueKey(current) !== queueKey(finalQueue)) {
    throw new Error("pilot measurement sessions do not form one complete initial-to-final queue history");
  }
  const expected = control.pilotPolicy.bundleIds;
  const observed = history.map((entry) => entry.bundleId).sort(compareCodePoint);
  if (JSON.stringify(observed) !== JSON.stringify(expected)) {
    throw new Error("pilot measurement sessions do not exactly cover the reviewed pilot bundles");
  }
  const finalBundles = finalQueue.state.acceptedSeals
    .map((entry) => entry.bundle_id).sort(compareCodePoint);
  if (JSON.stringify(finalBundles) !== JSON.stringify(expected)) {
    throw new Error("pilot measurement final queue does not exactly seal the reviewed pilot");
  }
  const historySeals = history.map((entry) => referenceKey(entry.seal.reference)).sort(compareCodePoint);
  const finalSeals = finalQueue.state.acceptedSeals.map(referenceKey).sort(compareCodePoint);
  if (JSON.stringify(historySeals) !== JSON.stringify(finalSeals)) {
    throw new Error("pilot measurement final queue seals differ from completed sessions");
  }
  return Object.freeze(history);
}

function deriveCounts(control, history) {
  const cohorts = new Map();
  const identities = new Set();
  let publicIdentities = 0;
  let internalIdentities = 0;
  for (const completion of history) {
    const bundle = control.bundles.get(completion.bundleId);
    if (!bundle) throw new Error(`${completion.bundleId}: missing reviewed pilot bundle`);
    const bundleCohort = bundle.identities.length > 0
      ? control.identities.get(bundle.identities[0])?.cohort : null;
    if (!bundleCohort) throw new Error(`${completion.bundleId}: pilot bundle has no identity cohort`);
    if (!cohorts.has(bundleCohort)) {
      cohorts.set(bundleCohort, {
        cohort: bundleCohort, bundles: 0, identities: 0,
        public_identities: 0, internal_identities: 0,
      });
    }
    const count = cohorts.get(bundleCohort);
    count.bundles += 1;
    for (const identity of bundle.identities) {
      if (identities.has(identity)) throw new Error(`${identity}: pilot identity belongs to two bundles`);
      identities.add(identity);
      const reviewed = control.identities.get(identity);
      if (!reviewed || reviewed.bundle_id !== completion.bundleId
        || reviewed.cohort !== bundleCohort) {
        throw new Error(`${identity}: pilot identity authority differs from its bundle`);
      }
      count.identities += 1;
      if (reviewed.public_identity.kind === "internal") {
        internalIdentities += 1;
        count.internal_identities += 1;
      } else {
        publicIdentities += 1;
        count.public_identities += 1;
      }
    }
  }
  const result = {
    bundles: history.length,
    identities: identities.size,
    public_identities: publicIdentities,
    internal_identities: internalIdentities,
    cohorts: [...cohorts.values()].sort(
      (left, right) => compareCodePoint(left.cohort, right.cohort),
    ),
  };
  if (JSON.stringify(result) !== JSON.stringify(control.pilotPolicy.counts)) {
    throw new Error("pilot measurement independently derived counts differ from reviewed policy");
  }
  return result;
}

function queueKey(queue) {
  return JSON.stringify(queueBinding(queue));
}

function referenceKey(value) {
  return `${value.path}\0${value.digest}`;
}

function assertUnique(seen, value, label) {
  if (seen.has(value)) throw new Error(`pilot measurement duplicates ${label} ${value}`);
  seen.add(value);
}

function assertSessionLeasesUnique(completion, leaseIds, leaseDigests) {
  const authorities = [
    completion.start.initialLeaseAuthority,
    completion.finalLeaseAuthority,
  ];
  const localIds = new Set(authorities.map((entry) => entry.lease.value.lease_id));
  const localDigests = new Set(authorities.map((entry) => entry.reference.digest));
  for (const leaseId of localIds) assertOwnedOnce(leaseIds, leaseId, completion.sessionId, "lease id");
  for (const digest of localDigests) {
    assertOwnedOnce(leaseDigests, digest, completion.sessionId, "lease digest");
  }
}

function assertOwnedOnce(owners, value, sessionId, label) {
  const owner = owners.get(value);
  if (owner !== undefined && owner !== sessionId) {
    throw new Error(`pilot measurement duplicates ${label} across sessions`);
  }
  owners.set(value, sessionId);
}
