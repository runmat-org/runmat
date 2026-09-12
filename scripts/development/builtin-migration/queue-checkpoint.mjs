import { assertValidatedControl } from "./control.mjs";
import { evidenceDigest } from "./evidence.mjs";
import { deepImmutable } from "./immutable.mjs";
import { assertValidatedQueueState, acceptedSealSet } from "./queue.mjs";
import {
  digest, exact, kind, repositoryPath, sourceRevision, stableId, uniqueStrings,
} from "./schema.mjs";

const VALIDATED_QUEUE_CHECKPOINTS = new WeakSet();

export function validateQueueCheckpoint(
  value, trustedDigest, queueState, control, loadPredecessorCheckpoint = () => null,
) {
  assertValidatedControl(control);
  const state = assertValidatedQueueState(queueState, control);
  kind(value, 1, "runmat-builtin-migration-queue-checkpoint", "queue checkpoint");
  exact(value, [
    "schema_version", "kind", "authority", "control_manifest_digest",
    "queue_state_digest", "predecessor_checkpoint_digest", "head_seal",
    "source_revision", "source_digest", "accepted_seal_set_digest", "review", "digest",
  ], "queue checkpoint");
  if (value.authority !== "reviewer-authored-current-queue-checkpoint") {
    throw new Error("queue checkpoint has invalid authority");
  }
  if (value.control_manifest_digest !== control.digest) {
    throw new Error("queue checkpoint belongs to another control manifest");
  }
  if (digest(trustedDigest, "trusted queue checkpoint digest") !== value.digest) {
    throw new Error("queue checkpoint differs from the trusted current checkpoint digest");
  }
  if (value.queue_state_digest !== state.stateDigest) {
    throw new Error("queue checkpoint does not bind the supplied queue state");
  }
  const accepted = acceptedSealSet(state, control);
  if (value.accepted_seal_set_digest !== accepted.value.digest) {
    throw new Error("queue checkpoint accepted seal set differs from its queue state");
  }
  const predecessorDigest = state.predecessor?.checkpoint_digest ?? null;
  if (value.predecessor_checkpoint_digest !== predecessorDigest) {
    throw new Error("queue checkpoint predecessor differs from its queue state");
  }
  if (state.predecessor === null) {
    if (value.head_seal !== null || value.source_revision !== control.baseline.revision
      || value.source_digest !== control.baseline.source_digest) {
      throw new Error("initial queue checkpoint must bind the frozen control baseline");
    }
  } else {
    const previous = loadPredecessorCheckpoint(state.predecessor);
    validateQueueCheckpoint(
      previous, state.predecessor.checkpoint_digest, state.predecessorState,
      control, loadPredecessorCheckpoint,
    );
    const head = parseHeadSeal(value.head_seal);
    const priorIds = new Set(state.predecessorState.acceptedSeals.map((entry) => entry.bundle_id));
    const appended = state.sealedBundles.filter((entry) =>
      !priorIds.has(entry.reference.bundle_id));
    const sealed = appended.length === 1 && JSON.stringify(appended[0].reference) === JSON.stringify(head)
      ? appended[0] : null;
    if (!sealed || sealed.integrated_revision !== value.source_revision
      || sealed.source_digest !== value.source_digest) {
      throw new Error("queue checkpoint head seal does not bind its integration source");
    }
  }
  sourceRevision(value.source_revision, "queue checkpoint source revision");
  digest(value.source_digest, "queue checkpoint source digest");
  digest(value.accepted_seal_set_digest, "queue checkpoint accepted seal-set digest");
  exact(value.review, ["status", "evidence"], "queue checkpoint review");
  if (value.review.status !== "reviewed") throw new Error("queue checkpoint must be reviewed");
  uniqueStrings(value.review.evidence, "queue checkpoint review evidence");
  digest(value.digest, "queue checkpoint digest");
  const { digest: _ignored, ...payload } = value;
  if (evidenceDigest(payload) !== value.digest) throw new Error("queue checkpoint digest mismatch");
  const result = deepImmutable({ value, digest: value.digest, queueState: state });
  VALIDATED_QUEUE_CHECKPOINTS.add(result);
  return result;
}

export function assertValidatedQueueCheckpoint(value, control, queueState) {
  assertValidatedControl(control);
  const state = assertValidatedQueueState(queueState, control);
  if (!VALIDATED_QUEUE_CHECKPOINTS.has(value)) {
    throw new Error("operation requires the exact validated queue checkpoint");
  }
  if (value.value.control_manifest_digest !== control.digest
    || value.value.queue_state_digest !== state.stateDigest) {
    throw new Error("queue checkpoint was validated for another control or queue state");
  }
  return value;
}

function parseHeadSeal(value) {
  exact(value, ["path", "artifact_id", "digest", "bundle_id"], "queue checkpoint head seal");
  return {
    path: repositoryPath(value.path, "queue checkpoint head seal path"),
    artifact_id: stableId(value.artifact_id, "queue checkpoint head seal artifact id"),
    digest: digest(value.digest, "queue checkpoint head seal digest"),
    bundle_id: stableId(value.bundle_id, "queue checkpoint head seal bundle id"),
  };
}
