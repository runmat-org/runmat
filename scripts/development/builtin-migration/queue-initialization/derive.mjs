import { assertValidatedControl } from "../control.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { validateQueueCheckpoint } from "../queue-checkpoint.mjs";
import { acceptedSealSet, emptyQueueState, validateQueueState } from "../queue.mjs";
import { assertLoadedInitialQueueReview } from "./review.mjs";

export function deriveInitialQueue({ session, control: controlValue, review: reviewValue }) {
  const control = assertValidatedControl(controlValue);
  const review = assertLoadedInitialQueueReview(reviewValue, { session, control });
  const stateValue = emptyQueueState(control);
  const state = validateQueueState(stateValue, control, () => null);
  const accepted = acceptedSealSet(state, control);
  const checkpointPayload = {
    schema_version: 2,
    kind: "runmat-builtin-migration-queue-checkpoint",
    authority: "reviewer-authored-current-queue-checkpoint",
    control_manifest_digest: control.digest,
    queue_state_digest: state.stateDigest,
    predecessor_checkpoint_digest: null,
    phase: "pilot",
    head_event: null,
    source_revision: control.baseline.revision,
    source_digest: control.baseline.source_digest,
    inventory_digest: control.baseline.inventory_digest,
    accepted_seal_set_digest: accepted.value.digest,
    review: review.value.review,
  };
  const checkpointValue = {
    ...checkpointPayload, digest: evidenceDigest(checkpointPayload),
  };
  const checkpoint = validateQueueCheckpoint(
    checkpointValue, checkpointValue.digest, state, control,
  );
  return Object.freeze({
    state, checkpoint, stateValue: state.value, checkpointValue: checkpoint.value,
  });
}
