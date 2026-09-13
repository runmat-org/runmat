import { assertLoadedPilotEvaluation, revalidatePilotEvaluation } from "../pilot-evaluation.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { assertValidatedControl } from "../control.mjs";
import { assertLoadedQueueAuthority } from "../queue-authority/index.mjs";

export function derivePilotTransition({ session, control: controlValue, evaluation: value }) {
  const control = assertValidatedControl(controlValue);
  const evaluation = assertLoadedPilotEvaluation(value, { session, control });
  const finalQueue = evaluation.measurement.finalQueue;
  assertLoadedQueueAuthority(finalQueue, { session, control });
  revalidatePilotEvaluation(evaluation, { session, control });
  const evaluationReference = publicEvaluationReference(evaluation);
  const predecessor = {
    state_path: finalQueue.observations.state.path,
    state_digest: finalQueue.observations.state.semanticDigest,
    checkpoint_path: finalQueue.observations.checkpoint.path,
    checkpoint_digest: finalQueue.observations.checkpoint.semanticDigest,
  };
  const statePayload = {
    schema_version: 4,
    kind: "runmat-builtin-migration-queue-state",
    authority: "reviewed-monotonic-scheduling-input",
    control_manifest_digest: control.digest,
    phase: "production",
    pilot_transition: { evaluation: evaluationReference },
    predecessor,
    bundles: finalQueue.state.value.bundles,
    seals: finalQueue.state.value.seals,
  };
  const state = { ...statePayload, digest: evidenceDigest(statePayload) };
  const checkpointPayload = {
    schema_version: 2,
    kind: "runmat-builtin-migration-queue-checkpoint",
    authority: "reviewer-authored-current-queue-checkpoint",
    control_manifest_digest: control.digest,
    queue_state_digest: state.digest,
    predecessor_checkpoint_digest: finalQueue.checkpoint.value.digest,
    phase: "production",
    head_event: { kind: "pilot-to-production", evaluation: evaluationReference },
    source_revision: finalQueue.checkpoint.value.source_revision,
    source_digest: finalQueue.checkpoint.value.source_digest,
    inventory_digest: finalQueue.checkpoint.value.inventory_digest,
    accepted_seal_set_digest: finalQueue.checkpoint.value.accepted_seal_set_digest,
    review: {
      status: "reviewed",
      evidence: ["Exact validated pilot evaluation authorizes the production transition."],
    },
  };
  const checkpoint = { ...checkpointPayload, digest: evidenceDigest(checkpointPayload) };
  return Object.freeze({ state, checkpoint, finalQueue, evaluation });
}

export function pilotTransitionValidator({ session, control: controlValue, evaluation: value }) {
  const control = assertValidatedControl(controlValue);
  const evaluation = assertLoadedPilotEvaluation(value, { session, control });
  const expectedQueue = evaluation.measurement.finalQueue;
  const expectedReference = publicEvaluationReference(evaluation);
  return ({ predecessorState, pilotTransition, acceptedSeals }) => {
    if (predecessorState.stateDigest !== expectedQueue.state.stateDigest
      || JSON.stringify(predecessorState.value) !== JSON.stringify(expectedQueue.state.value)) {
      throw new Error("pilot transition predecessor is not the evaluated final queue");
    }
    if (JSON.stringify(pilotTransition?.evaluation) !== JSON.stringify(expectedReference)) {
      throw new Error("pilot transition does not reference the exact loaded evaluation");
    }
    if (JSON.stringify(acceptedSeals) !== JSON.stringify(evaluation.value.seals)) {
      throw new Error("pilot transition seal set differs from the evaluated pilot");
    }
    revalidatePilotEvaluation(evaluation, { session, control });
    return evaluation;
  };
}

export function publicEvaluationReference(evaluation) {
  return Object.freeze({
    path: evaluation.reference.path,
    artifact_id: evaluation.value.artifact_id,
    digest: evaluation.reference.digest,
  });
}
