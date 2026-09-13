import {
  assertAuthorityLoadSession, revalidateObservedArtifacts,
} from "../authority-loading/index.mjs";
import { assertValidatedControl } from "../control.mjs";
import { assertLoadedPilotEvaluation } from "../pilot-evaluation.mjs";
import { loadQueueCheckpoint, openQueueCheckpointLoader } from "../queue-authority/checkpoint-loader.mjs";
import { openQueueReferenceIndex } from "../queue-authority/reference.mjs";
import { openQueueSealLoader } from "../queue-authority/seal-loader.mjs";
import { loadQueueState, openQueueStateLoader } from "../queue-authority/state-loader.mjs";
import { pilotTransitionPaths } from "./paths.mjs";
import { derivePilotTransition, pilotTransitionValidator } from "./validation.mjs";

const TRANSITIONS = new WeakMap();

export function loadPilotTransition({ session: sessionValue, control: controlValue, evaluation: value }) {
  const session = assertAuthorityLoadSession(sessionValue);
  const control = assertValidatedControl(controlValue);
  const evaluation = assertLoadedPilotEvaluation(value, { session, control });
  const paths = pilotTransitionPaths(evaluation.reference.digest);
  const expected = derivePilotTransition({ session, control, evaluation });
  const references = openQueueReferenceIndex(session);
  const seals = openQueueSealLoader({ session, references, control });
  const states = openQueueStateLoader({
    session, references, seals, control,
    validatePilotTransition: pilotTransitionValidator({ session, control, evaluation }),
  });
  const finalQueue = evaluation.measurement.finalQueue;
  loadQueueState(states, {
    path: finalQueue.observations.state.path,
    digest: finalQueue.observations.state.semanticDigest,
  });
  const state = loadQueueState(states, { path: paths.state, digest: null });
  const checkpoints = openQueueCheckpointLoader({ session, references, control });
  const checkpoint = loadQueueCheckpoint(checkpoints, {
    path: paths.checkpoint,
    digest: expected.checkpoint.digest,
  }, state.value);
  revalidateObservedArtifacts(session);
  const result = Object.freeze({
    state: state.value,
    checkpoint: checkpoint.value,
    observations: Object.freeze({
      state: observation(state), checkpoint: observation(checkpoint),
    }),
    evaluation,
  });
  TRANSITIONS.set(result, { session, control, evaluation });
  return result;
}

export function assertLoadedPilotTransition(value, {
  session: sessionValue, control: controlValue, evaluation: evaluationValue,
}) {
  const record = TRANSITIONS.get(value);
  if (!record || record.session !== assertAuthorityLoadSession(sessionValue)
    || record.control !== assertValidatedControl(controlValue)
    || record.evaluation !== evaluationValue) {
    throw new Error("operation requires the exact loaded pilot transition");
  }
  revalidateObservedArtifacts(record.session);
  return value;
}

function observation(value) {
  return Object.freeze({
    path: value.path,
    semanticDigest: value.semanticDigest,
    contentDigest: value.contentDigest,
  });
}
