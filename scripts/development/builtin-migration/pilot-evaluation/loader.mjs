import {
  assertAuthorityLoadSession, assertSessionArtifact, canonicalAuthorityPath,
  loadedJsonValue, loadJsonArtifact, revalidateObservedArtifacts,
  withAuthorityTraversal,
} from "../authority-loading/index.mjs";
import { assertValidatedControl } from "../control.mjs";
import { deepImmutable } from "../immutable.mjs";
import { loadPilotMeasurement } from "../pilot-measurement/index.mjs";
import { digest, exact, stableId } from "../schema.mjs";
import { loadedAuthorityBinding } from "./bindings.mjs";
import { loadPilotLimiterReforecast } from "./limiter-loader.mjs";
import { parseEvaluationValue, withEvaluationDigest } from "./schema.mjs";
import { pilotEvaluationPaths } from "./paths.mjs";
import { derivePilotEvaluation, evaluationArtifactId } from "./validation.mjs";

const EVALUATIONS = new WeakMap();
const SESSION_EVALUATIONS = new WeakMap();

export function loadPilotEvaluation({
  session: sessionValue, reference: referenceValue, control: controlValue, repository,
}) {
  const session = assertAuthorityLoadSession(sessionValue);
  const control = assertValidatedControl(controlValue);
  const reference = parseEvaluationReference(referenceValue);
  const cached = sessionEvaluations(session).get(reference.path);
  if (cached) {
    if (cached.reference.digest !== reference.digest
        || cached.reference.artifact_id !== reference.artifact_id) {
      throw new Error(`${reference.path}: pilot evaluation path was rebound`);
    }
    revalidateObservedArtifacts(session);
    return assertLoadedPilotEvaluation(cached, { session, control });
  }
  return withAuthorityTraversal(session, {
    domain: "pilot-evaluation", path: reference.path, semanticDigest: reference.digest,
  }, () => loadUncached({ session, reference, control, repository }));
}

export function assertLoadedPilotEvaluation(value, {
  session: sessionValue, control: controlValue,
}) {
  const session = assertAuthorityLoadSession(sessionValue);
  const control = assertValidatedControl(controlValue);
  const record = EVALUATIONS.get(value);
  if (!record || record.session !== session || record.control !== control) {
    throw new Error("operation requires the exact loaded pilot evaluation");
  }
  assertSessionArtifact(session, value.artifact, value.reference.path);
  return value;
}

export function revalidatePilotEvaluation(value, context) {
  const evaluation = assertLoadedPilotEvaluation(value, context);
  revalidateObservedArtifacts(context.session);
  return evaluation;
}

function loadUncached({ session, reference, control, repository }) {
  const artifact = loadJsonArtifact(session, reference.path, "pilot evaluation");
  assertSessionArtifact(session, artifact, reference.path);
  const value = parseEvaluationValue(loadedJsonValue(artifact, session.root));
  assertReference(value, reference, control);
  const measurement = loadPilotMeasurement({
    session,
    reference: {
      path: value.measurement.path,
      digest: value.measurement.semantic_digest,
    },
    control,
    repository,
  });
  if (JSON.stringify(loadedAuthorityBinding(measurement))
      !== JSON.stringify(value.measurement)) {
    throw new Error("pilot evaluation measurement observation mismatch");
  }
  const limiterReforecast = loadLimiter(value, { session, control, measurement });
  const expected = withEvaluationDigest(derivePilotEvaluation({
    session, control, measurement, limiterReforecast,
  }));
  if (JSON.stringify(value) !== JSON.stringify(expected)) {
    throw new Error("pilot evaluation differs from exact recomputed evidence");
  }
  revalidateObservedArtifacts(session);
  const immutable = deepImmutable({ value: expected, reference });
  const result = Object.freeze({
    ...immutable, artifact, control, measurement, limiterReforecast,
  });
  EVALUATIONS.set(result, { session, control });
  sessionEvaluations(session).set(reference.path, result);
  return result;
}

function loadLimiter(value, { session, control, measurement }) {
  if (value.limiter_reforecast === null) return null;
  const limiter = loadPilotLimiterReforecast({
    session,
    reference: {
      path: value.limiter_reforecast.path,
      digest: value.limiter_reforecast.semantic_digest,
    },
    control,
    measurement,
  });
  if (limiter.observation.contentDigest !== value.limiter_reforecast.content_digest) {
    throw new Error("pilot evaluation limiter observation mismatch");
  }
  return limiter;
}

function parseEvaluationReference(value) {
  exact(value, ["path", "artifact_id", "digest"], "pilot evaluation reference");
  return Object.freeze({
    path: canonicalAuthorityPath(value.path, "pilot evaluation reference path"),
    artifact_id: stableId(value.artifact_id, "pilot evaluation reference artifact id"),
    digest: digest(value.digest, "pilot evaluation reference digest"),
  });
}

function assertReference(value, reference, control) {
  if (value.digest !== reference.digest || value.artifact_id !== reference.artifact_id) {
    throw new Error("pilot evaluation reference does not match the persisted result");
  }
  if (value.control_manifest_digest !== control.digest
      || value.pilot_policy_digest !== control.pilotPolicyDigest
      || value.pilot_id !== control.pilotPolicy.pilotId) {
    throw new Error("pilot evaluation belongs to another control, policy, or pilot");
  }
  const paths = pilotEvaluationPaths(value.measurement.semantic_digest);
  if (reference.path !== paths.evaluation
      || reference.artifact_id !== evaluationArtifactId(value.measurement.semantic_digest)) {
    throw new Error("pilot evaluation reference is not deterministic for its measurement");
  }
}

function sessionEvaluations(session) {
  let evaluations = SESSION_EVALUATIONS.get(session);
  if (!evaluations) {
    evaluations = new Map();
    SESSION_EVALUATIONS.set(session, evaluations);
  }
  return evaluations;
}
