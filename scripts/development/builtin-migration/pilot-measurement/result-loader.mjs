import {
  assertAuthorityLoadSession, assertSessionArtifact, loadedJsonValue,
  loadJsonArtifact, revalidateObservedArtifacts, withAuthorityTraversal,
} from "../authority-loading/index.mjs";
import { assertValidatedControl } from "../control.mjs";
import { deepImmutable } from "../immutable.mjs";
import { reconstructPilotMeasurement, derivedResultPayload } from "./reconstruction.mjs";
import { pilotMeasurementResultPath } from "./paths.mjs";
import { loadPilotMeasurementReview } from "./review-loader.mjs";
import { parseReference, parseResultValue, withSelfDigest } from "./schema.mjs";

const RESULTS = new WeakMap();
const SESSION_RESULTS = new WeakMap();

export function loadPilotMeasurement({
  session: sessionValue, reference: referenceValue, control: controlValue, repository,
}) {
  const session = assertAuthorityLoadSession(sessionValue);
  const control = assertValidatedControl(controlValue);
  const reference = parseReference(referenceValue, "pilot measurement result reference");
  const cached = sessionResults(session).get(reference.path);
  if (cached) {
    if (cached.reference.digest !== reference.digest) {
      throw new Error(`${reference.path}: pilot measurement result path was rebound`);
    }
    revalidateObservedArtifacts(session);
    return assertLoadedPilotMeasurement(cached, { session, control });
  }
  return withAuthorityTraversal(session, {
    domain: "pilot-measurement-result", path: reference.path,
    semanticDigest: reference.digest,
  }, () => loadUncached({ session, reference, control, repository }));
}

export function assertLoadedPilotMeasurement(value, {
  session: sessionValue, control: controlValue,
}) {
  const session = assertAuthorityLoadSession(sessionValue);
  const control = assertValidatedControl(controlValue);
  const record = RESULTS.get(value);
  if (!record || record.session !== session || record.control !== control) {
    throw new Error("operation requires the exact loaded pilot measurement result");
  }
  assertSessionArtifact(session, value.artifact, value.reference.path);
  return value;
}

export function revalidatePilotMeasurement(value, context) {
  const result = assertLoadedPilotMeasurement(value, context);
  revalidateObservedArtifacts(context.session);
  return result;
}

function loadUncached({ session, reference, control, repository }) {
  const artifact = loadJsonArtifact(session, reference.path, "pilot measurement result");
  assertSessionArtifact(session, artifact, reference.path);
  const value = parseResultValue(loadedJsonValue(artifact, session.root));
  if (value.digest !== reference.digest) {
    throw new Error("pilot measurement result semantic digest mismatch");
  }
  if (value.control_manifest_digest !== control.digest
    || value.pilot_policy_digest !== control.pilotPolicyDigest) {
    throw new Error("pilot measurement result belongs to another control or policy");
  }
  const review = loadPilotMeasurementReview({
    session,
    reference: {
      path: value.review_manifest.path,
      digest: value.review_manifest.semantic_digest,
    },
    control,
  });
  if (review.artifact.contentDigest !== value.review_manifest.content_digest) {
    throw new Error("pilot measurement result review-manifest observation mismatch");
  }
  if (reference.path !== pilotMeasurementResultPath(review.reference.digest)) {
    throw new Error("pilot measurement result is outside its deterministic review path");
  }
  const reconstructed = reconstructPilotMeasurement({
    review, session, control, repository,
  });
  const expected = withSelfDigest(derivedResultPayload(review, reconstructed, control));
  if (JSON.stringify(value) !== JSON.stringify(expected)) {
    throw new Error("pilot measurement result differs from exact reconstructed evidence");
  }
  revalidateObservedArtifacts(session);
  const immutable = deepImmutable({ value: expected, reference });
  const result = Object.freeze({
    ...immutable,
    artifact,
    review,
    initialQueue: reconstructed.initialQueue,
    finalQueue: reconstructed.finalQueue,
    completions: reconstructed.history,
  });
  RESULTS.set(result, { session, control });
  sessionResults(session).set(reference.path, result);
  return result;
}

function sessionResults(session) {
  let results = SESSION_RESULTS.get(session);
  if (!results) {
    results = new Map();
    SESSION_RESULTS.set(session, results);
  }
  return results;
}
