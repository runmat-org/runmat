import {
  assertAuthorityLoadSession, assertSessionArtifact, loadedJsonValue,
  loadJsonArtifact, revalidateObservedArtifacts, withAuthorityTraversal,
} from "../authority-loading/index.mjs";
import { assertValidatedControl } from "../control.mjs";
import { deepImmutable } from "../immutable.mjs";
import { parseReference, parseReviewValue } from "./schema.mjs";

const REVIEWS = new WeakMap();
const SESSION_REVIEWS = new WeakMap();

export function loadPilotMeasurementReview({
  session: sessionValue, reference: referenceValue, control: controlValue,
}) {
  const session = assertAuthorityLoadSession(sessionValue);
  const control = assertValidatedControl(controlValue);
  const reference = parseReference(referenceValue, "pilot measurement review reference");
  const cached = sessionReviews(session).get(reference.path);
  if (cached) {
    if (cached.reference.digest !== reference.digest) {
      throw new Error(`${reference.path}: pilot measurement review path was rebound`);
    }
    revalidateObservedArtifacts(session);
    return assertLoadedPilotMeasurementReview(cached, { session, control });
  }
  return withAuthorityTraversal(session, {
    domain: "pilot-measurement-review", path: reference.path,
    semanticDigest: reference.digest,
  }, () => {
    const artifact = loadJsonArtifact(session, reference.path, "pilot measurement review");
    assertSessionArtifact(session, artifact, reference.path);
    const value = parseReviewValue(loadedJsonValue(artifact, session.root));
    if (value.digest !== reference.digest) {
      throw new Error("pilot measurement review semantic digest mismatch");
    }
    if (value.control_manifest_digest !== control.digest
      || value.pilot_policy_digest !== control.pilotPolicyDigest) {
      throw new Error("pilot measurement review belongs to another control or policy");
    }
    revalidateObservedArtifacts(session);
    const immutable = deepImmutable({
      value,
      reference,
      observation: {
        path: artifact.path,
        semanticDigest: reference.digest,
        contentDigest: artifact.contentDigest,
      },
    });
    const result = Object.freeze({ ...immutable, artifact, control });
    REVIEWS.set(result, { session, control });
    sessionReviews(session).set(reference.path, result);
    return result;
  });
}

export function assertLoadedPilotMeasurementReview(value, {
  session: sessionValue, control: controlValue,
}) {
  const session = assertAuthorityLoadSession(sessionValue);
  const control = assertValidatedControl(controlValue);
  const record = REVIEWS.get(value);
  if (!record || record.session !== session || record.control !== control) {
    throw new Error("operation requires the exact loaded pilot measurement review");
  }
  assertSessionArtifact(session, value.artifact, value.reference.path);
  return value;
}

function sessionReviews(session) {
  let reviews = SESSION_REVIEWS.get(session);
  if (!reviews) {
    reviews = new Map();
    SESSION_REVIEWS.set(session, reviews);
  }
  return reviews;
}
