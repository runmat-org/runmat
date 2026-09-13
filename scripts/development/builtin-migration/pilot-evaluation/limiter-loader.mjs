import {
  assertAuthorityLoadSession, assertSessionArtifact, loadedJsonValue,
  loadJsonArtifact, revalidateObservedArtifacts, withAuthorityTraversal,
} from "../authority-loading/index.mjs";
import { assertValidatedControl } from "../control.mjs";
import { deepImmutable } from "../immutable.mjs";
import { assertLoadedPilotMeasurement } from "../pilot-measurement/index.mjs";
import { parseReference } from "../pilot-measurement/schema.mjs";
import { loadedAuthorityBinding } from "./bindings.mjs";
import { parseLimiterValue } from "./limiter-schema.mjs";
import { pilotEvaluationPaths } from "./paths.mjs";

const LIMITERS = new WeakMap();
const SESSION_LIMITERS = new WeakMap();

export function loadPilotLimiterReforecast({
  session: sessionValue, reference: referenceValue, control: controlValue,
  measurement: measurementValue,
}) {
  const session = assertAuthorityLoadSession(sessionValue);
  const control = assertValidatedControl(controlValue);
  const measurement = assertLoadedPilotMeasurement(measurementValue, { session, control });
  const reference = parseReference(referenceValue, "pilot limiter reforecast reference");
  if (reference.path !== pilotEvaluationPaths(measurement.reference.digest).limiterReforecast) {
    throw new Error("pilot limiter reforecast is outside its deterministic measurement path");
  }
  const cached = sessionLimiters(session).get(reference.path);
  if (cached) {
    if (cached.reference.digest !== reference.digest) {
      throw new Error(`${reference.path}: pilot limiter reforecast path was rebound`);
    }
    revalidateObservedArtifacts(session);
    return assertLoadedPilotLimiterReforecast(cached, { session, control, measurement });
  }
  return withAuthorityTraversal(session, {
    domain: "pilot-limiter-reforecast", path: reference.path,
    semanticDigest: reference.digest,
  }, () => loadUncached({ session, reference, control, measurement }));
}

export function assertLoadedPilotLimiterReforecast(value, {
  session: sessionValue, control: controlValue, measurement: measurementValue,
}) {
  const session = assertAuthorityLoadSession(sessionValue);
  const control = assertValidatedControl(controlValue);
  const measurement = assertLoadedPilotMeasurement(measurementValue, { session, control });
  const record = LIMITERS.get(value);
  if (!record || record.session !== session || record.control !== control
      || record.measurement !== measurement) {
    throw new Error("operation requires the exact loaded pilot limiter reforecast");
  }
  assertSessionArtifact(session, value.artifact, value.reference.path);
  return value;
}

function loadUncached({ session, reference, control, measurement }) {
  const artifact = loadJsonArtifact(session, reference.path, "pilot limiter reforecast");
  assertSessionArtifact(session, artifact, reference.path);
  const value = parseLimiterValue(loadedJsonValue(artifact, session.root));
  if (value.digest !== reference.digest) {
    throw new Error("pilot limiter reforecast semantic digest mismatch");
  }
  if (value.control_manifest_digest !== control.digest
      || value.pilot_policy_digest !== control.pilotPolicyDigest
      || value.pilot_id !== control.pilotPolicy.pilotId) {
    throw new Error("pilot limiter reforecast belongs to another control, policy, or pilot");
  }
  if (JSON.stringify(value.measurement) !== JSON.stringify(loadedAuthorityBinding(measurement))) {
    throw new Error("pilot limiter reforecast measurement observation mismatch");
  }
  revalidateObservedArtifacts(session);
  const immutable = deepImmutable({ value, reference, observation: {
    path: artifact.path,
    semanticDigest: reference.digest,
    contentDigest: artifact.contentDigest,
  } });
  const result = Object.freeze({ ...immutable, artifact, control, measurement });
  LIMITERS.set(result, { session, control, measurement });
  sessionLimiters(session).set(reference.path, result);
  return result;
}

function sessionLimiters(session) {
  let limiters = SESSION_LIMITERS.get(session);
  if (!limiters) {
    limiters = new Map();
    SESSION_LIMITERS.set(session, limiters);
  }
  return limiters;
}
