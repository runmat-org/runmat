import path from "node:path";

import {
  assertAuthorityLoadSession, revalidateObservedArtifacts,
} from "../authority-loading/index.mjs";
import { publishEvidenceBytes } from "../atomic-evidence-publication.mjs";
import { assertValidatedControl } from "../control.mjs";
import { assertLoadedPilotMeasurement } from "../pilot-measurement/index.mjs";
import { loadPilotEvaluation } from "./loader.mjs";
import { pilotEvaluationPaths } from "./paths.mjs";
import { withEvaluationDigest } from "./schema.mjs";
import { derivePilotEvaluation } from "./validation.mjs";

export function recordPilotEvaluation(input) {
  assertRecorderInput(input);
  const {
    session: sessionValue, control: controlValue, measurement: measurementValue,
    limiterReforecast = null, repository,
  } = input;
  const session = assertAuthorityLoadSession(sessionValue);
  const control = assertValidatedControl(controlValue);
  const measurement = assertLoadedPilotMeasurement(measurementValue, { session, control });
  const value = withEvaluationDigest(derivePilotEvaluation({
    session, control, measurement, limiterReforecast,
  }));
  revalidateObservedArtifacts(session);
  const relativePath = pilotEvaluationPaths(measurement.reference.digest).evaluation;
  publishEvidenceBytes(
    path.join(session.root.path, ...relativePath.split("/")),
    `${JSON.stringify(value, null, 2)}\n`,
    { createParentDirectories: true },
  );
  return loadPilotEvaluation({
    session,
    reference: {
      path: relativePath,
      artifact_id: value.artifact_id,
      digest: value.digest,
    },
    control,
    repository,
  });
}

function assertRecorderInput(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("pilot evaluation recorder input must be an object");
  }
  const required = ["control", "measurement", "repository", "session"];
  const allowed = [...required, "limiterReforecast"];
  const keys = Object.keys(value).sort();
  if (required.some((field) => !keys.includes(field))
      || keys.some((field) => !allowed.includes(field))) {
    throw new Error(
      "pilot evaluation recorder accepts only session, control, measurement, "
      + "repository, and optional limiterReforecast",
    );
  }
}
