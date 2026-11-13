import path from "node:path";

import { loadPilotLimiterReforecast } from "../pilot-evaluation.mjs";
import { loadedAuthorityBinding } from "../pilot-evaluation/bindings.mjs";
import { withSelfDigest } from "../pilot-evaluation/limiter-schema.mjs";
import { pilotEvaluationPaths } from "../pilot-evaluation/paths.mjs";
import { pilotMeasurementFixture, writeJson } from "./pilot-measurement-fixture.mjs";

export function pilotEvaluationFixture({ belowTarget = false, loadLimiter = belowTarget } = {}) {
  const measurementFixture = belowTarget
    ? belowTargetMeasurementFixture()
    : pilotMeasurementFixture();
  const paths = pilotEvaluationPaths(measurementFixture.measurement.reference.digest);
  const limiterValue = limiterPayload(measurementFixture);
  if (loadLimiter) {
    writeJson(path.join(measurementFixture.root, paths.limiterReforecast), limiterValue);
  }
  const limiter = loadLimiter
    ? loadPilotLimiterReforecast({
      session: measurementFixture.session,
      reference: { path: paths.limiterReforecast, digest: limiterValue.digest },
      control: measurementFixture.control,
      measurement: measurementFixture.measurement,
    })
    : null;
  return { ...measurementFixture, paths, limiterValue, limiter };
}

function belowTargetMeasurementFixture() {
  const original = Date.now;
  let now = Date.parse("2026-01-01T00:00:00.000Z");
  Date.now = () => {
    const observed = now;
    now += 3_600_001;
    return observed;
  };
  try {
    return pilotMeasurementFixture();
  } finally {
    Date.now = original;
  }
}

export function limiterPayload(fixture) {
  return withSelfDigest({
    schema_version: 1,
    kind: "runmat-builtin-migration-pilot-limiter-reforecast",
    authority: "reviewer-authored-development-input",
    control_manifest_digest: fixture.control.digest,
    pilot_policy_digest: fixture.control.pilotPolicyDigest,
    pilot_id: fixture.control.pilotPolicy.pilotId,
    measurement: loadedAuthorityBinding(fixture.measurement),
    limiter: {
      category: "build-capacity",
      evidence: ["Observed serialized build capacity during the measured pilot."],
    },
    reforecast: {
      aggregate_worker_milliseconds: 3_600_000,
      elapsed_milliseconds: 3_600_000,
      evidence: ["A dedicated build lane supplies the revised measured-capacity forecast."],
    },
    review: {
      status: "reviewed",
      evidence: ["Reviewer accepted the limiter evidence and revised forecast."],
    },
  });
}
