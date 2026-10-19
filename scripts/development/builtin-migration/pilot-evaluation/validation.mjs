import { assertValidatedControl } from "../control.mjs";
import { assertLoadedPilotMeasurement } from "../pilot-measurement/index.mjs";
import { pilotAdmissionComparison } from "../pilot-policy.mjs";
import { artifactBinding, loadedAuthorityBinding } from "./bindings.mjs";
import { assertLoadedPilotLimiterReforecast } from "./limiter-loader.mjs";

export function derivePilotEvaluation({
  session, control: controlValue, measurement: measurementValue,
  limiterReforecast: limiterValue = null,
}) {
  const control = assertValidatedControl(controlValue);
  const measurement = assertLoadedPilotMeasurement(measurementValue, { session, control });
  const { counts, timing } = measurement.value;
  const comparison = pilotAdmissionComparison(control.pilotPolicy, {
    publicIdentities: counts.public_identities,
    aggregateWorkerMilliseconds: timing.aggregate_worker_ms,
    elapsedMilliseconds: timing.elapsed_ms,
  });
  const limiter = validateOutcomeAuthority({
    session, control, measurement, comparison, limiterValue,
  });
  const admission = control.pilotPolicy.admission;
  return {
    schema_version: 1,
    kind: "runmat-builtin-migration-pilot-evaluation",
    authority: "derived-from-persisted-pilot-measurement",
    control_manifest_digest: control.digest,
    pilot_policy_digest: control.pilotPolicyDigest,
    pilot_id: control.pilotPolicy.pilotId,
    artifact_id: evaluationArtifactId(measurement.reference.digest),
    measurement: loadedAuthorityBinding(measurement),
    policy: {
      minimum_public_identities_per_aggregate_hour: { ...admission.rate },
      maximum_elapsed_milliseconds: admission.maximumElapsedMilliseconds,
      required_waived_gate_count: 0,
      ordinary_gate_policy: "all-required-gates-must-pass",
    },
    counts: measurement.value.counts,
    timing: measurement.value.timing,
    final_queue: measurement.value.final_queue,
    seals: measurement.value.seals,
    seal_set_digest: measurement.finalQueue.checkpoint.value.accepted_seal_set_digest,
    source: measurement.value.source,
    admission_comparison: comparison,
    outcome: comparison.threshold_met ? "threshold-met" : "below-target",
    limiter_reforecast: limiter === null ? null : artifactBinding(limiter.observation),
  };
}

export function evaluationArtifactId(measurementDigest) {
  if (typeof measurementDigest !== "string" || !/^sha256:[a-f0-9]{64}$/.test(measurementDigest)) {
    throw new Error("pilot evaluation measurement digest must be a lowercase sha256 digest");
  }
  return `pilot-evaluation-${measurementDigest.slice("sha256:".length)}`;
}

function validateOutcomeAuthority({
  session, control, measurement, comparison, limiterValue,
}) {
  if (comparison.threshold_met) {
    if (limiterValue !== null) {
      throw new Error("threshold-met pilot evaluation must not include limiter data");
    }
    return null;
  }
  if (limiterValue === null) {
    throw new Error("below-target pilot evaluation requires a reviewed limiter and reforecast");
  }
  return assertLoadedPilotLimiterReforecast(limiterValue, {
    session, control, measurement,
  });
}
