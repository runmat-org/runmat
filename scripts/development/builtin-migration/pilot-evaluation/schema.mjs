import { canonicalAuthorityPath } from "../authority-loading/index.mjs";
import { evidenceDigest } from "../evidence.mjs";
import {
  array, digest, enumValue, exact, integer, kind, sourceRevision, stableId,
} from "../schema.mjs";
import { parseCohort } from "../topology/schema.mjs";
import { parseArtifactBinding } from "./bindings.mjs";

export const EVALUATION_KIND = "runmat-builtin-migration-pilot-evaluation";
export const EVALUATION_VERSION = 1;

export function parseEvaluationValue(value) {
  kind(value, EVALUATION_VERSION, EVALUATION_KIND, "pilot evaluation");
  exact(value, [
    "schema_version", "kind", "authority", "control_manifest_digest",
    "pilot_policy_digest", "pilot_id", "artifact_id", "measurement", "policy", "counts",
    "timing", "final_queue", "seals", "seal_set_digest", "source",
    "admission_comparison", "outcome", "limiter_reforecast", "digest",
  ], "pilot evaluation");
  if (value.authority !== "derived-from-persisted-pilot-measurement") {
    throw new Error("pilot evaluation has invalid authority");
  }
  digest(value.control_manifest_digest, "pilot evaluation control digest");
  digest(value.pilot_policy_digest, "pilot evaluation policy digest");
  stableId(value.pilot_id, "pilot evaluation pilot id");
  stableId(value.artifact_id, "pilot evaluation artifact id");
  parseArtifactBinding(value.measurement, "pilot evaluation measurement");
  parsePolicy(value.policy);
  parseCounts(value.counts);
  parseTiming(value.timing);
  parseQueue(value.final_queue);
  array(value.seals, "pilot evaluation seals").forEach(parseSeal);
  digest(value.seal_set_digest, "pilot evaluation seal-set digest");
  parseSource(value.source);
  parseComparison(value.admission_comparison);
  enumValue(value.outcome, ["threshold-met", "below-target"], "pilot evaluation outcome");
  if (value.limiter_reforecast !== null) {
    parseArtifactBinding(value.limiter_reforecast, "pilot evaluation limiter reforecast");
  }
  assertSelfDigest(value);
  return value;
}

export function withEvaluationDigest(payload) {
  return { ...payload, digest: evidenceDigest(payload) };
}

function parsePolicy(value) {
  exact(value, [
    "minimum_public_identities_per_aggregate_hour", "maximum_elapsed_milliseconds",
    "required_waived_gate_count", "ordinary_gate_policy",
  ], "pilot evaluation policy");
  exact(value.minimum_public_identities_per_aggregate_hour, [
    "numerator", "denominator",
  ], "pilot evaluation minimum rate");
  integer(value.minimum_public_identities_per_aggregate_hour.numerator,
    "pilot evaluation rate numerator", 1);
  integer(value.minimum_public_identities_per_aggregate_hour.denominator,
    "pilot evaluation rate denominator", 1);
  integer(value.maximum_elapsed_milliseconds,
    "pilot evaluation maximum elapsed milliseconds", 1);
  integer(value.required_waived_gate_count,
    "pilot evaluation required waived gate count", 0);
  if (value.required_waived_gate_count !== 0
      || value.ordinary_gate_policy !== "all-required-gates-must-pass") {
    throw new Error("pilot evaluation must preserve zero waivers and all required gates");
  }
}

function parseCounts(value) {
  exact(value, [
    "bundles", "identities", "public_identities", "internal_identities", "cohorts",
  ], "pilot evaluation counts");
  for (const field of ["bundles", "identities", "public_identities", "internal_identities"]) {
    integer(value[field], `pilot evaluation ${field}`, 0);
  }
  array(value.cohorts, "pilot evaluation cohort counts").forEach((entry, index) => {
    exact(entry, [
      "cohort", "bundles", "identities", "public_identities", "internal_identities",
    ], `pilot evaluation cohort ${index + 1}`);
    parseCohort(entry.cohort, `pilot evaluation cohort ${index + 1} id`);
    for (const field of ["bundles", "identities", "public_identities", "internal_identities"]) {
      integer(entry[field], `pilot evaluation cohort ${index + 1} ${field}`, 0);
    }
  });
}

function parseTiming(value) {
  exact(value, [
    "started_at", "ended_at", "elapsed_ms", "aggregate_worker_ms",
  ], "pilot evaluation timing");
  for (const field of ["started_at", "ended_at"]) {
    const parsed = Date.parse(value[field]);
    if (!Number.isSafeInteger(parsed) || new Date(parsed).toISOString() !== value[field]) {
      throw new Error(`pilot evaluation ${field} must be an exact UTC millisecond timestamp`);
    }
  }
  integer(value.elapsed_ms, "pilot evaluation elapsed milliseconds", 1);
  integer(value.aggregate_worker_ms, "pilot evaluation aggregate worker milliseconds", 1);
}

function parseQueue(value) {
  exact(value, ["state", "checkpoint"], "pilot evaluation final queue");
  parseArtifactBinding(value.state, "pilot evaluation final queue state");
  parseArtifactBinding(value.checkpoint, "pilot evaluation final queue checkpoint");
}

function parseSeal(value, index) {
  const label = `pilot evaluation seal ${index + 1}`;
  exact(value, ["path", "artifact_id", "digest", "bundle_id"], label);
  // Seal references are already reconstructed and validated by the measurement authority.
  canonicalAuthorityPath(value.path, `${label} path`);
  stableId(value.artifact_id, `${label} id`);
  digest(value.digest, `${label} digest`);
  stableId(value.bundle_id, `${label} bundle`);
}

function parseSource(value) {
  exact(value, ["revision", "source_digest", "inventory_digest"], "pilot evaluation source");
  sourceRevision(value.revision, "pilot evaluation source revision");
  digest(value.source_digest, "pilot evaluation source digest");
  digest(value.inventory_digest, "pilot evaluation inventory digest");
}

function parseComparison(value) {
  exact(value, ["rate", "elapsed", "threshold_met"], "pilot admission comparison");
  exact(value.rate, [
    "public_identities", "aggregate_worker_milliseconds", "minimum_rate_numerator",
    "minimum_rate_denominator", "milliseconds_per_hour", "measured_operand",
    "required_operand", "meets_minimum",
  ], "pilot admission rate comparison");
  for (const field of [
    "public_identities", "aggregate_worker_milliseconds", "minimum_rate_numerator",
    "minimum_rate_denominator", "milliseconds_per_hour",
  ]) {
    integer(
      value.rate[field],
      `pilot admission rate ${field}`,
      field === "public_identities" ? 0 : 1,
    );
  }
  decimal(value.rate.measured_operand, "pilot admission measured operand");
  decimal(value.rate.required_operand, "pilot admission required operand");
  boolean(value.rate.meets_minimum, "pilot admission rate decision");
  exact(value.elapsed, [
    "elapsed_milliseconds", "maximum_elapsed_milliseconds", "within_maximum",
  ], "pilot admission elapsed comparison");
  integer(value.elapsed.elapsed_milliseconds, "pilot admission elapsed milliseconds", 1);
  integer(value.elapsed.maximum_elapsed_milliseconds,
    "pilot admission maximum elapsed milliseconds", 1);
  boolean(value.elapsed.within_maximum, "pilot admission elapsed decision");
  boolean(value.threshold_met, "pilot admission threshold decision");
}

function decimal(value, label) {
  if (typeof value !== "string" || !/^(?:0|[1-9][0-9]*)$/.test(value)) {
    throw new Error(`${label} must be a canonical nonnegative decimal integer`);
  }
  return value;
}

function boolean(value, label) {
  if (typeof value !== "boolean") throw new Error(`${label} must be boolean`);
}

function assertSelfDigest(value) {
  digest(value.digest, "pilot evaluation digest");
  const { digest: _ignored, ...payload } = value;
  if (evidenceDigest(payload) !== value.digest) throw new Error("pilot evaluation digest mismatch");
}
