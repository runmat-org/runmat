import { evidenceDigest } from "../evidence.mjs";
import {
  array, digest, enumValue, exact, integer, kind, stableId,
} from "../schema.mjs";
import { canonicalStrings } from "../topology/schema.mjs";
import { parseArtifactBinding } from "./bindings.mjs";

export const LIMITER_KIND = "runmat-builtin-migration-pilot-limiter-reforecast";
export const LIMITER_VERSION = 1;

const LIMITER_CATEGORIES = Object.freeze([
  "build-capacity",
  "dependency-blocker",
  "implementation-complexity",
  "review-capacity",
  "test-capacity",
  "tooling-gap",
]);

export function parseLimiterValue(value) {
  kind(value, LIMITER_VERSION, LIMITER_KIND, "pilot limiter reforecast");
  exact(value, [
    "schema_version", "kind", "authority", "control_manifest_digest",
    "pilot_policy_digest", "pilot_id", "measurement", "limiter", "reforecast",
    "review", "digest",
  ], "pilot limiter reforecast");
  if (value.authority !== "reviewer-authored-development-input") {
    throw new Error("pilot limiter reforecast has invalid authority");
  }
  digest(value.control_manifest_digest, "pilot limiter control digest");
  digest(value.pilot_policy_digest, "pilot limiter policy digest");
  stableId(value.pilot_id, "pilot limiter pilot id");
  parseArtifactBinding(value.measurement, "pilot limiter measurement");
  parseLimiter(value.limiter);
  parseReforecast(value.reforecast);
  parseReview(value.review);
  assertSelfDigest(value, "pilot limiter reforecast");
  return value;
}

export function withSelfDigest(payload) {
  return { ...payload, digest: evidenceDigest(payload) };
}

function parseLimiter(value) {
  exact(value, ["category", "evidence"], "pilot limiter finding");
  enumValue(value.category, LIMITER_CATEGORIES, "pilot limiter category");
  concreteEvidence(value.evidence, "pilot limiter evidence");
}

function parseReforecast(value) {
  exact(value, [
    "aggregate_worker_milliseconds", "elapsed_milliseconds", "evidence",
  ], "pilot revised forecast");
  integer(
    value.aggregate_worker_milliseconds,
    "pilot forecast aggregate worker milliseconds",
    1,
  );
  integer(value.elapsed_milliseconds, "pilot forecast elapsed milliseconds", 1);
  concreteEvidence(value.evidence, "pilot revised forecast evidence");
}

function parseReview(value) {
  exact(value, ["status", "evidence"], "pilot limiter review");
  if (value.status !== "reviewed") {
    throw new Error("pilot limiter review status must be reviewed");
  }
  concreteEvidence(value.evidence, "pilot limiter review evidence");
}

function concreteEvidence(value, label) {
  canonicalStrings(array(value, label), label);
}

function assertSelfDigest(value, label) {
  digest(value.digest, `${label} digest`);
  const { digest: _ignored, ...payload } = value;
  if (evidenceDigest(payload) !== value.digest) throw new Error(`${label} digest mismatch`);
}
