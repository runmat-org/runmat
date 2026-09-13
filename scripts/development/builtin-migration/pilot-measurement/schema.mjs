import { canonicalAuthorityPath } from "../authority-loading/index.mjs";
import { compareCodePoint } from "../constants.mjs";
import { evidenceDigest } from "../evidence.mjs";
import {
  array, digest, exact, integer, kind, sourceRevision, stableId, uniqueStrings,
} from "../schema.mjs";
import { parseCohort, parseReviewedEvidence } from "../topology/schema.mjs";

export const REVIEW_KIND = "runmat-builtin-migration-pilot-measurement-review";
export const RESULT_KIND = "runmat-builtin-migration-pilot-measurement-result";
export const MEASUREMENT_VERSION = 1;

export function parseReviewValue(value) {
  kind(value, MEASUREMENT_VERSION, REVIEW_KIND, "pilot measurement review");
  exact(value, [
    "schema_version", "kind", "authority", "control_manifest_digest",
    "pilot_policy_digest", "initial_queue", "final_queue", "completed_sessions",
    "review", "digest",
  ], "pilot measurement review");
  if (value.authority !== "reviewer-authored-development-input") {
    throw new Error("pilot measurement review has invalid authority");
  }
  digest(value.control_manifest_digest, "pilot measurement control digest");
  digest(value.pilot_policy_digest, "pilot measurement policy digest");
  const sessions = array(value.completed_sessions, "pilot measurement sessions")
    .map((entry, index) => parseReference(entry, `pilot session reference ${index + 1}`));
  const keys = sessions.map(referenceKey);
  if (new Set(keys).size !== keys.length
    || JSON.stringify(keys) !== JSON.stringify([...keys].sort(compareCodePoint))) {
    throw new Error("pilot measurement session references must be unique and canonical");
  }
  parseReviewedEvidence(value.review, "pilot measurement review evidence");
  assertSelfDigest(value, "pilot measurement review");
  return Object.freeze({
    ...value,
    initial_queue: parseQueueBinding(value.initial_queue, "initial pilot queue"),
    final_queue: parseQueueBinding(value.final_queue, "final pilot queue"),
    completed_sessions: Object.freeze(sessions),
  });
}

export function parseResultValue(value) {
  kind(value, MEASUREMENT_VERSION, RESULT_KIND, "pilot measurement result");
  exact(value, [
    "schema_version", "kind", "authority", "control_manifest_digest",
    "pilot_policy_digest", "review_manifest", "initial_queue", "final_queue",
    "completed_sessions", "seals", "bundle_ids", "source", "counts", "timing",
    "digest",
  ], "pilot measurement result");
  if (value.authority !== "derived-from-reviewed-pilot-work-sessions") {
    throw new Error("pilot measurement result has invalid authority");
  }
  digest(value.control_manifest_digest, "pilot measurement result control digest");
  digest(value.pilot_policy_digest, "pilot measurement result policy digest");
  parseArtifactBinding(value.review_manifest, "pilot measurement review manifest");
  parseQueueBinding(value.initial_queue, "pilot measurement result initial queue");
  parseQueueBinding(value.final_queue, "pilot measurement result final queue");
  array(value.completed_sessions, "pilot measurement result sessions").forEach(
    (entry, index) => parseArtifactBinding(entry, `pilot result session ${index + 1}`),
  );
  array(value.seals, "pilot measurement result seals").forEach(parseSealReference);
  uniqueStrings(value.bundle_ids, "pilot measurement result bundle ids");
  parseSource(value.source);
  parseCounts(value.counts);
  parseTiming(value.timing);
  assertSelfDigest(value, "pilot measurement result");
  return value;
}

export function parseReference(value, label) {
  exact(value, ["path", "digest"], label);
  return Object.freeze({
    path: canonicalAuthorityPath(value.path, `${label} path`),
    digest: digest(value.digest, `${label} digest`),
  });
}

export function parseArtifactBinding(value, label) {
  exact(value, ["path", "semantic_digest", "content_digest"], label);
  return {
    path: canonicalAuthorityPath(value.path, `${label} path`),
    semantic_digest: digest(value.semantic_digest, `${label} semantic digest`),
    content_digest: digest(value.content_digest, `${label} content digest`),
  };
}

export function parseQueueBinding(value, label) {
  exact(value, ["state", "checkpoint"], label);
  return {
    state: parseArtifactBinding(value.state, `${label} state`),
    checkpoint: parseArtifactBinding(value.checkpoint, `${label} checkpoint`),
  };
}

export function artifactBinding(observation) {
  return {
    path: observation.path,
    semantic_digest: observation.semanticDigest,
    content_digest: observation.contentDigest,
  };
}

export function queueBinding(queue) {
  return {
    state: artifactBinding(queue.observations.state),
    checkpoint: artifactBinding(queue.observations.checkpoint),
  };
}

export function withSelfDigest(payload) {
  return { ...payload, digest: evidenceDigest(payload) };
}

function parseSealReference(value, index) {
  const label = `pilot measurement result seal ${index + 1}`;
  exact(value, ["path", "artifact_id", "digest", "bundle_id"], label);
  canonicalAuthorityPath(value.path, `${label} path`);
  stableId(value.artifact_id, `${label} id`);
  digest(value.digest, `${label} digest`);
  stableId(value.bundle_id, `${label} bundle`);
}

function parseSource(value) {
  exact(value, ["revision", "source_digest", "inventory_digest"], "pilot result source");
  sourceRevision(value.revision, "pilot result source revision");
  digest(value.source_digest, "pilot result source digest");
  digest(value.inventory_digest, "pilot result inventory digest");
}

function parseCounts(value) {
  exact(value, [
    "bundles", "identities", "public_identities", "internal_identities", "cohorts",
  ], "pilot result counts");
  for (const field of ["bundles", "identities", "public_identities", "internal_identities"]) {
    integer(value[field], `pilot result ${field}`, 0);
  }
  array(value.cohorts, "pilot result cohort counts").forEach((entry, index) => {
    exact(entry, [
      "cohort", "bundles", "identities", "public_identities", "internal_identities",
    ], `pilot result cohort ${index + 1}`);
    parseCohort(entry.cohort, `pilot result cohort ${index + 1} id`);
    for (const field of ["bundles", "identities", "public_identities", "internal_identities"]) {
      integer(entry[field], `pilot result cohort ${index + 1} ${field}`, 0);
    }
  });
}

function parseTiming(value) {
  exact(value, [
    "started_at", "ended_at", "elapsed_ms", "aggregate_worker_ms",
  ], "pilot result timing");
  if (!Number.isSafeInteger(Date.parse(value.started_at))
    || new Date(Date.parse(value.started_at)).toISOString() !== value.started_at
    || !Number.isSafeInteger(Date.parse(value.ended_at))
    || new Date(Date.parse(value.ended_at)).toISOString() !== value.ended_at) {
    throw new Error("pilot result timing timestamps must use safe UTC milliseconds");
  }
  integer(value.elapsed_ms, "pilot result elapsed milliseconds", 1);
  integer(value.aggregate_worker_ms, "pilot result aggregate worker milliseconds", 1);
}

function assertSelfDigest(value, label) {
  digest(value.digest, `${label} digest`);
  const { digest: _ignored, ...payload } = value;
  if (evidenceDigest(payload) !== value.digest) throw new Error(`${label} digest mismatch`);
}

function referenceKey(value) {
  return `${value.path}\0${value.digest}`;
}
