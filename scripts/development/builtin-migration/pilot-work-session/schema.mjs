import { evidenceDigest } from "../evidence.mjs";
import { canonicalAuthorityPath } from "../authority-loading/index.mjs";
import {
  digest, exact, kind, sourceRevision, stableId, timestamp,
} from "../schema.mjs";

export const PILOT_WORK_SESSION_START_KIND =
  "runmat-builtin-migration-pilot-work-session-start";
export const PILOT_WORK_SESSION_COMPLETION_KIND =
  "runmat-builtin-migration-pilot-work-session-completion";
export const PILOT_WORK_SESSION_VERSION = 2;

export function parseStartValue(value) {
  kind(value, PILOT_WORK_SESSION_VERSION, PILOT_WORK_SESSION_START_KIND,
    "pilot work-session start");
  exact(value, [
    "schema_version", "kind", "authority", "control_manifest_digest",
    "pilot_policy_digest", "pilot_id", "initial_queue", "initial_lease",
    "bundle_id", "session_id",
    "source", "started_at", "digest",
  ], "pilot work-session start");
  assertMachineAuthority(value.authority, "pilot work-session start");
  digest(value.control_manifest_digest, "pilot work-session control manifest digest");
  digest(value.pilot_policy_digest, "pilot work-session pilot policy digest");
  const parsed = {
    ...value,
    pilot_id: stableId(value.pilot_id, "pilot work-session start pilot id"),
    initial_queue: parseQueueBinding(
      value.initial_queue, "pilot work-session start initial queue",
    ),
    initial_lease: parseArtifactBinding(
      value.initial_lease, "pilot work-session start initial lease",
    ),
    bundle_id: stableId(value.bundle_id, "pilot work-session start bundle id"),
    session_id: stableId(value.session_id, "pilot work-session start session id"),
    source: parseSourceBinding(value.source, "pilot work-session start source"),
    started_at: timestamp(value.started_at, "pilot work-session started_at"),
  };
  assertSelfDigest(value, "pilot work-session start");
  return parsed;
}

export function parseCompletionValue(value) {
  kind(value, PILOT_WORK_SESSION_VERSION, PILOT_WORK_SESSION_COMPLETION_KIND,
    "pilot work-session completion");
  exact(value, [
    "schema_version", "kind", "authority", "control_manifest_digest",
    "pilot_policy_digest", "pilot_id", "start", "final_lease",
    "pre_integration_queue", "successor_queue", "seal", "bundle_id",
    "session_id", "source", "ended_at", "digest",
  ], "pilot work-session completion");
  assertMachineAuthority(value.authority, "pilot work-session completion");
  digest(value.control_manifest_digest, "pilot work-session control manifest digest");
  digest(value.pilot_policy_digest, "pilot work-session pilot policy digest");
  const parsed = {
    ...value,
    pilot_id: stableId(value.pilot_id, "pilot work-session completion pilot id"),
    start: parseArtifactBinding(value.start, "pilot work-session completion start"),
    final_lease: parseArtifactBinding(
      value.final_lease, "pilot work-session completion final lease",
    ),
    pre_integration_queue: parseQueueBinding(
      value.pre_integration_queue, "pilot work-session completion pre-integration queue",
    ),
    successor_queue: parseQueueBinding(
      value.successor_queue, "pilot work-session completion successor queue",
    ),
    seal: parseSealReference(value.seal),
    bundle_id: stableId(value.bundle_id, "pilot work-session completion bundle id"),
    session_id: stableId(value.session_id, "pilot work-session completion session id"),
    source: parseSourceBinding(value.source, "pilot work-session completion source"),
    ended_at: timestamp(value.ended_at, "pilot work-session ended_at"),
  };
  assertSelfDigest(value, "pilot work-session completion");
  return parsed;
}

export function queueBindingFromAuthority(queue) {
  return {
    state: observationBinding(queue.observations.state),
    checkpoint: observationBinding(queue.observations.checkpoint),
  };
}

export function artifactBinding(path, semanticDigest, contentDigest) {
  return { path, semantic_digest: semanticDigest, content_digest: contentDigest };
}

export function sourceBinding(checkpoint) {
  return {
    revision: checkpoint.value.source_revision,
    source_digest: checkpoint.value.source_digest,
    inventory_digest: checkpoint.value.inventory_digest,
  };
}

export function sealReference(sealed) {
  return { ...sealed.reference };
}

export function withSelfDigest(payload) {
  return { ...payload, digest: evidenceDigest(payload) };
}

function observationBinding(value) {
  return {
    path: value.path,
    semantic_digest: value.semanticDigest,
    content_digest: value.contentDigest,
  };
}

function parseQueueBinding(value, label) {
  exact(value, ["state", "checkpoint"], label);
  return {
    state: parseArtifactBinding(value.state, `${label} state`),
    checkpoint: parseArtifactBinding(value.checkpoint, `${label} checkpoint`),
  };
}

function parseArtifactBinding(value, label) {
  exact(value, ["path", "semantic_digest", "content_digest"], label);
  return {
    path: canonicalAuthorityPath(value.path, `${label} path`),
    semantic_digest: digest(value.semantic_digest, `${label} semantic digest`),
    content_digest: digest(value.content_digest, `${label} content digest`),
  };
}

function parseSourceBinding(value, label) {
  exact(value, ["revision", "source_digest", "inventory_digest"], label);
  return {
    revision: sourceRevision(value.revision, `${label} revision`),
    source_digest: digest(value.source_digest, `${label} source digest`),
    inventory_digest: digest(value.inventory_digest, `${label} inventory digest`),
  };
}

function parseSealReference(value) {
  exact(value, ["path", "artifact_id", "digest", "bundle_id"],
    "pilot work-session completion seal");
  return {
    path: canonicalAuthorityPath(value.path, "pilot work-session completion seal path"),
    artifact_id: stableId(value.artifact_id, "pilot work-session completion seal id"),
    digest: digest(value.digest, "pilot work-session completion seal digest"),
    bundle_id: stableId(value.bundle_id, "pilot work-session completion seal bundle"),
  };
}

function assertMachineAuthority(value, label) {
  if (value !== "machine-observed-development-evidence-only") {
    throw new Error(`${label} has invalid authority`);
  }
}

function assertSelfDigest(value, label) {
  digest(value.digest, `${label} digest`);
  const { digest: _ignored, ...payload } = value;
  if (evidenceDigest(payload) !== value.digest) throw new Error(`${label} digest mismatch`);
}
