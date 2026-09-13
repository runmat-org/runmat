import {
  assertAuthorityLoadSession, assertSessionArtifact, canonicalAuthorityPath,
  loadJsonArtifact, loadedJsonValue, revalidateObservedArtifacts,
  withAuthorityTraversal,
} from "../authority-loading/index.mjs";
import { assertValidatedControl } from "../control.mjs";
import { deepImmutable } from "../immutable.mjs";
import { loadLeaseAuthority } from "../lease-authority.mjs";
import { loadQueueAuthority } from "../queue-authority/index.mjs";
import { digest } from "../schema.mjs";
import { parseCompletionValue, parseStartValue } from "./schema.mjs";
import { sessionPaths } from "./paths.mjs";
import {
  validateCompletionBindings, validateStartBindings,
} from "./validation.mjs";

const STARTS = new WeakMap();
const COMPLETIONS = new WeakMap();
const SESSION_LOADERS = new WeakMap();

export function loadPilotWorkSessionStart({
  session: sessionValue, reference: referenceValue, control: controlValue, repository,
}) {
  const session = assertAuthorityLoadSession(sessionValue);
  const control = assertValidatedControl(controlValue);
  const reference = parseReference(referenceValue, "pilot work-session start reference");
  const cached = cachedAuthority(session, "starts", reference);
  if (cached) {
    revalidateObservedArtifacts(session);
    return assertLoadedPilotWorkSessionStart(cached, { session, control });
  }
  return withAuthorityTraversal(session, {
    domain: "pilot-work-session-start", path: reference.path,
    semanticDigest: reference.digest,
  }, () => {
    const { artifact, value } = loadValue(session, reference, "pilot work-session start");
    const parsed = parseStartValue(value);
    const initialQueue = loadQueueAuthority({
      session,
      statePath: parsed.initial_queue.state.path,
      checkpointPath: parsed.initial_queue.checkpoint.path,
      trustedCheckpointDigest: parsed.initial_queue.checkpoint.semantic_digest,
      control,
    });
    const initialLeaseAuthority = loadLeaseAuthority(
      session,
      { path: parsed.initial_lease.path, digest: parsed.initial_lease.semantic_digest },
      control,
      repository,
    );
    if (reference.path !== sessionPaths(
      control.digest, control.pilotPolicyDigest, parsed.pilot_id, parsed.bundle_id,
    ).start) {
      throw new Error("pilot work-session start is outside its deterministic pilot/bundle session path");
    }
    validateStartBindings({
      value: parsed, session, control, queueAuthority: initialQueue,
      leaseAuthority: initialLeaseAuthority,
    });
    revalidateObservedArtifacts(session);
    const immutable = deepImmutable({
      value: parsed,
      reference,
      observation: {
        path: artifact.path,
        semanticDigest: reference.digest,
        contentDigest: artifact.contentDigest,
      },
      bundleId: parsed.bundle_id,
      pilotId: parsed.pilot_id,
      sessionId: parsed.session_id,
      startedAt: parsed.started_at,
    });
    const result = Object.freeze({
      ...immutable, artifact, control, initialQueue, initialLeaseAuthority,
    });
    STARTS.set(result, { session, control });
    cacheAuthority(session, "starts", reference, result);
    return result;
  });
}

export function loadPilotWorkSessionCompletion({
  session: sessionValue, reference: referenceValue, control: controlValue, repository,
}) {
  const session = assertAuthorityLoadSession(sessionValue);
  const control = assertValidatedControl(controlValue);
  const reference = parseReference(referenceValue, "pilot work-session completion reference");
  const cached = cachedAuthority(session, "completions", reference);
  if (cached) {
    revalidateObservedArtifacts(session);
    return assertLoadedPilotWorkSessionCompletion(cached, { session, control });
  }
  return withAuthorityTraversal(session, {
    domain: "pilot-work-session-completion", path: reference.path,
    semanticDigest: reference.digest,
  }, () => {
    const { artifact, value } = loadValue(session, reference, "pilot work-session completion");
    const parsed = parseCompletionValue(value);
    const start = loadPilotWorkSessionStart({
      session,
      reference: {
        path: parsed.start.path,
        digest: parsed.start.semantic_digest,
      },
      control,
      repository,
    });
    if (reference.path !== sessionPaths(
      control.digest, control.pilotPolicyDigest, start.pilotId, start.bundleId,
    ).completion) {
      throw new Error("pilot work-session completion is outside its deterministic pilot/bundle session path");
    }
    const finalLeaseAuthority = loadLeaseAuthority(
      session,
      { path: parsed.final_lease.path, digest: parsed.final_lease.semantic_digest },
      control,
      repository,
    );
    const preIntegrationQueue = loadQueueAuthority({
      session,
      statePath: parsed.pre_integration_queue.state.path,
      checkpointPath: parsed.pre_integration_queue.checkpoint.path,
      trustedCheckpointDigest: parsed.pre_integration_queue.checkpoint.semantic_digest,
      control,
    });
    const successorQueue = loadQueueAuthority({
      session,
      statePath: parsed.successor_queue.state.path,
      checkpointPath: parsed.successor_queue.checkpoint.path,
      trustedCheckpointDigest: parsed.successor_queue.checkpoint.semantic_digest,
      control,
    });
    const { sealed } = validateCompletionBindings({
      value: parsed, session, control, start, finalLeaseAuthority,
      preIntegrationQueue, successorQueue,
    });
    revalidateObservedArtifacts(session);
    const immutable = deepImmutable({
      value: parsed,
      reference,
      observation: {
        path: artifact.path,
        semanticDigest: reference.digest,
        contentDigest: artifact.contentDigest,
      },
      bundleId: parsed.bundle_id,
      pilotId: parsed.pilot_id,
      sessionId: parsed.session_id,
      endedAt: parsed.ended_at,
    });
    const result = Object.freeze({
      ...immutable, artifact, control, start, finalLeaseAuthority,
      preIntegrationQueue, successorQueue, seal: sealed.seal,
    });
    COMPLETIONS.set(result, { session, control });
    cacheAuthority(session, "completions", reference, result);
    return result;
  });
}

export function assertLoadedPilotWorkSessionStart(value, {
  session: sessionValue, control: controlValue,
}) {
  return assertLoaded(value, STARTS, sessionValue, controlValue, "start");
}

export function assertLoadedPilotWorkSessionCompletion(value, {
  session: sessionValue, control: controlValue,
}) {
  return assertLoaded(value, COMPLETIONS, sessionValue, controlValue, "completion");
}

function assertLoaded(value, registry, sessionValue, controlValue, role) {
  const session = assertAuthorityLoadSession(sessionValue);
  const control = assertValidatedControl(controlValue);
  const record = registry.get(value);
  if (!record || record.session !== session || record.control !== control) {
    throw new Error(`operation requires the exact loaded pilot work-session ${role}`);
  }
  assertSessionArtifact(session, value.artifact, value.reference.path);
  return value;
}

function loadValue(session, reference, label) {
  const artifact = loadJsonArtifact(session, reference.path, label);
  assertSessionArtifact(session, artifact, reference.path);
  const value = loadedJsonValue(artifact, session.root);
  if (value?.digest !== reference.digest) {
    throw new Error(`${label} semantic digest mismatch`);
  }
  return { artifact, value };
}

function parseReference(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)
    || JSON.stringify(Object.keys(value).sort()) !== JSON.stringify(["digest", "path"])) {
    throw new Error(`${label} fields must be exactly path, digest`);
  }
  return Object.freeze({
    path: canonicalAuthorityPath(value.path, `${label} path`),
    digest: digest(value.digest, `${label} digest`),
  });
}

function loaderRecord(session) {
  let record = SESSION_LOADERS.get(session);
  if (!record) {
    record = { starts: new Map(), completions: new Map() };
    SESSION_LOADERS.set(session, record);
  }
  return record;
}

function cachedAuthority(session, role, reference) {
  const entry = loaderRecord(session)[role].get(reference.path);
  if (!entry) return null;
  if (entry.digest !== reference.digest) {
    throw new Error(`${reference.path}: pilot work-session authority path was rebound`);
  }
  return entry.value;
}

function cacheAuthority(session, role, reference, value) {
  loaderRecord(session)[role].set(reference.path, {
    digest: reference.digest, value,
  });
}
