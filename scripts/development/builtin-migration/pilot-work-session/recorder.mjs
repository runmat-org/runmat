import path from "node:path";

import {
  assertAuthorityLoadSession, revalidateObservedArtifacts,
} from "../authority-loading/index.mjs";
import { publishEvidenceBytes } from "../atomic-evidence-publication.mjs";
import { assertValidatedControl } from "../control.mjs";
import {
  assertActivePilotLeaseAuthority, bindPilotLeaseAuthority,
} from "../lease-authority.mjs";
import { assertLoadedQueueAuthority } from "../queue-authority/index.mjs";
import { sessionPaths } from "./paths.mjs";
import {
  artifactBinding, parseCompletionValue, parseStartValue, queueBindingFromAuthority,
  sealReference, sourceBinding, withSelfDigest,
} from "./schema.mjs";
import {
  assertLoadedPilotWorkSessionStart, loadPilotWorkSessionCompletion,
  loadPilotWorkSessionStart,
} from "./loader.mjs";
import {
  directSuccessorSeal, validateCompletionBindings, validateStartBindings,
} from "./validation.mjs";

export function recordPilotWorkSessionStart(input) {
  assertRecorderInput(input, [
    "session", "control", "queueAuthority", "leaseAuthority", "repository",
  ], "pilot work-session start recorder");
  const {
    session: sessionValue, control: controlValue, queueAuthority: queueValue,
    leaseAuthority: leaseValue, repository,
  } = input;
  const session = assertAuthorityLoadSession(sessionValue);
  const control = assertValidatedControl(controlValue);
  const queueAuthority = assertLoadedQueueAuthority(
    queueValue, { session, control },
  );
  const leaseAuthority = leaseValue;
  const bound = bindPilotLeaseAuthority({
    leaseAuthority, queueAuthority, session, control,
  });
  assertActivePilotLeaseAuthority(bound, { session, control });
  const bundleId = leaseAuthority.lease.value.bundle_id;
  const paths = sessionPaths(
    control.digest, control.pilotPolicyDigest, control.pilotPolicy.pilotId, bundleId,
  );
  const payload = withSelfDigest({
    schema_version: 2,
    kind: "runmat-builtin-migration-pilot-work-session-start",
    authority: "machine-observed-development-evidence-only",
    control_manifest_digest: control.digest,
    pilot_policy_digest: control.pilotPolicyDigest,
    pilot_id: control.pilotPolicy.pilotId,
    initial_queue: queueBindingFromAuthority(queueAuthority),
    initial_lease: artifactBinding(
      leaseAuthority.reference.path,
      leaseAuthority.reference.digest,
      leaseAuthority.artifact.contentDigest,
    ),
    bundle_id: bundleId,
    session_id: paths.sessionId,
    source: sourceBinding(queueAuthority.checkpoint),
    started_at: machineTimestamp(),
  });
  const parsed = parseStartValue(payload);
  validateStartBindings({
    value: parsed, session, control, queueAuthority, leaseAuthority,
  });
  publish(session, paths.start, payload);
  return loadPilotWorkSessionStart({
    session,
    reference: { path: paths.start, digest: payload.digest },
    control,
    repository,
  });
}

export function recordPilotWorkSessionCompletion(input) {
  assertRecorderInput(input, [
    "session", "control", "start", "finalLeaseAuthority",
    "preIntegrationQueue", "successorQueue", "repository",
  ], "pilot work-session completion recorder");
  const {
    session: sessionValue, control: controlValue, start: startValue,
    finalLeaseAuthority: finalLeaseValue, preIntegrationQueue: preIntegrationValue,
    successorQueue: successorValue, repository,
  } = input;
  const session = assertAuthorityLoadSession(sessionValue);
  const control = assertValidatedControl(controlValue);
  const start = assertLoadedPilotWorkSessionStart(startValue, { session, control });
  const preIntegrationQueue = assertLoadedQueueAuthority(
    preIntegrationValue, { session, control },
  );
  const finalLeaseAuthority = finalLeaseValue;
  const finalBound = bindPilotLeaseAuthority({
    leaseAuthority: finalLeaseAuthority,
    queueAuthority: preIntegrationQueue,
    session,
    control,
  });
  assertActivePilotLeaseAuthority(finalBound, { session, control });
  const successorQueue = assertLoadedQueueAuthority(
    successorValue, { session, control },
  );
  const sealed = directSuccessorSeal(preIntegrationQueue, successorQueue);
  const paths = sessionPaths(
    control.digest, control.pilotPolicyDigest, start.pilotId, start.bundleId,
  );
  if (start.reference.path !== paths.start) {
    throw new Error(
      "pilot work-session start is outside its deterministic pilot/bundle session path",
    );
  }
  const payload = withSelfDigest({
    schema_version: 2,
    kind: "runmat-builtin-migration-pilot-work-session-completion",
    authority: "machine-observed-development-evidence-only",
    control_manifest_digest: control.digest,
    pilot_policy_digest: control.pilotPolicyDigest,
    pilot_id: start.pilotId,
    start: artifactBinding(
      start.reference.path, start.reference.digest, start.artifact.contentDigest,
    ),
    final_lease: artifactBinding(
      finalLeaseAuthority.reference.path,
      finalLeaseAuthority.reference.digest,
      finalLeaseAuthority.artifact.contentDigest,
    ),
    pre_integration_queue: queueBindingFromAuthority(preIntegrationQueue),
    successor_queue: queueBindingFromAuthority(successorQueue),
    seal: sealReference(sealed),
    bundle_id: start.bundleId,
    session_id: start.sessionId,
    source: sourceBinding(successorQueue.checkpoint),
    ended_at: machineTimestamp(),
  });
  const parsed = parseCompletionValue(payload);
  validateCompletionBindings({
    value: parsed, session, control, start, finalLeaseAuthority,
    preIntegrationQueue, successorQueue,
  });
  revalidateObservedArtifacts(session);
  publish(session, paths.completion, payload);
  return loadPilotWorkSessionCompletion({
    session,
    reference: { path: paths.completion, digest: payload.digest },
    control,
    repository,
  });
}

function publish(session, relativePath, value) {
  const target = path.join(session.root.path, ...relativePath.split("/"));
  publishEvidenceBytes(target, `${JSON.stringify(value, null, 2)}\n`, {
    createParentDirectories: true,
  });
}

function machineTimestamp() {
  return new Date(Date.now()).toISOString();
}

function assertRecorderInput(value, fields, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)
    || JSON.stringify(Object.keys(value).sort())
      !== JSON.stringify([...fields].sort())) {
    throw new Error(`${label} fields must be exactly ${fields.join(", ")}`);
  }
}
