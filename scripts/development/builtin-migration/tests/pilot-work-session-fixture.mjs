import fs from "node:fs";
import path from "node:path";

import {
  openAuthorityLoadSession, openAuthorityRoot,
} from "../authority-loading/index.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { loadLeaseAuthority } from "../lease-authority.mjs";
import {
  loadPilotWorkSessionCompletion, loadPilotWorkSessionStart,
  recordPilotWorkSessionCompletion, recordPilotWorkSessionStart,
} from "../pilot-work-session/index.mjs";
import { sessionPaths } from "../pilot-work-session/paths.mjs";
import { loadQueueAuthority } from "../queue-authority/index.mjs";
import { acceptFirstBundle } from "./sequential-shared-parent-authority-fixture.mjs";
import { sequentialSharedParentFixture } from "./sequential-shared-parent-fixture.mjs";
import { createTemporaryDirectory } from "./temporary-directories.mjs";

export function workSessionFixture({ fixture = null } = {}) {
  const core = fixture ?? sequentialSharedParentFixture();
  const root = createTemporaryDirectory("runmat-pilot-work-session-");
  writeJson(path.join(root, "test-artifacts/queue-state-0.json"), core.queueState.value);
  writeJson(
    path.join(root, "test-artifacts/queue-checkpoint-0.json"),
    core.queueCheckpointValue,
  );
  const session = openAuthorityLoadSession(openAuthorityRoot(root));
  const queue = loadQueueAuthority({
    session,
    statePath: "test-artifacts/queue-state-0.json",
    checkpointPath: "test-artifacts/queue-checkpoint-0.json",
    trustedCheckpointDigest: core.queueCheckpointValue.digest,
    control: core.control,
  });
  const lease = installAndLoadLease(
    { root, session, control: core.control, repository: core.repository },
    "leases/lease-alpha.json",
    core.firstLease,
  );
  return {
    fixture: core,
    root,
    session,
    queue,
    lease,
    control: core.control,
    repository: core.repository,
    bundleId: core.bundleIds[0],
  };
}

export function startRecorderInput(fixture, leaseAuthority = fixture.lease) {
  return {
    session: fixture.session,
    control: fixture.control,
    queueAuthority: fixture.queue,
    leaseAuthority,
    repository: fixture.repository,
  };
}

export function recordStart(fixture, leaseAuthority = fixture.lease) {
  return recordPilotWorkSessionStart(startRecorderInput(fixture, leaseAuthority));
}

export function recordCompletion(
  fixture, start, finalLeaseAuthority, preIntegrationQueue, successorQueue,
) {
  return recordPilotWorkSessionCompletion({
    session: fixture.session,
    control: fixture.control,
    start,
    finalLeaseAuthority,
    preIntegrationQueue,
    successorQueue,
    repository: fixture.repository,
  });
}

export function acceptedAuthority(fixture) {
  const bundle = fixture.control.bundles.get(fixture.bundleId);
  const reviewed = bundle.integration_outputs.map(
    ({ product_id, path: outputPath, producer }) => ({
      product_id, path: outputPath, producer,
    }),
  );
  const phases = {
    lease_base_revision: fixture.fixture.inventory.source.revision,
    authored_revision: fixture.fixture.inventory.source.revision,
    integrated_revision: fixture.fixture.inventory.source.revision,
    authored_changed_paths: [],
    integration_changed_paths: [],
    reviewed_authored_write_set: structuredClone(bundle.authored_write_set),
    reviewed_source_migrations: structuredClone(bundle.source_migrations),
    reviewed_integration_outputs: reviewed,
    authored_write_set_digest: evidenceDigest(bundle.authored_write_set),
    source_migrations_digest: evidenceDigest(bundle.source_migrations),
    integration_outputs_digest: evidenceDigest(reviewed),
  };
  return acceptFirstBundle({
    fixture: fixture.fixture,
    subjectInventory: fixture.fixture.inventory,
    phases,
  });
}

export function installAcceptedFiles(root, accepted, index = 1, sealName = "alpha") {
  writeJson(path.join(root, `test-artifacts/seal-${sealName}.json`), accepted.seal);
  writeJson(path.join(root, `queue-state-${index}.json`), accepted.queueStateValue);
  writeJson(path.join(root, `queue-checkpoint-${index}.json`), accepted.checkpointValue);
}

export function installAndLoadLease(fixture, relativePath, leaseRecord) {
  writeJson(path.join(fixture.root, relativePath), leaseRecord.value);
  return loadLeaseAuthority(
    fixture.session,
    { path: relativePath, digest: evidenceDigest(leaseRecord.value) },
    fixture.control,
    fixture.repository,
  );
}

export function loadQueue(fixture, index, trustedDigest) {
  return loadQueueAuthority({
    session: fixture.session,
    statePath: `queue-state-${index}.json`,
    checkpointPath: `queue-checkpoint-${index}.json`,
    trustedCheckpointDigest: trustedDigest,
    control: fixture.control,
  });
}

export function installSuccessor(fixture) {
  const accepted = acceptedAuthority(fixture);
  installAcceptedFiles(fixture.root, accepted);
  return loadQueue(fixture, 1, accepted.checkpointValue.digest);
}

export function completedFixture() {
  const fixture = workSessionFixture();
  const start = recordStart(fixture);
  const successor = installSuccessor(fixture);
  const completion = recordCompletion(
    fixture, start, fixture.lease, fixture.queue, successor,
  );
  return { ...fixture, start, successor, completion };
}

export function replaceCompletion(fixture, value) {
  const target = path.join(fixture.root, fixture.completion.reference.path);
  fs.renameSync(target, `${target}.prior`);
  writeJson(target, value);
}

export function reloadCompletion(fixture, value) {
  const session = openAuthorityLoadSession(openAuthorityRoot(fixture.root));
  return loadPilotWorkSessionCompletion({
    session,
    reference: { path: fixture.completion.reference.path, digest: value.digest },
    control: fixture.control,
    repository: fixture.repository,
  });
}

export function reloadStart(fixture, relativePath, value) {
  const session = openAuthorityLoadSession(openAuthorityRoot(fixture.root));
  return loadPilotWorkSessionStart({
    session,
    reference: { path: relativePath, digest: value.digest },
    control: fixture.control,
    repository: fixture.repository,
  });
}

export function startValue(fixture) {
  const start = recordStart(fixture);
  const target = path.join(fixture.root, start.reference.path);
  fs.renameSync(target, `${target}.prior`);
  return structuredClone(start.value);
}

export function fixtureSessionPaths(fixture) {
  return sessionPaths(
    fixture.control.digest,
    fixture.control.pilotPolicyDigest,
    fixture.control.pilotPolicy.pilotId,
    fixture.bundleId,
  );
}

export function reseal(value) {
  delete value.digest;
  value.digest = evidenceDigest(value);
}

export function resealSealAndSuccessor(accepted) {
  const oldReference = accepted.queueStateValue.seals[0];
  const newSealDigest = evidenceDigest(accepted.seal);
  const newReference = { ...oldReference, digest: newSealDigest };
  accepted.reference = newReference;
  accepted.queueStateValue.seals = [newReference];
  reseal(accepted.queueStateValue);
  accepted.checkpointValue.queue_state_digest = accepted.queueStateValue.digest;
  accepted.checkpointValue.head_event.seal = newReference;
  accepted.checkpointValue.accepted_seal_set_digest = evidenceDigest({
    schema_version: 1,
    kind: "runmat-builtin-migration-accepted-seal-set",
    authority: "derived-from-validated-seals",
    control_manifest_digest: accepted.seal.control_manifest_digest,
    seals: [newReference],
  });
  reseal(accepted.checkpointValue);
}

export function writeJson(target, value) {
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, `${JSON.stringify(value, null, 2)}\n`);
}

export function fakeDigest(character) {
  return `sha256:${character.repeat(64)}`;
}
