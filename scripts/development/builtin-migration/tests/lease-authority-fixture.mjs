import fs from "node:fs";
import path from "node:path";

import {
  openAuthorityLoadSession, openAuthorityRoot,
} from "../authority-loading/index.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { bindPilotLeaseAuthority, loadLeaseAuthority } from "../lease-authority.mjs";
import { loadQueueAuthority } from "../queue-authority/index.mjs";
import { controlledFixture } from "./helpers.mjs";
import { acceptFirstBundle } from "./sequential-shared-parent-authority-fixture.mjs";
import { sequentialSharedParentFixture } from "./sequential-shared-parent-fixture.mjs";
import { createTemporaryDirectory } from "./temporary-directories.mjs";

export function leaseAuthorityFixture({ expired = false, mutate = null } = {}) {
  const controlled = controlledFixture();
  const leaseValue = structuredClone(controlled.leaseValue);
  if (expired) {
    leaseValue.request.issued_at = "2020-01-01T00:00:00.000Z";
    leaseValue.request.expires_at = "2020-01-02T00:00:00.000Z";
    leaseValue.issued_at = leaseValue.request.issued_at;
    leaseValue.expires_at = leaseValue.request.expires_at;
    resealLeaseValue(leaseValue);
  }
  mutate?.(leaseValue, controlled);
  const root = createTemporaryDirectory("runmat-lease-authority-");
  const relativePath = "leases/pilot-foo.json";
  const target = path.join(root, relativePath);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  const bytes = Buffer.from(`${JSON.stringify(leaseValue, null, 2)}\n`);
  fs.writeFileSync(target, bytes);
  writeJson(path.join(root, "queue-state.json"), controlled.queueState.value);
  writeJson(path.join(root, "queue-checkpoint.json"), controlled.queueCheckpointValue);
  const session = openAuthorityLoadSession(openAuthorityRoot(root));
  const queueAuthority = loadQueueAuthority({
    session,
    statePath: "queue-state.json",
    checkpointPath: "queue-checkpoint.json",
    trustedCheckpointDigest: controlled.queueCheckpointValue.digest,
    control: controlled.control,
  });
  const reference = { path: relativePath, digest: evidenceDigest(leaseValue) };
  return {
    ...controlled, leaseValue, root, target, bytes, session, reference, queueAuthority,
  };
}

export function loadFixtureLease(fixture) {
  return loadLeaseAuthority(
    fixture.session, fixture.reference, fixture.control, fixture.repository,
  );
}

export function bindFixtureLease(fixture, leaseAuthority) {
  return bindPilotLeaseAuthority({
    leaseAuthority,
    queueAuthority: fixture.queueAuthority,
    session: fixture.session,
    control: fixture.control,
  });
}

export function sealedLeaseAuthorityFixture() {
  const controlled = sequentialSharedParentFixture();
  const bundle = controlled.control.bundles.get(controlled.bundleIds[0]);
  const outputs = bundle.integration_outputs.map(
    ({ product_id, path: outputPath, producer }) => ({
      product_id, path: outputPath, producer,
    }),
  );
  const phases = {
    lease_base_revision: controlled.inventory.source.revision,
    authored_revision: controlled.inventory.source.revision,
    integrated_revision: controlled.inventory.source.revision,
    authored_changed_paths: [],
    integration_changed_paths: [],
    reviewed_authored_write_set: structuredClone(bundle.authored_write_set),
    reviewed_integration_outputs: outputs,
    authored_write_set_digest: evidenceDigest(bundle.authored_write_set),
    integration_outputs_digest: evidenceDigest(outputs),
  };
  const accepted = acceptFirstBundle({
    fixture: controlled, subjectInventory: controlled.inventory, phases,
  });
  const root = createTemporaryDirectory("runmat-sealed-lease-authority-");
  const artifacts = path.join(root, "test-artifacts");
  fs.mkdirSync(artifacts);
  writeJson(path.join(artifacts, "queue-state-0.json"), controlled.queueState.value);
  writeJson(path.join(artifacts, "queue-checkpoint-0.json"), controlled.queueCheckpointValue);
  writeJson(path.join(artifacts, "seal-alpha.json"), accepted.seal);
  writeJson(path.join(root, "queue-state.json"), accepted.queueStateValue);
  writeJson(path.join(root, "queue-checkpoint.json"), accepted.checkpointValue);
  const leaseValue = controlled.firstLease.value;
  const relativePath = "leases/lease-alpha.json";
  const target = path.join(root, relativePath);
  fs.mkdirSync(path.dirname(target));
  const bytes = Buffer.from(`${JSON.stringify(leaseValue, null, 2)}\n`);
  fs.writeFileSync(target, bytes);
  const session = openAuthorityLoadSession(openAuthorityRoot(root));
  const queueAuthority = loadQueueAuthority({
    session, statePath: "queue-state.json", checkpointPath: "queue-checkpoint.json",
    trustedCheckpointDigest: accepted.checkpointValue.digest, control: controlled.control,
  });
  return {
    ...controlled, leaseValue, root, target, bytes, session, queueAuthority,
    reference: { path: relativePath, digest: evidenceDigest(leaseValue) },
  };
}

function resealLeaseValue(value) {
  delete value.digest;
  value.digest = evidenceDigest(value);
}

function writeJson(target, value) {
  fs.writeFileSync(target, `${JSON.stringify(value, null, 2)}\n`);
}
