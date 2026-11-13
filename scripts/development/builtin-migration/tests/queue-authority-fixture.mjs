import fs from "node:fs";
import path from "node:path";

import { evidenceDigest } from "../evidence.mjs";
import { loadQueueAuthorityFromCliPaths } from "../queue-authority/index.mjs";
import { acceptFirstBundle } from "./sequential-shared-parent-authority-fixture.mjs";
import { createTemporaryDirectory } from "./temporary-directories.mjs";

export function writeInitialQueueAuthority(fixture) {
  const root = createTemporaryDirectory("runmat-queue-authority-");
  const state = path.join(root, "state.json");
  const checkpoint = path.join(root, "checkpoint.json");
  writeQueueJson(state, fixture.queueState.value);
  writeQueueJson(checkpoint, fixture.queueCheckpointValue);
  return { root, state, checkpoint };
}

export function loadInitialQueueAuthority(files, fixture) {
  return loadQueueAuthorityFromCliPaths({
    statePath: files.state,
    checkpointPath: files.checkpoint,
    trustedCheckpointDigest: fixture.queueCheckpointValue.digest,
    control: fixture.control,
  });
}

export function acceptedQueueAuthority(fixture) {
  const bundle = fixture.control.bundles.get(fixture.bundleIds[0]);
  const reviewedOutputs = bundle.integration_outputs.map(
    ({ product_id, path: outputPath, producer }) => ({
      product_id, path: outputPath, producer,
    }),
  );
  const phases = {
    lease_base_revision: fixture.inventory.source.revision,
    authored_revision: fixture.inventory.source.revision,
    integrated_revision: fixture.inventory.source.revision,
    authored_changed_paths: [],
    integration_changed_paths: [],
    reviewed_authored_write_set: structuredClone(bundle.authored_write_set),
    reviewed_source_migrations: structuredClone(bundle.source_migrations),
    reviewed_integration_outputs: reviewedOutputs,
    authored_write_set_digest: evidenceDigest(bundle.authored_write_set),
    source_migrations_digest: evidenceDigest(bundle.source_migrations),
    integration_outputs_digest: evidenceDigest(reviewedOutputs),
  };
  return acceptFirstBundle({
    fixture, subjectInventory: fixture.inventory, phases,
  });
}

export function writeAcceptedQueueAuthority(fixture, accepted) {
  const root = createTemporaryDirectory("runmat-queue-authority-chain-");
  const artifacts = path.join(root, "test-artifacts");
  fs.mkdirSync(artifacts);
  writeQueueJson(path.join(artifacts, "queue-state-0.json"), fixture.queueState.value);
  writeQueueJson(path.join(artifacts, "queue-checkpoint-0.json"), fixture.queueCheckpointValue);
  writeQueueJson(path.join(artifacts, "seal-alpha.json"), accepted.seal);
  writeQueueJson(path.join(root, "queue-state-1.json"), accepted.queueStateValue);
  writeQueueJson(path.join(root, "queue-checkpoint-1.json"), accepted.checkpointValue);
  return root;
}

export function resealQueueArtifact(value) {
  delete value.digest;
  value.digest = evidenceDigest(value);
}

export function writeQueueJson(target, value) {
  fs.writeFileSync(target, `${JSON.stringify(value, null, 2)}\n`);
}
