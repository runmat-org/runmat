import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";

import {
  loadJsonArtifact, openAuthorityLoadSession, openAuthorityRoot,
} from "../authority-loading/index.mjs";
import { evidenceDigest } from "../evidence.mjs";
import {
  loadQueueAuthorityFromCliPaths, revalidateLoadedQueueAuthority,
} from "../queue-authority/index.mjs";
import {
  bindQueueReference, openQueueReferenceIndex,
} from "../queue-authority/reference.mjs";
import { cleanupRepositoryFixtures, controlledFixture } from "./helpers.mjs";
import {
  acceptedQueueAuthority, loadInitialQueueAuthority, resealQueueArtifact,
  writeAcceptedQueueAuthority, writeInitialQueueAuthority, writeQueueJson,
} from "./queue-authority-fixture.mjs";
import { sequentialSharedParentFixture } from "./sequential-shared-parent-fixture.mjs";
import { createTemporaryDirectory } from "./temporary-directories.mjs";

test.afterEach(cleanupRepositoryFixtures);

test("semantic digest, control, and wire-schema mismatches fail closed", async (context) => {
  for (const scenario of [
    { name: "state digest", mutate(state) { state.digest = `sha256:${"0".repeat(64)}`; } },
    { name: "state legacy schema", mutate(state) { state.schema_version = 3; resealQueueArtifact(state); } },
    { name: "state future schema", mutate(state) { state.schema_version = 5; resealQueueArtifact(state); } },
  ]) {
    await context.test(scenario.name, () => {
      const fixture = controlledFixture();
      const files = writeInitialQueueAuthority(fixture);
      const state = structuredClone(fixture.queueState.value);
      scenario.mutate(state);
      writeQueueJson(files.state, state);
      assert.throws(
        () => loadInitialQueueAuthority(files, fixture),
        /digest mismatch|schema_version 4/,
      );
    });
  }
  for (const version of [1, 3]) {
    await context.test(`checkpoint schema ${version}`, () => {
      const fixture = controlledFixture();
      const files = writeInitialQueueAuthority(fixture);
      const checkpoint = structuredClone(fixture.queueCheckpointValue);
      checkpoint.schema_version = version;
      resealQueueArtifact(checkpoint);
      writeQueueJson(files.checkpoint, checkpoint);
      assert.throws(() => loadQueueAuthorityFromCliPaths({
        statePath: files.state,
        checkpointPath: files.checkpoint,
        trustedCheckpointDigest: checkpoint.digest,
        control: fixture.control,
      }), /schema_version 2/);
    });
  }
  await context.test("trusted checkpoint digest", () => {
    const fixture = controlledFixture();
    const files = writeInitialQueueAuthority(fixture);
    assert.throws(() => loadQueueAuthorityFromCliPaths({
      statePath: files.state,
      checkpointPath: files.checkpoint,
      trustedCheckpointDigest: `sha256:${"f".repeat(64)}`,
      control: fixture.control,
    }), /semantic digest mismatch/);
  });
  await context.test("control", () => {
    const fixture = controlledFixture();
    const other = controlledFixture({ identity: "bar" });
    const files = writeInitialQueueAuthority(fixture);
    assert.throws(
      () => loadInitialQueueAuthority(files, other), /another control manifest/,
    );
  });
});

test("state references cannot alias another authority domain", () => {
  const fixture = sequentialSharedParentFixture();
  const accepted = acceptedQueueAuthority(fixture);
  const root = writeAcceptedQueueAuthority(fixture, accepted);
  const state = structuredClone(accepted.queueStateValue);
  state.seals[0].path = state.predecessor.state_path;
  resealQueueArtifact(state);
  writeQueueJson(path.join(root, "queue-state-1.json"), state);
  assert.throws(
    () => loadAccepted(root, accepted, fixture), /path was rebound|digest mismatch/,
  );
});

test("state predecessor recursion rejects a repeated active path", () => {
  const fixture = sequentialSharedParentFixture();
  const accepted = acceptedQueueAuthority(fixture);
  const root = writeAcceptedQueueAuthority(fixture, accepted);
  const state = structuredClone(accepted.queueStateValue);
  state.predecessor.state_path = "queue-state-1.json";
  resealQueueArtifact(state);
  writeQueueJson(path.join(root, "queue-state-1.json"), state);
  assert.throws(
    () => loadAccepted(root, accepted, fixture), /authority path cycle/,
  );
});

test("checkpoint predecessor recursion rejects a repeated active path", () => {
  const fixture = sequentialSharedParentFixture();
  const accepted = acceptedQueueAuthority(fixture);
  const root = writeAcceptedQueueAuthority(fixture, accepted);
  const state = structuredClone(accepted.queueStateValue);
  state.predecessor.checkpoint_path = "queue-checkpoint-1.json";
  resealQueueArtifact(state);
  writeQueueJson(path.join(root, "queue-state-1.json"), state);
  const checkpoint = structuredClone(accepted.checkpointValue);
  checkpoint.queue_state_digest = state.digest;
  resealQueueArtifact(checkpoint);
  writeQueueJson(path.join(root, "queue-checkpoint-1.json"), checkpoint);
  assert.throws(
    () => loadAccepted(root, { checkpointValue: checkpoint }, fixture),
    /authority path cycle/,
  );
});

test("one semantic authority digest cannot be rebound to another observed path", () => {
  const rootPath = createTemporaryDirectory("runmat-queue-authority-rebind-");
  writeQueueJson(path.join(rootPath, "first.json"), { value: 1 });
  writeQueueJson(path.join(rootPath, "second.json"), { value: 1 });
  const session = openAuthorityLoadSession(openAuthorityRoot(rootPath));
  loadJsonArtifact(session, "first.json");
  loadJsonArtifact(session, "second.json");
  const references = openQueueReferenceIndex(session);
  const semanticDigest = evidenceDigest({ value: 1 });
  bindQueueReference(references, "queue-state", "first.json", semanticDigest);
  assert.throws(
    () => bindQueueReference(
      references, "queue-state", "first.json", `sha256:${"f".repeat(64)}`,
    ),
    /queue authority path was rebound/,
  );
  assert.throws(
    () => bindQueueReference(references, "queue-state", "second.json", semanticDigest),
    /semantic authority digest was rebound/,
  );
});

test("a returned snapshot detects later artifact mutation and replacement", async (context) => {
  for (const operation of ["mutate", "replace"]) {
    await context.test(operation, () => {
      const fixture = controlledFixture();
      const files = writeInitialQueueAuthority(fixture);
      const snapshot = loadInitialQueueAuthority(files, fixture);
      if (operation === "mutate") {
        fs.writeFileSync(files.state, `${JSON.stringify(fixture.queueState.value)}\n`);
      } else {
        fs.renameSync(files.checkpoint, path.join(files.root, "prior-checkpoint.json"));
        writeQueueJson(files.checkpoint, fixture.queueCheckpointValue);
      }
      assert.throws(
        () => revalidateLoadedQueueAuthority(snapshot), /changed after observation/,
      );
    });
  }
});

function loadAccepted(root, accepted, fixture) {
  return loadQueueAuthorityFromCliPaths({
    statePath: path.join(root, "queue-state-1.json"),
    checkpointPath: path.join(root, "queue-checkpoint-1.json"),
    trustedCheckpointDigest: accepted.checkpointValue.digest,
    control: fixture.control,
  });
}
