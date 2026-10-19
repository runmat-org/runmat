import assert from "node:assert/strict";
import path from "node:path";
import test from "node:test";

import {
  openAuthorityLoadSession, openAuthorityRoot,
} from "../authority-loading/index.mjs";
import {
  assertLoadedQueueAuthority, loadQueueAuthority, loadQueueAuthorityFromCliPaths,
} from "../queue-authority/index.mjs";
import { assertValidatedQueueCheckpoint } from "../queue-checkpoint.mjs";
import { assertValidatedQueueSeal } from "../queue-seal.mjs";
import { assertValidatedQueueState } from "../queue.mjs";
import {
  cleanupRepositoryFixtures, controlledFixture,
} from "./helpers.mjs";
import {
  acceptedQueueAuthority, writeAcceptedQueueAuthority, writeInitialQueueAuthority,
  writeQueueJson,
} from "./queue-authority-fixture.mjs";
import { sequentialSharedParentFixture } from "./sequential-shared-parent-fixture.mjs";

test.afterEach(cleanupRepositoryFixtures);

test("initial authority binds byte observations to exact semantic capabilities", () => {
  const fixture = controlledFixture();
  const files = writeInitialQueueAuthority(fixture);
  const root = openAuthorityRoot(files.root);
  const session = openAuthorityLoadSession(root);
  const snapshot = loadQueueAuthority({
    session,
    statePath: "state.json",
    checkpointPath: "checkpoint.json",
    trustedCheckpointDigest: fixture.queueCheckpointValue.digest,
    control: fixture.control,
  });

  assert.equal(assertValidatedQueueState(snapshot.state, fixture.control), snapshot.state);
  assert.equal(
    assertValidatedQueueCheckpoint(snapshot.checkpoint, fixture.control, snapshot.state),
    snapshot.checkpoint,
  );
  assert.notEqual(
    snapshot.observations.state.contentDigest,
    snapshot.observations.state.semanticDigest,
  );
  assert.equal(snapshot.observations.state.path, "state.json");
  assert.equal(assertLoadedQueueAuthority(snapshot, {
    session,
    control: fixture.control,
    statePath: "state.json",
    stateDigest: fixture.queueState.stateDigest,
    checkpointPath: "checkpoint.json",
    checkpointDigest: fixture.queueCheckpointValue.digest,
  }), snapshot);
  for (const mismatch of [
    { statePath: "other-state.json" },
    { stateDigest: `sha256:${"a".repeat(64)}` },
    { checkpointPath: "other-checkpoint.json" },
    { checkpointDigest: `sha256:${"b".repeat(64)}` },
  ]) {
    assert.throws(
      () => assertLoadedQueueAuthority(snapshot, mismatch), /snapshot .* mismatch/,
    );
  }
  assert.throws(
    () => assertLoadedQueueAuthority(structuredClone(snapshot)),
    /exact loaded queue authority snapshot/,
  );
  assert.throws(
    () => assertLoadedQueueAuthority(snapshot, {
      session: openAuthorityLoadSession(root),
    }),
    /another load session/,
  );
  assert.throws(
    () => assertLoadedQueueAuthority(snapshot, {
      control: { digest: fixture.control.digest },
    }),
    /exact validated control manifest/,
  );
});

test("recursive authority returns exact predecessor and seal capabilities", () => {
  const fixture = sequentialSharedParentFixture();
  const accepted = acceptedQueueAuthority(fixture);
  const root = writeAcceptedQueueAuthority(fixture, accepted);
  const snapshot = loadQueueAuthorityFromCliPaths({
    statePath: path.join(root, "queue-state-1.json"),
    checkpointPath: path.join(root, "queue-checkpoint-1.json"),
    trustedCheckpointDigest: accepted.checkpointValue.digest,
    control: fixture.control,
  });

  assert.equal(
    assertValidatedQueueState(snapshot.state.predecessorState, fixture.control),
    snapshot.state.predecessorState,
  );
  const seal = snapshot.state.sealedBundles[0].seal;
  assert.equal(assertValidatedQueueSeal(seal, fixture.control), seal);
  assert.equal(snapshot.checkpoint.queueState, snapshot.state);
  assert.equal(snapshot.state.predecessorState.stateDigest, fixture.queueState.stateDigest);
  assert.equal(seal.reference.digest, accepted.reference.digest);
});

test("CLI resolves sibling state and checkpoint paths from the caller directory", () => {
  const fixture = controlledFixture();
  const files = writeInitialQueueAuthority(fixture);
  const snapshot = loadQueueAuthorityFromCliPaths({
    statePath: path.relative(process.cwd(), files.state),
    checkpointPath: path.relative(process.cwd(), files.checkpoint),
    trustedCheckpointDigest: fixture.queueCheckpointValue.digest,
    control: fixture.control,
  });

  assert.equal(snapshot.observations.state.path, "state.json");
  assert.equal(snapshot.observations.checkpoint.path, "checkpoint.json");
});

test("CLI paths cannot escape the state authority root", () => {
  const fixture = controlledFixture();
  const files = writeInitialQueueAuthority(fixture);
  const outside = path.join(path.dirname(files.root), "outside-checkpoint.json");
  writeQueueJson(outside, fixture.queueCheckpointValue);
  for (const checkpointPath of [outside, "../outside-checkpoint.json"]) {
    assert.throws(() => loadQueueAuthorityFromCliPaths({
      statePath: files.state,
      checkpointPath,
      trustedCheckpointDigest: fixture.queueCheckpointValue.digest,
      control: fixture.control,
    }), /below the queue authority root/);
  }
});
