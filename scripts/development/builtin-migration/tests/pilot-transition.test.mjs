import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";

import { evidenceDigest } from "../evidence.mjs";
import { loadQueueAuthority } from "../queue-authority/index.mjs";
import {
  assertLoadedPilotTransition, loadPilotTransition, pilotTransitionPaths,
  recordPilotTransition,
} from "../pilot-transition.mjs";
import {
  pilotTransitionFixture, writeTransitionArtifact,
} from "./pilot-transition-fixture.mjs";

test("records the exact irreversible pilot-to-production transition", () => {
  const fixture = pilotTransitionFixture();
  const result = record(fixture);
  assert.equal(result.state.value.phase, "production");
  assert.deepEqual(result.state.value.bundles, fixture.derived.finalQueue.state.value.bundles);
  assert.deepEqual(result.state.value.seals, fixture.derived.finalQueue.state.value.seals);
  assert.equal(result.checkpoint.value.source_digest,
    fixture.derived.finalQueue.checkpoint.value.source_digest);
  assert.equal(result.checkpoint.value.inventory_digest,
    fixture.derived.finalQueue.checkpoint.value.inventory_digest);
  assert.equal(result.checkpoint.value.accepted_seal_set_digest,
    fixture.derived.finalQueue.checkpoint.value.accepted_seal_set_digest);
  assert.equal(result.state.pilotEvaluation, fixture.evaluation);
  assert.equal(assertLoadedPilotTransition(result, {
    session: fixture.session, control: fixture.control, evaluation: fixture.evaluation,
  }), result);
});

test("below-target transition requires the exact reviewed limiter-bearing evaluation", () => {
  const fixture = pilotTransitionFixture({ belowTarget: true });
  const result = record(fixture);
  assert.equal(result.evaluation.value.outcome, "below-target");
  assert.ok(result.evaluation.limiterReforecast);
  assert.equal(result.state.value.phase, "production");
});

test("ordinary queue loading remains fail-closed for phase transitions", () => {
  const fixture = pilotTransitionFixture();
  const result = record(fixture);
  assert.throws(() => loadQueueAuthority({
    session: fixture.session,
    statePath: result.observations.state.path,
    checkpointPath: result.observations.checkpoint.path,
    trustedCheckpointDigest: result.observations.checkpoint.semanticDigest,
    control: fixture.control,
  }), /requires branded evaluation authority/);
});

test("byte-identical retry verifies and continues without overwriting", () => {
  const fixture = pilotTransitionFixture();
  const first = record(fixture);
  const before = fs.statSync(path.join(fixture.root, first.observations.state.path));
  const second = record(fixture);
  const after = fs.statSync(path.join(fixture.root, second.observations.state.path));
  assert.equal(after.ino, before.ino);
  assert.equal(second.state.value.digest, first.state.value.digest);
  assert.equal(second.checkpoint.value.digest, first.checkpoint.value.digest);
});

test("an exact orphan state can recover by publishing the missing checkpoint", () => {
  const fixture = pilotTransitionFixture();
  const paths = pilotTransitionPaths(fixture.evaluation.reference.digest);
  writeTransitionArtifact(fixture, paths.state, fixture.derived.state);
  const result = record(fixture);
  assert.equal(result.checkpoint.value.phase, "production");
  assert.ok(fs.existsSync(path.join(fixture.root, paths.checkpoint)));
});

test("divergent orphan state is rejected rather than repaired", () => {
  const fixture = pilotTransitionFixture();
  const paths = pilotTransitionPaths(fixture.evaluation.reference.digest);
  const changed = structuredClone(fixture.derived.state);
  changed.bundles = {};
  writeTransitionArtifact(fixture, paths.state, changed);
  assert.throws(() => record(fixture), /differs from expected bytes/);
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(fixture.root, paths.state), "utf8")), changed);
  assert.equal(fs.existsSync(path.join(fixture.root, paths.checkpoint)), false);
});

test("transition loading rejects evaluation, predecessor, and checkpoint drift", async (t) => {
  const cases = [
    ["evaluation reference", (state) => {
      state.pilot_transition.evaluation.digest = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    }],
    ["predecessor", (state) => {
      state.predecessor.state_digest = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    }],
    ["accepted seals", (state) => { state.seals = []; }],
    ["source", (_state, checkpoint) => {
      checkpoint.source_digest = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    }],
  ];
  for (const [name, mutate] of cases) {
    await t.test(name, () => {
      const fixture = pilotTransitionFixture();
      const state = structuredClone(fixture.derived.state);
      const checkpoint = structuredClone(fixture.derived.checkpoint);
      mutate(state, checkpoint);
      const persistedState = withDigest(state);
      checkpoint.queue_state_digest = persistedState.digest;
      const persistedCheckpoint = withDigest(checkpoint);
      writeTransitionArtifact(fixture,
        pilotTransitionPaths(fixture.evaluation.reference.digest).state,
        persistedState);
      writeTransitionArtifact(fixture,
        pilotTransitionPaths(fixture.evaluation.reference.digest).checkpoint,
        persistedCheckpoint);
      assert.throws(() => loadPilotTransition({
        session: fixture.session, control: fixture.control,
        evaluation: fixture.evaluation,
      }));
    });
  }
});

function record(fixture) {
  return recordPilotTransition({
    session: fixture.session, control: fixture.control, evaluation: fixture.evaluation,
  });
}

function withDigest(value) {
  const copy = structuredClone(value);
  const { digest: _ignored, ...payload } = copy;
  copy.digest = evidenceDigest(payload);
  return copy;
}
