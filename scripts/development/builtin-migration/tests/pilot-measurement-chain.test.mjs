import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";

import { openAuthorityLoadSession, openAuthorityRoot } from "../authority-loading/index.mjs";
import { compareCodePoint } from "../constants.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { publishPilotMeasurement } from "../pilot-measurement.mjs";
import { cleanupRepositoryFixtures } from "./helpers.mjs";
import { pilotMeasurementChainFixture } from "./pilot-measurement-chain-fixture.mjs";
import { writeJson } from "./pilot-measurement-fixture.mjs";

test.afterEach(cleanupRepositoryFixtures);

test("derives overlapping timing and queue order from two independent pilot sessions", () => {
  const fixture = pilotMeasurementChainFixture();
  const result = fixture.measurement.value;
  const canonicalPaths = fixture.canonicalReferences.map((entry) => entry.path);
  const historyPaths = result.completed_sessions.map((entry) => entry.path);
  assert.deepEqual(canonicalPaths, [...canonicalPaths].sort(compareCodePoint));
  assert.deepEqual(historyPaths, fixture.history.map((entry) => entry.reference.path));
  assert.deepEqual(result.bundle_ids, [fixture.chain.fixture.bundleIds[1], fixture.chain.fixture.bundleIds[0]]);

  const expectedTiming = timingFrom(fixture.history);
  assert.deepEqual(result.timing, expectedTiming);
  assert.ok(result.timing.aggregate_worker_ms > result.timing.elapsed_ms);
  assert.ok(Date.parse(fixture.alphaStart.startedAt) <= Date.parse(fixture.betaStart.startedAt));
  assert.ok(Date.parse(fixture.betaCompletion.endedAt) <= Date.parse(fixture.alphaCompletion.endedAt));

  assert.deepEqual(
    result.seals.map(referenceKey).sort(compareCodePoint),
    fixture.queue2.state.acceptedSeals.map(referenceKey).sort(compareCodePoint),
  );
});

test("rejects an omitted session and the resulting queue-history gap", () => {
  const fixture = pilotMeasurementChainFixture({ publish: false });
  rewriteReview(fixture, (value) => {
    value.completed_sessions = [structuredClone(fixture.alphaCompletion.reference)];
  });
  assert.throws(() => publishFresh(fixture), /complete initial-to-final queue history/);
});

test("rejects a completion that claims a forked predecessor", () => {
  const fixture = pilotMeasurementChainFixture({ publish: false });
  replaceCompletion(fixture, fixture.alphaCompletion, (value) => {
    value.pre_integration_queue = structuredClone(fixture.reviewValue.initial_queue);
  });
  assert.throws(
    () => publishFresh(fixture),
    /exact queue checkpoint|direct successor/,
  );
});

test("rejects duplicate bundle and session identities", () => {
  for (const [mutate, pattern] of [
    [(value, fixture) => { value.bundle_id = fixture.betaCompletion.bundleId; }, /identity differs/],
    [(value, fixture) => { value.session_id = fixture.betaCompletion.sessionId; }, /identity differs/],
  ]) {
    const fixture = pilotMeasurementChainFixture({ publish: false });
    replaceCompletion(fixture, fixture.alphaCompletion, (value) => mutate(value, fixture));
    assert.throws(() => publishFresh(fixture), pattern);
  }
});

test("rejects a production-valid lease id reused across pilot sessions", () => {
  const fixture = pilotMeasurementChainFixture({
    publish: false,
    duplicateInitialLeaseId: true,
  });
  assert.equal(
    fixture.alphaStart.initialLeaseAuthority.lease.value.lease_id,
    fixture.betaStart.initialLeaseAuthority.lease.value.lease_id,
  );
  assert.throws(
    () => publishFresh(fixture),
    /duplicates lease id across sessions/,
  );
});

function timingFrom(history) {
  const starts = history.map((entry) => Date.parse(entry.start.startedAt));
  const ends = history.map((entry) => Date.parse(entry.endedAt));
  return {
    started_at: new Date(Math.min(...starts)).toISOString(),
    ended_at: new Date(Math.max(...ends)).toISOString(),
    elapsed_ms: Math.max(...ends) - Math.min(...starts),
    aggregate_worker_ms: history.reduce((sum, entry) =>
      sum + Date.parse(entry.endedAt) - Date.parse(entry.start.startedAt), 0),
  };
}

function replaceCompletion(fixture, completion, mutate) {
  const changed = structuredClone(completion.value);
  mutate(changed);
  reseal(changed);
  const target = path.join(fixture.root, completion.reference.path);
  fs.renameSync(target, `${target}.prior`);
  writeJson(target, changed);
  rewriteReview(fixture, (value) => {
    const reference = value.completed_sessions.find(
      (entry) => entry.path === completion.reference.path,
    );
    reference.digest = changed.digest;
  });
}

function rewriteReview(fixture, mutate) {
  mutate(fixture.reviewValue);
  reseal(fixture.reviewValue);
  writeJson(path.join(fixture.root, fixture.reviewPath), fixture.reviewValue);
  fixture.manifestReference.digest = fixture.reviewValue.digest;
}

function publishFresh(fixture) {
  return publishPilotMeasurement({
    session: openAuthorityLoadSession(openAuthorityRoot(fixture.root)),
    manifestReference: fixture.manifestReference,
    control: fixture.control,
    repository: fixture.repository,
  });
}

function reseal(value) {
  delete value.digest;
  value.digest = evidenceDigest(value);
}

function referenceKey(value) {
  return `${value.path}\0${value.digest}`;
}
