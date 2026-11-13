import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";

import {
  openAuthorityLoadSession, openAuthorityRoot,
} from "../authority-loading/index.mjs";
import { evidenceDigest } from "../evidence.mjs";
import {
  assertLoadedPilotMeasurement, assertLoadedPilotMeasurementReview,
  loadPilotMeasurement, loadPilotMeasurementReview, publishPilotMeasurement,
  revalidatePilotMeasurement,
} from "../pilot-measurement.mjs";
import { parseReviewValue } from "../pilot-measurement/schema.mjs";
import { derivePilotTiming } from "../pilot-measurement/timing.mjs";
import { cleanupRepositoryFixtures, controlledFixture } from "./helpers.mjs";
import {
  pilotMeasurementFixture, writeJson,
} from "./pilot-measurement-fixture.mjs";

test.afterEach(cleanupRepositoryFixtures);

test("publishes and reloads exact reviewed pilot measurement authority", () => {
  const fixture = pilotMeasurementFixture();
  const measurement = fixture.measurement;
  const observedDuration = Date.parse(fixture.completion.endedAt)
    - Date.parse(fixture.start.startedAt);
  assert.ok(observedDuration > 0);
  assert.equal(measurement.value.timing.elapsed_ms, observedDuration);
  assert.equal(measurement.value.timing.aggregate_worker_ms, observedDuration);
  assert.deepEqual(measurement.value.counts, fixture.control.pilotPolicy.counts);
  assert.deepEqual(measurement.value.bundle_ids, [fixture.bundleId]);
  assert.match(
    measurement.reference.path,
    new RegExp(`^pilot-measurements/${fixture.reviewValue.digest.slice(7)}/measurement\\.json$`),
  );
  assert.equal(assertLoadedPilotMeasurement(measurement, fixture), measurement);
  assert.equal(revalidatePilotMeasurement(measurement, fixture), measurement);
  assert.equal(loadPilotMeasurement({
    session: fixture.session,
    reference: measurement.reference,
    control: fixture.control,
    repository: fixture.repository,
  }), measurement);
  assert.throws(() => { measurement.value.timing.elapsed_ms = 1; }, TypeError);
});

test("review loading preserves canonical authored references and rejects clones", () => {
  const fixture = pilotMeasurementFixture({ publish: false });
  const review = loadPilotMeasurementReview({
    session: fixture.session,
    reference: fixture.manifestReference,
    control: fixture.control,
  });
  assert.deepEqual(review.value.completed_sessions, fixture.reviewValue.completed_sessions);
  assert.equal(assertLoadedPilotMeasurementReview(review, fixture), review);
  assert.throws(
    () => assertLoadedPilotMeasurementReview({ ...review }, fixture),
    /exact loaded pilot measurement review/,
  );
  const anotherSession = openAuthorityLoadSession(openAuthorityRoot(fixture.root));
  assert.throws(
    () => assertLoadedPilotMeasurementReview(review, {
      session: anotherSession, control: fixture.control,
    }),
    /exact loaded pilot measurement review/,
  );
});

test("review schema rejects versions, extra fields, duplicates, and noncanonical references", () => {
  const base = pilotMeasurementFixture({ publish: false }).reviewValue;
  for (const [mutate, pattern] of [
    [(value) => { value.schema_version = 0; }, /schema_version 1/],
    [(value) => { value.schema_version = 2; }, /schema_version 1/],
    [(value) => { value.repair = true; }, /fields must be exactly/],
    [(value) => { value.completed_sessions.push(structuredClone(value.completed_sessions[0])); },
      /unique and canonical/],
    [(value) => {
      value.completed_sessions = [
        { path: "z/completion.json", digest: fakeDigest("a") },
        { path: "a/completion.json", digest: fakeDigest("b") },
      ];
    }, /unique and canonical/],
  ]) {
    const value = structuredClone(base);
    mutate(value);
    reseal(value);
    assert.throws(() => parseReviewValue(value), pattern);
  }
});

test("review authority rejects digest, content, root queue, and final queue substitution", () => {
  const digestFixture = pilotMeasurementFixture({ publish: false });
  assert.throws(() => publishPilotMeasurement({
    ...publisherInput(digestFixture),
    manifestReference: { ...digestFixture.manifestReference, digest: fakeDigest("d") },
  }), /semantic digest mismatch/);

  for (const [mutate, pattern] of [
    [(value) => { value.initial_queue.state.content_digest = fakeDigest("c"); },
      /initial queue observation mismatch/],
    [(value) => { value.initial_queue = structuredClone(value.final_queue); },
      /empty reviewed pilot root/],
    [(value) => { value.final_queue = structuredClone(value.initial_queue); },
      /complete initial-to-final queue history/],
  ]) {
    const fixture = pilotMeasurementFixture({ publish: false });
    replaceReview(fixture, mutate);
    assert.throws(() => publishPilotMeasurement(publisherInput(fixture)), pattern);
  }
});

test("session selection must exactly cover the pilot without duplicates or omissions", () => {
  for (const [mutate, pattern] of [
    [(value) => { value.completed_sessions = []; }, /nonempty array/],
    [(value) => { value.completed_sessions.push(structuredClone(value.completed_sessions[0])); },
      /unique and canonical/],
  ]) {
    const fixture = pilotMeasurementFixture({ publish: false });
    replaceReview(fixture, mutate);
    assert.throws(() => publishPilotMeasurement(publisherInput(fixture)), pattern);
  }
});

test("result derived timing, counts, and path cannot be authored or relocated", () => {
  for (const [mutate, pathMutation, pattern] of [
    [(value) => { value.timing.elapsed_ms += 1; }, null, /exact reconstructed evidence/],
    [(value) => { value.counts.public_identities = 0; }, null, /exact reconstructed evidence/],
    [(value) => { value.bundle_ids = []; }, null, /nonempty array/],
    [() => {}, "other/measurement.json", /deterministic review path/],
  ]) {
    const fixture = pilotMeasurementFixture();
    const changed = structuredClone(fixture.measurement.value);
    mutate(changed);
    reseal(changed);
    const relative = pathMutation ?? fixture.measurement.reference.path;
    if (relative === fixture.measurement.reference.path) {
      fs.renameSync(path.join(fixture.root, relative), path.join(fixture.root, `${relative}.prior`));
    }
    writeJson(path.join(fixture.root, relative), changed);
    const session = openAuthorityLoadSession(openAuthorityRoot(fixture.root));
    assert.throws(() => loadPilotMeasurement({
      session,
      reference: { path: relative, digest: changed.digest },
      control: fixture.control,
      repository: fixture.repository,
    }), pattern);
  }
});

test("dependency replacement and capability/session substitution fail closed", () => {
  const fixture = pilotMeasurementFixture();
  fs.writeFileSync(
    path.join(fixture.root, fixture.completion.reference.path),
    `${JSON.stringify(fixture.completion.value)}\n`,
  );
  assert.throws(
    () => revalidatePilotMeasurement(fixture.measurement, fixture),
    /changed after observation/,
  );

  const clean = pilotMeasurementFixture();
  const other = controlledFixture();
  assert.throws(
    () => assertLoadedPilotMeasurement(clean.measurement, {
      session: clean.session, control: other.control,
    }),
    /exact loaded pilot measurement result/,
  );
});

test("timing rejects zero work, reversal, and unsafe aggregate arithmetic", () => {
  assert.throws(
    () => derivePilotTiming([
      fakeCompletion("2026-01-01T00:00:00.000Z", "2026-01-01T00:00:00.000Z"),
    ]),
    /aggregate worker milliseconds must be positive/,
  );
  assert.throws(
    () => derivePilotTiming([
      fakeCompletion("2026-01-01T01:00:00.000Z", "2026-01-01T00:00:00.000Z"),
    ]),
    /duration must be an integer >= 0/,
  );
  const earliest = new Date(-8_640_000_000_000_000).toISOString();
  const latest = new Date(8_640_000_000_000_000).toISOString();
  assert.throws(() => derivePilotTiming([
    fakeCompletion(earliest, "1970-01-01T00:00:00.000Z"),
    fakeCompletion("1970-01-01T00:00:00.000Z", latest),
  ]), /must be an integer/);
});

function publisherInput(fixture) {
  return {
    session: fixture.session,
    manifestReference: fixture.manifestReference,
    control: fixture.control,
    repository: fixture.repository,
  };
}

function replaceReview(fixture, mutate) {
  mutate(fixture.reviewValue);
  reseal(fixture.reviewValue);
  writeJson(path.join(fixture.root, fixture.reviewPath), fixture.reviewValue);
  fixture.manifestReference.digest = fixture.reviewValue.digest;
}

function reseal(value) {
  delete value.digest;
  value.digest = evidenceDigest(value);
}

function fakeCompletion(startedAt, endedAt) {
  return { start: { startedAt }, endedAt };
}

function fakeDigest(character) {
  return `sha256:${character.repeat(64)}`;
}
