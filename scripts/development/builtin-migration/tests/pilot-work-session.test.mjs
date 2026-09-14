import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";

import {
  openAuthorityLoadSession, openAuthorityRoot,
} from "../authority-loading/index.mjs";
import { evidenceDigest } from "../evidence.mjs";
import {
  assertLoadedPilotWorkSessionCompletion, assertLoadedPilotWorkSessionStart,
  loadPilotWorkSessionCompletion, loadPilotWorkSessionStart,
  recordPilotWorkSessionStart,
} from "../pilot-work-session/index.mjs";
import { pilotWorkSessionId } from "../pilot-work-session/paths.mjs";
import {
  artifactBinding, queueBindingFromAuthority, sealReference, sourceBinding,
  withSelfDigest,
} from "../pilot-work-session/schema.mjs";
import { directSuccessorSeal } from "../pilot-work-session/validation.mjs";
import { cleanupRepositoryFixtures } from "./helpers.mjs";
import {
  acceptedAuthority, completedFixture, fakeDigest, fixtureSessionPaths,
  installAcceptedFiles, installSuccessor, loadQueue, recordCompletion, recordStart,
  reloadCompletion, reloadStart, replaceCompletion, reseal,
  resealSealAndSuccessor, startRecorderInput, startValue, workSessionFixture,
  writeJson,
} from "./pilot-work-session-fixture.mjs";

test.afterEach(cleanupRepositoryFixtures);

test("machine recorders publish and reload one exact logical session", () => {
  const fixture = workSessionFixture();
  const start = recordStart(fixture);
  assert.equal(start.bundleId, fixture.bundleId);
  assert.equal(start.sessionId, pilotWorkSessionId(
    fixture.control.pilotPolicy.pilotId, fixture.bundleId,
  ));
  assert.equal(assertLoadedPilotWorkSessionStart(start, fixture), start);
  const successor = installSuccessor(fixture);
  const completion = recordCompletion(
    fixture, start, fixture.lease, fixture.queue, successor,
  );
  assert.equal(completion.start, start);
  assert.deepEqual(
    completion.seal.reference, successor.state.sealedBundles[0].seal.reference,
  );
  assert.equal(assertLoadedPilotWorkSessionCompletion(completion, fixture), completion);
  assert.throws(() => recordStart(fixture), /evidence target already exists/);
  assert.throws(() => recordCompletion(
    fixture, start, fixture.lease, fixture.queue, successor,
  ), /evidence target already exists/);
  assert.throws(() => recordPilotWorkSessionStart({
    ...startRecorderInput(fixture), started_at: "2020-01-01T00:00:00.000Z",
  }), /fields must be exactly/);
});

test("loaded capabilities reject clones, another session, and changed bytes", () => {
  const fixture = workSessionFixture();
  const start = recordStart(fixture);
  assert.throws(
    () => assertLoadedPilotWorkSessionStart({ ...start }, fixture),
    /exact loaded pilot work-session start/,
  );
  const otherSession = openAuthorityLoadSession(openAuthorityRoot(fixture.root));
  assert.throws(
    () => assertLoadedPilotWorkSessionStart(start, {
      session: otherSession, control: fixture.control,
    }),
    /exact loaded pilot work-session start/,
  );
  fs.writeFileSync(
    path.join(fixture.root, start.reference.path),
    `${JSON.stringify(start.value)}\n`,
  );
  assert.throws(() => loadPilotWorkSessionStart({
    ...fixture, reference: start.reference,
  }), /changed after observation/);
});

test("completion rejects a non-successor queue and a seal from another lease", () => {
  const fixture = workSessionFixture();
  const start = recordStart(fixture);
  assert.throws(() => recordCompletion(
    fixture, start, fixture.lease, fixture.queue, fixture.queue,
  ), /exactly one appended queue seal/);
  const accepted = acceptedAuthority(fixture);
  accepted.seal.lease_id = "different-lease";
  resealSealAndSuccessor(accepted);
  installAcceptedFiles(fixture.root, accepted);
  const successor = loadQueue(fixture, 1, accepted.checkpointValue.digest);
  assert.throws(() => recordCompletion(
    fixture, start, fixture.lease, fixture.queue, successor,
  ), /lease|seal/i);
});

test("persisted starts reject schema, policy, timestamp, path, and digest drift", async (context) => {
  const scenarios = [
    ["legacy schema", (value) => { value.schema_version = 1; }],
    ["future schema", (value) => { value.schema_version = 3; }],
    ["policy", (value) => { value.pilot_policy_digest = fakeDigest("e"); }],
    ["timestamp", (value) => { value.started_at = "2100-01-01T00:00:00.000Z"; }],
    ["queue observation", (value) => {
      value.initial_queue.state.content_digest = fakeDigest("d");
    }],
    ["portable authority path", (value) => { value.initial_lease.path = "con/lease.json"; }],
  ];
  for (const [name, mutate] of scenarios) {
    await context.test(name, () => {
      const fixture = workSessionFixture();
      const value = startValue(fixture);
      mutate(value);
      reseal(value);
      const paths = fixtureSessionPaths(fixture);
      writeJson(path.join(fixture.root, paths.start), value);
      assert.throws(() => reloadStart(fixture, paths.start, value),
        /schema_version 2|policy|active lease interval|observation mismatch|canonical relative POSIX path/);
    });
  }
  await context.test("non-deterministic path", () => {
    const fixture = workSessionFixture();
    const value = startValue(fixture);
    writeJson(path.join(fixture.root, "other/start.json"), value);
    assert.throws(() => reloadStart(fixture, "other/start.json", value),
      /deterministic pilot\/bundle session path/);
  });
});

test("completion rejects clock reversal and integration outputs outside the final lease", () => {
  const reversal = completedFixture();
  const reversed = structuredClone(reversal.completion.value);
  reversed.ended_at = "2020-01-01T00:00:00.000Z";
  reseal(reversed);
  replaceCompletion(reversal, reversed);
  assert.throws(() => reloadCompletion(reversal, reversed), /end precedes/);
  const drift = workSessionFixture();
  const start = recordStart(drift);
  const accepted = acceptedAuthority(drift);
  accepted.seal.phases.reviewed_integration_outputs = [];
  accepted.seal.phases.integration_outputs_digest = evidenceDigest([]);
  resealSealAndSuccessor(accepted);
  installAcceptedFiles(drift.root, accepted);
  const successor = loadQueue(drift, 1, accepted.checkpointValue.digest);
  assert.throws(() => recordCompletion(
    drift, start, drift.lease, drift.queue, successor,
  ), /seal integration-output digest differs from its exact final lease/);

  const migrationDrift = workSessionFixture();
  const migrationStart = recordStart(migrationDrift);
  const migrationAccepted = acceptedAuthority(migrationDrift);
  migrationAccepted.seal.phases.reviewed_source_migrations = [{
    strategy: "module-support-reparent",
    source_path: "crates/runtime/shared.rs",
    source_baseline_digest: fakeDigest("1"),
    promoted_target: { kind: "authored", path: "crates/runtime/shared/mod.rs" },
    destination_paths: ["crates/runtime/shared/mod.rs", "crates/runtime/shared/support.rs"],
    support_destinations: [{
      destination_path: "crates/runtime/shared/support.rs", reexports: [],
    }],
    identity_destinations: [],
    reason: "Fixture source-migration drift",
    review: { status: "reviewed", evidence: ["fixture:source-migration-drift"] },
  }];
  migrationAccepted.seal.phases.source_migrations_digest = evidenceDigest(
    migrationAccepted.seal.phases.reviewed_source_migrations,
  );
  resealSealAndSuccessor(migrationAccepted);
  installAcceptedFiles(migrationDrift.root, migrationAccepted);
  const migrationSuccessor = loadQueue(
    migrationDrift, 1, migrationAccepted.checkpointValue.digest,
  );
  assert.throws(() => recordCompletion(
    migrationDrift, migrationStart, migrationDrift.lease, migrationDrift.queue,
    migrationSuccessor,
  ), /seal source-migrations digest differs from its exact final lease/);
});

test("persisted completion rejects schema and exact-start observation drift", async (context) => {
  for (const [name, mutate, expected] of [
    ["legacy schema", (value) => { value.schema_version = 1; }, /schema_version 2/],
    ["future schema", (value) => { value.schema_version = 3; }, /schema_version 2/],
    [
      "start observation",
      (value) => { value.start.content_digest = fakeDigest("c"); },
      /completion start observation mismatch/,
    ],
    [
      "final lease observation",
      (value) => { value.final_lease.content_digest = fakeDigest("b"); },
      /completion final lease observation mismatch/,
    ],
    [
      "pre-integration queue observation",
      (value) => { value.pre_integration_queue.state.content_digest = fakeDigest("a"); },
      /completion pre-integration queue state observation mismatch/,
    ],
    [
      "end outside final lease",
      (value) => { value.ended_at = "2100-01-01T00:00:00.000Z"; },
      /outside its active lease interval/,
    ],
  ]) {
    await context.test(name, () => {
      const fixture = completedFixture();
      const changed = structuredClone(fixture.completion.value);
      mutate(changed);
      reseal(changed);
      replaceCompletion(fixture, changed);
      assert.throws(() => reloadCompletion(fixture, changed), expected);
    });
  }
});

test("recorder and persisted completion reject seal provenance outside the final lease", () => {
  const fixture = workSessionFixture();
  const start = recordStart(fixture);
  const accepted = acceptedAuthority(fixture);
  accepted.seal.lease_base_inventory_digest = fakeDigest("9");
  resealSealAndSuccessor(accepted);
  installAcceptedFiles(fixture.root, accepted);
  const successor = loadQueue(fixture, 1, accepted.checkpointValue.digest);
  assert.throws(() => recordCompletion(
    fixture, start, fixture.lease, fixture.queue, successor,
  ), /seal lease base inventory digest differs from its exact final lease/);

  const sealed = directSuccessorSeal(fixture.queue, successor);
  const payload = withSelfDigest({
    schema_version: 2,
    kind: "runmat-builtin-migration-pilot-work-session-completion",
    authority: "machine-observed-development-evidence-only",
    control_manifest_digest: fixture.control.digest,
    pilot_policy_digest: fixture.control.pilotPolicyDigest,
    pilot_id: start.pilotId,
    start: artifactBinding(
      start.reference.path, start.reference.digest, start.artifact.contentDigest,
    ),
    final_lease: artifactBinding(
      fixture.lease.reference.path,
      fixture.lease.reference.digest,
      fixture.lease.artifact.contentDigest,
    ),
    pre_integration_queue: queueBindingFromAuthority(fixture.queue),
    successor_queue: queueBindingFromAuthority(successor),
    seal: sealReference(sealed),
    bundle_id: start.bundleId,
    session_id: start.sessionId,
    source: sourceBinding(successor.checkpoint),
    ended_at: new Date().toISOString(),
  });
  const target = fixtureSessionPaths(fixture).completion;
  writeJson(path.join(fixture.root, target), payload);
  assert.throws(() => loadPilotWorkSessionCompletion({
    session: fixture.session,
    reference: { path: target, digest: payload.digest },
    control: fixture.control,
    repository: fixture.repository,
  }), /seal lease base inventory digest differs from its exact final lease/);
});
