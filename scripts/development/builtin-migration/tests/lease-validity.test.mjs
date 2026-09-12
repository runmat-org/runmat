import assert from "node:assert/strict";
import test from "node:test";

import { auditMigration } from "../audit.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { runGateProducer } from "../gate-adapter.mjs";
import { parseGateResult } from "../gate-result.mjs";
import { captureMigrationPhases } from "../integration-phases.mjs";
import { assertActiveLease, issueLease, parseLease } from "../lease.mjs";
import { prepareIdentity } from "../prepare.mjs";
import { sealBundle } from "../seal.mjs";
import {
  cleanupRepositoryFixtures, controlledFixture, gate,
} from "./helpers.mjs";
import { createTemporaryDirectory } from "./temporary-directories.mjs";

const DIGEST = `sha256:${"a".repeat(64)}`;

test.afterEach(cleanupRepositoryFixtures);

test("active lease authorization uses an injected clock and half-open interval", () => {
  const fixture = controlledFixture();
  const issued = Date.parse(fixture.lease.value.issued_at);
  const expires = Date.parse(fixture.lease.value.expires_at);
  let calls = 0;
  const atIssue = () => { calls += 1; return new Date(issued); };
  assert.equal(assertActiveLease(fixture.lease, fixture.control, atIssue), fixture.lease);
  assert.equal(calls, 1, "authorization must take one deterministic clock observation");
  assert.doesNotThrow(() => assertActiveLease(
    fixture.lease, fixture.control, () => expires - 1,
  ));
  assert.throws(
    () => assertActiveLease(fixture.lease, fixture.control, () => issued - 1),
    /not active yet/,
  );
  assert.throws(
    () => assertActiveLease(fixture.lease, fixture.control, () => expires),
    /has expired/,
  );
});

test("lease issuance authorizes the reviewed interval at the injected time", () => {
  const fixture = controlledFixture();
  const request = structuredClone(fixture.leaseRequest);
  request.lease_id = "future-window";
  request.issued_at = "2026-10-01T00:00:00.000Z";
  request.expires_at = "2026-10-02T00:00:00.000Z";
  const issue = (clock) => issueLease(
    request, fixture.control, fixture.repository, fixture.inventory,
    fixture.queueState, fixture.queueCheckpoint, clock,
  );
  assert.throws(() => issue(() => Date.parse("2026-09-30T23:59:59.999Z")), /not active yet/);
  assert.doesNotThrow(() => issue(() => Date.parse(request.issued_at)));
  assert.throws(() => issue(() => Date.parse(request.expires_at)), /has expired/);
});

test("historical lease parsing remains deterministic after expiration", () => {
  const fixture = controlledFixture();
  const historical = historicalLease(fixture);
  const parsed = parseLease(historical, fixture.control, fixture.repository);
  assert.equal(parsed.value.digest, historical.digest);
  assert.throws(
    () => assertActiveLease(parsed, fixture.control, () => Date.parse("2026-09-12T00:00:00.000Z")),
    /has expired/,
  );
});

test("live prepare, gate, audit, phase capture, and seal reject an expired lease", () => {
  const fixture = controlledFixture();
  const expired = parseLease(historicalLease(fixture), fixture.control, fixture.repository);
  const clock = () => Date.parse("2026-09-12T00:00:00.000Z");
  const output = createTemporaryDirectory("expired-lease-");
  assert.throws(() => prepareIdentity(
    fixture.repository, fixture.inventory, fixture.control, expired, "foo", output, clock,
  ), /has expired/);
  assert.throws(() => runGateProducer({
    control: fixture.control,
    lease: expired,
    queue_state: fixture.queueState,
    queue_checkpoint: fixture.queueCheckpoint,
    control_baseline_inventory: fixture.inventory,
    lease_base_inventory: fixture.inventory,
    subject_inventory: fixture.inventory,
    bundle_id: fixture.bundleId,
    gate: "architecture",
    artifact_id: "expired-gate",
    inputs: null,
  }, clock), /has expired/);
  const batch = {
    schema_version: 1,
    kind: "runmat-builtin-migration-batch",
    identities: ["foo"],
  };
  assert.throws(() => auditMigration(
    fixture.repository, fixture.inventory, fixture.inventory, fixture.inventory,
    fixture.control, expired, batch,
    { artifact_id: "expired-audit", authored_revision: fixture.inventory.source.revision },
    clock,
  ), /has expired/);
  assert.throws(() => captureMigrationPhases(
    fixture.repository, expired, fixture.control, fixture.inventory,
    fixture.inventory.source.revision, clock,
  ), /has expired/);
  const phases = captureMigrationPhases(
    fixture.repository, fixture.lease, fixture.control, fixture.inventory,
    fixture.inventory.source.revision,
  );
  assert.throws(() => sealBundle(
    sealManifest(fixture, expired, phases), null, [], [], fixture.control,
    fixture.repository, expired, clock,
  ), /has expired/);
});

test("gate results reject the unreachable unavailable state", () => {
  const fixture = controlledFixture();
  const aggregate = gate(fixture, "architecture");
  aggregate.result = "unavailable";
  assert.throws(() => parseGateResult(aggregate), /gate result must be one of pass, fail/);
  const check = gate(fixture, "architecture");
  check.checks[0].result = "unavailable";
  check.result = "fail";
  assert.throws(() => parseGateResult(check), /gate check result must be one of pass, fail/);
});

function historicalLease(fixture) {
  const value = structuredClone(fixture.lease.value);
  value.request.issued_at = "2020-01-01T00:00:00.000Z";
  value.request.expires_at = "2020-01-02T00:00:00.000Z";
  value.issued_at = value.request.issued_at;
  value.expires_at = value.request.expires_at;
  delete value.digest;
  value.digest = evidenceDigest(value);
  return value;
}

function sealManifest(fixture, lease, phases) {
  return {
    schema_version: 5,
    kind: "runmat-builtin-migration-seal-manifest",
    authority: "reviewed-integration-request",
    seal_id: "expired-seal",
    bundle_id: fixture.bundleId,
    lease_id: lease.value.lease_id,
    lease_digest: lease.value.digest,
    identities: ["foo"],
    source_revision: fixture.inventory.source.revision,
    source_digest: fixture.inventory.source.digest,
    control_baseline_inventory_digest: fixture.inventory.digest,
    lease_base_inventory_digest: fixture.inventory.digest,
    subject_inventory_digest: fixture.inventory.digest,
    control_manifest_digest: fixture.control.digest,
    accepted_seals: lease.value.accepted_seals,
    accepted_seal_set_digest: lease.value.accepted_seal_set_digest,
    barrier_seals: lease.value.barrier_seals,
    barrier_seal_set_digest: lease.value.barrier_seal_set_digest,
    phases,
    verification: { path: "verification.json", artifact_id: "verification", digest: DIGEST },
    integration_gates: [{
      path: "integration-gate.json",
      artifact_id: "integration-gate",
      digest: DIGEST,
    }],
    review: { status: "reviewed", evidence: ["expiry boundary review"] },
  };
}
