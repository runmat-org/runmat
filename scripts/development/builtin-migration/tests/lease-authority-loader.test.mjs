import fs from "node:fs";
import assert from "node:assert/strict";
import test from "node:test";

import {
  openAuthorityLoadSession, openAuthorityRoot,
} from "../authority-loading/index.mjs";
import { contentDigest, evidenceDigest } from "../evidence.mjs";
import {
  assertActivePilotLeaseAuthority, assertBoundPilotLeaseAuthority,
  assertLoadedLeaseAuthority, loadLeaseAuthority,
  revalidateLeaseAuthority,
} from "../lease-authority.mjs";
import { buildAcceptedSealSet, buildBarrierSealSet } from "../seal-set-schema.mjs";
import {
  cleanupRepositoryFixtures, controlledFixture,
} from "./helpers.mjs";
import {
  bindFixtureLease as bind, leaseAuthorityFixture as authorityFixture,
  loadFixtureLease as load, sealedLeaseAuthorityFixture as sealedLeaseFixture,
} from "./lease-authority-fixture.mjs";

test.afterEach(cleanupRepositoryFixtures);

test("lease loading binds canonical bytes, semantic digest, and production lease validation", () => {
  const fixture = authorityFixture();
  const loaded = load(fixture);
  assert.equal(loaded.reference.path, fixture.reference.path);
  assert.equal(loaded.reference.digest, evidenceDigest(fixture.leaseValue));
  assert.equal(loaded.artifact.contentDigest, contentDigest(fixture.bytes));
  assert.equal(loaded.lease.value.digest, fixture.leaseValue.digest);
  assert.equal(load(fixture), loaded, "one session/path/digest has one authority capability");
  assert.equal(assertLoadedLeaseAuthority(loaded, fixture.session, fixture.control), loaded);
  assert.equal(revalidateLeaseAuthority(loaded, fixture.session, fixture.control), loaded);
  assert.throws(() => { loaded.reference.path = "changed.json"; }, TypeError);
  const bound = bind(fixture, loaded);
  assert.equal(
    bind(fixture, loaded), bound,
    "one lease/queue/session/control tuple has one bound capability",
  );
  assert.equal(assertBoundPilotLeaseAuthority(bound, fixture), bound);
  assert.equal(assertActivePilotLeaseAuthority(bound, fixture), bound);
});

test("historical loading is deterministic after expiry while active use is not", () => {
  const fixture = authorityFixture({ expired: true });
  const loaded = load(fixture);
  const bound = bind(fixture, loaded);
  assert.equal(loaded.lease.value.expires_at, "2020-01-02T00:00:00.000Z");
  assert.throws(
    () => assertActivePilotLeaseAuthority(bound, fixture),
    /has expired/,
  );
  const active = authorityFixture();
  const activeLoaded = load(active);
  const activeBound = bind(active, activeLoaded);
  assert.equal(
    assertActivePilotLeaseAuthority(activeBound, active), activeBound,
  );
});

test("references reject path and semantic digest substitution", () => {
  const fixture = authorityFixture();
  assert.throws(
    () => loadLeaseAuthority(
      fixture.session, { ...fixture.reference, path: "../lease.json" },
      fixture.control, fixture.repository,
    ),
    /canonical relative POSIX path/,
  );
  assert.throws(
    () => loadLeaseAuthority(
      fixture.session, { ...fixture.reference, digest: fakeDigest("f") },
      fixture.control, fixture.repository,
    ),
    /semantic digest mismatch/,
  );
});

test("production schema and inner self-digest remain mandatory", () => {
  for (const mutate of [
    (value) => { value.schema_version = 4; },
    (value) => { value.bundle_id = "changed-bundle"; },
  ]) {
    const fixture = authorityFixture({ mutate });
    assert.throws(
      () => load(fixture),
      /schema_version 6|unknown bundle|authored lease digest mismatch|differs from the reviewed request/,
    );
  }
});

test("loaded authority rejects clones and capabilities from another session or control", () => {
  const fixture = authorityFixture();
  const loaded = load(fixture);
  const secondSession = openAuthorityLoadSession(openAuthorityRoot(fixture.root));
  assert.throws(
    () => assertLoadedLeaseAuthority({ ...loaded }, fixture.session, fixture.control),
    /exact loaded lease authority/,
  );
  assert.throws(
    () => assertLoadedLeaseAuthority(loaded, secondSession, fixture.control),
    /exact loaded lease authority/,
  );
  const other = controlledFixture();
  assert.throws(
    () => assertLoadedLeaseAuthority(loaded, fixture.session, other.control),
    /exact loaded lease authority/,
  );
});

test("revalidation detects replacement or mutation after observation", () => {
  const fixture = authorityFixture();
  const loaded = load(fixture);
  fs.writeFileSync(fixture.target, `${JSON.stringify(fixture.leaseValue)}\n`);
  assert.throws(
    () => revalidateLeaseAuthority(loaded, fixture.session, fixture.control),
    /changed after observation/,
  );
});

test("queue binding rejects stale checkpoint, phase, seal sets, and base authority", () => {
  for (const [mutate, pattern] of [
    [(value) => updateLease(value, { queue_checkpoint_digest: fakeDigest("a") }),
      /exact queue checkpoint/],
    [(value, controlled) => {
      const barrier = buildBarrierSealSet({
        controlManifestDigest: controlled.control.digest,
        bundleId: controlled.bundleId,
        queuePhase: "production",
        seals: value.barrier_seals,
      });
      updateLease(value, {
        queue_phase: "production", barrier_seal_set_digest: barrier.digest,
      });
    },
      /all be in pilot phase/],
    [(value, controlled) => {
      const seals = [fakeSealReference()];
      const accepted = buildAcceptedSealSet(controlled.control.digest, seals);
      updateLease(value, {
        accepted_seals: seals, accepted_seal_set_digest: accepted.digest,
      });
    }, /accepted seals differ/],
    [(value) => {
      value.lease_base_inventory.inventory_digest = fakeDigest("c");
      value.request.lease_base_inventory.inventory_digest = fakeDigest("c");
      resealLease(value);
    }, /lease base differs/],
  ]) {
    const fixture = authorityFixture({ mutate });
    const loaded = load(fixture);
    assert.throws(() => bind(fixture, loaded), pattern);
  }
});

test("loaded leases are not operational authorization without exact queue binding", () => {
  const fixture = authorityFixture();
  const loaded = load(fixture);
  assert.throws(
    () => assertActivePilotLeaseAuthority(loaded, fixture),
    /exact bound pilot lease authority/,
  );
  const bound = bind(fixture, loaded);
  const anotherSession = openAuthorityLoadSession(openAuthorityRoot(fixture.root));
  assert.throws(
    () => assertBoundPilotLeaseAuthority(bound, {
      session: anotherSession, control: fixture.control,
    }),
    /exact bound pilot lease authority/,
  );
});

test("an accepted seal makes its originating lease target ineligible", () => {
  const fixture = sealedLeaseFixture();
  const loaded = load(fixture);
  assert.throws(() => bind(fixture, loaded), /target is already sealed/);
});

function updateLease(value, fields) {
  Object.assign(value, structuredClone(fields));
  Object.assign(value.request, structuredClone(fields));
  resealLease(value);
}

function resealLease(value) {
  delete value.digest;
  value.digest = evidenceDigest(value);
}

function fakeSealReference() {
  return {
    path: "seals/fake.json", artifact_id: "seal-fake", digest: fakeDigest("d"),
    bundle_id: "fake-bundle",
  };
}

function fakeDigest(character) {
  return `sha256:${character.repeat(64)}`;
}
