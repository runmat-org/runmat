import assert from "node:assert/strict";
import test from "node:test";

import { evidenceDigest } from "../evidence.mjs";
import {
  acceptedSealSet, barrierSealSet, buildQueue, cohortBarrierBlockers,
  emptyQueueState, requiredBarrierBundleIds, validateQueueState,
} from "../queue.mjs";
import { cleanupRepositoryFixtures, controlledFixture } from "./helpers.mjs";

test.afterEach(cleanupRepositoryFixtures);

test("cohort barriers require every earlier-cohort bundle to be sealed", () => {
  const control = fixtureControl();
  const active = control.bundles.get("c02-active");
  assert.deepEqual(cohortBarrierBlockers(active, control, new Set()), [
    "cohort-barrier:C00:c00-foundation",
    "cohort-barrier:C01:c01-first",
    "cohort-barrier:C01:c01-second",
  ]);
  assert.deepEqual(cohortBarrierBlockers(active, control, new Set([
    "c00-foundation", "c01-first",
  ])), ["cohort-barrier:C01:c01-second"]);
  assert.deepEqual(cohortBarrierBlockers(active, control, new Set([
    "c00-foundation", "c01-first", "c01-second",
  ])), []);
});

test("bundles in one cohort do not block one another", () => {
  const control = fixtureControl();
  assert.deepEqual(
    cohortBarrierBlockers(
      control.bundles.get("c01-second"), control, new Set(["c00-foundation"]),
    ),
    [],
  );
});

test("only content-addressed passing seals can unblock queue barriers", () => {
  const fixture = controlledFixture();
  const seal = passingSeal(fixture);
  const state = queueState(fixture, reference(seal));
  const validated = validateQueueState(
    state, fixture.control, () => seal, () => fixture.queueState,
  );
  assert.deepEqual(validated.acceptedSeals, state.seals);
  assert.equal(validated.sealedBundles[0].integrated_revision, seal.phases.integrated_revision);
  assert.equal(validated.sealedBundles[0].source_digest, seal.source_digest);
  const accepted = acceptedSealSet(validated, fixture.control);
  assert.deepEqual(accepted.value.seals, state.seals);
  assert.equal(accepted.sealedBundles[0].source_digest, seal.source_digest);
  assert.equal(evidenceDigest({
    schema_version: accepted.value.schema_version,
    kind: accepted.value.kind,
    authority: accepted.value.authority,
    control_manifest_digest: accepted.value.control_manifest_digest,
    seals: accepted.value.seals,
  }), accepted.value.digest);
  assert.equal(buildQueue(fixture.inventory, fixture.control, validated).rows[0].migration_state, "sealed");
  const barriers = barrierSealSet(validated, fixture.control, fixture.bundleId);
  assert.deepEqual(requiredBarrierBundleIds(fixture.control, fixture.bundleId), []);
  assert.deepEqual(barriers.value.seals, []);
  assert.equal(evidenceDigest({
    schema_version: barriers.value.schema_version,
    kind: barriers.value.kind,
    authority: barriers.value.authority,
    control_manifest_digest: barriers.value.control_manifest_digest,
    bundle_id: barriers.value.bundle_id,
    seals: barriers.value.seals,
  }), barriers.value.digest);

  assert.throws(
    () => validateQueueState(state, fixture.control, () => null, () => fixture.queueState),
    /is missing/,
  );
  const changedBytes = structuredClone(seal);
  changedBytes.source_digest = `sha256:${"f".repeat(64)}`;
  assert.throws(
    () => validateQueueState(
      state, fixture.control, () => changedBytes, () => fixture.queueState,
    ),
    /digest mismatch/,
  );
});

test("caller-authored progress cannot claim or substitute seal authority", () => {
  const fixture = controlledFixture();
  const unvalidated = emptyQueueState(fixture.control);
  assert.throws(
    () => buildQueue(fixture.inventory, fixture.control, unvalidated),
    /exact validated queue state/,
  );

  const forgedProgress = emptyQueueState(fixture.control);
  forgedProgress.bundles[fixture.bundleId] = { artifact: "claimed-seal", state: "sealed" };
  assert.throws(
    () => validateQueueState(forgedProgress, fixture.control, () => null),
    /invalid queue state entry/,
  );
});

test("queue seals reject forged identity, artifact, bundle, control, and baseline authority", () => {
  const fixture = controlledFixture();
  for (const [mutate, pattern] of [
    [(value) => { value.extra = true; }, /fields must be exactly/],
    [(value) => { value.result = "fail"; }, /not a passing seal result/],
    [(value) => { value.seal_id = "another-seal"; }, /artifact id mismatch/],
    [(value) => { value.bundle_id = "another-bundle"; }, /bundle mismatch/],
    [(value) => { value.control_manifest_digest = `sha256:${"e".repeat(64)}`; }, /another control manifest/],
    [(value) => { value.control_baseline_inventory_digest = `sha256:${"d".repeat(64)}`; }, /stale control baseline inventory/],
    [(value) => { value.accepted_seal_set_digest = `sha256:${"c".repeat(64)}`; }, /accepted seal-set digest mismatch/],
    [(value) => { value.barrier_seal_set_digest = `sha256:${"b".repeat(64)}`; }, /barrier seal-set digest mismatch/],
    [(value) => { value.identities = ["another_identity"]; }, /identity coverage mismatch/],
  ]) {
    const seal = passingSeal(fixture);
    mutate(seal);
    const sealReference = reference(seal, fixture.bundleId);
    assert.throws(
      () => validateQueueState(
        queueState(fixture, sealReference), fixture.control, () => seal,
        () => fixture.queueState,
      ),
      pattern,
    );
  }
});

test("queue seal references are canonical, unique, and repository-relative", () => {
  const fixture = controlledFixture();
  const seal = passingSeal(fixture);
  const sealReference = reference(seal);

  const escaping = queueState(fixture, { ...sealReference, path: "../seal-fixture.json" });
  assert.throws(
    () => validateQueueState(
      escaping, fixture.control, () => seal, () => fixture.queueState,
    ),
    /normalized safe repository-relative path/,
  );

  const duplicate = queueState(fixture, sealReference);
  duplicate.seals.push(structuredClone(sealReference));
  resealQueueState(duplicate);
  assert.throws(
    () => validateQueueState(
      duplicate, fixture.control, () => seal, () => fixture.queueState,
    ),
    /duplicate bundle/,
  );
});

function fixtureControl() {
  const cohorts = new Map([
    ["C00", { id: "C00", order: 0 }],
    ["C01", { id: "C01", order: 1 }],
    ["C02", { id: "C02", order: 2 }],
  ]);
  const bundles = new Map([
    ["c00-foundation", bundle("c00-foundation", "foundation")],
    ["c01-first", bundle("c01-first", "first")],
    ["c01-second", bundle("c01-second", "second")],
    ["c02-active", bundle("c02-active", "active")],
  ]);
  const identities = new Map([
    ["foundation", { cohort: "C00" }],
    ["first", { cohort: "C01" }],
    ["second", { cohort: "C01" }],
    ["active", { cohort: "C02" }],
  ]);
  return { cohorts, bundles, identities };
}

function bundle(id, identity) {
  return { id, identities: [identity] };
}

function passingSeal(fixture) {
  const bundle = fixture.control.bundles.get(fixture.bundleId);
  const reviewedAuthoredWriteSet = structuredClone(bundle.authored_write_set);
  const reviewedIntegrationOutputs = bundle.integration_outputs.map(
    ({ product_id, path, producer }) => ({ product_id, path, producer }),
  );
  const phases = {
    lease_base_revision: fixture.lease.value.base_revision,
    authored_revision: fixture.inventory.source.revision,
    integrated_revision: fixture.inventory.source.revision,
    authored_changed_paths: [],
    integration_changed_paths: [],
    reviewed_authored_write_set: reviewedAuthoredWriteSet,
    reviewed_integration_outputs: reviewedIntegrationOutputs,
    authored_write_set_digest: evidenceDigest(reviewedAuthoredWriteSet),
    integration_outputs_digest: evidenceDigest(reviewedIntegrationOutputs),
  };
  const barrierSeals = [];
  const acceptedSeals = [];
  const acceptedSealSet = {
    schema_version: 1,
    kind: "runmat-builtin-migration-accepted-seal-set",
    authority: "derived-from-validated-seals",
    control_manifest_digest: fixture.control.digest,
    seals: acceptedSeals,
  };
  const barrierSealSet = {
    schema_version: 1,
    kind: "runmat-builtin-migration-barrier-seal-set",
    authority: "derived-from-validated-seals",
    control_manifest_digest: fixture.control.digest,
    bundle_id: fixture.bundleId,
    seals: barrierSeals,
  };
  return {
    schema_version: 5,
    kind: "runmat-builtin-migration-seal-result",
    authority: "development-integration-evidence-only",
    seal_id: "seal-fixture",
    bundle_id: fixture.bundleId,
    identities: [fixture.id],
    lease_id: fixture.lease.value.lease_id,
    lease_digest: fixture.lease.value.digest,
    phases,
    source_revision: fixture.inventory.source.revision,
    source_digest: fixture.inventory.source.digest,
    control_baseline_inventory_digest: fixture.inventory.digest,
    lease_base_inventory_digest: fixture.inventory.digest,
    subject_inventory_digest: fixture.inventory.digest,
    control_manifest_digest: fixture.control.digest,
    verification_digest: `sha256:${"a".repeat(64)}`,
    accepted_seals: acceptedSeals,
    accepted_seal_set_digest: evidenceDigest(acceptedSealSet),
    barrier_seals: barrierSeals,
    barrier_seal_set_digest: evidenceDigest(barrierSealSet),
    integration_gate_artifacts: ["generated-products", "inventory-delta"],
    result: "pass",
    failures: [],
  };
}

function reference(seal, bundleId = seal.bundle_id) {
  return {
    path: `${seal.seal_id}.json`,
    artifact_id: "seal-fixture",
    digest: evidenceDigest(seal),
    bundle_id: bundleId,
  };
}

function queueState(fixture, sealReference) {
  const payload = {
    schema_version: 3,
    kind: "runmat-builtin-migration-queue-state",
    authority: "reviewed-monotonic-scheduling-input",
    control_manifest_digest: fixture.control.digest,
    predecessor: {
      state_path: "queue-state-0.json",
      state_digest: fixture.queueState.stateDigest,
      checkpoint_path: "queue-checkpoint-0.json",
      checkpoint_digest: fixture.queueCheckpoint.digest,
    },
    bundles: {},
    seals: [sealReference],
  };
  return { ...payload, digest: evidenceDigest(payload) };
}

function resealQueueState(value) {
  delete value.digest;
  value.digest = evidenceDigest(value);
}
