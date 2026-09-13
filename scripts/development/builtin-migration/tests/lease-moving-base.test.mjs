import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";

import { dispositionInputFromControl } from "../dispositions.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { captureMigrationPhases } from "../integration-phases.mjs";
import { buildInventory } from "../inventory.mjs";
import { issueLease, parseLease, validateAcceptedIntegrationCheckpoint } from "../lease.mjs";
import { acceptedSealSet, barrierSealSet } from "../queue.mjs";
import { cleanupRepositoryFixtures, controlledFixture, leaseBaseInventoryBinding } from "./helpers.mjs";

test.afterEach(cleanupRepositoryFixtures);

test("a lease base rejects arbitrary clean commits that are not accepted seal checkpoints", () => {
  const fixture = controlledFixture();
  const repository = fixture.repository;
  const originalBase = fixture.inventory.source.revision;

  fs.writeFileSync(path.join(repository, "prior-integrated-work.txt"), "already integrated\n");
  commit(repository, "prior integrated bundle");
  const baseInventory = inventoryAtHead(fixture);

  assert.throws(
    () => issueLease(request(fixture, fixture.inventory, "stale-base"), fixture.control, repository, fixture.inventory, fixture.queueState, fixture.queueCheckpoint),
    /base revision is stale/,
  );

  assert.throws(
    () => issueLease(request(fixture, baseInventory, "unsealed-base"), fixture.control, repository, baseInventory, fixture.queueState, fixture.queueCheckpoint),
    /lease base differs from the trusted current queue checkpoint source|first lease base must equal the exact frozen control baseline inventory/,
  );
});

test("a same-cohort accepted seal admits its exact checkpoint but not later commits or reverts", () => {
  const fixture = controlledFixture();
  const runtimePath = "crates/runmat-runtime/src/builtins/math/basic/foo.rs";
  const original = fs.readFileSync(path.join(fixture.repository, runtimePath), "utf8");
  fs.appendFileSync(path.join(fixture.repository, runtimePath), "\n// sealed same-cohort work\n");
  commit(fixture.repository, "seal same-cohort bundle");
  const sealedInventory = inventoryAtHead(fixture);
  const sealed = [{
    reference: { path: "seals/c01-peer.json", artifact_id: "seal-c01-peer", digest: `sha256:${"a".repeat(64)}`, bundle_id: "c01-peer" },
    integrated_revision: sealedInventory.source.revision,
    source_digest: sealedInventory.source.digest,
    subject_inventory_digest: sealedInventory.digest,
  }];
  const frozen = { revision: fixture.inventory.source.revision, inventory_digest: fixture.inventory.digest };
  assert.doesNotThrow(() => validateAcceptedIntegrationCheckpoint(
    fixture.repository, sealedInventory.source.revision, sealedInventory, frozen, sealed,
  ));

  fs.appendFileSync(path.join(fixture.repository, runtimePath), "\n// unsealed follow-up\n");
  commit(fixture.repository, "unsealed follow-up");
  const unsealed = inventoryAtHead(fixture);
  assert.throws(() => validateAcceptedIntegrationCheckpoint(
    fixture.repository, unsealed.source.revision, unsealed, frozen, sealed,
  ), /must equal an accepted sealed integration checkpoint/);

  fs.writeFileSync(path.join(fixture.repository, runtimePath), original);
  commit(fixture.repository, "revert sealed work without a seal");
  const reverted = inventoryAtHead(fixture);
  assert.throws(() => validateAcceptedIntegrationCheckpoint(
    fixture.repository, reverted.source.revision, reverted, frozen, sealed,
  ), /must equal an accepted sealed integration checkpoint/);
});

test("lease issuance rejects dirty and divergent integration bases", () => {
  const dirty = controlledFixture();
  fs.appendFileSync(
    path.join(dirty.repository, "crates/runmat-runtime/src/builtins/math/basic/foo.rs"),
    "\n// uncommitted\n",
  );
  assert.throws(
    () => issueLease(
      request(dirty, dirty.inventory, "dirty-base"),
      dirty.control,
      dirty.repository,
      dirty.inventory,
      dirty.queueState,
      dirty.queueCheckpoint,
    ),
    /clean worktree/,
  );

  const divergent = controlledFixture();
  const tree = git(divergent.repository, "rev-parse", "HEAD^{tree}");
  const unrelated = execFileSync("git", ["commit-tree", tree, "-m", "unrelated root"], {
    cwd: divergent.repository,
    encoding: "utf8",
    env: commitEnvironment(),
  }).trim();
  git(divergent.repository, "checkout", "--quiet", "--detach", unrelated);
  const divergentInventory = inventoryAtHead(divergent);
  assert.throws(
    () => issueLease(
      request(divergent, divergentInventory, "divergent-base"),
      divergent.control,
      divergent.repository,
      divergentInventory,
      divergent.queueState,
      divergent.queueCheckpoint,
    ),
    /base inventory revision differs|does not descend from the frozen control baseline/,
  );
});

test("lease parsing and subject admission reject forged or stale base provenance", () => {
  const fixture = controlledFixture();
  const base = fixture.inventory.source.revision;
  const value = fixture.leaseValue;

  const mismatched = structuredClone(value);
  mismatched.lease_base_inventory.inventory_digest = `sha256:${"0".repeat(64)}`;
  reseal(mismatched);
  assert.throws(
    () => parseLease(mismatched, fixture.control, fixture.repository),
    /differs from the reviewed request/,
  );

  const tree = git(fixture.repository, "rev-parse", "HEAD^{tree}");
  const unrelated = execFileSync("git", ["commit-tree", tree, "-m", "forged root"], {
    cwd: fixture.repository,
    encoding: "utf8",
    env: commitEnvironment(),
  }).trim();
  const forged = structuredClone(value);
  forged.base_revision = `git:${unrelated}`;
  forged.request.base_revision = forged.base_revision;
  reseal(forged);
  assert.throws(
    () => parseLease(forged, fixture.control, fixture.repository),
    /base inventory revision differs|does not descend from the frozen control baseline/,
  );

  const lease = parseLease(value, fixture.control, fixture.repository);
  fs.writeFileSync(path.join(fixture.repository, "prior-integrated-work.txt"), "already integrated\n");
  commit(fixture.repository, "advance beyond lease base");
  const staleSubject = structuredClone(fixture.inventory);
  staleSubject.source.revision = fixture.inventory.source.revision;
  assert.throws(
    () => captureMigrationPhases(
      fixture.repository,
      lease,
      fixture.control,
      staleSubject,
      base,
    ),
    /differs from the repository HEAD/,
  );
  const dirtySubject = structuredClone(fixture.inventory);
  dirtySubject.source.revision = base;
  dirtySubject.source.dirty = true;
  assert.throws(
    () => captureMigrationPhases(
      fixture.repository,
      lease,
      fixture.control,
      dirtySubject,
      base,
    ),
    /clean committed source snapshot/,
  );
});

function request(fixture, baseInventory, leaseId) {
  const accepted = acceptedSealSet(fixture.queueState, fixture.control);
  const barriers = barrierSealSet(fixture.queueState, fixture.control, fixture.bundleId);
  return {
    schema_version: 5,
    kind: "runmat-builtin-migration-lease-request",
    authority: "reviewed-development-request",
    control_manifest_digest: fixture.control.digest,
    bundle_id: fixture.bundleId,
    lease_id: leaseId,
    owner: "fixture-migrator",
    base_revision: baseInventory.source.revision,
    lease_base_inventory: leaseBaseInventoryBinding(baseInventory),
    queue_checkpoint_digest: fixture.queueCheckpoint.digest,
    queue_phase: fixture.queueState.value.phase,
    accepted_seals: accepted.value.seals,
    accepted_seal_set_digest: accepted.value.digest,
    barrier_seals: barriers.value.seals,
    barrier_seal_set_digest: barriers.value.digest,
    issued_at: "2020-01-01T00:00:00.000Z",
    expires_at: "2099-01-01T00:00:00.000Z",
    review: { status: "reviewed", evidence: ["fixture integration-base review"] },
  };
}

function inventoryAtHead(fixture) {
  return buildInventory(fixture.repository, dispositionInputFromControl(fixture.control), {
    revision: revision(fixture.repository),
    compiledInventory: fixture.compiledInventory,
  });
}

function revision(repository) {
  return `git:${git(repository, "rev-parse", "HEAD")}`;
}

function commit(repository, message) {
  git(repository, "add", ".");
  execFileSync("git", ["commit", "--quiet", "-m", message], {
    cwd: repository,
    env: commitEnvironment(),
  });
}

function git(repository, ...arguments_) {
  return execFileSync("git", arguments_, { cwd: repository, encoding: "utf8" }).trim();
}

function commitEnvironment() {
  return {
    ...process.env,
    GIT_AUTHOR_NAME: "RunMat Test",
    GIT_AUTHOR_EMAIL: "test@runmat.invalid",
    GIT_COMMITTER_NAME: "RunMat Test",
    GIT_COMMITTER_EMAIL: "test@runmat.invalid",
    GIT_CONFIG_COUNT: "1",
    GIT_CONFIG_KEY_0: "commit.gpgsign",
    GIT_CONFIG_VALUE_0: "false",
  };
}

function reseal(value) {
  delete value.digest;
  value.digest = evidenceDigest(value);
}
