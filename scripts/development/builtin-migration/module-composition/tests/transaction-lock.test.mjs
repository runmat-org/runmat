import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  COMPOSITION_REPOSITORY_LOCK_LEASE_MS, renewCompositionRepositoryLock,
  withCompositionRepositoryLock,
} from "../transaction-lock.mjs";
import {
  createLockOwner, readLockOwner, replaceLockOwner,
} from "../transaction-lock-owner.mjs";
import { acquireProcessFence, releaseProcessFence } from "../transaction-lock-fence.mjs";
import {
  claimRecoveryReservation, releaseRecoveryReservation,
} from "../transaction-recovery-reservation.mjs";

test("a live lease renews from an injected clock and retains its original acquisition", () => withRoot((root) => {
  let now = Date.UTC(2026, 0, 1);
  withCompositionRepositoryLock(root, (lock) => {
    const initial = owner(root);
    now += 60_000;
    const renewed = renewCompositionRepositoryLock(lock);
    assert.equal(renewed.acquired_at, initial.acquired_at);
    assert.equal(renewed.renewed_at, new Date(now).toISOString());
    assert.equal(Date.parse(renewed.expires_at), now + COMPOSITION_REPOSITORY_LOCK_LEASE_MS);
  }, lockOptions(() => now, ["1".repeat(32)]));
  assert.equal(fs.existsSync(lockPath(root)), false);
}));

test("elapsed metadata expiry cannot revoke a process-fenced active handle", () => withRoot((root) => {
  let now = Date.UTC(2026, 0, 1);
  withCompositionRepositoryLock(root, (lock) => {
    now += COMPOSITION_REPOSITORY_LOCK_LEASE_MS;
    assert.throws(() => withCompositionRepositoryLock(
      root, () => assert.fail("a successor ran while the original process fence was live"),
      lockOptions(() => now, ["f".repeat(32)]),
    ), /retains the repository process fence/);
    assert.equal(Date.parse(renewCompositionRepositoryLock(lock).renewed_at), now);
  }, lockOptions(() => now, ["0".repeat(32)]));
  assert.equal(fs.existsSync(lockPath(root)), false);
}));

test("an expired complete foreign-host owner fails closed without automatic takeover", () => withRoot((root) => {
  const acquired = Date.UTC(2026, 0, 1);
  const stale = createOwner("2", acquired, 55, "foreign-host");
  strand(root, stale);
  assert.throws(() => withCompositionRepositoryLock(
    root, () => {}, lockOptions(() => acquired + 1, ["3".repeat(32)]),
  ), /holds the repository lock lease/);
  assert.throws(() => withCompositionRepositoryLock(
    root, () => assert.fail("an expired complete owner was automatically displaced"),
    lockOptions(() => acquired + COMPOSITION_REPOSITORY_LOCK_LEASE_MS, ["4".repeat(32)]),
  ), /complete owner; automatic recovery is disabled/);
  assert.deepEqual(owner(root), stale);
}));

test("PID reuse cannot authorize replacement of a complete owner", () => withRoot((root) => {
  const acquired = Date.UTC(2026, 0, 1);
  const stale = createOwner("6", acquired, process.pid, os.hostname());
  strand(root, stale);
  assert.throws(() => withCompositionRepositoryLock(
    root, () => {}, lockOptions(() => acquired + 1, ["7".repeat(32)], process.pid),
  ), /holds the repository lock lease/);
  assert.throws(() => withCompositionRepositoryLock(
    root, () => assert.fail("PID reuse displaced the complete owner"),
    lockOptions(() => acquired + COMPOSITION_REPOSITORY_LOCK_LEASE_MS, ["8".repeat(32)], process.pid),
  ), /complete owner; automatic recovery is disabled/);
  assert.deepEqual(owner(root), stale);
}));

test("complete-owner expiry performs no quarantine or filesystem mutation", () => withRoot((root) => {
  const acquired = Date.UTC(2026, 0, 1);
  const stale = createOwner("a", acquired, 72, "foreign-host");
  strand(root, stale);
  const now = acquired + COMPOSITION_REPOSITORY_LOCK_LEASE_MS;
  let quarantined = false;
  assert.throws(() => withCompositionRepositoryLock(root, () => {}, {
    ...lockOptions(() => now, ["b".repeat(32)]),
    afterQuarantine() { quarantined = true; },
  }), /complete owner; automatic recovery is disabled/);
  assert.equal(quarantined, false);
  assert.deepEqual(owner(root), stale);
}));

test("claim rechecks recovery admission after mkdir and removes only its empty directory", () => withRoot((root) => {
  const acquired = Date.UTC(2026, 0, 1);
  const stale = createOwner("9", acquired, 72, "stopped-host");
  strand(root, stale);
  const quarantine = `${lockPath(root)}.test-quarantine`;
  let reservation;
  try {
    assert.throws(() => withCompositionRepositoryLock(root, () => {
      assert.fail("claim entered after recovery reserved the repository");
    }, {
      ...lockOptions(() => acquired + COMPOSITION_REPOSITORY_LOCK_LEASE_MS, ["8".repeat(32)]),
      afterFirstAdmission() {
        reservation = claimRecoveryReservation(fs.realpathSync(root));
        fs.renameSync(lockPath(root), quarantine);
      },
    }), /stopped-owner recovery is in progress/);
    assert.equal(fs.existsSync(lockPath(root)), false);
    assert.deepEqual(readLockOwner(quarantine, COMPOSITION_REPOSITORY_LOCK_LEASE_MS), stale);
  } finally {
    if (reservation) releaseRecoveryReservation(fs.realpathSync(root), reservation);
  }
}));

test("owner tampering is rejected without deleting the lock", () => withRoot((root) => {
  const acquired = Date.UTC(2026, 0, 1);
  strand(root, { ...createOwner("c", acquired, 81, "host"), unexpected: true });
  assert.throws(() => withCompositionRepositoryLock(
    root, () => {}, lockOptions(() => acquired + COMPOSITION_REPOSITORY_LOCK_LEASE_MS, ["d".repeat(32)]),
  ), /lock owner is invalid/);
  assert.equal(fs.existsSync(lockPath(root)), true);
}));

test("release validates the active token and preserves a replacement owner", () => withRoot((root) => {
  const now = Date.UTC(2026, 0, 1);
  assert.throws(() => withCompositionRepositoryLock(root, (lock) => {
    const active = owner(root);
    const replacement = createLockOwner({
      token: "f".repeat(32), acquiredAt: null, now,
      pid: active.pid, hostname: active.hostname,
      fencePort: active.fence_port,
      leaseMs: COMPOSITION_REPOSITORY_LOCK_LEASE_MS,
    });
    replaceLockOwner(lockPath(root), replacement, { expected: active });
    assert.notEqual(lock.token, replacement.token);
  }, lockOptions(() => now, ["e".repeat(32)])), /lock owner changed/);
  assert.equal(owner(root).token, "f".repeat(32));
}));

test("an interrupted owner creation has a bounded acquisition lease", () => withRoot((root) => {
  const lock = lockPath(root);
  const modified = Date.UTC(2026, 0, 1);
  fs.mkdirSync(lock);
  fs.writeFileSync(path.join(lock, `.owner-${"1".repeat(32)}.new`), "{\"partial\":");
  fs.utimesSync(lock, new Date(modified), new Date(modified));
  assert.throws(() => withCompositionRepositoryLock(
    root, () => {}, lockOptions(() => modified + 1, ["2".repeat(32)]),
  ), /acquisition lease is incomplete/);
  withCompositionRepositoryLock(root, () => {}, lockOptions(
    () => modified + COMPOSITION_REPOSITORY_LOCK_LEASE_MS,
    ["3".repeat(32), "4".repeat(32)],
  ));
  assert.equal(fs.existsSync(lock), false);
}));

test("a complete owner draft retains its live process fence before publication", () => withRoot((root) => {
  const lock = lockPath(root);
  const acquired = Date.UTC(2026, 0, 1);
  const token = "a".repeat(32);
  const fence = acquireProcessFence(token);
  const draftOwner = createLockOwner({
    token, acquiredAt: null, now: acquired, pid: 45, hostname: "draft-host",
    fencePort: fence.port, leaseMs: COMPOSITION_REPOSITORY_LOCK_LEASE_MS,
  });
  fs.mkdirSync(lock);
  fs.writeFileSync(path.join(lock, `.owner-${token}.new`), `${JSON.stringify(draftOwner)}\n`);
  fs.utimesSync(lock, new Date(acquired), new Date(acquired));
  try {
    assert.throws(() => withCompositionRepositoryLock(
      root, () => assert.fail("a successor ran while the draft process fence was live"),
      lockOptions(() => acquired + COMPOSITION_REPOSITORY_LOCK_LEASE_MS, ["b".repeat(32)]),
    ), /retains the repository process fence/);
  } finally { releaseProcessFence(fence); }
  assert.throws(() => withCompositionRepositoryLock(
    root, () => assert.fail("a complete owner draft was automatically displaced"),
    lockOptions(() => acquired + COMPOSITION_REPOSITORY_LOCK_LEASE_MS, ["c".repeat(32)]),
  ), /complete owner draft; automatic recovery is disabled/);
}));

test("incomplete recovery restores quarantined state after replacement or initialization failure", () => {
  for (const mode of ["replacement", "initialization"]) withRoot((root) => {
    const lock = lockPath(root);
    const modified = Date.UTC(2026, 0, 1);
    const draft = `.owner-${"5".repeat(32)}.new`;
    fs.mkdirSync(lock);
    fs.writeFileSync(path.join(lock, draft), "partial");
    fs.utimesSync(lock, new Date(modified), new Date(modified));
    const tokens = mode === "replacement" ? ["6".repeat(32)] : ["7".repeat(32)];
    assert.throws(() => withCompositionRepositoryLock(root, () => {}, {
      ...lockOptions(() => modified + COMPOSITION_REPOSITORY_LOCK_LEASE_MS, tokens),
      afterQuarantine(quarantine) {
        if (mode === "replacement") fs.appendFileSync(path.join(quarantine, draft), " changed");
      },
    }), mode === "replacement" ? /incomplete lock changed/ : /token source returned an invalid token/);
    assert.equal(fs.existsSync(lock), true);
    assert.match(fs.readFileSync(path.join(lock, draft), "utf8"), /^partial/);
  });
});

function withRoot(run) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-composition-lock-"));
  try { return run(root); } finally { fs.rmSync(root, { recursive: true, force: true }); }
}
function lockOptions(clock, tokens, pid = 91) { return { clock, pid, hostname: "test-host", randomToken: () => tokens.shift() }; }
function lockPath(root) { return path.join(fs.realpathSync(root), ".runmat-module-composition.lock"); }
function owner(root) { return readLockOwner(lockPath(root), COMPOSITION_REPOSITORY_LOCK_LEASE_MS); }
function createOwner(character, now, pid, hostname) { return createLockOwner({ token: character.repeat(32), acquiredAt: null, now, pid, hostname, fencePort: 32_000, leaseMs: COMPOSITION_REPOSITORY_LOCK_LEASE_MS }); }
function strand(root, value) { const lock = lockPath(root); fs.mkdirSync(lock); fs.writeFileSync(path.join(lock, "owner.json"), `${JSON.stringify(value)}\n`); }
