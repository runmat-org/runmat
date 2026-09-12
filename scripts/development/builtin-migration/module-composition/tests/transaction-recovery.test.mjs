import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { evidenceDigest } from "../../evidence.mjs";
import { moduleCompositionProductRegistry } from "../registry.mjs";
import { inspectMaterializationTargets } from "../repository-state.mjs";
import {
  installCompositionSet, recoverCompositionAfterStoppedOwner, renderCompositionSetTwice,
  withCompositionTransaction,
} from "../transaction.mjs";
import { writeTransactionJournal } from "../transaction-journal.mjs";
import {
  COMPOSITION_REPOSITORY_LOCK_LEASE_MS, withCompositionRepositoryLock,
} from "../transaction-lock.mjs";
import { createLockOwner, readLockOwner, replaceLockOwner } from "../transaction-lock-owner.mjs";
import {
  clearStoppedOwnerRecoveryReservation, withStoppedOwnerRecovery,
} from "../transaction-lock-recovery.mjs";
import { claimRecoveryReservation } from "../transaction-recovery-reservation.mjs";

test("installation persists its complete journal before creating stage artifacts", () => withRepository((root, product) => {
  withCompositionTransaction(root, (lock) => {
    const before = inspectMaterializationTargets(root, [product]);
    const rendered = renderCompositionSetTwice([product]);
    assert.throws(() => installCompositionSet(root, rendered, before, lock, {
      afterJournal({ journal }) {
        assert.equal(fs.existsSync(path.join(root, JOURNAL)), true);
        const target = path.join(root, product.path);
        assert.equal(fs.existsSync(`${target}.runmat-stage-${journal.token}`), false);
        throw new Error("stop after durable journal");
      },
    }), /stop after durable journal/);
    assert.equal(fs.existsSync(path.join(root, JOURNAL)), false);
    assert.equal(fs.existsSync(path.join(root, product.path)), false);
  });
}));

test("authority guard receives only the transaction-owned immutable artifact set", () => withRepository((root, product) => {
  withCompositionTransaction(root, (lock) => {
    const before = inspectMaterializationTargets(root, [product]);
    const rendered = renderCompositionSetTwice([product]);
    let observed = null;
    installCompositionSet(root, rendered, before, lock, {}, (authority) => {
      observed = authority;
      assert.equal(Object.isFrozen(authority), true);
      assert.equal(Object.isFrozen(authority.transactionArtifacts), true);
      assert.throws(() => authority.transactionArtifacts.push("unreviewed"), /not extensible/);
    });
    const canonicalRoot = fs.realpathSync(root);
    const target = path.join(canonicalRoot, product.path);
    assert.equal(observed.transactionArtifacts.length, 2);
    assert.equal(observed.transactionArtifacts[0], path.join(canonicalRoot, JOURNAL));
    assert.match(observed.transactionArtifacts[1], new RegExp(`^${escapeRegex(target)}\\.runmat-stage-[0-9]+-[a-f0-9]{24}$`));
  });
}));

test("an existing target has a recoverable durable backup before replacement publication", () => withRepository((root, product) => {
  const target = path.join(root, product.path);
  const original = "pub mod alpha;\n";
  write(root, product.path, original);
  withCompositionTransaction(root, (lock) => {
    const before = inspectMaterializationTargets(root, [product]);
    const rendered = renderCompositionSetTwice([product]);
    assert.throws(() => installCompositionSet(root, rendered, before, lock, {
      afterBackupDurable({ backup }) {
        assert.equal(fs.existsSync(target), false);
        assert.equal(fs.readFileSync(backup, "utf8"), original);
        throw new Error("stop after durable backup");
      },
    }), /stop after durable backup/);
    assert.equal(fs.readFileSync(target, "utf8"), original);
    assert.equal(fs.readdirSync(path.dirname(target)).some((name) => name.includes(".runmat-backup-")), false);
  });
}));

test("metadata expiry during installation cannot admit a successor past the process fence", () => withRepository((root, product) => {
  const target = path.join(root, product.path);
  write(root, product.path, "pub mod original;\n");
  let now = Date.UTC(2026, 0, 1);
  withCompositionRepositoryLock(root, (lock) => {
    const before = inspectMaterializationTargets(root, [product]);
    const rendered = renderCompositionSetTwice([product]);
    installCompositionSet(root, rendered, before, lock, {
      beforeInstall() {
        const journal = fs.readFileSync(path.join(root, JOURNAL));
        const original = fs.readFileSync(target);
        now += COMPOSITION_REPOSITORY_LOCK_LEASE_MS;
        assert.throws(() => withCompositionRepositoryLock(
          root, () => assert.fail("a successor entered during installation"),
          lockOptions(() => now, ["2".repeat(32)]),
        ), /retains the repository process fence/);
        assert.deepEqual(fs.readFileSync(path.join(root, JOURNAL)), journal);
        assert.deepEqual(fs.readFileSync(target), original);
      },
    });
  }, lockOptions(() => now, ["1".repeat(32)]));
  assert.match(fs.readFileSync(target, "utf8"), /alpha/);
}));

test("metadata expiry during cleanup cannot expose the journal or target to a successor", () => withRepository((root, product) => {
  const target = path.join(root, product.path);
  write(root, product.path, "pub mod original;\n");
  let now = Date.UTC(2026, 0, 1);
  withCompositionRepositoryLock(root, (lock) => {
    const before = inspectMaterializationTargets(root, [product]);
    const rendered = renderCompositionSetTwice([product]);
    installCompositionSet(root, rendered, before, lock, {
      beforeCleanup() {
        const journal = fs.readFileSync(path.join(root, JOURNAL));
        const installed = fs.readFileSync(target);
        now += COMPOSITION_REPOSITORY_LOCK_LEASE_MS;
        assert.throws(() => withCompositionRepositoryLock(
          root, () => assert.fail("a successor entered during committed cleanup"),
          lockOptions(() => now, ["4".repeat(32)]),
        ), /retains the repository process fence/);
        assert.deepEqual(fs.readFileSync(path.join(root, JOURNAL)), journal);
        assert.deepEqual(fs.readFileSync(target), installed);
      },
    });
  }, lockOptions(() => now, ["3".repeat(32)]));
  assert.equal(fs.existsSync(path.join(root, JOURNAL)), false);
  assert.match(fs.readFileSync(target, "utf8"), /alpha/);
}));

test("recovery removes partial staging created after a durable journal", () => withRepository((root) => {
  const relative = "crates/runmat-runtime/src/builtins/math/mod.rs";
  const target = path.join(root, relative);
  const original = "// original\n";
  const replacement = "// replacement\n";
  write(root, relative, original);
  const stat = fs.statSync(target);
  const token = `999-${"e".repeat(24)}`;
  writeTransactionJournal(root, transactionJournal(token, relative, true, original, replacement, stat));
  const stage = `${target}.runmat-stage-${token}`;
  fs.writeFileSync(stage, "// partial");
  withCompositionTransaction(root, (_lock, recovery) => {
    assert.deepEqual(recovery, { recovered: true, phase: "installing" });
    assert.equal(fs.readFileSync(target, "utf8"), original);
    assert.equal(fs.existsSync(stage), false);
  });
}));

test("recovery removes bounded journal drafts without touching unrelated files", () => withRepository((root) => {
  const draft = path.join(root, `${JOURNAL}.new`);
  const legacy = path.join(root, `${JOURNAL}.new-123-0123456789abcdef`);
  const unrelated = path.join(root, `${JOURNAL}.new-untrusted`);
  const external = path.join(root, "external-journal-draft-target");
  fs.writeFileSync(draft, "{\"partial\":");
  fs.writeFileSync(external, "external\n");
  fs.symlinkSync(external, legacy);
  fs.writeFileSync(unrelated, "preserve\n");
  withCompositionTransaction(root, (_lock, recovery) => {
    assert.deepEqual(recovery, { recovered: false });
    assert.equal(fs.existsSync(draft), false);
    assert.equal(fs.existsSync(legacy), false);
    assert.equal(fs.readFileSync(external, "utf8"), "external\n");
    assert.equal(fs.readFileSync(unrelated, "utf8"), "preserve\n");
  });
}));

test("recovery refuses a non-file at the bounded journal draft path", () => withRepository((root) => {
  const draft = path.join(root, `${JOURNAL}.new`);
  fs.mkdirSync(draft);
  assert.throws(() => withCompositionTransaction(root, () => {}), /journal draft is not a regular file or symbolic link/);
  assert.equal(fs.statSync(draft).isDirectory(), true);
}));

test("explicit stopped-owner recovery rolls back a journal while holding a replacement fence", () => withRepository((root) => {
  const relative = "crates/runmat-runtime/src/builtins/math/mod.rs";
  const target = path.join(root, relative);
  const original = "// original\n";
  const replacement = "// replacement\n";
  write(root, relative, original);
  const transactionToken = `999-${"a".repeat(24)}`;
  const stat = fs.statSync(target);
  writeTransactionJournal(root, transactionJournal(transactionToken, relative, true, original, replacement, stat));
  fs.renameSync(target, `${target}.runmat-backup-${transactionToken}`);
  fs.writeFileSync(target, replacement);
  const acquired = Date.UTC(2026, 0, 1);
  const ownerToken = "b".repeat(32);
  strandOwner(root, ownerToken, acquired);
  let reservationChecked = false;
  let quarantineChecked = false;
  const result = recoverCompositionAfterStoppedOwner(root, {
    token: ownerToken, ownerCannotResume: true,
    clock: () => acquired + COMPOSITION_REPOSITORY_LOCK_LEASE_MS,
    afterReservation() {
      assert.throws(() => withCompositionRepositoryLock(root, () => {
        assert.fail("a contender entered before stale-owner quarantine");
      }), /stopped-owner recovery is in progress/);
      reservationChecked = true;
    },
    afterQuarantine() {
      assert.throws(() => withCompositionRepositoryLock(root, () => {
        assert.fail("a contender entered before successor publication");
      }), /stopped-owner recovery is in progress/);
      quarantineChecked = true;
    },
  });
  assert.equal(reservationChecked, true);
  assert.equal(quarantineChecked, true);
  assert.deepEqual(result, { recovered: true, phase: "installing" });
  assert.equal(fs.readFileSync(target, "utf8"), original);
  assert.equal(fs.existsSync(path.join(root, JOURNAL)), false);
  assert.equal(fs.existsSync(lockPath(root)), false);
}));

test("stopped-owner recovery requires external fencing proof and exact stale authority", () => {
  for (const mode of ["confirmation", "token", "unexpired", "malformed"]) withRepository((root) => {
    const acquired = Date.UTC(2026, 0, 1);
    const token = "c".repeat(32);
    strandOwner(root, token, acquired);
    if (mode === "malformed") fs.writeFileSync(path.join(lockPath(root), "owner.json"), "{}\n");
    const options = {
      token: mode === "token" ? "d".repeat(32) : token,
      ownerCannotResume: mode !== "confirmation",
      clock: () => acquired + (mode === "unexpired" ? 1 : COMPOSITION_REPOSITORY_LOCK_LEASE_MS),
    };
    assert.throws(() => recoverCompositionAfterStoppedOwner(root, options), {
      confirmation: /explicit assertion/, token: /token differs/,
      unexpired: /expired owner lease/, malformed: /owner is invalid/,
    }[mode]);
    assert.equal(fs.existsSync(lockPath(root)), true);
  });
});

test("stopped-owner recovery cannot displace an expired owner with a live process fence", () => withRepository((root) => {
  let now = Date.UTC(2026, 0, 1);
  withCompositionRepositoryLock(root, (lock) => {
    now += COMPOSITION_REPOSITORY_LOCK_LEASE_MS;
    assert.throws(() => recoverCompositionAfterStoppedOwner(root, {
      token: lock.token, ownerCannotResume: true, clock: () => now,
    }), /refuses a live repository process fence/);
  }, lockOptions(() => now, ["e".repeat(32)]));
}));

test("stopped-owner recovery restores an owner changed during quarantine", () => withRepository((root) => {
  const acquired = Date.UTC(2026, 0, 1);
  const token = "f".repeat(32);
  strandOwner(root, token, acquired);
  assert.throws(() => recoverCompositionAfterStoppedOwner(root, {
    token, ownerCannotResume: true,
    clock: () => acquired + COMPOSITION_REPOSITORY_LOCK_LEASE_MS,
    afterQuarantine(quarantine) {
      const observed = readLockOwner(quarantine, COMPOSITION_REPOSITORY_LOCK_LEASE_MS);
      const changed = createLockOwner({
        token, acquiredAt: observed.acquired_at,
        now: acquired + COMPOSITION_REPOSITORY_LOCK_LEASE_MS,
        pid: observed.pid, hostname: observed.hostname, fencePort: observed.fence_port,
        leaseMs: COMPOSITION_REPOSITORY_LOCK_LEASE_MS,
      });
      replaceLockOwner(quarantine, changed, { expected: observed });
    },
  }), /owner changed during stopped-owner recovery/);
  assert.equal(fs.existsSync(lockPath(root)), true);
}));

test("stopped-owner callback failure restores the exact stale owner and releases its reservation", () => withRepository((root) => {
  const acquired = Date.UTC(2026, 0, 1);
  const token = "1".repeat(32);
  strandOwner(root, token, acquired);
  const before = fs.readFileSync(path.join(lockPath(root), "owner.json"));
  assert.throws(() => withStoppedOwnerRecovery(root, {
    token, ownerCannotResume: true,
    clock: () => acquired + COMPOSITION_REPOSITORY_LOCK_LEASE_MS,
  }, () => { throw new Error("injected recovery callback failure"); }), /injected recovery callback failure/);
  assert.deepEqual(fs.readFileSync(path.join(lockPath(root), "owner.json")), before);
  assert.equal(fs.existsSync(path.join(root, ".runmat-module-composition.recovery.lock")), false);
}));

test("an operator can clear an interrupted recovery reservation only while the exact stale owner remains", () => withRepository((root) => {
  const acquired = Date.UTC(2026, 0, 1);
  const ownerToken = "2".repeat(32);
  strandOwner(root, ownerToken, acquired);
  const reservation = claimRecoveryReservation(fs.realpathSync(root));
  assert.throws(() => clearStoppedOwnerRecoveryReservation(root, {
    ownerToken, reservationToken: "3".repeat(32), ownerCannotResume: true,
    recoveryProcessCannotResume: true,
    clock: () => acquired + COMPOSITION_REPOSITORY_LOCK_LEASE_MS,
  }), /cleanup token differs/);
  const result = clearStoppedOwnerRecoveryReservation(root, {
    ownerToken, reservationToken: reservation.token, ownerCannotResume: true,
    recoveryProcessCannotResume: true,
    clock: () => acquired + COMPOSITION_REPOSITORY_LOCK_LEASE_MS,
  });
  assert.deepEqual(result, { cleared: true, owner_token: ownerToken });
  assert.equal(fs.existsSync(path.join(root, ".runmat-module-composition.recovery.lock")), false);
  assert.equal(readLockOwner(lockPath(root), COMPOSITION_REPOSITORY_LOCK_LEASE_MS).token, ownerToken);
}));

const JOURNAL = ".runmat-module-composition.transaction.json";

function withRepository(run) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-composition-transaction-"));
  const definition = moduleCompositionProductRegistry().find((entry) => entry.product_id === "runtime-math");
  const product = { ...structuredClone(definition), children: [child("alpha", "crates/runmat-runtime/src/builtins/math/alpha/mod.rs")] };
  try {
    write(root, product.children[0].source_path, "pub fn value() {}\n");
    return run(root, product);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
}

function transactionJournal(token, relative, existed, before, replacement, stat) {
  return {
    schema_version: 1, kind: "runmat-module-composition-transaction", phase: "installing",
    registry_digest: evidenceDigest(moduleCompositionProductRegistry()), token, entries: [{
      product_id: "runtime-math", path: relative, existed,
      before_digest: existed ? sha256(before) : null,
      before_file_identity: existed ? { device: String(stat.dev), inode: String(stat.ino) } : null,
      new_digest: sha256(replacement),
    }],
  };
}

function child(module, sourcePath) {
  return { module, source_kind: "directory", source_path: sourcePath, role: "group", visibility: "public", declaration_condition: { kind: "always" }, declaration_order: 0, macro_use: false, reexports: [], aggregation_sources: [] };
}
function write(root, relative, contents) { const target = path.join(root, relative); fs.mkdirSync(path.dirname(target), { recursive: true }); fs.writeFileSync(target, contents); }
function sha256(value) { return `sha256:${crypto.createHash("sha256").update(value).digest("hex")}`; }
function escapeRegex(value) { return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }
function lockOptions(clock, tokens) { return { clock, pid: 91, hostname: "test-host", randomToken: () => tokens.shift() }; }
function lockPath(root) { return path.join(fs.realpathSync(root), ".runmat-module-composition.lock"); }
function strandOwner(root, token, now) {
  const lock = lockPath(root);
  const owner = createLockOwner({
    token, acquiredAt: null, now, pid: 72, hostname: "stopped-host", fencePort: 32_000,
    leaseMs: COMPOSITION_REPOSITORY_LOCK_LEASE_MS,
  });
  fs.mkdirSync(lock);
  fs.writeFileSync(path.join(lock, "owner.json"), `${JSON.stringify(owner)}\n`);
}
