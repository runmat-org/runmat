import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

import { syncDirectory } from "./durability.mjs";
import { renderModuleCompositionProduct } from "./generate.mjs";
import { assertProductStateUnchanged } from "./repository-state.mjs";
import { journalPath, markTransactionCommitted, recoverTransactionJournal, removeTransactionJournal, writeTransactionJournal } from "./transaction-journal.mjs";
import {
  assertCompositionRepositoryLock, renewCompositionRepositoryLock,
  withCompositionRepositoryLock,
} from "./transaction-lock.mjs";
import { withStoppedOwnerRecovery } from "./transaction-lock-recovery.mjs";
import {
  planTransactionEntry,
  transactionJournalEntry,
  transactionRegistryDigest,
} from "./transaction-plan.mjs";
import { verifyModuleCompositionProduct } from "./verify.mjs";

export function withCompositionTransaction(repository, callback) {
  return withCompositionRepositoryLock(repository, (lock) => {
    const recovery = recoverTransactionJournal(lock.repository);
    renewCompositionRepositoryLock(lock);
    return callback(lock, recovery);
  });
}

export function recoverCompositionAfterStoppedOwner(repository, options) {
  return withStoppedOwnerRecovery(repository, options, (lock) => {
    const recovery = recoverTransactionJournal(lock.repository);
    renewCompositionRepositoryLock(lock);
    return recovery;
  });
}

export function renderCompositionSetTwice(products, render = renderModuleCompositionProduct) {
  return products.map((product) => {
    const first = render(product);
    const second = render(product);
    if (first !== second) throw new Error(`${product.product_id}: module composition rendering is nondeterministic`);
    verifyModuleCompositionProduct(product, first);
    verifyModuleCompositionProduct(product, second);
    return { product, content: first };
  });
}

export function installCompositionSet(
  repository, rendered, beforeStates, lock, hooks = {}, authorityGuard = null,
) {
  const root = assertCompositionRepositoryLock(lock, repository);
  renewCompositionRepositoryLock(lock);
  if (fs.existsSync(journalPath(root))) throw new Error("unrecovered module composition transaction journal remains");
  if (!rendered.length) return { installed: [], cleanup: "complete", cleanup_errors: [] };
  const before = new Map(beforeStates.map((entry) => [entry.product_id, entry]));
  if (before.size !== beforeStates.length || before.size !== rendered.length
    || rendered.some((entry) => !before.has(entry.product.product_id))) {
    throw new Error("composition before-state does not exactly cover rendered products");
  }
  const token = `${process.pid}-${crypto.randomBytes(12).toString("hex")}`;
  const planned = rendered.map((entry) => planTransactionEntry(
    root, token, entry, before.get(entry.product.product_id),
  ));
  const journal = {
    schema_version: 2,
    kind: "runmat-module-composition-transaction",
    phase: "installing",
    registry_digest: transactionRegistryDigest(),
    token,
    entries: planned.map(transactionJournalEntry),
  };
  writeTransactionJournal(root, journal);
  const staged = [];
  try {
    hooks.afterJournal?.({ journal });
    renewCompositionRepositoryLock(lock);
    for (const entry of planned) {
      renewCompositionRepositoryLock(lock);
      staged.push(stageEntry(root, entry, hooks, lock));
    }
    renewCompositionRepositoryLock(lock);
    if (authorityGuard !== null) {
      if (typeof authorityGuard !== "function") throw new Error("composition authority guard must be a function");
      const transactionArtifacts = Object.freeze([
        journalPath(root), ...staged.filter((entry) => entry.new_digest !== null)
          .map((entry) => entry.temporary),
      ]);
      authorityGuard(Object.freeze({ repository: root, transactionArtifacts }));
    }
  } catch (error) {
    recoverAfterFailure(root, error, "module composition staging and recovery failed", lock, token);
  }
  try {
    for (let index = 0; index < staged.length; index += 1) {
      renewCompositionRepositoryLock(lock);
      installEntry(root, staged[index], index, hooks, lock);
    }
    renewCompositionRepositoryLock(lock);
    markTransactionCommitted(root, token);
  } catch (error) {
    let recoveryError = null;
    try { renewCompositionRepositoryLock(lock); recoverTransactionJournal(root, token); }
    catch (observed) { recoveryError = observed; }
    if (recoveryError) throw new AggregateError([error, recoveryError], "module composition install and recovery failed");
    throw error;
  }
  renewCompositionRepositoryLock(lock);
  const cleanupErrors = cleanupCommitted(staged, hooks, lock);
  if (!cleanupErrors.length) {
    renewCompositionRepositoryLock(lock);
    removeTransactionJournal(root, token);
  }
  return { installed: staged.map((entry) => ({ product_id: entry.product.product_id, path: entry.product.path, state: entry.desired_state })), cleanup: cleanupErrors.length ? "recovery-required" : "complete", cleanup_errors: cleanupErrors.map((error) => error instanceof Error ? error.message : String(error)) };
}

function stageEntry(root, entry, hooks, lock) {
  assertProductStateUnchanged(root, entry.before);
  if (entry.desired_state === "absent") return entry;
  writeExclusive(entry.temporary, entry.content);
  hooks.afterStage?.(entry);
  renewCompositionRepositoryLock(lock);
  const observed = readRegularNoFollow(entry.temporary);
  if (observed.contents !== entry.content) throw new Error(`${entry.product.product_id}: staged composition bytes changed`);
  verifyModuleCompositionProduct(entry.product, observed.contents);
  return entry;
}

function installEntry(root, entry, index, hooks, lock) {
  hooks.beforeInstall?.({ ...entry, index });
  renewCompositionRepositoryLock(lock);
  assertProductStateUnchanged(root, entry.before);
  if (entry.desired_state === "absent") {
    if (entry.before.state !== "present") {
      throw new Error(`${entry.product.product_id}: absent target cannot be deleted`);
    }
    fs.renameSync(entry.target, entry.backup);
    fsyncDirectory(path.dirname(entry.target));
    hooks.afterBackupDurable?.({ ...entry, index });
    renewCompositionRepositoryLock(lock);
    assertBackup(entry);
  } else if (entry.before.state === "absent") {
    fs.linkSync(entry.temporary, entry.target);
    fs.unlinkSync(entry.temporary);
  } else {
    fs.renameSync(entry.target, entry.backup);
    fsyncDirectory(path.dirname(entry.target));
    try {
      hooks.afterBackupDurable?.({ ...entry, index });
      renewCompositionRepositoryLock(lock);
      assertBackup(entry);
      fs.renameSync(entry.temporary, entry.target);
    }
    catch (error) { if (!fs.existsSync(entry.target) && fs.existsSync(entry.backup)) fs.renameSync(entry.backup, entry.target); throw error; }
  }
  fsyncDirectory(path.dirname(entry.target));
}

function cleanupCommitted(entries, hooks, lock) {
  const errors = [];
  for (let index = 0; index < entries.length; index += 1) {
    const entry = entries[index];
    try {
      hooks.beforeCleanup?.({ ...entry, index });
      renewCompositionRepositoryLock(lock);
      unlinkIfPresent(entry.temporary);
      unlinkIfPresent(entry.backup);
      fsyncDirectory(path.dirname(entry.target));
    }
    catch (error) { errors.push(error); }
  }
  return errors;
}

function recoverAfterFailure(root, error, message, lock, token) {
  try { renewCompositionRepositoryLock(lock); recoverTransactionJournal(root, token); }
  catch (recoveryError) { throw new AggregateError([error, recoveryError], message); }
  throw error;
}

function assertBackup(entry) { const observed = readRegularNoFollow(entry.backup); if (digest(observed.contents) !== entry.before.content_digest || String(observed.stat.dev) !== entry.before.file_identity.device || String(observed.stat.ino) !== entry.before.file_identity.inode) throw new Error(`${entry.product.product_id}: composition target changed during install`); }
function writeExclusive(target, contents) { const descriptor = fs.openSync(target, fs.constants.O_CREAT | fs.constants.O_EXCL | fs.constants.O_WRONLY | noFollow(), 0o600); try { fs.writeFileSync(descriptor, contents); fs.fsyncSync(descriptor); } finally { fs.closeSync(descriptor); } fsyncDirectory(path.dirname(target)); }
function readRegularNoFollow(target) { const prior = fs.lstatSync(target); if (!prior.isFile()) throw new Error(`${target} is not a regular staged file`); const descriptor = fs.openSync(target, fs.constants.O_RDONLY | noFollow()); try { const stat = fs.fstatSync(descriptor); if (!stat.isFile() || stat.dev !== prior.dev || stat.ino !== prior.ino) throw new Error(`${target} changed while opening the staged file`); return { contents: fs.readFileSync(descriptor, "utf8"), stat }; } finally { fs.closeSync(descriptor); } }
function digest(value) { return `sha256:${crypto.createHash("sha256").update(value).digest("hex")}`; }
function unlinkIfPresent(target) { try { fs.unlinkSync(target); } catch (error) { if (error?.code !== "ENOENT") throw error; } }
function fsyncDirectory(directory) { syncDirectory(directory); }
function noFollow() { return fs.constants.O_NOFOLLOW ?? 0; }
