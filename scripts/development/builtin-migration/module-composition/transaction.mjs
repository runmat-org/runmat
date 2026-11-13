import crypto from "node:crypto";
import fs from "node:fs";

import { companionDefinitions, companionSourcesForParent, parseDerivedCompanionProduct } from "./companions.mjs";
import { renderModuleCompositionCompanion, renderModuleCompositionProduct } from "./generate.mjs";
import { resolveRepositoryProduct } from "./repository-state.mjs";
import { journalPath, markTransactionCommitted, recoverTransactionJournal, removeTransactionJournal, writeTransactionJournal } from "./transaction-journal.mjs";
import { cleanupCommitted, installEntry, stageEntry } from "./transaction-install.mjs";
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
  const twice = products.map((product) => {
    const companion = Object.hasOwn(product, "companion_parent")
      ? parseDerivedCompanionProduct(product) : null;
    const first = companion
      ? renderModuleCompositionCompanion(companion.companion_parent, companion.companion_section)
      : render(product);
    const second = companion
      ? renderModuleCompositionCompanion(companion.companion_parent, companion.companion_section)
      : render(product);
    if (first !== second) throw new Error(`${product.product_id}: module composition rendering is nondeterministic`);
    return { product, first, second };
  });
  const firstSet = twice.map(({ product, first }) => ({ product, content: first }));
  const secondSet = twice.map(({ product, second }) => ({ product, content: second }));
  for (const { product, first, second } of twice) {
    const firstCompanions = Object.hasOwn(product, "companion_parent")
      ? new Map() : companionSourcesForParent(product, firstSet);
    const secondCompanions = Object.hasOwn(product, "companion_parent")
      ? new Map() : companionSourcesForParent(product, secondSet);
    verifyModuleCompositionProduct(product, first, firstCompanions);
    verifyModuleCompositionProduct(product, second, secondCompanions);
  }
  return firstSet;
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
  const selectedParents = new Map(rendered.filter((entry) =>
    !Object.hasOwn(entry.product, "companion_parent"))
    .map((entry) => [entry.product.product_id, entry.product]));
  for (const entry of rendered.filter((item) => Object.hasOwn(item.product, "companion_parent"))) {
    const companion = parseDerivedCompanionProduct(entry.product);
    const parent = selectedParents.get(companion.companion_parent.product_id);
    if (!parent || JSON.stringify(parent) !== JSON.stringify(companion.companion_parent)) {
      throw new Error(`${companion.product_id}: companion is not bound to its selected parent`);
    }
  }
  for (const parent of selectedParents.values()) for (const definition of companionDefinitions()
    .filter((item) => item.parent_id === parent.product_id)) {
    const target = resolveRepositoryProduct(root, definition.path);
    let exists;
    try { exists = fs.lstatSync(target) !== null; }
    catch (error) { if (error?.code === "ENOENT") exists = false; else throw error; }
    if (exists && !rendered.some((entry) => entry.product.product_id
      === `${definition.parent_id}-${definition.section}`)) {
      throw new Error(`${parent.product_id}: prior companion must be included in the transition`);
    }
  }
  const token = `${process.pid}-${crypto.randomBytes(12).toString("hex")}`;
  const planned = rendered.map((entry) => planTransactionEntry(
    root, token, entry, before.get(entry.product.product_id),
  ));
  const companionSourcesByParent = new Map(rendered.filter((entry) =>
    !Object.hasOwn(entry.product, "companion_parent") && entry.product.state === "present")
    .map((entry) => [entry.product.product_id, companionSourcesForParent(entry.product, rendered)]));
  const journal = {
    schema_version: 3,
    kind: "runmat-module-composition-transaction",
    phase: "installing",
    registry_digest: transactionRegistryDigest(),
    token,
    entries: planned.map((entry) => transactionJournalEntry(entry, planned)),
  };
  writeTransactionJournal(root, journal);
  const staged = [];
  try {
    hooks.afterJournal?.({ journal });
    renewCompositionRepositoryLock(lock);
    for (const entry of planned) {
      renewCompositionRepositoryLock(lock);
      staged.push(stageEntry(root, entry, hooks, lock, companionSourcesByParent));
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

function recoverAfterFailure(root, error, message, lock, token) {
  try { renewCompositionRepositoryLock(lock); recoverTransactionJournal(root, token); }
  catch (recoveryError) { throw new AggregateError([error, recoveryError], message); }
  throw error;
}
