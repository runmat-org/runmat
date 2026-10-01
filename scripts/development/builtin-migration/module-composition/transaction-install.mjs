import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

import { syncDirectory } from "./durability.mjs";
import { assertProductStateUnchanged } from "./repository-state.mjs";
import { renewCompositionRepositoryLock } from "./transaction-lock.mjs";
import { verifyModuleCompositionProduct } from "./verify.mjs";

export function stageEntry(root, entry, hooks, lock, companionSourcesByParent) {
  assertProductStateUnchanged(root, entry.before);
  if (entry.desired_state === "absent") return entry;
  writeExclusive(entry.temporary, entry.content);
  hooks.afterStage?.(entry);
  renewCompositionRepositoryLock(lock);
  const observed = readRegularNoFollow(entry.temporary);
  if (observed.contents !== entry.content) throw new Error(`${entry.product.product_id}: staged composition bytes changed`);
  verifyModuleCompositionProduct(entry.product, observed.contents,
    companionSourcesByParent.get(entry.product.product_id) ?? new Map());
  return entry;
}

export function installEntry(root, entry, index, hooks, lock) {
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
    catch (error) {
      if (!fs.existsSync(entry.target) && fs.existsSync(entry.backup)) {
        fs.renameSync(entry.backup, entry.target);
      }
      throw error;
    }
  }
  fsyncDirectory(path.dirname(entry.target));
}

export function cleanupCommitted(entries, hooks, lock) {
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

function assertBackup(entry) {
  const observed = readRegularNoFollow(entry.backup);
  if (digest(observed.contents) !== entry.before.content_digest
    || String(observed.stat.dev) !== entry.before.file_identity.device
    || String(observed.stat.ino) !== entry.before.file_identity.inode) {
    throw new Error(`${entry.product.product_id}: composition target changed during install`);
  }
}

function writeExclusive(target, contents) {
  const descriptor = fs.openSync(target,
    fs.constants.O_CREAT | fs.constants.O_EXCL | fs.constants.O_WRONLY | noFollow(), 0o600);
  try {
    fs.writeFileSync(descriptor, contents);
    fs.fsyncSync(descriptor);
  } finally { fs.closeSync(descriptor); }
  fsyncDirectory(path.dirname(target));
}

function readRegularNoFollow(target) {
  const prior = fs.lstatSync(target);
  if (!prior.isFile()) throw new Error(`${target} is not a regular staged file`);
  const descriptor = fs.openSync(target, fs.constants.O_RDONLY | noFollow());
  try {
    const stat = fs.fstatSync(descriptor);
    if (!stat.isFile() || stat.dev !== prior.dev || stat.ino !== prior.ino) {
      throw new Error(`${target} changed while opening the staged file`);
    }
    return { contents: fs.readFileSync(descriptor, "utf8"), stat };
  } finally { fs.closeSync(descriptor); }
}

function digest(value) {
  return `sha256:${crypto.createHash("sha256").update(value).digest("hex")}`;
}

function unlinkIfPresent(target) {
  try { fs.unlinkSync(target); }
  catch (error) { if (error?.code !== "ENOENT") throw error; }
}

function fsyncDirectory(directory) { syncDirectory(directory); }
function noFollow() { return fs.constants.O_NOFOLLOW ?? 0; }
