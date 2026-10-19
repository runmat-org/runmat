import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { syncDirectory } from "./durability.mjs";
import {
  acquireProcessFence, assertProcessFence, processFenceIsActive,
  releaseProcessFence,
} from "./transaction-lock-fence.mjs";
import {
  createLockOwner, inspectIncompleteLockDirectory, ownerEquals, readLockOwner,
  removeIncompleteLockDirectory, removeOwnedLockDirectory, replaceLockOwner,
} from "./transaction-lock-owner.mjs";
import { assertRecoveryAdmission } from "./transaction-recovery-reservation.mjs";

const LOCK = ".runmat-module-composition.lock";
const ACTIVE_LOCKS = new WeakMap();
export const COMPOSITION_REPOSITORY_LOCK_LEASE_MS = 10 * 60 * 1_000;

export function withCompositionRepositoryLock(repository, callback, options = {}) {
  const root = fs.realpathSync(path.resolve(repository));
  const settings = lockSettings(options);
  const context = claim(path.join(root, LOCK), root, settings);
  const handle = Object.freeze({ repository: root, token: context.owner.token });
  ACTIVE_LOCKS.set(handle, context);
  let result;
  let failure = null;
  try { result = callback(handle); } catch (error) { failure = error; }
  try { release(handle); } catch (error) {
    ACTIVE_LOCKS.delete(handle);
    if (failure) throw new AggregateError([failure, error], "composition operation and lock release failed");
    throw error;
  }
  ACTIVE_LOCKS.delete(handle);
  if (failure) throw failure;
  return result;
}

export function renewCompositionRepositoryLock(handle) {
  const context = activeContext(handle);
  const now = context.settings.now();
  assertAuthority(context, now);
  const owner = createLockOwner({
    token: context.owner.token, acquiredAt: context.owner.acquired_at,
    now, pid: context.owner.pid, hostname: context.owner.hostname,
    fencePort: context.owner.fence_port,
    leaseMs: COMPOSITION_REPOSITORY_LOCK_LEASE_MS,
  });
  replaceLockOwner(context.lock, owner, { expected: context.owner });
  context.owner = owner;
  assertAuthority(context, now);
  return Object.freeze(structuredClone(owner));
}

export function assertCompositionRepositoryLock(handle, repository) {
  const context = activeContext(handle);
  const root = fs.realpathSync(path.resolve(repository));
  if (context.root !== root) throw new Error("module composition install requires its active repository lock");
  assertAuthority(context, context.settings.now());
  return root;
}

function claim(lock, root, settings) {
  assertRecoveryAdmission(root, settings.recoveryToken);
  settings.afterFirstAdmission?.();
  try {
    fs.mkdirSync(lock, { mode: 0o700 });
  } catch (error) {
    if (error?.code !== "EEXIST") throw error;
    return claimExisting(lock, root, settings);
  }
  const identity = directoryIdentity(lock);
  try { assertRecoveryAdmission(root, settings.recoveryToken); }
  catch (error) { rollbackEmptyClaim(lock, identity, root, error); }
  try { return initializeClaim(lock, root, settings); }
  catch (error) { rollbackEmptyClaim(lock, identity, root, error); }
}

function claimExisting(lock, root, settings) {
  const incomplete = inspectIncompleteLockDirectory(lock);
  if (incomplete !== null) return reclaimIncomplete(lock, root, incomplete, settings);
  const observed = readLockOwner(lock, COMPOSITION_REPOSITORY_LOCK_LEASE_MS);
  if (processFenceIsActive(observed.fence_port, observed.token)) {
    throw new Error("another module composition materializer retains the repository process fence");
  }
  if (settings.now() < Date.parse(observed.expires_at)) {
    throw new Error("another module composition materializer holds the repository lock lease");
  }
  throw new Error("expired module composition lock has a complete owner; automatic recovery is disabled because the owning process may resume");
}

function initializeClaim(lock, root, settings) {
  const now = settings.now();
  const token = settings.randomToken();
  const fence = acquireProcessFence(token);
  let owner;
  try {
    owner = createLockOwner({
      token, acquiredAt: null, now,
      pid: settings.pid, hostname: settings.hostname,
      fencePort: fence.port,
      leaseMs: COMPOSITION_REPOSITORY_LOCK_LEASE_MS,
    });
    replaceLockOwner(lock, owner, { initial: true });
    syncDirectory(root);
  }
  catch (error) { releaseProcessFence(fence); throw error; }
  return { lock, root, owner, fence, directoryIdentity: directoryIdentity(lock), settings };
}

function reclaimIncomplete(lock, root, observed, settings) {
  const draftOwner = observed.artifact?.owner;
  if (draftOwner && processFenceIsActive(draftOwner.fence_port, draftOwner.token)) {
    throw new Error("another module composition materializer retains the repository process fence");
  }
  if (draftOwner) {
    throw new Error("interrupted module composition lock has a complete owner draft; automatic recovery is disabled because the owning process may resume");
  }
  if (settings.now() < observed.modified_at + COMPOSITION_REPOSITORY_LOCK_LEASE_MS) {
    throw new Error("module composition repository lock acquisition lease is incomplete");
  }
  const quarantine = `${lock}.expired-incomplete-${settings.randomToken().slice(0, 16)}`;
  fs.renameSync(lock, quarantine);
  syncDirectory(root);
  validateQuarantine(lock, quarantine, root, settings, () => {
    const quarantined = inspectIncompleteLockDirectory(quarantine);
    if (JSON.stringify(observed) !== JSON.stringify(quarantined)) throw new Error("module composition incomplete lock changed during recovery");
  });
  try { fs.mkdirSync(lock, { mode: 0o700 }); }
  catch (error) { if (error?.code === "EEXIST") throw new Error("another materializer acquired the repository lock during incomplete recovery"); throw error; }
  let context;
  try { context = initializeClaim(lock, root, settings); }
  catch (error) { recoverFailedReplacement(lock, quarantine, root, error); }
  try { removeIncompleteLockDirectory(quarantine, observed); }
  catch (error) { recoverFailedReplacement(lock, quarantine, root, error, context); }
  syncDirectory(root);
  return context;
}

function release(handle) {
  const context = activeContext(handle);
  let failure = null;
  try { assertAuthority(context, null); removeOwnedLockDirectory(context.lock, context.owner); syncDirectory(context.root); }
  catch (error) { failure = error; }
  try { releaseProcessFence(context.fence); }
  catch (error) { if (failure) throw new AggregateError([failure, error], "composition lock release and process-fence release failed"); throw error; }
  if (failure) throw failure;
}

function assertAuthority(context, now) {
  assertProcessFence(context.fence);
  if (!sameIdentity(context.directoryIdentity, directoryIdentity(context.lock))) {
    throw new Error("module composition repository lock directory was replaced");
  }
  const owner = readLockOwner(context.lock, COMPOSITION_REPOSITORY_LOCK_LEASE_MS);
  if (!ownerEquals(owner, context.owner)) throw new Error("module composition repository lock owner changed");
  if (now !== null && now < Date.parse(owner.renewed_at)) throw new Error("module composition repository lock clock moved backwards");
}

function activeContext(handle) {
  const context = ACTIVE_LOCKS.get(handle);
  if (!context) throw new Error("module composition operation requires its active repository lock");
  return context;
}

function lockSettings(options) {
  const clock = options.clock ?? Date.now;
  const randomToken = options.randomToken ?? (() => crypto.randomBytes(16).toString("hex"));
  const settings = {
    now() { const value = clock(); if (!Number.isSafeInteger(value) || value < 0) throw new Error("composition lock clock must return a nonnegative integer timestamp"); return value; },
    randomToken() { const value = randomToken(); if (!/^[a-f0-9]{32}$/.test(value)) throw new Error("composition lock token source returned an invalid token"); return value; },
    pid: options.pid ?? process.pid, hostname: options.hostname ?? os.hostname(),
    afterQuarantine: options.afterQuarantine,
    afterFirstAdmission: options.afterFirstAdmission,
    recoveryToken: options.recoveryToken ?? null,
  };
  if (!Number.isSafeInteger(settings.pid) || settings.pid <= 0 || typeof settings.hostname !== "string" || !settings.hostname) throw new Error("composition lock diagnostics are invalid");
  return settings;
}

function restoreQuarantine(lock, quarantine, root) { if (!fs.existsSync(lock)) { fs.renameSync(quarantine, lock); syncDirectory(root); } }
function validateQuarantine(lock, quarantine, root, settings, validate) { try { settings.afterQuarantine?.(quarantine); validate(); } catch (error) { restoreQuarantine(lock, quarantine, root); throw error; } }
function recoverFailedReplacement(lock, quarantine, root, error, context = null) { try { removeFailedClaim(lock); if (context) releaseProcessFence(context.fence); syncDirectory(root); restoreQuarantine(lock, quarantine, root); } catch (cleanupError) { throw new AggregateError([error, cleanupError], "module composition lock replacement and recovery failed"); } throw error; }
function removeFailedClaim(lock) { const incomplete = inspectIncompleteLockDirectory(lock); if (incomplete) removeIncompleteLockDirectory(lock, incomplete); else removeOwnedLockDirectory(lock, readLockOwner(lock, COMPOSITION_REPOSITORY_LOCK_LEASE_MS)); }
function rollbackEmptyClaim(lock, identity, root, error) { try { removeNewEmptyLock(lock, identity, root); } catch (cleanupError) { throw new AggregateError([error, cleanupError], "composition lock acquisition and rollback failed"); } throw error; }
function removeNewEmptyLock(lock, expectedIdentity, root) { if (!sameIdentity(directoryIdentity(lock), expectedIdentity) || fs.readdirSync(lock).length) return; fs.rmdirSync(lock); syncDirectory(root); }
function directoryIdentity(target) { const stat = fs.statSync(target); return { device: String(stat.dev), inode: String(stat.ino) }; }
function sameIdentity(left, right) { return left.device === right.device && left.inode === right.inode; }
