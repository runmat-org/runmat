import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

import { syncDirectory } from "./durability.mjs";
import { processFenceIsActive } from "./transaction-lock-fence.mjs";
import {
  inspectIncompleteLockDirectory, ownerEquals, readLockOwner, removeOwnedLockDirectory,
} from "./transaction-lock-owner.mjs";
import {
  COMPOSITION_REPOSITORY_LOCK_LEASE_MS, withCompositionRepositoryLock,
} from "./transaction-lock.mjs";
import {
  claimRecoveryReservation, clearStoppedRecoveryReservation, releaseRecoveryReservation,
} from "./transaction-recovery-reservation.mjs";

const LOCK = ".runmat-module-composition.lock";

export function withStoppedOwnerRecovery(repository, options, callback) {
  const root = fs.realpathSync(path.resolve(repository));
  const token = options?.token;
  if (!/^[a-f0-9]{32}$/.test(token ?? "")) {
    throw new Error("stopped-owner recovery requires the exact recorded owner token");
  }
  if (options?.ownerCannotResume !== true) {
    throw new Error("stopped-owner recovery requires an explicit assertion that the owning process cannot resume");
  }
  if (typeof callback !== "function") throw new Error("stopped-owner recovery requires a recovery callback");
  const now = (options.clock ?? Date.now)();
  if (!Number.isSafeInteger(now) || now < 0) throw new Error("stopped-owner recovery clock is invalid");
  const reservation = claimRecoveryReservation(root);
  try { return recoverReserved(root, options, callback, token, now, reservation); }
  finally { releaseRecoveryReservation(root, reservation); }
}

export function clearStoppedOwnerRecoveryReservation(repository, options) {
  const root = fs.realpathSync(path.resolve(repository));
  if (options?.ownerCannotResume !== true || options?.recoveryProcessCannotResume !== true) {
    throw new Error("reservation cleanup requires explicit assertions that both owning processes cannot resume");
  }
  const lock = path.join(root, LOCK);
  if (inspectIncompleteLockDirectory(lock) !== null) {
    throw new Error("reservation cleanup requires the original complete published owner");
  }
  const owner = readLockOwner(lock, COMPOSITION_REPOSITORY_LOCK_LEASE_MS);
  if (owner.token !== options.ownerToken) throw new Error("reservation cleanup owner token differs from the recorded owner");
  const now = (options.clock ?? Date.now)();
  if (!Number.isSafeInteger(now) || now < Date.parse(owner.expires_at)) {
    throw new Error("reservation cleanup requires an expired owner lease");
  }
  if (processFenceIsActive(owner.fence_port, owner.token)) {
    throw new Error("reservation cleanup refuses a live owner process fence");
  }
  clearStoppedRecoveryReservation(root, options.reservationToken);
  return Object.freeze({ cleared: true, owner_token: owner.token });
}

function recoverReserved(root, options, callback, token, now, reservation) {
  const lock = path.join(root, LOCK);
  options.afterReservation?.();
  if (inspectIncompleteLockDirectory(lock) !== null) {
    throw new Error("stopped-owner recovery requires a complete published owner");
  }
  const observed = readLockOwner(lock, COMPOSITION_REPOSITORY_LOCK_LEASE_MS);
  if (observed.token !== token) throw new Error("stopped-owner recovery token differs from the recorded owner");
  if (now < Date.parse(observed.expires_at)) throw new Error("stopped-owner recovery requires an expired owner lease");
  if (processFenceIsActive(observed.fence_port, observed.token)) {
    throw new Error("stopped-owner recovery refuses a live repository process fence");
  }
  const quarantine = `${lock}.operator-recovery-${token}-${crypto.randomBytes(8).toString("hex")}`;
  fs.renameSync(lock, quarantine);
  syncDirectory(root);
  try {
    options.afterQuarantine?.(quarantine);
    const quarantined = readLockOwner(quarantine, COMPOSITION_REPOSITORY_LOCK_LEASE_MS);
    if (!ownerEquals(observed, quarantined)) {
      throw new Error("module composition lock owner changed during stopped-owner recovery");
    }
  } catch (error) {
    restore(lock, quarantine, root);
    throw error;
  }
  try {
    return withCompositionRepositoryLock(root, (active) => {
      const result = callback(active);
      removeOwnedLockDirectory(quarantine, observed);
      syncDirectory(root);
      return result;
    }, { recoveryToken: reservation.token });
  } catch (error) {
    if (fs.existsSync(quarantine) && !fs.existsSync(lock)) restore(lock, quarantine, root);
    throw error;
  }
}

function restore(lock, quarantine, root) {
  if (fs.existsSync(lock)) throw new Error("cannot restore stopped owner over an active repository lock");
  fs.renameSync(quarantine, lock);
  syncDirectory(root);
}
