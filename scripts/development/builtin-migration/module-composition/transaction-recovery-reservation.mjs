import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

import { syncDirectory } from "./durability.mjs";

const RESERVATION = ".runmat-module-composition.recovery.lock";

export function claimRecoveryReservation(repository) {
  const token = crypto.randomBytes(16).toString("hex");
  const target = path.join(repository, RESERVATION);
  const descriptor = fs.openSync(
    target, fs.constants.O_CREAT | fs.constants.O_EXCL | fs.constants.O_WRONLY | noFollow(), 0o600,
  );
  try { fs.writeFileSync(descriptor, `${token}\n`); fs.fsyncSync(descriptor); }
  catch (error) { fs.closeSync(descriptor); unlinkIfPresent(target); throw error; }
  fs.closeSync(descriptor);
  syncDirectory(repository);
  return Object.freeze({ target, token, identity: fileIdentity(target) });
}

export function assertRecoveryAdmission(repository, admittedToken = null) {
  const target = path.join(repository, RESERVATION);
  let observed;
  try { observed = readReservation(target); }
  catch (error) { if (error?.code === "ENOENT") return; throw error; }
  if (admittedToken === null || observed.token !== admittedToken) {
    throw new Error("module composition stopped-owner recovery is in progress");
  }
}

export function releaseRecoveryReservation(repository, reservation) {
  const target = path.join(repository, RESERVATION);
  const observed = readReservation(target);
  if (observed.token !== reservation.token || !sameIdentity(observed.identity, reservation.identity)) {
    throw new Error("module composition recovery reservation changed before release");
  }
  fs.unlinkSync(target);
  syncDirectory(repository);
}

export function clearStoppedRecoveryReservation(repository, expectedToken) {
  if (!/^[a-f0-9]{32}$/.test(expectedToken ?? "")) {
    throw new Error("recovery reservation cleanup requires its exact token");
  }
  const target = path.join(repository, RESERVATION);
  const observed = readReservation(target);
  if (observed.token !== expectedToken) throw new Error("recovery reservation cleanup token differs from the recorded token");
  const quarantine = `${target}.clear-${expectedToken}-${crypto.randomBytes(8).toString("hex")}`;
  fs.renameSync(target, quarantine);
  syncDirectory(repository);
  try {
    const moved = readReservation(quarantine);
    if (moved.token !== observed.token || !sameIdentity(moved.identity, observed.identity)) {
      throw new Error("recovery reservation changed during cleanup");
    }
    fs.unlinkSync(quarantine);
    syncDirectory(repository);
  } catch (error) {
    if (!fs.existsSync(target) && fs.existsSync(quarantine)) {
      fs.renameSync(quarantine, target);
      syncDirectory(repository);
    }
    throw error;
  }
}

function readReservation(target) {
  const prior = fs.lstatSync(target);
  if (!prior.isFile()) throw new Error("module composition recovery reservation is not a regular file");
  const descriptor = fs.openSync(target, fs.constants.O_RDONLY | noFollow());
  try {
    const stat = fs.fstatSync(descriptor);
    if (!stat.isFile() || stat.dev !== prior.dev || stat.ino !== prior.ino) throw new Error("module composition recovery reservation changed while opening");
    const value = fs.readFileSync(descriptor, "utf8");
    if (!/^[a-f0-9]{32}\n$/.test(value)) throw new Error("module composition recovery reservation is invalid");
    return { token: value.trim(), identity: { device: String(stat.dev), inode: String(stat.ino) } };
  } finally { fs.closeSync(descriptor); }
}

function fileIdentity(target) { const stat = fs.statSync(target); return { device: String(stat.dev), inode: String(stat.ino) }; }
function sameIdentity(left, right) { return left.device === right.device && left.inode === right.inode; }
function unlinkIfPresent(target) { try { fs.unlinkSync(target); } catch (error) { if (error?.code !== "ENOENT") throw error; } }
function noFollow() { return fs.constants.O_NOFOLLOW ?? 0; }
