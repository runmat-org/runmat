import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

import { syncDirectory } from "./durability.mjs";

const OWNER = "owner.json";
const OWNER_KEYS = [
  "acquired_at", "expires_at", "fence_port", "hostname", "kind", "pid", "renewed_at",
  "schema_version", "token",
];

export function createLockOwner({ token, acquiredAt, now, pid, hostname, fencePort, leaseMs }) {
  const renewedAt = timestamp(now);
  return Object.freeze({
    schema_version: 1, kind: "runmat-module-composition-lock-owner",
    token, acquired_at: acquiredAt ?? renewedAt, renewed_at: renewedAt,
    expires_at: timestamp(now + leaseMs), fence_port: fencePort, pid, hostname,
  });
}

export function readLockOwner(lock, leaseMs) {
  const target = path.join(lock, OWNER);
  const prior = fs.lstatSync(target);
  if (!prior.isFile()) throw new Error("module composition lock owner is not a regular file");
  const descriptor = fs.openSync(target, fs.constants.O_RDONLY | noFollow());
  let value;
  try {
    const stat = fs.fstatSync(descriptor);
    if (!stat.isFile() || stat.dev !== prior.dev || stat.ino !== prior.ino) {
      throw new Error("module composition lock owner changed while opening");
    }
    value = JSON.parse(fs.readFileSync(descriptor, "utf8"));
  } finally { fs.closeSync(descriptor); }
  return validateOwner(value, leaseMs);
}

export function replaceLockOwner(lock, owner, options = {}) {
  const target = path.join(lock, OWNER);
  const temporary = path.join(lock, `.owner-${owner.token}.new`);
  const before = directoryIdentity(lock);
  const descriptor = fs.openSync(temporary, fs.constants.O_CREAT | fs.constants.O_EXCL | fs.constants.O_WRONLY | noFollow(), 0o600);
  try { fs.writeFileSync(descriptor, `${JSON.stringify(owner)}\n`); fs.fsyncSync(descriptor); }
  finally { fs.closeSync(descriptor); }
  try {
    if (!sameIdentity(before, directoryIdentity(lock))) throw new Error("module composition lock directory changed during owner write");
    if (options.initial !== true) {
      const observed = readLockOwner(lock, ownerLeaseMs(owner));
      if (!options.expected || !ownerEquals(observed, options.expected)) {
        throw new Error("module composition lock owner changed before renewal");
      }
    }
    fs.renameSync(temporary, target);
    syncDirectory(lock);
  } catch (error) { unlinkIfPresent(temporary); throw error; }
}

export function removeOwnedLockDirectory(lock, expected) {
  const observed = readLockOwner(lock, ownerLeaseMs(expected));
  if (!ownerEquals(observed, expected)) throw new Error("module composition lock release owner differs from the active lease");
  const allowedDraft = `.owner-${expected.token}.new`;
  for (const name of fs.readdirSync(lock)) {
    if (name !== OWNER && name !== allowedDraft) throw new Error("module composition lock directory contains an unexpected entry");
  }
  unlinkIfPresent(path.join(lock, allowedDraft));
  fs.unlinkSync(path.join(lock, OWNER));
  fs.rmdirSync(lock);
}

export function inspectIncompleteLockDirectory(lock) {
  if (exists(path.join(lock, OWNER))) return null;
  const entries = fs.readdirSync(lock);
  if (entries.length > 1 || (entries.length === 1 && !/^\.owner-[a-f0-9]{32}\.new$/.test(entries[0]))) {
    throw new Error("module composition incomplete lock directory contains an unexpected entry");
  }
  const stat = fs.statSync(lock);
  const artifact = entries.length ? observeArtifact(path.join(lock, entries[0])) : null;
  return Object.freeze({ directory: directoryIdentity(lock), modified_at: Math.ceil(stat.mtimeMs), artifact });
}

export function removeIncompleteLockDirectory(lock, expected) {
  const observed = inspectIncompleteLockDirectory(lock);
  if (!observed || JSON.stringify(observed) !== JSON.stringify(expected)) {
    throw new Error("module composition incomplete lock changed during recovery");
  }
  if (observed.artifact) fs.unlinkSync(path.join(lock, observed.artifact.name));
  fs.rmdirSync(lock);
}

export function ownerEquals(left, right) { return JSON.stringify(left) === JSON.stringify(right); }

function validateOwner(value, leaseMs) {
  if (!value || JSON.stringify(Object.keys(value).sort()) !== JSON.stringify([...OWNER_KEYS].sort())
    || value.schema_version !== 1 || value.kind !== "runmat-module-composition-lock-owner"
    || !/^[a-f0-9]{32}$/.test(value.token) || !Number.isSafeInteger(value.pid) || value.pid <= 0
    || !Number.isInteger(value.fence_port) || value.fence_port < 1 || value.fence_port > 65_535
    || typeof value.hostname !== "string" || !value.hostname) {
    throw new Error("module composition lock owner is invalid");
  }
  const acquired = parseTimestamp(value.acquired_at);
  const renewed = parseTimestamp(value.renewed_at);
  const expires = parseTimestamp(value.expires_at);
  if (acquired > renewed || expires - renewed !== leaseMs) throw new Error("module composition lock owner lease interval is invalid");
  return Object.freeze(structuredClone(value));
}

function ownerLeaseMs(owner) { return Date.parse(owner.expires_at) - Date.parse(owner.renewed_at); }
function timestamp(value) { return new Date(value).toISOString(); }
function parseTimestamp(value) { const parsed = Date.parse(value); if (!Number.isSafeInteger(parsed) || new Date(parsed).toISOString() !== value) throw new Error("module composition lock owner timestamp is invalid"); return parsed; }
function directoryIdentity(target) { const stat = fs.statSync(target); return { device: String(stat.dev), inode: String(stat.ino) }; }
function sameIdentity(left, right) { return left.device === right.device && left.inode === right.inode; }
function observeArtifact(target) {
  const prior = fs.lstatSync(target);
  if (!prior.isFile()) throw new Error("module composition incomplete lock artifact is not a regular file");
  const descriptor = fs.openSync(target, fs.constants.O_RDONLY | noFollow());
  try {
    const stat = fs.fstatSync(descriptor);
    if (stat.dev !== prior.dev || stat.ino !== prior.ino) {
      throw new Error("module composition incomplete lock artifact changed while opening");
    }
    const bytes = fs.readFileSync(descriptor);
    let owner = null;
    try {
      const candidate = JSON.parse(bytes.toString("utf8"));
      owner = validateOwner(candidate, ownerLeaseMs(candidate));
    } catch {
      // A partial or malformed draft has no process-fence claim.
    }
    return {
      name: path.basename(target), device: String(stat.dev), inode: String(stat.ino),
      digest: crypto.createHash("sha256").update(bytes).digest("hex"), owner,
    };
  } finally { fs.closeSync(descriptor); }
}
function exists(target) { try { fs.lstatSync(target); return true; } catch (error) { if (error?.code === "ENOENT") return false; throw error; } }
function unlinkIfPresent(target) { try { fs.unlinkSync(target); } catch (error) { if (error?.code !== "ENOENT") throw error; } }
function noFollow() { return fs.constants.O_NOFOLLOW ?? 0; }
