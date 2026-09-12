import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

import { evidenceDigest } from "../evidence.mjs";
import { syncDirectory } from "./durability.mjs";
import { moduleCompositionProductRegistry } from "./registry.mjs";
import { resolveRepositoryProduct } from "./repository-state.mjs";

const JOURNAL = ".runmat-module-composition.transaction.json";
const JOURNAL_DRAFT = `${JOURNAL}.new`;
const COMMITTED_DRAFT = `${JOURNAL}.committed.new`;

export function journalPath(repository) { return path.join(repository, JOURNAL); }
function commitPath(repository) { return `${journalPath(repository)}.committed`; }

export function writeTransactionJournal(repository, value) {
  const target = journalPath(repository);
  const temporary = path.join(repository, JOURNAL_DRAFT);
  const descriptor = fs.openSync(temporary, fs.constants.O_CREAT | fs.constants.O_EXCL | fs.constants.O_WRONLY | noFollow(), 0o600);
  try {
    fs.writeFileSync(descriptor, `${JSON.stringify(value)}\n`);
    fs.fsyncSync(descriptor);
  } catch (error) {
    fs.closeSync(descriptor);
    unlinkIfPresent(temporary);
    throw error;
  }
  fs.closeSync(descriptor);
  try { fs.renameSync(temporary, target); fsyncDirectory(repository); }
  catch (error) { unlinkIfPresent(temporary); throw error; }
}

export function recoverTransactionJournal(repository, expectedToken = null) {
  const target = journalPath(repository);
  if (lstatOrNull(target) === null) {
    const orphan = commitPath(repository);
    if (lstatOrNull(orphan) !== null) { committedMarker(repository); fs.unlinkSync(orphan); fsyncDirectory(repository); }
    cleanupJournalDrafts(repository);
    return { recovered: false };
  }
  const journal = readJournal(target);
  assertExpectedToken(journal, expectedToken);
  cleanupJournalDrafts(repository);
  const committed = committedMarker(repository);
  if (committed) finishCommitted(repository, journal);
  else rollbackInstalling(repository, journal);
  fs.unlinkSync(target);
  unlinkIfPresent(commitPath(repository));
  fsyncDirectory(repository);
  return { recovered: true, phase: committed ? "committed" : "installing" };
}

export function removeTransactionJournal(repository, expectedToken) {
  assertExpectedToken(readJournal(journalPath(repository)), expectedToken);
  fs.unlinkSync(journalPath(repository));
  unlinkIfPresent(commitPath(repository));
  cleanupJournalDrafts(repository);
  fsyncDirectory(repository);
}

export function markTransactionCommitted(repository, expectedToken) {
  assertExpectedToken(readJournal(journalPath(repository)), expectedToken);
  const target = commitPath(repository);
  const temporary = path.join(repository, COMMITTED_DRAFT);
  const descriptor = fs.openSync(temporary, fs.constants.O_CREAT | fs.constants.O_EXCL | fs.constants.O_WRONLY | noFollow(), 0o600);
  try {
    fs.writeFileSync(descriptor, "committed\n");
    fs.fsyncSync(descriptor);
  } catch (error) {
    fs.closeSync(descriptor);
    unlinkIfPresent(temporary);
    throw error;
  }
  fs.closeSync(descriptor);
  try { fs.renameSync(temporary, target); fsyncDirectory(repository); }
  catch (error) { unlinkIfPresent(temporary); throw error; }
}

function finishCommitted(repository, journal) {
  for (const entry of journal.entries) {
    const paths = entryPaths(repository, journal.token, entry);
    assertDigest(paths.target, entry.new_digest, `${entry.product_id}: committed target`);
    unlinkIfPresent(paths.temporary);
    unlinkIfPresent(paths.backup);
    fsyncDirectory(path.dirname(paths.target));
  }
}

function rollbackInstalling(repository, journal) {
  for (const entry of [...journal.entries].reverse()) {
    const paths = entryPaths(repository, journal.token, entry);
    if (lstatOrNull(paths.backup) !== null) {
      assertBeforeFile(paths.backup, entry, `${entry.product_id}: recovery backup`);
      unlinkKnownTarget(paths.target, entry);
      fs.renameSync(paths.backup, paths.target);
    } else if (entry.existed) {
      assertBeforeFile(paths.target, entry, `${entry.product_id}: recovery target`);
    } else {
      unlinkKnownTarget(paths.target, entry);
    }
    unlinkIfPresent(paths.temporary);
    fsyncDirectory(path.dirname(paths.target));
  }
}

function readJournal(target) {
  const prior = fs.lstatSync(target);
  if (!prior.isFile()) throw new Error("module composition recovery journal is not a regular file");
  const descriptor = fs.openSync(target, fs.constants.O_RDONLY | noFollow());
  let value;
  try { const stat = fs.fstatSync(descriptor); if (stat.dev !== prior.dev || stat.ino !== prior.ino) throw new Error("module composition recovery journal changed while opening"); value = JSON.parse(fs.readFileSync(descriptor, "utf8")); } finally { fs.closeSync(descriptor); }
  const keys = ["entries", "kind", "phase", "registry_digest", "schema_version", "token"];
  if (value?.schema_version !== 1 || value.kind !== "runmat-module-composition-transaction"
    || value.phase !== "installing" || value.registry_digest !== registryDigest()
    || !/^[0-9]+-[a-f0-9]{24}$/.test(value.token) || !Array.isArray(value.entries)
    || JSON.stringify(Object.keys(value).sort()) !== JSON.stringify(keys)) {
    throw new Error("module composition recovery journal is invalid");
  }
  validateRegistryEntries(value.entries);
  return value;
}

function validateRegistryEntries(entries) {
  const registry = new Map(moduleCompositionProductRegistry()
    .map((entry) => [entry.product_id, entry.path]));
  const productIds = new Set();
  const paths = new Set();
  for (const entry of entries) {
    validateJournalEntry(entry);
    if (registry.get(entry?.product_id) !== entry?.path) {
      throw new Error("module composition recovery journal product differs from the fixed registry");
    }
    if (productIds.has(entry.product_id) || paths.has(entry.path)) {
      throw new Error("module composition recovery journal contains duplicate products");
    }
    productIds.add(entry.product_id);
    paths.add(entry.path);
  }
}

function assertExpectedToken(journal, expectedToken) {
  if (expectedToken !== null && journal.token !== expectedToken) {
    throw new Error("module composition transaction journal owner changed");
  }
}

function entryPaths(repository, token, entry) {
  validateJournalEntry(entry);
  const target = resolveRepositoryProduct(repository, entry.path);
  return { target, temporary: `${target}.runmat-stage-${token}`, backup: `${target}.runmat-backup-${token}` };
}

function validateJournalEntry(entry) {
  const keys = ["before_digest", "before_file_identity", "existed", "new_digest", "path", "product_id"];
  const identityValid = entry?.before_file_identity && typeof entry.before_file_identity.device === "string" && typeof entry.before_file_identity.inode === "string" && JSON.stringify(Object.keys(entry.before_file_identity).sort()) === JSON.stringify(["device", "inode"]);
  if (!entry || JSON.stringify(Object.keys(entry).sort()) !== JSON.stringify(keys) || typeof entry.product_id !== "string" || typeof entry.path !== "string" || typeof entry.existed !== "boolean" || !digestValue(entry.new_digest) || (entry.existed && (!digestValue(entry.before_digest) || !identityValid)) || (!entry.existed && (entry.before_digest !== null || entry.before_file_identity !== null))) throw new Error("module composition recovery journal entry is invalid");
}

function unlinkKnownTarget(target, entry) {
  if (lstatOrNull(target) === null) return;
  const observed = digestBytes(readRegularNoFollow(target).bytes);
  const allowed = [entry.new_digest, entry.before_digest].filter(Boolean);
  if (!allowed.includes(observed)) throw new Error(`${entry.product_id}: recovery target has unknown content`);
  fs.unlinkSync(target);
}
function assertDigest(target, expected, label) { const observed = readRegularNoFollow(target); if (digestBytes(observed.bytes) !== expected) throw new Error(`${label} content differs from the recovery journal`); }
function assertBeforeFile(target, entry, label) { const observed = readRegularNoFollow(target); if (digestBytes(observed.bytes) !== entry.before_digest) throw new Error(`${label} content differs from the recovery journal`); if (String(observed.stat.dev) !== entry.before_file_identity.device || String(observed.stat.ino) !== entry.before_file_identity.inode) throw new Error(`${label} identity differs from the recovery journal`); }
function committedMarker(repository) { const target = commitPath(repository); if (lstatOrNull(target) === null) return false; if (readRegularNoFollow(target).bytes.toString("utf8") !== "committed\n") throw new Error("module composition commit marker is invalid"); return true; }
function cleanupJournalDrafts(repository) {
  const names = fs.readdirSync(repository).filter((name) => name === JOURNAL_DRAFT || name === COMMITTED_DRAFT
    || /^\.runmat-module-composition\.transaction\.json\.new-[0-9]+-[a-f0-9]{16}$/.test(name));
  for (const name of names) unlinkBoundedArtifact(path.join(repository, name));
  if (names.length) fsyncDirectory(repository);
}
function unlinkBoundedArtifact(target) {
  const observed = lstatOrNull(target);
  if (observed === null) return;
  if (!observed.isFile() && !observed.isSymbolicLink()) {
    throw new Error("module composition journal draft is not a regular file or symbolic link");
  }
  fs.unlinkSync(target);
}
function digestValue(value) { return typeof value === "string" && /^sha256:[a-f0-9]{64}$/.test(value); }
function readRegularNoFollow(target) { const prior = fs.lstatSync(target); if (!prior.isFile()) throw new Error(`${target} is not a regular recovery file`); const descriptor = fs.openSync(target, fs.constants.O_RDONLY | noFollow()); try { const stat = fs.fstatSync(descriptor); if (!stat.isFile() || stat.dev !== prior.dev || stat.ino !== prior.ino) throw new Error(`${target} changed while opening for recovery`); return { bytes: fs.readFileSync(descriptor), stat }; } finally { fs.closeSync(descriptor); } }
function digestBytes(bytes) { return `sha256:${crypto.createHash("sha256").update(bytes).digest("hex")}`; }
function registryDigest() { return evidenceDigest(moduleCompositionProductRegistry()); }
function lstatOrNull(target) { try { return fs.lstatSync(target); } catch (error) { if (["ENOENT", "ENOTDIR"].includes(error?.code)) return null; throw error; } }
function unlinkIfPresent(target) { try { fs.unlinkSync(target); } catch (error) { if (error?.code !== "ENOENT") throw error; } }
function fsyncDirectory(directory) { syncDirectory(directory); }
function noFollow() { return fs.constants.O_NOFOLLOW ?? 0; }
