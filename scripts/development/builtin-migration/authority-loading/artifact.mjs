import fs from "node:fs";
import path from "node:path";
import { TextDecoder } from "node:util";

import { contentDigest } from "../evidence.mjs";
import { assertAuthorityRoot } from "./root.mjs";

const ARTIFACTS = new WeakMap();
const UTF8 = new TextDecoder("utf-8", { fatal: true });

export function canonicalAuthorityPath(value, label = "authority artifact path") {
  if (typeof value !== "string" || value.length === 0 || value.includes("\\")
      || value.includes("\0") || value.includes(":")) {
    throw new Error(`${label} must be a canonical relative POSIX path`);
  }
  const components = value.split("/");
  if (path.posix.isAbsolute(value) || path.posix.normalize(value) !== value
      || value === "." || components.some(invalidPortableComponent)) {
    throw new Error(`${label} must be a canonical relative POSIX path`);
  }
  return value;
}

function invalidPortableComponent(value) {
  if (value === "" || value === ".." || /[ .]$/.test(value)) return true;
  return /^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(value);
}

export function observeJsonArtifact(rootValue, relativePath, label = "authority artifact") {
  const root = assertAuthorityRoot(rootValue);
  const relative = canonicalAuthorityPath(relativePath, `${label} path`);
  const observed = readArtifact(root, relative, label);
  let parsed;
  try {
    parsed = JSON.parse(UTF8.decode(observed.bytes));
  } catch (error) {
    throw new Error(`${label} is not valid UTF-8 JSON: ${error.message}`);
  }
  const artifact = Object.freeze({
    path: relative,
    byteLength: observed.bytes.length,
    contentDigest: contentDigest(observed.bytes),
  });
  ARTIFACTS.set(artifact, {
    root,
    identity: observed.identity,
    value: structuredClone(parsed),
  });
  return artifact;
}

export function assertLoadedJsonArtifact(value, rootValue, relativePath = null) {
  const root = assertAuthorityRoot(rootValue);
  const record = ARTIFACTS.get(value);
  if (!record || record.root !== root) {
    throw new Error("operation requires an exact loaded JSON authority artifact");
  }
  if (relativePath !== null
      && value.path !== canonicalAuthorityPath(relativePath, "expected authority artifact path")) {
    throw new Error("loaded authority artifact belongs to another path");
  }
  return value;
}

export function loadedJsonValue(value, rootValue) {
  assertLoadedJsonArtifact(value, rootValue);
  return structuredClone(ARTIFACTS.get(value).value);
}

export function revalidateJsonArtifact(value, rootValue) {
  const artifact = assertLoadedJsonArtifact(value, rootValue);
  const record = ARTIFACTS.get(artifact);
  let observed;
  try {
    observed = readArtifact(
      record.root, artifact.path, `loaded authority artifact ${artifact.path}`,
    );
  } catch (error) {
    throw new Error(`${artifact.path}: cannot revalidate loaded authority artifact: ${error.message}`);
  }
  if (!sameIdentity(observed.identity, record.identity)
      || observed.bytes.length !== artifact.byteLength
      || contentDigest(observed.bytes) !== artifact.contentDigest) {
    throw new Error(`${artifact.path}: loaded authority artifact changed after observation`);
  }
  return artifact;
}

function readArtifact(root, relative, label) {
  // Node cannot anchor each descendant lookup to an open directory descriptor on every
  // supported platform. These checks protect a tree managed by cooperative writers;
  // callers must revalidate the session immediately before using its authority.
  const components = relative.split("/");
  let current = root.path;
  for (const component of components.slice(0, -1)) {
    current = path.join(current, component);
    const state = fs.lstatSync(current);
    if (state.isSymbolicLink() || !state.isDirectory()) {
      throw new Error(`${label} has a symbolic-link or non-directory ancestor`);
    }
  }
  const target = path.join(current, components.at(-1));
  const state = fs.lstatSync(target);
  if (state.isSymbolicLink() || !state.isFile()) {
    throw new Error(`${label} must be a real regular file`);
  }
  const flags = fs.constants.O_RDONLY | noFollowFlag() | nonblockingFlag();
  const descriptor = fs.openSync(target, flags);
  try {
    const before = fs.fstatSync(descriptor);
    if (!before.isFile() || !sameIdentity(before, state)) {
      throw new Error(`${label} changed while opening`);
    }
    const bytes = fs.readFileSync(descriptor);
    const after = fs.fstatSync(descriptor);
    if (!after.isFile() || !sameStableFile(before, after)
        || bytes.length !== after.size) {
      throw new Error(`${label} changed while reading`);
    }
    return { bytes, identity: fileIdentity(after) };
  } finally {
    fs.closeSync(descriptor);
  }
}

function noFollowFlag() {
  // Windows does not expose O_NOFOLLOW through Node. The surrounding lstat/open/fstat
  // checks are the explicit cooperative-writer policy on that platform.
  if (process.platform === "win32") return 0;
  if (fs.constants.O_NOFOLLOW === undefined) {
    throw new Error("no-follow authority artifact reads are unavailable on this platform");
  }
  return fs.constants.O_NOFOLLOW;
}

function nonblockingFlag() {
  if (process.platform === "win32") return 0;
  return fs.constants.O_NONBLOCK ?? 0;
}

function sameIdentity(left, right) {
  return left.dev === right.dev && left.ino === right.ino;
}

function sameStableFile(left, right) {
  return sameIdentity(left, right) && left.size === right.size
    && left.mtimeMs === right.mtimeMs && left.ctimeMs === right.ctimeMs;
}

function fileIdentity(value) {
  return Object.freeze({
    dev: value.dev,
    ino: value.ino,
    size: value.size,
    mtimeMs: value.mtimeMs,
    ctimeMs: value.ctimeMs,
  });
}
