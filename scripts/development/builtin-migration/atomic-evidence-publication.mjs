import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

import { syncDirectory } from "./directory-durability.mjs";

export function canonicalEvidencePath(target, options = {}) {
  const filesystem = options.filesystem ?? fs;
  const remainder = [];
  let existing = path.resolve(target);
  while (!filesystem.existsSync(existing)) {
    const parent = path.dirname(existing);
    if (parent === existing) throw new Error(`cannot resolve an existing ancestor for ${target}`);
    remainder.unshift(path.basename(existing));
    existing = parent;
  }
  return path.join(filesystem.realpathSync(existing), ...remainder);
}

export function publishEvidenceBytes(target, bytes, options = {}) {
  const filesystem = options.filesystem ?? fs;
  const platform = options.platform ?? process.platform;
  const mode = publicationMode(options.mode);
  const absolute = path.resolve(target);
  const parent = path.dirname(absolute);
  if (absolute === parent) throw new Error("evidence target must name a file");
  const contents = exactBytes(bytes);
  ensureDirectory(parent, Boolean(options.createParentDirectories), filesystem, platform);
  rejectExistingTarget(absolute, filesystem);

  const token = options.temporaryToken ?? crypto.randomBytes(16).toString("hex");
  if (!/^[A-Za-z0-9_-]+$/.test(token)) throw new Error("temporary token is invalid");
  const temporary = path.join(parent, `.${path.basename(absolute)}.runmat-new-${token}`);
  const staged = stageBytes(temporary, contents, mode, filesystem, platform);
  let published = false;
  try {
    options.hooks?.afterStage?.({ target: absolute, temporary });
    ensureDirectory(parent, false, filesystem, platform);
    rejectExistingTarget(absolute, filesystem);
    assertOwnedTemporary(temporary, staged, filesystem, platform);
    options.hooks?.beforePublish?.({ target: absolute, temporary });
    filesystem.linkSync(temporary, absolute);
    published = true;
    filesystem.unlinkSync(temporary);
    const durability = syncDirectory(parent, { filesystem, platform });
    return Object.freeze({ path: absolute, byteLength: contents.length, durability });
  } catch (error) {
    const cleanup = removeOwnedTemporary(temporary, staged, filesystem);
    if (cleanup !== null) {
      throw new AggregateError([error, cleanup], "evidence publication and temporary cleanup failed");
    }
    if (published) {
      throw new Error(
        `evidence target was published but publication finalization failed: ${error.message}`,
        { cause: error },
      );
    }
    throw error;
  }
}

function publicationMode(value = 0o666) {
  if (!Number.isInteger(value) || value < 0 || value > 0o777) {
    throw new TypeError("evidence publication mode must contain only Unix permission bits");
  }
  return value;
}

function exactBytes(value) {
  if (typeof value === "string") return Buffer.from(value, "utf8");
  if (Buffer.isBuffer(value) || value instanceof Uint8Array) return Buffer.from(value);
  throw new TypeError("evidence bytes must be a string, Buffer, or Uint8Array");
}

function ensureDirectory(directory, create, filesystem, platform) {
  const parsed = path.parse(directory);
  let current = parsed.root;
  for (const component of directory.slice(parsed.root.length).split(path.sep).filter(Boolean)) {
    const parent = current;
    current = path.join(current, component);
    let state = lstatOrNull(current, filesystem);
    if (state === null) {
      if (!create) throw new Error(`evidence parent directory does not exist: ${current}`);
      filesystem.mkdirSync(current, { mode: 0o755 });
      syncDirectory(parent, { filesystem, platform });
      state = filesystem.lstatSync(current);
    }
    if (state.isSymbolicLink() || !state.isDirectory()) {
      throw new Error(`evidence parent component is not a real directory: ${current}`);
    }
  }
}

function rejectExistingTarget(target, filesystem) {
  const state = lstatOrNull(target, filesystem);
  if (state !== null) throw new Error(`evidence target already exists: ${target}`);
}

function stageBytes(temporary, contents, mode, filesystem, platform) {
  const constants = filesystem.constants ?? fs.constants;
  if (platform !== "win32" && constants.O_NOFOLLOW === undefined) {
    throw new Error("exclusive no-follow staging is unavailable on this platform");
  }
  const flags = constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL
    | (constants.O_NOFOLLOW ?? 0);
  let descriptor = null;
  let identity = null;
  let failure = null;
  try {
    descriptor = filesystem.openSync(temporary, flags, mode);
    identity = filesystem.fstatSync(descriptor);
    if (!identity.isFile()) throw new Error("evidence temporary is not a regular file");
    filesystem.writeFileSync(descriptor, contents);
    filesystem.fsyncSync(descriptor);
  } catch (error) {
    failure = error;
  } finally {
    if (descriptor !== null) {
      try { filesystem.closeSync(descriptor); } catch (error) { failure ??= error; }
    }
  }
  if (failure !== null) {
    const cleanup = identity === null ? null : removeOwnedTemporary(temporary, identity, filesystem);
    if (cleanup !== null) {
      throw new AggregateError([failure, cleanup], "evidence staging and temporary cleanup failed");
    }
    throw failure;
  }
  return { dev: identity.dev, ino: identity.ino };
}

function removeOwnedTemporary(temporary, identity, filesystem) {
  try {
    const state = lstatOrNull(temporary, filesystem);
    if (state === null) return null;
    if (!state.isFile() || state.isSymbolicLink()
        || state.dev !== identity.dev || state.ino !== identity.ino) {
      return new Error("evidence temporary changed before cleanup");
    }
    filesystem.unlinkSync(temporary);
    return null;
  } catch (error) {
    return error;
  }
}

function assertOwnedTemporary(temporary, identity, filesystem, platform) {
  const state = filesystem.lstatSync(temporary);
  if (!state.isFile() || state.isSymbolicLink()
      || state.dev !== identity.dev || state.ino !== identity.ino) {
    throw new Error("evidence temporary changed before publication");
  }
  const constants = filesystem.constants ?? fs.constants;
  if (platform !== "win32" && constants.O_NOFOLLOW === undefined) {
    throw new Error("no-follow temporary verification is unavailable on this platform");
  }
  const descriptor = filesystem.openSync(
    temporary, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0),
  );
  try {
    const opened = filesystem.fstatSync(descriptor);
    if (!opened.isFile() || opened.dev !== identity.dev || opened.ino !== identity.ino) {
      throw new Error("evidence temporary changed while opening for publication");
    }
  } finally {
    filesystem.closeSync(descriptor);
  }
}

function lstatOrNull(target, filesystem) {
  try {
    return filesystem.lstatSync(target);
  } catch (error) {
    if (["ENOENT", "ENOTDIR"].includes(error?.code)) return null;
    throw error;
  }
}
