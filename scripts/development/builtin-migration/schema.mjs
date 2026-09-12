import { sorted } from "./constants.mjs";
import path from "node:path";

export const SAFE_IDENTITY = /^(?:[A-Za-z]|__)[A-Za-z0-9_.]*$/;
export const SAFE_STABLE_ID = /^[a-z][a-z0-9_.-]*$/;
export const SAFE_PATH = /^(?!\/)(?!.*(?:^|\/)\.\.(?:\/|$))[A-Za-z0-9_.+@/-]+$/;
export const DIGEST = /^sha256:[a-f0-9]{64}$/;
export const SOURCE_REVISION = /^git:[a-f0-9]{40}$/;
export const FILESYSTEM_IDENTITY = /^(?:posix-dev:[1-9][0-9]*|windows-volume:[A-Fa-f0-9-]{8,})$/;

export function object(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value;
}

export function exact(value, fields, label) {
  object(value, label);
  const actual = sorted(Object.keys(value));
  const expected = sorted(fields);
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(`${label} fields must be exactly ${fields.join(", ")}`);
  }
}

export function kind(value, version, expectedKind, label = expectedKind) {
  object(value, label);
  if (value.schema_version !== version || value.kind !== expectedKind) {
    throw new Error(`${label} must use schema_version ${version} and kind ${expectedKind}`);
  }
}

export function nonempty(value, label) {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${label} must be a nonempty string`);
  return value.trim();
}

export function enumValue(value, allowed, label) {
  if (!allowed.includes(value)) throw new Error(`${label} must be one of ${allowed.join(", ")}`);
  return value;
}

export function digest(value, label) {
  const result = nonempty(value, label);
  if (!DIGEST.test(result)) throw new Error(`${label} must be a lowercase sha256 digest`);
  return result;
}

export function sourceRevision(value, label) {
  const result = nonempty(value, label);
  if (!SOURCE_REVISION.test(result)) throw new Error(`${label} must be git:<40 lowercase hex characters>`);
  return result;
}

export function identity(value, label) {
  const result = nonempty(value, label);
  if (!SAFE_IDENTITY.test(result)) throw new Error(`${label} is not a safe builtin identity`);
  return result;
}

export function stableId(value, label) {
  const result = nonempty(value, label);
  if (!SAFE_STABLE_ID.test(result)) throw new Error(`${label} is not a safe stable identifier`);
  return result;
}

export function repositoryPath(value, label) {
  const result = nonempty(value, label);
  if (!SAFE_PATH.test(result)
    || result.includes("//")
    || result === "."
    || result.endsWith("/")
    || path.posix.normalize(result) !== result) {
    throw new Error(`${label} is not a normalized safe repository-relative path`);
  }
  return result;
}

export function absolutePath(value, label) {
  const result = nonempty(value, label);
  if (!path.isAbsolute(result) || path.normalize(result) !== result) throw new Error(`${label} must be a normalized absolute path`);
  return result;
}

export function filesystemIdentity(value, label) {
  const result = nonempty(value, label);
  if (!FILESYSTEM_IDENTITY.test(result)) throw new Error(`${label} must identify a POSIX device or Windows volume`);
  return result;
}

export function timestamp(value, label) {
  const result = nonempty(value, label);
  const parsed = Date.parse(result);
  if (!Number.isFinite(parsed) || new Date(parsed).toISOString() !== result) throw new Error(`${label} must be an ISO-8601 UTC timestamp`);
  return result;
}

export function array(value, label, { empty = false } = {}) {
  if (!Array.isArray(value) || (!empty && value.length === 0)) {
    throw new Error(`${label} must be ${empty ? "an" : "a nonempty"} array`);
  }
  return value;
}

export function uniqueStrings(value, label, { empty = false, pattern = null, lower = false } = {}) {
  const entries = array(value, label, { empty }).map((entry) => nonempty(entry, label));
  if (pattern && entries.some((entry) => !pattern.test(entry))) throw new Error(`${label} contains an invalid value`);
  const normalized = entries.map((entry) => lower ? entry.toLowerCase() : entry);
  if (new Set(normalized).size !== normalized.length) throw new Error(`${label} must be unique`);
  return normalized;
}

export function boolean(value, label) {
  if (typeof value !== "boolean") throw new Error(`${label} must be boolean`);
  return value;
}

export function integer(value, label, minimum = 0) {
  if (!Number.isSafeInteger(value) || value < minimum) throw new Error(`${label} must be an integer >= ${minimum}`);
  return value;
}
