import path from "node:path";

import {
  assertAuthorityLoadSession, canonicalAuthorityPath,
} from "../authority-loading/index.mjs";
import { digest } from "../schema.mjs";

const INDEXES = new WeakMap();

export function openQueueReferenceIndex(sessionValue) {
  const session = assertAuthorityLoadSession(sessionValue);
  const index = Object.freeze({ session });
  INDEXES.set(index, {
    session,
    byPath: new Map(),
    bySemanticAuthority: new Map(),
  });
  return index;
}

export function bindQueueReference(indexValue, domain, pathValue, digestValue) {
  const index = indexRecord(indexValue);
  const authorityDomain = domainId(domain);
  const authorityPath = canonicalAuthorityPath(pathValue, `${authorityDomain} path`);
  const semanticDigest = digest(digestValue, `${authorityDomain} semantic digest`);
  const binding = Object.freeze({
    domain: authorityDomain,
    path: authorityPath,
    semanticDigest,
  });
  const pathBinding = index.byPath.get(authorityPath);
  if (pathBinding && !sameBinding(pathBinding, binding)) {
    throw new Error(`${authorityPath}: queue authority path was rebound`);
  }
  const semanticKey = `${authorityDomain}\0${semanticDigest}`;
  const semanticPath = index.bySemanticAuthority.get(semanticKey);
  if (semanticPath && semanticPath !== authorityPath) {
    throw new Error(`${authorityDomain}: semantic authority digest was rebound to another path`);
  }
  if (!pathBinding) index.byPath.set(authorityPath, binding);
  if (!semanticPath) index.bySemanticAuthority.set(semanticKey, authorityPath);
  return pathBinding ?? binding;
}

export function assertQueueReferenceIndex(value, sessionValue = null) {
  const record = indexRecord(value);
  if (sessionValue !== null && record.session !== assertAuthorityLoadSession(sessionValue)) {
    throw new Error("queue reference index belongs to another authority load session");
  }
  return value;
}

export function canonicalCliChildPath(basePath, suppliedPath, label) {
  const base = path.resolve(basePath);
  const absolute = path.resolve(base, suppliedPath);
  const relative = path.relative(base, absolute);
  if (relative === "" || relative === ".." || relative.startsWith(`..${path.sep}`)
      || path.isAbsolute(relative)) {
    throw new Error(`${label} must name a file below the queue authority root`);
  }
  return canonicalAuthorityPath(relative.split(path.sep).join("/"), label);
}

function indexRecord(value) {
  const record = INDEXES.get(value);
  if (!record) throw new Error("operation requires an exact queue reference index");
  return record;
}

function domainId(value) {
  if (typeof value !== "string" || !/^queue-[a-z][a-z0-9-]*$/.test(value)) {
    throw new Error("queue authority domain is invalid");
  }
  return value;
}

function sameBinding(left, right) {
  return left.domain === right.domain && left.path === right.path
    && left.semanticDigest === right.semanticDigest;
}
