import { compareCodePoint } from "../constants.mjs";
import { digest } from "../schema.mjs";
import {
  assertLoadedJsonArtifact, canonicalAuthorityPath, observeJsonArtifact,
  revalidateJsonArtifact,
} from "./artifact.mjs";
import { assertAuthorityRoot, revalidateAuthorityRoot } from "./root.mjs";

const SESSIONS = new WeakMap();

export function openAuthorityLoadSession(rootValue) {
  const root = assertAuthorityRoot(rootValue);
  const session = Object.freeze({ root });
  SESSIONS.set(session, {
    root,
    artifacts: new Map(),
    activePaths: new Set(),
    activeSemanticAuthorities: new Set(),
  });
  return session;
}

export function assertAuthorityLoadSession(value) {
  if (!SESSIONS.has(value)) {
    throw new Error("operation requires an exact authority load session");
  }
  return value;
}

export function loadJsonArtifact(sessionValue, relativePath, label = "authority artifact") {
  const session = sessionRecord(sessionValue);
  const relative = canonicalAuthorityPath(relativePath, `${label} path`);
  const existing = session.artifacts.get(relative);
  if (existing) return existing;
  const artifact = observeJsonArtifact(session.root, relative, label);
  session.artifacts.set(relative, artifact);
  return artifact;
}

export function withAuthorityTraversal(
  sessionValue, { domain, path, semanticDigest }, operation,
) {
  const session = sessionRecord(sessionValue);
  const domainId = authorityDomain(domain);
  const relative = canonicalAuthorityPath(path, "authority traversal path");
  const authorityDigest = digest(semanticDigest, "authority traversal semantic digest");
  if (typeof operation !== "function") throw new TypeError("authority traversal requires an operation");
  const semanticKey = `${domainId}\0${authorityDigest}`;
  if (session.activePaths.has(relative)) {
    throw new Error(`${relative}: authority path cycle detected`);
  }
  if (session.activeSemanticAuthorities.has(semanticKey)) {
    throw new Error(`${domainId}: semantic authority cycle detected`);
  }
  session.activePaths.add(relative);
  session.activeSemanticAuthorities.add(semanticKey);
  try {
    return operation();
  } finally {
    session.activePaths.delete(relative);
    session.activeSemanticAuthorities.delete(semanticKey);
  }
}

export function revalidateObservedArtifacts(sessionValue) {
  const session = sessionRecord(sessionValue);
  revalidateAuthorityRoot(session.root);
  const paths = [...session.artifacts.keys()].sort(compareCodePoint);
  for (const relative of paths) {
    revalidateJsonArtifact(session.artifacts.get(relative), session.root);
  }
  return Object.freeze({ artifacts: paths.length });
}

export function assertSessionArtifact(sessionValue, artifactValue, relativePath = null) {
  const session = sessionRecord(sessionValue);
  const artifact = assertLoadedJsonArtifact(artifactValue, session.root, relativePath);
  if (session.artifacts.get(artifact.path) !== artifact) {
    throw new Error("loaded authority artifact does not belong to this session");
  }
  return artifact;
}

function sessionRecord(value) {
  assertAuthorityLoadSession(value);
  return SESSIONS.get(value);
}

function authorityDomain(value) {
  if (typeof value !== "string" || !/^[a-z][a-z0-9-]*$/.test(value)) {
    throw new Error("authority traversal domain must be a stable lowercase id");
  }
  return value;
}
