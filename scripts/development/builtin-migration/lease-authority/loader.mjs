import {
  assertAuthorityLoadSession, assertSessionArtifact, loadedJsonValue,
  loadJsonArtifact, revalidateObservedArtifacts, withAuthorityTraversal,
} from "../authority-loading/index.mjs";
import { assertValidatedControl } from "../control.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { deepImmutable } from "../immutable.mjs";
import { assertValidatedLease, parseLease } from "../lease.mjs";
import { parseLeaseAuthorityReference } from "./reference.mjs";

const LOADED_LEASE_AUTHORITIES = new WeakMap();
const SESSION_AUTHORITIES = new WeakMap();

export function loadLeaseAuthority(sessionValue, referenceValue, controlValue, repository) {
  const session = assertAuthorityLoadSession(sessionValue);
  const control = assertValidatedControl(controlValue);
  const reference = parseLeaseAuthorityReference(referenceValue);
  const authorities = sessionAuthorities(session);
  const key = `${reference.path}\0${reference.digest}`;
  const existing = authorities.get(key);
  if (existing) {
    assertLoadedLeaseAuthority(existing, session, control);
    return existing;
  }
  return withAuthorityTraversal(session, {
    domain: "authored-lease",
    path: reference.path,
    semanticDigest: reference.digest,
  }, () => {
    const artifact = loadJsonArtifact(session, reference.path, "authored lease authority");
    assertSessionArtifact(session, artifact, reference.path);
    const serialized = loadedJsonValue(artifact, session.root);
    if (evidenceDigest(serialized) !== reference.digest) {
      throw new Error("authored lease authority semantic digest mismatch");
    }
    const lease = parseLease(serialized, control, repository);
    const immutable = deepImmutable({ reference, semanticDigest: reference.digest });
    const result = Object.freeze({ ...immutable, artifact, lease, control });
    LOADED_LEASE_AUTHORITIES.set(result, { session, control });
    authorities.set(key, result);
    return result;
  });
}

export function assertLoadedLeaseAuthority(value, sessionValue, controlValue) {
  const session = assertAuthorityLoadSession(sessionValue);
  const control = assertValidatedControl(controlValue);
  const record = LOADED_LEASE_AUTHORITIES.get(value);
  if (!record || record.session !== session || record.control !== control) {
    throw new Error("operation requires the exact loaded lease authority for this session and control");
  }
  assertSessionArtifact(session, value.artifact, value.reference.path);
  assertValidatedLease(value.lease, control);
  return value;
}

export function revalidateLeaseAuthority(value, sessionValue, controlValue) {
  const authority = assertLoadedLeaseAuthority(value, sessionValue, controlValue);
  revalidateObservedArtifacts(sessionValue);
  return authority;
}

function sessionAuthorities(session) {
  let authorities = SESSION_AUTHORITIES.get(session);
  if (!authorities) {
    authorities = new Map();
    SESSION_AUTHORITIES.set(session, authorities);
  }
  return authorities;
}
