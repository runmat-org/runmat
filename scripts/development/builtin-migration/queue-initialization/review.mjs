import {
  assertAuthorityLoadSession, assertSessionArtifact, canonicalAuthorityPath, loadedJsonValue,
  loadJsonArtifact, revalidateObservedArtifacts, withAuthorityTraversal,
} from "../authority-loading/index.mjs";
import { assertValidatedControl } from "../control.mjs";
import { deepImmutable } from "../immutable.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { digest, exact, kind } from "../schema.mjs";
import { parseCanonicalReviewedEvidence } from "../reviewed-evidence.mjs";

const REVIEWS = new WeakMap();
const SESSION_REVIEWS = new WeakMap();

export function loadInitialQueueReview({ session: sessionValue, reference, control: controlValue }) {
  const session = assertAuthorityLoadSession(sessionValue);
  const control = assertValidatedControl(controlValue);
  const parsedReference = parseReference(reference);
  const cached = reviewsFor(session).get(parsedReference.path);
  if (cached) {
    if (cached.reference.digest !== parsedReference.digest) {
      throw new Error(`${parsedReference.path}: initial queue review path was rebound`);
    }
    revalidateObservedArtifacts(session);
    return assertLoadedInitialQueueReview(cached, { session, control });
  }
  return withAuthorityTraversal(session, {
    domain: "initial-queue-review", path: parsedReference.path,
    semanticDigest: parsedReference.digest,
  }, () => {
    const artifact = loadJsonArtifact(session, parsedReference.path, "initial queue review");
    assertSessionArtifact(session, artifact, parsedReference.path);
    const value = parseInitialQueueReview(loadedJsonValue(artifact, session.root));
    if (value.digest !== parsedReference.digest) {
      throw new Error("initial queue review semantic digest mismatch");
    }
    if (value.control_manifest_digest !== control.digest) {
      throw new Error("initial queue review belongs to another control manifest");
    }
    revalidateObservedArtifacts(session);
    const result = Object.freeze({
      ...deepImmutable({ value, reference: parsedReference }), artifact, control,
    });
    REVIEWS.set(result, { session, control });
    reviewsFor(session).set(parsedReference.path, result);
    return result;
  });
}

export function assertLoadedInitialQueueReview(value, { session, control }) {
  const authoritySession = assertAuthorityLoadSession(session);
  const validatedControl = assertValidatedControl(control);
  const record = REVIEWS.get(value);
  if (!record || record.session !== authoritySession || record.control !== validatedControl) {
    throw new Error("operation requires the exact loaded initial queue review");
  }
  assertSessionArtifact(authoritySession, value.artifact, value.reference.path);
  return value;
}

export function parseInitialQueueReview(value) {
  kind(value, 1, "runmat-builtin-migration-initial-queue-review", "initial queue review");
  exact(value, [
    "schema_version", "kind", "authority", "control_manifest_digest", "review", "digest",
  ], "initial queue review");
  if (value.authority !== "reviewer-authored-development-input") {
    throw new Error("initial queue review has invalid authority");
  }
  digest(value.control_manifest_digest, "initial queue review control digest");
  parseCanonicalReviewedEvidence(value.review, "initial queue review evidence");
  const { digest: observed, ...payload } = value;
  digest(observed, "initial queue review digest");
  if (evidenceDigest(payload) !== observed) throw new Error("initial queue review digest mismatch");
  return deepImmutable(value);
}

function parseReference(value) {
  exact(value, ["path", "digest"], "initial queue review reference");
  return Object.freeze({
    path: canonicalAuthorityPath(value.path, "initial queue review reference path"),
    digest: digest(value.digest, "initial queue review reference digest"),
  });
}

function reviewsFor(session) {
  let reviews = SESSION_REVIEWS.get(session);
  if (!reviews) {
    reviews = new Map();
    SESSION_REVIEWS.set(session, reviews);
  }
  return reviews;
}
