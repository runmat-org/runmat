import { evidenceDigest } from "../evidence.mjs";
import { deepImmutable } from "../immutable.mjs";
import { parseControlAttestation } from "./attestation.mjs";
import { parseControlCandidate } from "./compose.mjs";

const VALIDATED_CONTROL_REVIEWS = new WeakSet();

export function validateControlReviewChain(candidateValue, attestationValue, expectedCandidate) {
  const candidate = parseControlCandidate(candidateValue, expectedCandidate);
  const attestation = parseControlAttestation(attestationValue, candidate);
  const payload = {
    schema_version: 2,
    kind: "runmat-builtin-migration-control-manifest",
    authority: "reviewed-development-control",
    program: candidate.program,
    inputs: structuredClone(candidate.inputs),
    topology_digest: candidate.inputs.reviewed_topology_digest,
    candidate_digest: candidate.digest,
    attestation_digest: attestation.digest,
    baseline_context: structuredClone(candidate.baseline_context),
    cohorts: structuredClone(candidate.cohorts),
    bundle_controls: structuredClone(candidate.bundle_controls),
    identity_controls: structuredClone(candidate.identity_controls),
    migration_findings: structuredClone(candidate.migration_findings),
    exception_manifest: structuredClone(candidate.exception_manifest),
    storage_policy: structuredClone(candidate.storage_policy),
    review: structuredClone(attestation.review),
  };
  const controlValue = { ...payload, digest: evidenceDigest(payload) };
  const context = deepImmutable({
    candidate,
    attestation,
    controlValue,
    digest: evidenceDigest({ candidate_digest: candidate.digest, attestation_digest: attestation.digest }),
  });
  VALIDATED_CONTROL_REVIEWS.add(context);
  return context;
}

export function assertValidatedControlReview(value) {
  if (!VALIDATED_CONTROL_REVIEWS.has(value)) {
    throw new Error("operation requires the exact deterministically validated control review chain");
  }
  return value;
}
