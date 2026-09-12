import { evidenceDigest } from "../evidence.mjs";
import { parseControlAttestation } from "./attestation.mjs";
import { assertValidatedControlCandidate, controlCandidateInputDigests } from "./compose.mjs";

export function buildControlAttestationTemplate(candidate) {
  assertValidatedControlCandidate(candidate);
  return {
    schema_version: 1,
    kind: "runmat-builtin-migration-control-attestation",
    authority: "reviewer-authored-development-input",
    program: "RM-1064/C00-C07",
    candidate_digest: candidate.digest,
    input_digests: controlCandidateInputDigests(candidate),
    review: { status: "unreviewed", evidence: [] },
  };
}

export function sealControlAttestation(reviewValue, candidate) {
  assertValidatedControlCandidate(candidate);
  if (Object.hasOwn(reviewValue, "digest")) throw new Error("control attestation review must not supply its own digest");
  const value = { ...structuredClone(reviewValue), digest: evidenceDigest(reviewValue) };
  parseControlAttestation(value, candidate);
  return value;
}
