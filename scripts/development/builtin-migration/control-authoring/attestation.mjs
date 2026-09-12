import { evidenceDigest } from "../evidence.mjs";
import { deepImmutable } from "../immutable.mjs";
import { digest, exact, kind, uniqueStrings } from "../schema.mjs";
import { controlCandidateInputDigests } from "./compose.mjs";

const PROGRAM = "RM-1064/C00-C07";

export function parseControlAttestation(value, candidate) {
  kind(value, 1, "runmat-builtin-migration-control-attestation", "control attestation");
  exact(value, ["schema_version", "kind", "authority", "program", "candidate_digest", "input_digests", "review", "digest"], "control attestation");
  if (value.authority !== "reviewer-authored-development-input" || value.program !== PROGRAM) {
    throw new Error("control attestation has invalid authority or program");
  }
  if (digest(value.candidate_digest, "control attestation candidate digest") !== candidate.digest) {
    throw new Error("control attestation does not bind the exact candidate");
  }
  if (JSON.stringify(value.input_digests) !== JSON.stringify(controlCandidateInputDigests(candidate))) {
    throw new Error("control attestation input digests differ from the control candidate");
  }
  exact(value.review, ["status", "evidence"], "control attestation review");
  if (value.review.status !== "reviewed") throw new Error("control attestation must be reviewed");
  uniqueStrings(value.review.evidence, "control attestation review evidence");
  digest(value.digest, "control attestation digest");
  const { digest: _ignored, ...payload } = value;
  if (evidenceDigest(payload) !== value.digest) throw new Error("control attestation digest mismatch");
  return deepImmutable(structuredClone(value));
}
