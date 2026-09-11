import { evidenceDigest } from "../evidence.mjs";
import { digest, exact, kind, object } from "../schema.mjs";
import { parseTopologyCandidate } from "./candidate.mjs";
import { parseReviewedEvidence, TOPOLOGY_PROGRAM } from "./schema.mjs";

export const TOPOLOGY_ATTESTATION_KIND = "runmat-builtin-topology-attestation";
export const REVIEWED_TOPOLOGY_KIND = "runmat-builtin-migration-reviewed-topology";

const INPUT_DIGEST_FIELDS = Object.freeze([
  "inventory",
  "component_graph",
  "control_draft",
  "c01_c03_review",
  "c04_c05_review",
  "c06_c07_review",
  "reconciliation",
  "stability_corrections",
]);

export function freezeReviewedTopology(candidateValue, attestationValue, expectedCandidate) {
  if (!expectedCandidate) throw new Error("topology freeze requires deterministic recomposition of the expected candidate");
  const candidate = parseTopologyCandidate(candidateValue, expectedCandidate);
  const attestation = parseTopologyAttestation(attestationValue, candidate);
  const payload = {
    schema_version: 1,
    kind: REVIEWED_TOPOLOGY_KIND,
    authority: "reviewed-development-topology",
    program: TOPOLOGY_PROGRAM,
    candidate_digest: candidate.digest,
    attestation_digest: evidenceDigest(attestation),
    baseline: structuredClone(candidate.baseline),
    inputs: structuredClone(candidate.inputs),
    bundles: structuredClone(candidate.bundles),
    identities: structuredClone(candidate.identities),
    summary: structuredClone(candidate.summary),
    review: structuredClone(attestation.review),
  };
  return { ...payload, digest: evidenceDigest(payload) };
}

export function parseTopologyAttestation(value, candidate) {
  kind(value, 1, TOPOLOGY_ATTESTATION_KIND, "topology attestation");
  exact(value, ["schema_version", "kind", "authority", "program", "candidate_digest", "input_digests", "review"], "topology attestation");
  if (value.authority !== "reviewer-authored-development-input" || value.program !== TOPOLOGY_PROGRAM) {
    throw new Error("topology attestation has invalid authority or program");
  }
  digest(value.candidate_digest, "topology attestation candidate digest");
  if (value.candidate_digest !== candidate.digest) throw new Error("topology attestation does not bind the exact candidate");
  exact(value.input_digests, INPUT_DIGEST_FIELDS, "topology attestation input digests");
  const expected = candidateInputDigests(candidate);
  for (const field of INPUT_DIGEST_FIELDS) {
    digest(value.input_digests[field], `topology attestation ${field} digest`);
    if (value.input_digests[field] !== expected[field]) {
      throw new Error(`topology attestation ${field} digest differs from the candidate`);
    }
  }
  parseReviewedEvidence(value.review, "topology attestation review");
  return value;
}

export function parseReviewedTopology(value, candidateValue, attestationValue, expectedCandidate) {
  if (!candidateValue || !attestationValue || !expectedCandidate) {
    throw new Error("reviewed topology validation requires the candidate, attestation, and deterministic recomposition");
  }
  kind(value, 1, REVIEWED_TOPOLOGY_KIND, "reviewed topology");
  exact(value, ["schema_version", "kind", "authority", "program", "candidate_digest", "attestation_digest", "baseline", "inputs", "bundles", "identities", "summary", "review", "digest"], "reviewed topology");
  if (value.authority !== "reviewed-development-topology" || value.program !== TOPOLOGY_PROGRAM) {
    throw new Error("reviewed topology has invalid authority or program");
  }
  digest(value.candidate_digest, "reviewed topology candidate digest");
  digest(value.attestation_digest, "reviewed topology attestation digest");
  object(value.baseline, "reviewed topology baseline");
  object(value.inputs, "reviewed topology inputs");
  object(value.bundles, "reviewed topology bundles");
  object(value.identities, "reviewed topology identities");
  object(value.summary, "reviewed topology summary");
  parseReviewedEvidence(value.review, "reviewed topology review");
  digest(value.digest, "reviewed topology digest");
  const { digest: _ignored, ...payload } = value;
  if (evidenceDigest(payload) !== value.digest) throw new Error("reviewed topology digest mismatch");
  const expected = freezeReviewedTopology(candidateValue, attestationValue, expectedCandidate);
  if (JSON.stringify(value) !== JSON.stringify(expected)) {
    throw new Error("reviewed topology differs from the deterministically reconstructed topology");
  }
  return value;
}

export function candidateInputDigests(candidate) {
  return {
    inventory: candidate.baseline.inventory_digest,
    component_graph: candidate.baseline.component_graph_digest,
    control_draft: candidate.baseline.control_draft_digest,
    c01_c03_review: candidate.inputs.c01_c03_review,
    c04_c05_review: candidate.inputs.c04_c05_review,
    c06_c07_review: candidate.inputs.c06_c07_review,
    reconciliation: candidate.inputs.reconciliation,
    stability_corrections: candidate.inputs.stability_corrections,
  };
}
