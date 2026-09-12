import assert from "node:assert/strict";
import test from "node:test";

import { evidenceDigest } from "../../evidence.mjs";
import { candidateInputDigests, freezeReviewedTopology, parseReviewedTopology } from "../freeze.mjs";

test("attestation promotes only the exact content-bound candidate", () => {
  const candidate = candidateFixture();
  const attestation = attestationFixture(candidate);
  const frozen = freezeReviewedTopology(candidate, attestation, candidate);
  assert.equal(frozen.candidate_digest, candidate.digest);
  assert.equal(frozen.attestation_digest, evidenceDigest(attestation));
  assert.deepEqual(parseReviewedTopology(frozen, candidate, attestation, candidate), frozen);
});

test("attestation rejects candidate and transitive-input drift", () => {
  const candidate = candidateFixture();
  assert.throws(() => freezeReviewedTopology(candidate, attestationFixture(candidate)), /requires deterministic recomposition/);
  const wrongCandidate = attestationFixture(candidate);
  wrongCandidate.candidate_digest = digest("f");
  assert.throws(() => freezeReviewedTopology(candidate, wrongCandidate, candidate), /exact candidate/);

  const staleInput = attestationFixture(candidate);
  staleInput.input_digests.component_graph = digest("f");
  assert.throws(() => freezeReviewedTopology(candidate, staleInput, candidate), /component_graph digest differs/);
});

test("frozen topology rejects mutation behind its digest", () => {
  const candidate = candidateFixture();
  const frozen = freezeReviewedTopology(candidate, attestationFixture(candidate), candidate);
  frozen.summary.identities = 2;
  assert.throws(() => parseReviewedTopology(frozen, candidate, attestationFixture(candidate), candidate), /digest mismatch/);
});

test("frozen topology validation requires and matches deterministic source inputs", () => {
  const candidate = candidateFixture();
  const attestation = attestationFixture(candidate);
  const frozen = freezeReviewedTopology(candidate, attestation, candidate);
  assert.throws(() => parseReviewedTopology(frozen), /requires the candidate, attestation, and deterministic recomposition/);

  const fabricatedPayload = { ...structuredClone(frozen), authority: "reviewed-development-topology" };
  fabricatedPayload.summary.identities = 2;
  delete fabricatedPayload.digest;
  const fabricated = { ...fabricatedPayload, digest: evidenceDigest(fabricatedPayload) };
  assert.throws(
    () => parseReviewedTopology(fabricated, candidate, attestation, candidate),
    /deterministically reconstructed topology/,
  );
});

function candidateFixture() {
  const payload = {
    schema_version: 1,
    kind: "runmat-builtin-migration-topology-candidate",
    authority: "composed-unreviewed-candidate",
    program: "RM-1064/C00-C07",
    baseline: {
      revision: `git:${"a".repeat(40)}`,
      inventory_digest: digest("1"),
      component_graph_digest: digest("2"),
      control_draft_digest: digest("3"),
    },
    inputs: {
      c01_c03_review: digest("4"),
      c04_c05_review: digest("5"),
      c06_c07_review: digest("6"),
      reconciliation: digest("7"),
      stability_corrections: digest("8"),
    },
    bundles: {},
    identities: {},
    summary: {},
    review: { status: "unreviewed", evidence: [] },
  };
  return { ...payload, digest: evidenceDigest(payload) };
}

function attestationFixture(candidate) {
  return {
    schema_version: 1,
    kind: "runmat-builtin-topology-attestation",
    authority: "reviewer-authored-development-input",
    program: "RM-1064/C00-C07",
    candidate_digest: candidate.digest,
    input_digests: candidateInputDigests(candidate),
    review: { status: "reviewed", evidence: ["exact root review"] },
  };
}

function digest(character) {
  return `sha256:${character.repeat(64)}`;
}
