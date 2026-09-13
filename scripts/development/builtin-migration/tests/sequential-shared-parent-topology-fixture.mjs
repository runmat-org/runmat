import { evidenceDigest } from "../evidence.mjs";
import {
  candidateInputDigests, freezeReviewedTopology, parseReviewedTopology,
  reviewedTopologyView,
} from "../topology/freeze.mjs";
import {
  SEQUENTIAL_BUNDLES, SEQUENTIAL_IDENTITIES,
} from "./sequential-shared-parent-definition-fixture.mjs";

export function sequentialReviewedTopology(inventory, controlDraftDigest) {
  const bundles = Object.fromEntries(SEQUENTIAL_BUNDLES.map((bundleId, index) => {
    const identity = SEQUENTIAL_IDENTITIES[index];
    return [bundleId, topologyBundle(bundleId, identity)];
  }));
  const identities = Object.fromEntries(SEQUENTIAL_IDENTITIES.map((identity, index) => [
    identity, topologyIdentity(identity, SEQUENTIAL_BUNDLES[index]),
  ]));
  const payload = {
    schema_version: 1,
    kind: "runmat-builtin-migration-topology-candidate",
    authority: "composed-unreviewed-candidate",
    program: "RM-1064/C00-C07",
    baseline: {
      revision: inventory.source.revision,
      inventory_digest: inventory.digest,
      component_graph_digest: `sha256:${"e".repeat(64)}`,
      control_draft_digest: controlDraftDigest,
    },
    inputs: {
      c01_c03_review: `sha256:${"1".repeat(64)}`,
      c04_c05_review: `sha256:${"2".repeat(64)}`,
      c06_c07_review: `sha256:${"3".repeat(64)}`,
      reconciliation: `sha256:${"4".repeat(64)}`,
      stability_corrections: `sha256:${"5".repeat(64)}`,
    },
    bundles,
    identities,
    summary: { components: 2, identities: 2, bundles: 2, cohorts: ["C01"], target_packages: 1 },
    review: { status: "unreviewed", evidence: [] },
  };
  const candidate = { ...payload, digest: evidenceDigest(payload) };
  const attestation = {
    schema_version: 1,
    kind: "runmat-builtin-topology-attestation",
    authority: "reviewer-authored-development-input",
    program: "RM-1064/C00-C07",
    candidate_digest: candidate.digest,
    input_digests: candidateInputDigests(candidate),
    review: { status: "reviewed", evidence: ["sequential fixture topology review"] },
  };
  return reviewedTopologyView(parseReviewedTopology(
    freezeReviewedTopology(candidate, attestation, candidate), candidate, attestation, candidate,
  ));
}

function topologyBundle(bundleId, identity) {
  return {
    id: bundleId,
    cohort: "C01",
    authority_components: [`component-${identity}`],
    identities: [identity],
    atomic_reason: "One runtime and catalog contract",
    composition: {
      kind: "single-component",
      target_packages: [{ domain: "math", family: "basic" }],
      authored_write_set: [],
      shared_authority_sources: [],
      evidence: ["sequential fixture topology review"],
    },
    identity_targets: [{
      identity, domain: "math", family: "basic", classification: "preserved",
      evidence: ["sequential fixture topology review"],
    }],
    review: { status: "reviewed", evidence: ["sequential fixture topology review"] },
  };
}

function topologyIdentity(identity, bundleId) {
  return {
    identity,
    bundle_id: bundleId,
    cohort: "C01",
    domain: "math",
    family: "basic",
    disposition: { kind: "canonical", canonical: null, reason: null, source: "reviewed-input" },
    classification: "preserved",
    evidence: ["sequential fixture topology review"],
  };
}
