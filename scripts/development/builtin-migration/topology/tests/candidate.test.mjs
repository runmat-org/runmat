import assert from "node:assert/strict";
import test from "node:test";

import { evidenceDigest } from "../../evidence.mjs";
import { parseTopologyCandidate } from "../candidate.mjs";

test("candidate parsing requires deterministic recomposition", () => {
  const candidate = fixture();
  assert.throws(() => parseTopologyCandidate(candidate), /requires deterministic recomposition/);
  assert.equal(parseTopologyCandidate(candidate, structuredClone(candidate)), candidate);
});

test("candidate parsing rejects semantic edits and arbitrary substituted candidates", () => {
  const expected = fixture();
  const edited = structuredClone(expected);
  edited.summary.identities = 2;
  assert.throws(() => parseTopologyCandidate(edited, expected), /digest mismatch/);

  const substituted = structuredClone(expected);
  substituted.summary.identities = 2;
  const { digest: _ignored, ...payload } = substituted;
  substituted.digest = evidenceDigest(payload);
  assert.throws(() => parseTopologyCandidate(substituted, expected), /differs from deterministic composition/);
});

function fixture() {
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

function digest(character) {
  return `sha256:${character.repeat(64)}`;
}
