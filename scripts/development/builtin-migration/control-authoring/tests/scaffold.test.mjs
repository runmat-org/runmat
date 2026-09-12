import assert from "node:assert/strict";
import test, { afterEach } from "node:test";

import { buildControlDraft } from "../../control-draft.mjs";
import { evidenceDigest } from "../../evidence.mjs";
import { buildInventory } from "../../inventory.mjs";
import {
  REVISION, cleanupRepositoryFixtures, compiledInventoryFixture, controlledFixture,
  repositoryFixture, topologyFixture,
} from "../../tests/helpers.mjs";
import { composeTopologyCandidate } from "../../topology/compose.mjs";
import {
  candidateInputDigests,
  freezeReviewedTopology,
  parseReviewedTopology,
  reviewedTopologyView,
} from "../../topology/freeze.mjs";
import { fullTopologyChainFixture } from "../../topology/tests/full-chain-fixture.mjs";
import { buildControlOverlayScaffold, parseControlOverlayScaffold } from "../scaffold.mjs";

afterEach(cleanupRepositoryFixtures);

test("builds a deterministic topology-bound scaffold with no reviewed decisions", () => {
  const fixture = inputs();
  const first = buildControlOverlayScaffold(fixture.inventory, fixture.draft, fixture.topology);
  const second = buildControlOverlayScaffold(fixture.inventory, fixture.draft, fixture.topology);

  assert.deepEqual(first, second);
  assert.equal(first.bindings.inventory_digest, fixture.inventory.digest);
  assert.equal(first.bindings.control_draft_digest, fixture.draft.digest);
  assert.equal(first.bindings.reviewed_topology_digest, fixture.topology.digest);
  assert.equal(first.bundle_rows.length, 7);
  assert.equal(first.identity_rows.length, 9);
  assert.deepEqual(first.identity_rows.map((row) => row.identity), [
    "alpha", "beta", "betaaux", "datetime", "delta", "gamma", "linalg", "parallel", "shape",
  ]);

  const alphaSource = fixture.inventory.identities.find((row) => row.identity === "alpha");
  const alphaTopology = fixture.topology.identities.get("alpha");
  const alpha = first.identity_rows.find((row) => row.identity === "alpha");
  assert.equal(alpha.source_row_digest, evidenceDigest(alphaSource));
  assert.equal(alpha.topology_row_digest, evidenceDigest(alphaTopology));
  assert.deepEqual(alpha.candidate_paths.observed.runtime, ["crates/runmat-runtime/src/builtins/fixture/alpha.rs"]);
  assert.deepEqual(alpha.candidate_paths.topology_target_proposals, {
    catalog_package: "crates/runmat-builtins/src/catalog/entries/math/core/alpha/mod.rs",
    runtime_owner: "crates/runmat-runtime/src/builtins/math/core/alpha.rs",
    canonical_target: null,
    basis: "reviewed-topology-target-package-only",
  });
  assert.deepEqual(Object.keys(alpha.observations.source.authority_counts), [
    "catalog_entries", "catalog_constants", "legacy_functions", "legacy_documentation",
    "canonical_runtime_bindings", "canonical_runtime_constants", "implementation_provenance",
    "observed_runtime_registrations", "observed_wasm_registrations",
    "canonical_provider_records", "canonical_fusion_records",
    "observed_provider_paths", "observed_fusion_paths",
  ]);
  assert.deepEqual(Object.values(alpha.decisions), Array(9).fill({ status: "unresolved" }));

  const math = first.bundle_rows.find((row) => row.bundle_id === "c01-math-core");
  assert.deepEqual(math.observations.identities, ["alpha", "delta"]);
  assert.equal(math.source_rows_digest, evidenceDigest([
    fixture.inventory.identities.find((row) => row.identity === "alpha"),
    fixture.inventory.identities.find((row) => row.identity === "delta"),
  ]));
  assert.deepEqual(math.candidate_paths.observed.runtime, [
    "crates/runmat-runtime/src/builtins/fixture/alpha.rs",
    "crates/runmat-runtime/src/builtins/fixture/delta.rs",
  ]);
  assert.equal(math.review.status, "unreviewed");
  assert.ok(Object.values(math.decisions).every((decision) => decision.status === "unresolved"));
  assert.ok(Object.isFrozen(first));
  assert.deepEqual(parseControlOverlayScaffold(first, fixture.inventory, fixture.draft, fixture.topology), first);
});

test("keeps alias and internal target proposals separate from canonical authority", () => {
  const alias = dispositionScaffold({
    disposition: "alias", canonical: "canonical_target", domain: null, family: null, reason: null,
  }, {
    kind: "alias", canonical: "canonical_target", reason: null, source: "reviewed-input",
  });
  assert.deepEqual(alias.candidate_paths.topology_target_proposals, {
    catalog_package: null,
    runtime_owner: null,
    canonical_target: "canonical_target",
    basis: "reviewed-alias-target-no-copied-authority-proposal",
  });

  const internal = dispositionScaffold({
    disposition: "internal", canonical: null, domain: null, family: null,
    reason: "Implementation-only registration",
  }, {
    kind: "internal", canonical: null, reason: "Implementation-only registration",
    source: "reviewed-input",
  });
  assert.deepEqual(internal.candidate_paths.topology_target_proposals, {
    catalog_package: null,
    runtime_owner: null,
    canonical_target: null,
    basis: "reviewed-internal-identity-observed-paths-only",
  });
});

test("counts canonical authority independently from observed registrations", () => {
  const base = controlledFixture();
  const inventory = mutable(base.inventory);
  const source = inventory.identities[0];
  source.semantic_authority.runtime_bindings = [];
  source.semantic_authority.runtime_constants = ["constant_a", "constant_b"];
  source.registrations.wasm = [];
  reseal(inventory);
  const draft = buildControlDraft(inventory);
  const topology = topologyFixture(inventory, base.bundleId, base.id, {
    controlDraftDigest: draft.digest,
  });
  const counts = buildControlOverlayScaffold(inventory, draft, topology)
    .identity_rows[0].observations.source.authority_counts;
  assert.deepEqual(counts, {
    catalog_entries: 1,
    catalog_constants: 0,
    legacy_functions: 0,
    legacy_documentation: 0,
    canonical_runtime_bindings: 0,
    canonical_runtime_constants: 2,
    implementation_provenance: 1,
    observed_runtime_registrations: 1,
    observed_wasm_registrations: 0,
    canonical_provider_records: 0,
    canonical_fusion_records: 0,
    observed_provider_paths: 0,
    observed_fusion_paths: 0,
  });
});

test("parsing reconstructs the scaffold and rejects authority, review, and decision forgery", () => {
  const fixture = inputs();
  const scaffold = buildControlOverlayScaffold(fixture.inventory, fixture.draft, fixture.topology);

  const authority = mutable(scaffold);
  authority.authority = "reviewed-development-control";
  reseal(authority);
  assert.throws(
    () => parseControlOverlayScaffold(authority, fixture.inventory, fixture.draft, fixture.topology),
    /cannot claim or accept reviewed authority/,
  );

  const review = mutable(scaffold);
  review.identity_rows[0].review = { status: "reviewed", evidence: ["fabricated"] };
  reseal(review);
  assert.throws(
    () => parseControlOverlayScaffold(review, fixture.inventory, fixture.draft, fixture.topology),
    /differs from deterministic reconstruction/,
  );

  const decision = mutable(scaffold);
  decision.bundle_rows[0].decisions.owner_role = { status: "resolved", value: "migrator" };
  reseal(decision);
  assert.throws(
    () => parseControlOverlayScaffold(decision, fixture.inventory, fixture.draft, fixture.topology),
    /differs from deterministic reconstruction/,
  );
});

test("requires the exact inventory, pre-topology draft, and branded reviewed topology", () => {
  const fixture = inputs();
  assert.throws(
    () => buildControlOverlayScaffold(fixture.inventory, fixture.draft, {
      ...fixture.topology,
      identities: fixture.topology.identities,
      bundles: fixture.topology.bundles,
    }),
    /deterministically validated topology view/,
  );

  const driftedInventory = mutable(fixture.inventory);
  driftedInventory.source.dirty = true;
  reseal(driftedInventory);
  assert.throws(
    () => buildControlOverlayScaffold(driftedInventory, fixture.draft, fixture.topology),
    /baseline differs from the inventory|does not bind the exact inventory/,
  );
});

test("carries exact baseline evidence digests and topology target proposals without resolving them", () => {
  const base = controlledFixture();
  const draft = buildControlDraft(base.inventory);
  const topology = topologyFixture(base.inventory, base.bundleId, base.id, {
    controlDraftDigest: draft.digest,
  });
  const scaffold = buildControlOverlayScaffold(base.inventory, draft, topology);
  const row = scaffold.identity_rows[0];
  const sourceFiles = new Map(base.inventory.source.files.map((file) => [file.path, file.content_digest]));

  assert.ok(row.observations.source.baseline_evidence_candidates.length > 0);
  for (const evidence of row.observations.source.baseline_evidence_candidates) {
    assert.equal(evidence.content_digest, sourceFiles.get(evidence.path));
  }
  assert.equal(row.candidate_paths.topology_target_proposals.runtime_owner,
    "crates/runmat-runtime/src/builtins/math/basic/foo.rs");
  assert.equal(row.decisions.runtime_owner.status, "unresolved");
  assert.equal(row.decisions.baseline_evidence.status, "unresolved");
});

function inputs() {
  const chain = fullTopologyChainFixture();
  const candidate = composeTopologyCandidate(chain);
  const attestation = {
    schema_version: 1,
    kind: "runmat-builtin-topology-attestation",
    authority: "reviewer-authored-development-input",
    program: "RM-1064/C00-C07",
    candidate_digest: candidate.digest,
    input_digests: candidateInputDigests(candidate),
    review: { status: "reviewed", evidence: ["fixture root review"] },
  };
  const frozen = freezeReviewedTopology(candidate, attestation, candidate);
  const topology = reviewedTopologyView(parseReviewedTopology(frozen, candidate, attestation, candidate));
  return { inventory: chain.baselineInventory, draft: chain.controlDraft, topology };
}

function dispositionScaffold(disposition, topologyDisposition) {
  const id = "foo";
  const repository = repositoryFixture({ identity: id });
  const compiledInventory = compiledInventoryFixture(id);
  const dispositions = {
    schema_version: 1,
    kind: "runmat-builtin-dispositions",
    authority: "review-input-only",
    identities: {
      [id]: {
        ...disposition,
        review: { status: "reviewed", evidence: ["fixture disposition review"] },
      },
    },
  };
  const inventory = buildInventory(repository, dispositions, { revision: REVISION, compiledInventory });
  const draft = buildControlDraft(inventory);
  const topology = topologyFixture(inventory, "disposition-fixture", id, {
    controlDraftDigest: draft.digest,
    identity: {
      identity: id,
      bundle_id: "disposition-fixture",
      cohort: "C01",
      domain: "math",
      family: "basic",
      disposition: topologyDisposition,
      classification: "preserved",
      evidence: ["fixture topology review"],
    },
  });
  return buildControlOverlayScaffold(inventory, draft, topology).identity_rows[0];
}

function mutable(value) {
  return structuredClone(value);
}

function reseal(value) {
  delete value.digest;
  value.digest = evidenceDigest(value);
}
