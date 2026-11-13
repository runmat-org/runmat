import assert from "node:assert/strict";
import test from "node:test";
import { auditMigration } from "../audit.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { prepareIdentity } from "../prepare.mjs";
import { sealBundle } from "../seal.mjs";
import { verifyBatch } from "../verify.mjs";
import { parseVerificationResult } from "../verification-result.mjs";
import { cleanupRepositoryFixtures, controlledFixture, digestReference, gate } from "./helpers.mjs";
import { createTemporaryDirectory } from "./temporary-directories.mjs";

test.afterEach(cleanupRepositoryFixtures);

function evidence() {
  const fixture = controlledFixture();
  const prepared = prepareIdentity(fixture.repository, fixture.inventory, fixture.control, fixture.lease, "foo", createTemporaryDirectory("verify-"));
  const gates = ["catalog-contract", "runtime-binding", "documentation-cutover", "native-link", "architecture", "focused-tests", "format-diff", "strict-clippy"].map((name) => gate(fixture, name));
  const batch = { schema_version: 1, kind: "runmat-builtin-migration-batch", identities: ["foo"] };
  const audit = auditMigration(fixture.repository, fixture.inventory, fixture.inventory, fixture.inventory, fixture.control, fixture.lease, batch, { artifact_id: "audit-foo", authored_revision: fixture.inventory.source.revision, prepare_results: [prepared], source_dispositions: [], gate_results: gates });
  const auditReference = digestReference("audit.json", "audit-foo", audit);
  const gateReferences = gates.map((value) => digestReference(`${value.artifact_id}.json`, value.artifact_id, value));
  const manifest = {
    schema_version: 8, kind: "runmat-builtin-migration-verification-manifest", authority: "reviewed-verification-request",
    batch: { artifact_id: "verify-foo", source_revision: fixture.inventory.source.revision, source_digest: fixture.inventory.source.digest, control_baseline_inventory_digest: fixture.inventory.digest, lease_base_inventory_digest: fixture.inventory.digest, subject_inventory_digest: fixture.inventory.digest, control_manifest_digest: fixture.control.digest, bundle_id: fixture.bundleId, lease_id: fixture.lease.value.lease_id, lease_digest: fixture.lease.value.digest, queue_phase: fixture.lease.value.queue_phase, accepted_seals: fixture.lease.value.accepted_seals, accepted_seal_set_digest: fixture.lease.value.accepted_seal_set_digest, barrier_seals: fixture.lease.value.barrier_seals, barrier_seal_set_digest: fixture.lease.value.barrier_seal_set_digest, identities: ["foo"], phases: audit.phases },
    audit: auditReference, gate_results: gateReferences,
    expectations: [{ identity: "foo", required_gates: ["architecture", "catalog-contract", "documentation-cutover", "focused-tests", "format-diff", "native-link", "runtime-binding", "strict-clippy"] }],
  };
  const loadedAudit = { reference: auditReference, value: audit };
  const loadedGates = gateReferences.map((reference, index) => ({ reference, value: gates[index] }));
  return { fixture, audit, gates, manifest, loadedAudit, loadedGates };
}

test("verification v8 passes only exact, content-addressed audit and gate evidence", () => {
  const input = evidence();
  const result = verifyBatch(input.manifest, input.loadedAudit, input.loadedGates, input.fixture.control);
  assert.equal(result.schema_version, 8);
  assert.equal(result.result, "pass");
});

test("verification rejects missing, duplicate, stale, and mutated evidence", () => {
  const input = evidence();
  assert.equal(verifyBatch(input.manifest, input.loadedAudit, input.loadedGates.slice(1), input.fixture.control).result, "fail");
  const duplicate = structuredClone(input.manifest); duplicate.gate_results.push(duplicate.gate_results[0]);
  assert.throws(() => verifyBatch(duplicate, input.loadedAudit, input.loadedGates, input.fixture.control), /unique/);
  const mutated = structuredClone(input.loadedGates); mutated[0].value.checks[0].evidence_digest = `sha256:${"f".repeat(64)}`;
  assert.ok(verifyBatch(input.manifest, input.loadedAudit, mutated, input.fixture.control).global_failures.some((entry) => entry.code === "gate-digest-mismatch"));
  const future = structuredClone(input.manifest); future.schema_version = 9;
  assert.throws(() => verifyBatch(future, input.loadedAudit, input.loadedGates, input.fixture.control), /schema_version 8/);
  const legacy = structuredClone(input.manifest); legacy.schema_version = 7;
  assert.throws(() => verifyBatch(legacy, input.loadedAudit, input.loadedGates, input.fixture.control), /schema_version 8/);
  const inconsistentAudit = structuredClone(input.loadedAudit);
  inconsistentAudit.value.identities[0].failures.push({ code: "forged", detail: "ignored" });
  inconsistentAudit.reference.digest = evidenceDigest(inconsistentAudit.value);
  const inconsistentManifest = structuredClone(input.manifest);
  inconsistentManifest.audit.digest = inconsistentAudit.reference.digest;
  assert.ok(verifyBatch(inconsistentManifest, inconsistentAudit, input.loadedGates, input.fixture.control).global_failures.some((entry) => entry.code === "audit-invalid"));
});

test("verification rejects a different lease payload that reuses the same lease id", () => {
  const input = evidence();
  const forged = structuredClone(input.manifest);
  forged.batch.lease_digest = `sha256:${"e".repeat(64)}`;
  const result = verifyBatch(forged, input.loadedAudit, input.loadedGates, input.fixture.control);
  assert.ok(result.global_failures.some((entry) => entry.code === "audit-invalid"));
  assert.ok(result.global_failures.some((entry) => entry.code === "gate-invalid"));
});

test("closed verification results bind reviewed coverage to their exact gate inputs", () => {
  const input = evidence();
  const result = verifyBatch(
    input.manifest, input.loadedAudit, input.loadedGates, input.fixture.control,
  );
  assert.doesNotThrow(() => parseVerificationResult(result, input.fixture.control));

  const changedReference = structuredClone(result);
  changedReference.ordinary_gate_proof.gate_results[0].path = "gates/substitute.json";
  const { digest: _oldProofDigest, ...proofPayload } = changedReference.ordinary_gate_proof;
  changedReference.ordinary_gate_proof.digest = evidenceDigest(proofPayload);
  assert.throws(
    () => parseVerificationResult(changedReference, input.fixture.control),
    /gate proof references evidence outside the verification inputs/,
  );

  const changedCoverage = structuredClone(result);
  changedCoverage.identity_results[0].required_gates.pop();
  assert.throws(
    () => parseVerificationResult(changedCoverage, input.fixture.control),
    /required gates differ from reviewed control/,
  );

  const changedPhase = structuredClone(result);
  changedPhase.queue_phase = "production";
  assert.throws(
    () => parseVerificationResult(changedPhase, input.fixture.control),
    /barrier seals.*digest mismatch|queue phase mismatch/,
  );
});

test("a passing product gate is rejected below either reviewed storage pause watermark", () => {
  const input = evidence();
  const low = structuredClone(input.loadedGates);
  low[0].value.storage_admission.volumes[0].available_bytes = 0;
  low[0].value.storage_admission.volumes[0].status = "rejected";
  low[0].reference.digest = evidenceDigest(low[0].value);
  const manifest = structuredClone(input.manifest);
  manifest.gate_results[0].digest = low[0].reference.digest;
  assert.match(verifyBatch(manifest, input.loadedAudit, low, input.fixture.control).global_failures.find((entry) => entry.code === "gate-invalid").detail, /cannot pass below/);
});

test("seal requires a passing exact verification plus deterministic products and inventory delta", () => {
  const input = evidence();
  const verification = verifyBatch(input.manifest, input.loadedAudit, input.loadedGates, input.fixture.control);
  const integration = [gate(input.fixture, "deterministic-products"), gate(input.fixture, "inventory-delta")];
  const references = integration.map((value) => digestReference(`${value.artifact_id}.json`, value.artifact_id, value));
  const manifest = {
    schema_version: 7, kind: "runmat-builtin-migration-seal-manifest", authority: "reviewed-integration-request", seal_id: "seal-foo", bundle_id: input.fixture.bundleId, lease_id: input.fixture.lease.value.lease_id, lease_digest: input.fixture.lease.value.digest, queue_phase: input.fixture.lease.value.queue_phase, identities: ["foo"], phases: verification.phases,
    source_revision: input.fixture.inventory.source.revision, source_digest: input.fixture.inventory.source.digest, control_baseline_inventory_digest: input.fixture.inventory.digest, lease_base_inventory_digest: input.fixture.inventory.digest, subject_inventory_digest: input.fixture.inventory.digest, control_manifest_digest: input.fixture.control.digest,
    accepted_seals: input.fixture.lease.value.accepted_seals, accepted_seal_set_digest: input.fixture.lease.value.accepted_seal_set_digest,
    barrier_seals: input.fixture.lease.value.barrier_seals, barrier_seal_set_digest: input.fixture.lease.value.barrier_seal_set_digest,
    verification: digestReference("verification.json", verification.artifact_id, verification), integration_gates: references, review: { status: "reviewed", evidence: ["integration review"] },
  };
  const loaded = references.map((reference, index) => ({ reference, value: integration[index] }));
  assert.equal(sealBundle(
    manifest, verification, loaded, [], input.fixture.control, input.fixture.repository,
    input.fixture.lease,
  ).result, "pass");
  const wrongLease = structuredClone(manifest);
  wrongLease.lease_digest = `sha256:${"e".repeat(64)}`;
  assert.throws(() => sealBundle(
    wrongLease, verification, loaded, [], input.fixture.control, input.fixture.repository,
    input.fixture.lease,
  ), /exact active authored lease/);
  assert.equal(sealBundle(
    manifest, verification, loaded.slice(1), [], input.fixture.control,
    input.fixture.repository, input.fixture.lease,
  ).result, "fail");
  const partial = structuredClone(manifest); partial.identities = [];
  assert.throws(() => sealBundle(
    partial, verification, loaded, [], input.fixture.control, input.fixture.repository,
    input.fixture.lease,
  ), /nonempty array/);
  const inventedBarrier = structuredClone(manifest);
  inventedBarrier.barrier_seals = [{ path: "seal-other.json", artifact_id: "seal-other", digest: `sha256:${"a".repeat(64)}`, bundle_id: "other" }];
  assert.throws(() => sealBundle(
    inventedBarrier, verification, loaded, [], input.fixture.control,
    input.fixture.repository, input.fixture.lease,
  ), /control DAG/);
});
