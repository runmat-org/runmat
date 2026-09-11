import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { auditMigration } from "../audit.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { prepareIdentity } from "../prepare.mjs";
import { sealBundle } from "../seal.mjs";
import { verifyBatch } from "../verify.mjs";
import { controlledFixture, digestReference, gate } from "./helpers.mjs";

function evidence() {
  const fixture = controlledFixture();
  const prepared = prepareIdentity(fixture.repository, fixture.inventory, fixture.control, fixture.lease, "foo", fs.mkdtempSync(path.join(os.tmpdir(), "verify-")));
  const gates = ["catalog-contract", "runtime-binding", "documentation-cutover", "architecture"].map((name) => gate(fixture, name));
  const batch = { schema_version: 1, kind: "runmat-builtin-migration-batch", identities: ["foo"] };
  const audit = auditMigration(fixture.repository, fixture.inventory, fixture.control, fixture.lease, batch, { artifact_id: "audit-foo", changed_paths: [], prepare_results: [prepared], source_dispositions: [], gate_results: gates });
  const auditReference = digestReference("audit.json", "audit-foo", audit);
  const gateReferences = gates.map((value) => digestReference(`${value.artifact_id}.json`, value.artifact_id, value));
  const manifest = {
    schema_version: 2, kind: "runmat-builtin-migration-verification-manifest", authority: "reviewed-verification-request",
    batch: { artifact_id: "verify-foo", source_revision: fixture.inventory.source.revision, source_digest: fixture.inventory.source.digest, inventory_digest: fixture.inventory.digest, control_manifest_digest: fixture.control.digest, bundle_id: fixture.bundleId, identities: ["foo"] },
    audit: auditReference, gate_results: gateReferences,
    expectations: [{ identity: "foo", required_gates: ["catalog-contract", "runtime-binding", "documentation-cutover", "architecture"] }],
  };
  const loadedAudit = { reference: auditReference, value: audit };
  const loadedGates = gateReferences.map((reference, index) => ({ reference, value: gates[index] }));
  return { fixture, audit, gates, manifest, loadedAudit, loadedGates };
}

test("verification v2 passes only exact, content-addressed audit and gate evidence", () => {
  const input = evidence();
  const result = verifyBatch(input.manifest, input.loadedAudit, input.loadedGates);
  assert.equal(result.schema_version, 2);
  assert.equal(result.result, "pass");
});

test("verification rejects missing, duplicate, stale, and mutated evidence", () => {
  const input = evidence();
  assert.equal(verifyBatch(input.manifest, input.loadedAudit, input.loadedGates.slice(1)).result, "fail");
  const duplicate = structuredClone(input.manifest); duplicate.gate_results.push(duplicate.gate_results[0]);
  assert.throws(() => verifyBatch(duplicate, input.loadedAudit, input.loadedGates), /unique/);
  const mutated = structuredClone(input.loadedGates); mutated[0].value.checks[0].evidence_digest = `sha256:${"f".repeat(64)}`;
  assert.ok(verifyBatch(input.manifest, input.loadedAudit, mutated).global_failures.some((entry) => entry.code === "gate-digest-mismatch"));
  const future = structuredClone(input.manifest); future.schema_version = 3;
  assert.throws(() => verifyBatch(future, input.loadedAudit, input.loadedGates), /schema_version 2/);
  const inconsistentAudit = structuredClone(input.loadedAudit);
  inconsistentAudit.value.identities[0].failures.push({ code: "forged", detail: "ignored" });
  inconsistentAudit.reference.digest = evidenceDigest(inconsistentAudit.value);
  const inconsistentManifest = structuredClone(input.manifest);
  inconsistentManifest.audit.digest = inconsistentAudit.reference.digest;
  assert.ok(verifyBatch(inconsistentManifest, inconsistentAudit, input.loadedGates).global_failures.some((entry) => entry.code === "audit-invalid"));
});

test("a passing product gate is rejected below either reviewed storage pause watermark", () => {
  const input = evidence();
  const low = structuredClone(input.loadedGates);
  low[0].value.storage_admission.volumes[0].available_bytes = 0;
  low[0].value.storage_admission.volumes[0].status = "paused";
  low[0].reference.digest = evidenceDigest(low[0].value);
  const manifest = structuredClone(input.manifest);
  manifest.gate_results[0].digest = low[0].reference.digest;
  assert.match(verifyBatch(manifest, input.loadedAudit, low).global_failures.find((entry) => entry.code === "gate-invalid").detail, /cannot pass below/);
});

test("seal requires a passing exact verification plus deterministic products and inventory delta", () => {
  const input = evidence();
  const verification = verifyBatch(input.manifest, input.loadedAudit, input.loadedGates);
  const integration = [gate(input.fixture, "deterministic-products"), gate(input.fixture, "inventory-delta")];
  const references = integration.map((value) => digestReference(`${value.artifact_id}.json`, value.artifact_id, value));
  const manifest = {
    schema_version: 1, kind: "runmat-builtin-migration-seal-manifest", authority: "reviewed-integration-request", seal_id: "seal-foo", bundle_id: input.fixture.bundleId, identities: ["foo"],
    source_revision: input.fixture.inventory.source.revision, source_digest: input.fixture.inventory.source.digest, inventory_digest: input.fixture.inventory.digest, control_manifest_digest: input.fixture.control.digest,
    verification: digestReference("verification.json", verification.artifact_id, verification), integration_gates: references, prerequisite_seals: [], review: { status: "reviewed", evidence: ["integration review"] },
  };
  const loaded = references.map((reference, index) => ({ reference, value: integration[index] }));
  assert.equal(sealBundle(manifest, verification, loaded, [], input.fixture.control, input.fixture.repository).result, "pass");
  assert.equal(sealBundle(manifest, verification, loaded.slice(1), [], input.fixture.control, input.fixture.repository).result, "fail");
  const partial = structuredClone(manifest); partial.identities = [];
  assert.throws(() => sealBundle(partial, verification, loaded, [], input.fixture.control, input.fixture.repository), /nonempty array/);
  const inventedPrerequisite = structuredClone(manifest);
  inventedPrerequisite.prerequisite_seals = [{ path: "seal-other.json", artifact_id: "seal-other", digest: `sha256:${"a".repeat(64)}`, bundle_id: "other" }];
  assert.throws(() => sealBundle(inventedPrerequisite, verification, loaded, [], input.fixture.control, input.fixture.repository), /control DAG/);
});
