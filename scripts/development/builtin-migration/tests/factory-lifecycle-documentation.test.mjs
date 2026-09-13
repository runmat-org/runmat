import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { auditMigration, parseBatch } from "../audit.mjs";
import { contentDigest, evidenceDigest } from "../evidence.mjs";
import { buildDocumentationCutoverArtifact, documentationCutoverChecks, parseDocumentationCutoverArtifact } from "../documentation-cutover.mjs";
import { issueLease, parseLease, parseLeaseRequest, validateLeaseDiff } from "../lease.mjs";
import { prepareIdentity } from "../prepare.mjs";
import { buildQueue } from "../queue.mjs";
import { parseCompletedSourceDisposition, sourceFieldBaselineDigest, sourceFieldBaselineSource } from "../source-fields.mjs";
import { cleanupRepositoryFixtures, controlledFixture, gate, REVISION } from "./helpers.mjs";
import { reviewed } from "./factory-workflow-fixture.mjs";
import { createTemporaryDirectory } from "./temporary-directories.mjs";

test.afterEach(cleanupRepositoryFixtures);

test("queue is derived by bundle and carries prerequisite, scope, and maturity facts", () => {
  const fixture = controlledFixture();
  const queue = buildQueue(fixture.inventory, fixture.control);
  assert.equal(queue.schema_version, 4);
  assert.equal(queue.rows.length, 1);
  assert.deepEqual(queue.rows[0].identities, ["foo"]);
  assert.equal(queue.rows[0].inventory_observations[0].discovery_only, true);
  assert.ok(queue.rows[0].applicable_maturity.foo.includes("catalog-contract"));
});

test("reviewed control resolves disposition and package-layout inventory ambiguity", () => {
  const unresolved = [
    "disposition", "domain", "domain-conflict", "family", "family-conflict",
  ];
  const fixture = controlledFixture({ unresolved });
  const queue = buildQueue(fixture.inventory, fixture.control);
  assert.equal(queue.rows[0].migration_state, "ready");
  assert.deepEqual(queue.rows[0].blockers, []);
  assert.deepEqual(queue.rows[0].inventory_observations[0].unresolved, unresolved);

  const blockedFixture = controlledFixture({ unresolved: [...unresolved, "future-unreviewed-field"] });
  const blocked = buildQueue(blockedFixture.inventory, blockedFixture.control);
  assert.deepEqual(blocked.rows[0].blockers, ["unresolved-inventory:foo"]);
});

test("lease rejects integration output authorship and paths outside reviewed scope", () => {
  const fixture = controlledFixture();
  assert.throws(() => validateLeaseDiff(fixture.lease, fixture.control, ["README.md"]), /lease violation/);
  assert.throws(() => validateLeaseDiff(fixture.lease, fixture.control, ["crates/runmat-runtime/src/builtins/generated_wasm_registry.rs"]), /lease violation/);
  const forgedLease = {
    value: fixture.lease.value,
    bundle: {
      integration_outputs: [],
      authored_write_set: [{ kind: "tree", path: "crates" }],
    },
  };
  assert.throws(
    () => validateLeaseDiff(forgedLease, fixture.control, ["crates/unreviewed.rs"]),
    /exact validated authored lease/,
  );
  const extra = structuredClone(fixture.leaseValue); extra.extra = true;
  assert.throws(() => parseLease(extra, fixture.control, fixture.repository), /fields must be exactly/);
});

test("lease issuance derives immutable scopes from a reviewed request and control", () => {
  const fixture = controlledFixture();
  const request = { ...structuredClone(fixture.leaseRequest), lease_id: "lane-17", owner: "builtin-migrator", issued_at: "2020-01-01T00:00:00.000Z", expires_at: "2099-01-01T00:00:00.000Z", review: { status: "reviewed", evidence: ["C00 assignment review"] } };
  assert.doesNotThrow(() => parseLeaseRequest(request, fixture.control));
  const first = issueLease(request, fixture.control, fixture.repository, fixture.inventory, fixture.queueState, fixture.queueCheckpoint);
  const second = issueLease(request, fixture.control, fixture.repository, fixture.inventory, fixture.queueState, fixture.queueCheckpoint);
  assert.deepEqual(first, second);
  assert.deepEqual(first.authored_write_set, fixture.control.bundles.get(fixture.bundleId).authored_write_set);
  assert.doesNotThrow(() => parseLease(first, fixture.control, fixture.repository));

  const injected = structuredClone(request); injected.authored_write_set = [{ kind: "tree", path: "." }];
  assert.throws(() => issueLease(injected, fixture.control, fixture.repository, fixture.inventory, fixture.queueState, fixture.queueCheckpoint), /fields must be exactly/);
  const unreviewed = structuredClone(request); unreviewed.review = { status: "unreviewed", evidence: [] };
  assert.throws(() => issueLease(unreviewed, fixture.control, fixture.repository, fixture.inventory, fixture.queueState, fixture.queueCheckpoint), /must be reviewed/);
  const widened = structuredClone(first); widened.authored_write_set.push({ kind: "tree", path: "crates" });
  const { digest: _old, ...payload } = widened; widened.digest = evidenceDigest(payload);
  assert.throws(() => parseLease(widened, fixture.control, fixture.repository), /differs from reviewed bundle scope/);
});

test("prepare v3 is source-neutral and inventories every legacy JSON leaf", () => {
  const fixture = controlledFixture({ sidecar: true });
  const output = createTemporaryDirectory("runmat-review-");
  const result = prepareIdentity(fixture.repository, fixture.inventory, fixture.control, fixture.lease, "foo", output);
  assert.equal(result.schema_version, 3);
  assert.equal(result.source_changes, false);
  const checklistPath = path.join(result.workspace, "source-field-disposition.json");
  const checklist = JSON.parse(fs.readFileSync(checklistPath));
  assert.ok(checklist.sources[0].leaves.length >= 3);
  assert.ok(checklist.sources[0].leaves.every((leaf) => leaf.disposition === "pending"));
  assert.throws(() => parseCompletedSourceDisposition(checklist, "foo", result.checklist_baseline_digest), /not reviewed/);
});

test("completed field dispositions require exact prepared leaves and review evidence", () => {
  const fixture = controlledFixture({ sidecar: true });
  const output = createTemporaryDirectory("runmat-review-");
  const result = prepareIdentity(fixture.repository, fixture.inventory, fixture.control, fixture.lease, "foo", output);
  const checklist = JSON.parse(fs.readFileSync(path.join(result.workspace, "source-field-disposition.json")));
  checklist.review = { status: "reviewed", evidence: ["review"] };
  for (const source of checklist.sources) for (const leaf of source.leaves) { leaf.disposition = "preserved"; leaf.destination = { kind: "catalog-documentation", catalog_identity: "foo", pointer: leaf.pointer, value_digest: leaf.value_digest }; }
  assert.doesNotThrow(() => parseCompletedSourceDisposition(checklist, "foo", result.checklist_baseline_digest));
  checklist.sources[0].leaves.pop();
  assert.throws(() => parseCompletedSourceDisposition(checklist, "foo", result.checklist_baseline_digest), /prepared baseline/);
});

test("audit v8 cannot pass on file presence or example tokens without exact gate evidence", () => {
  const fixture = controlledFixture();
  const output = createTemporaryDirectory("runmat-review-");
  const prepared = prepareIdentity(fixture.repository, fixture.inventory, fixture.control, fixture.lease, "foo", output);
  const batch = { schema_version: 1, kind: "runmat-builtin-migration-batch", identities: ["foo"] };
  const absent = auditMigration(fixture.repository, fixture.inventory, fixture.inventory, fixture.inventory, fixture.control, fixture.lease, batch, { artifact_id: "audit-foo", authored_revision: fixture.inventory.source.revision, prepare_results: [prepared], source_dispositions: [], gate_results: [] });
  assert.equal(absent.result, "fail");
  assert.ok(absent.identities[0].failures.some((entry) => entry.code === "required-gate-missing"));
  const gates = ["catalog-contract", "runtime-binding", "documentation-cutover", "native-link", "architecture", "focused-tests", "format-diff", "strict-clippy"].map((name) => gate(fixture, name));
  const passed = auditMigration(fixture.repository, fixture.inventory, fixture.inventory, fixture.inventory, fixture.control, fixture.lease, batch, { artifact_id: "audit-foo", authored_revision: fixture.inventory.source.revision, prepare_results: [prepared], source_dispositions: [], gate_results: gates });
  assert.equal(passed.result, "pass");
  const stale = structuredClone(gates); stale[0].source_digest = `sha256:${"b".repeat(64)}`;
  assert.equal(auditMigration(fixture.repository, fixture.inventory, fixture.inventory, fixture.inventory, fixture.control, fixture.lease, batch, { artifact_id: "audit-foo", authored_revision: fixture.inventory.source.revision, prepare_results: [prepared], source_dispositions: [], gate_results: stale }).result, "fail");
  const impersonated = structuredClone(gates); impersonated[0].producer_evidence.contract.producer_source_digest = `sha256:${"e".repeat(64)}`;
  const rejected = auditMigration(fixture.repository, fixture.inventory, fixture.inventory, fixture.inventory, fixture.control, fixture.lease, batch, { artifact_id: "audit-foo", authored_revision: fixture.inventory.source.revision, prepare_results: [prepared], source_dispositions: [], gate_results: impersonated });
  assert.ok(rejected.global_failures.some((entry) => entry.detail.includes("differs from the reviewed gate plan")));
});

test("source-field destinations require value-digest reconciliation from the documentation producer", () => {
  const fixture = controlledFixture({ sidecar: true });
  const output = createTemporaryDirectory("runmat-review-");
  const prepared = prepareIdentity(fixture.repository, fixture.inventory, fixture.control, fixture.lease, "foo", output);
  const disposition = JSON.parse(fs.readFileSync(path.join(prepared.workspace, "source-field-disposition.json")));
  disposition.review = { status: "reviewed", evidence: ["review"] };
  for (const source of disposition.sources) for (const leaf of source.leaves) { leaf.disposition = "preserved"; leaf.destination = { kind: "catalog-documentation", catalog_identity: "foo", pointer: leaf.pointer, value_digest: leaf.value_digest }; }
  const gates = ["catalog-contract", "runtime-binding", "documentation-cutover", "native-link", "architecture", "focused-tests", "format-diff", "strict-clippy"].map((name) => gate(fixture, name));
  const batch = { schema_version: 1, kind: "runmat-builtin-migration-batch", identities: ["foo"] };
  const missing = auditMigration(fixture.repository, fixture.inventory, fixture.inventory, fixture.inventory, fixture.control, fixture.lease, batch, { artifact_id: "audit-destination", authored_revision: fixture.inventory.source.revision, prepare_results: [prepared], source_dispositions: [disposition], gate_results: gates });
  assert.ok(missing.identities[0].failures.some((entry) => entry.code === "destination-proof-missing"));
  const documentation = gates.find((entry) => entry.gate === "documentation-cutover");
  const sourcePath = disposition.sources[0].path;
  const sourceBytes = fs.readFileSync(path.join(fixture.repository, sourcePath));
  const baselineSource = sourceFieldBaselineSource(sourcePath, sourceBytes);
  const baselineDigest = evidenceDigest({ identity: "foo", sources: [baselineSource] });
  const catalogExport = { schema_version: 1, inventory: { documents: 1, catalog_identities: 1, legacy_sidecars: 0, missing_catalog_documentation: [] }, builtins: [{ key: "foo", authority: "catalog", ...JSON.parse(sourceBytes) }] };
  const artifact = buildDocumentationCutoverArtifact({
    catalog_export: catalogExport,
    catalog_export_bytes: JSON.stringify(catalogExport),
    source_dispositions: [{ baseline_digest: baselineDigest, value: disposition }],
    expected_sources: { foo: [baselineSource] },
    provenance: {
      source_revision: fixture.inventory.source.revision,
      source_digest: fixture.inventory.source.digest,
      compiled_inventory_digest: fixture.inventory.compiled_inventory.digest,
      control_manifest_digest: fixture.control.digest,
      bundle_id: fixture.bundleId,
      identities: ["foo"],
    },
  });
  const artifactBytes = Buffer.from(`${JSON.stringify(artifact)}\n`);
  fs.writeFileSync(documentation.artifacts[0].path, artifactBytes);
  documentation.artifacts[0].byte_length = artifactBytes.length;
  documentation.artifacts[0].content_digest = contentDigest(artifactBytes);
  documentation.checks = documentationCutoverChecks(artifact);
  documentation.producer_evidence.process.stdout_digest = artifact.catalog_export.evidence_digest;
  documentation.producer_evidence.captured_process_digest = evidenceDigest(documentation.producer_evidence.process);
  for (const invalidDocumentation of [
    { ...structuredClone(documentation), checks: documentation.checks.slice(0, -1) },
    { ...structuredClone(documentation), checks: [...documentation.checks, { id: "destination:forged", result: "pass", evidence_digest: `sha256:${"f".repeat(64)}` }] },
    { ...structuredClone(documentation), checks: documentation.checks.map((check, index) => index === 0 ? check : { ...check, evidence_digest: `sha256:${"f".repeat(64)}` }) },
  ]) {
    const invalidGates = gates.map((gateResult) => gateResult.gate === "documentation-cutover" ? invalidDocumentation : gateResult);
    const invalid = auditMigration(fixture.repository, fixture.inventory, fixture.inventory, fixture.inventory, fixture.control, fixture.lease, batch, { artifact_id: "audit-destination-invalid", authored_revision: fixture.inventory.source.revision, prepare_results: [prepared], source_dispositions: [disposition], gate_results: invalidGates });
    assert.ok(invalid.global_failures.some((entry) => entry.code === "invalid-gate-evidence"));
  }

  for (const mutate of [
    (value) => { value.provenance.compiled_inventory_digest = `sha256:${"9".repeat(64)}`; },
    (value) => { value.provenance.source_dispositions_digest = `sha256:${"8".repeat(64)}`; },
  ]) {
    const forgedArtifact = structuredClone(artifact);
    mutate(forgedArtifact);
    const { digest: _forgedDigest, ...forgedPayload } = forgedArtifact;
    forgedArtifact.digest = evidenceDigest(forgedPayload);
    const forgedDocumentation = structuredClone(documentation);
    const forgedBytes = Buffer.from(`${JSON.stringify(forgedArtifact)}\n`);
    fs.writeFileSync(forgedDocumentation.artifacts[0].path, forgedBytes);
    forgedDocumentation.artifacts[0].byte_length = forgedBytes.length;
    forgedDocumentation.artifacts[0].content_digest = contentDigest(forgedBytes);
    forgedDocumentation.checks = documentationCutoverChecks(forgedArtifact);
    const forgedGates = gates.map((gateResult) => gateResult.gate === "documentation-cutover" ? forgedDocumentation : gateResult);
    const invalid = auditMigration(fixture.repository, fixture.inventory, fixture.inventory, fixture.inventory, fixture.control, fixture.lease, batch, { artifact_id: "audit-destination-forged", authored_revision: fixture.inventory.source.revision, prepare_results: [prepared], source_dispositions: [disposition], gate_results: forgedGates });
    assert.ok(invalid.global_failures.some((entry) => entry.code === "invalid-gate-evidence"));
  }
  fs.writeFileSync(documentation.artifacts[0].path, artifactBytes);
  const reconciled = auditMigration(fixture.repository, fixture.inventory, fixture.inventory, fixture.inventory, fixture.control, fixture.lease, batch, { artifact_id: "audit-destination", authored_revision: fixture.inventory.source.revision, prepare_results: [prepared], source_dispositions: [disposition], gate_results: gates });
  assert.ok(!reconciled.identities[0].failures.some((entry) => entry.code === "destination-proof-missing"));
});

test("documentation cutover evidence reconciles every nested leaf by exact escaped pointer and digest", () => {
  const valueDigest = (value) => evidenceDigest(value);
  const legacyBytes = Buffer.from(`${JSON.stringify({ nested: { "a/b": ["kept", "old"] }, obsolete: true })}\n`);
  const baselineSource = sourceFieldBaselineSource("docs/builtins/reference/foo.json", legacyBytes);
  const disposition = {
    schema_version: 2, kind: "runmat-builtin-source-field-disposition", authority: "review-workspace-only", identity: "foo",
    sources: [{ path: baselineSource.path, digest: baselineSource.digest, leaves: [
      { pointer: "/nested/a~1b/0", value_digest: valueDigest("kept"), disposition: "preserved", destination: { kind: "catalog-documentation", catalog_identity: "foo", pointer: "/nested/a~1b/0", value_digest: valueDigest("kept") }, reason: null, evidence: [] },
      { pointer: "/nested/a~1b/1", value_digest: valueDigest("old"), disposition: "normalized", destination: { kind: "catalog-documentation", catalog_identity: "foo", pointer: "/nested/a~1b/1", value_digest: valueDigest("new") }, reason: "Normalize wording", evidence: ["review"] },
      { pointer: "/obsolete", value_digest: valueDigest(true), disposition: "removed", destination: null, reason: "Obsolete field", evidence: ["review"] },
    ] }], review: { status: "reviewed", evidence: ["owner review"] },
  };
  const catalogExport = { schema_version: 1, inventory: { documents: 1, catalog_identities: 1, legacy_sidecars: 0, missing_catalog_documentation: [] }, builtins: [{ key: "foo", authority: "catalog", nested: { "a/b": ["kept", "new"] } }] };
  const catalogBytes = `${JSON.stringify(catalogExport)}\n`;
  const provenance = { source_revision: REVISION, source_digest: `sha256:${"2".repeat(64)}`, compiled_inventory_digest: `sha256:${"3".repeat(64)}`, control_manifest_digest: `sha256:${"4".repeat(64)}`, bundle_id: "math-basic-foo", identities: ["foo"] };
  const input = { catalog_export: catalogExport, catalog_export_bytes: catalogBytes, source_dispositions: [{ baseline_digest: sourceFieldBaselineDigest(disposition), value: disposition }], expected_sources: { foo: [baselineSource] }, provenance };
  const artifact = buildDocumentationCutoverArtifact(input);
  assert.equal(artifact.result, "pass");
  assert.equal(artifact.rows.length, 3);
  assert.ok(documentationCutoverChecks(artifact).some((entry) => entry.id.includes("/nested/a~1b/0")));
  const expected = { ...provenance, catalog_export_digest: contentDigest(Buffer.from(catalogBytes)), source_dispositions: input.source_dispositions };
  assert.doesNotThrow(() => parseDocumentationCutoverArtifact(artifact, expected));

  const mismatch = structuredClone(input); mismatch.catalog_export.builtins[0].nested["a/b"][0] = "nearby kept token"; mismatch.catalog_export_bytes = `${JSON.stringify(mismatch.catalog_export)}\n`;
  assert.equal(buildDocumentationCutoverArtifact(mismatch).result, "fail");
  const omitted = structuredClone(input); omitted.source_dispositions[0].value.sources[0].leaves.pop();
  assert.throws(() => buildDocumentationCutoverArtifact(omitted), /prepared baseline inventory/);
  const duplicated = structuredClone(input); duplicated.source_dispositions[0].value.sources[0].leaves.push(structuredClone(duplicated.source_dispositions[0].value.sources[0].leaves[0]));
  assert.throws(() => buildDocumentationCutoverArtifact(duplicated), /duplicate source-field leaf/);
  const changed = structuredClone(input); changed.source_dispositions[0].value.sources[0].leaves[0].value_digest = valueDigest("changed");
  changed.source_dispositions[0].baseline_digest = sourceFieldBaselineDigest(changed.source_dispositions[0].value);
  assert.throws(() => buildDocumentationCutoverArtifact(changed), /frozen source documents/);
  assert.throws(() => parseDocumentationCutoverArtifact(artifact, { ...provenance, catalog_export_digest: `sha256:${"9".repeat(64)}` }), /catalog export is stale/);

  const schemaMismatch = structuredClone(artifact);
  schemaMismatch.catalog_export.schema_version = 2;
  const { digest: _schemaDigest, ...schemaPayload } = schemaMismatch;
  schemaMismatch.digest = evidenceDigest(schemaPayload);
  assert.throws(() => parseDocumentationCutoverArtifact(schemaMismatch, expected), /schema versions differ/);

  const omittedArtifact = structuredClone(artifact);
  omittedArtifact.rows.pop();
  omittedArtifact.summary = { identities: 1, leaves: 2, passed: 2 };
  const { digest: _oldDigest, ...omittedPayload } = omittedArtifact;
  omittedArtifact.digest = evidenceDigest(omittedPayload);
  assert.throws(() => parseDocumentationCutoverArtifact(omittedArtifact, expected), /exactly cover/);

  const alteredArtifact = structuredClone(artifact);
  alteredArtifact.rows[0].destination.pointer = "/nested/a~1b/1";
  const { digest: _alteredDigest, ...alteredPayload } = alteredArtifact;
  alteredArtifact.digest = evidenceDigest(alteredPayload);
  assert.throws(() => parseDocumentationCutoverArtifact(alteredArtifact, expected), /exactly cover|captured catalog export/);
});

test("batch schema rejects unsafe and case-colliding identities", () => {
  assert.deepEqual(parseBatch({ schema_version: 1, kind: "runmat-builtin-migration-batch", identities: ["__helper"] }), ["__helper"]);
  assert.throws(() => parseBatch({ schema_version: 1, kind: "runmat-builtin-migration-batch", identities: ["foo", "Foo"] }), /unique/);
  assert.throws(() => parseBatch({ schema_version: 1, kind: "runmat-builtin-migration-batch", identities: ["../foo"] }), /invalid/);
});
