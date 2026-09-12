import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { auditMigration, parseBatch } from "../audit.mjs";
import {
  MATURITY_GATES, assertControlSubject, parseControlManifest, validateControlManifestStructure,
} from "../control.mjs";
import { buildControlDraft, freezeReviewedControl, parseControlDraft } from "../control-draft.mjs";
import { validateControlReviewChain } from "../control-authoring/authority.mjs";
import { composeControlCandidate, controlCandidateInputDigests } from "../control-authoring/compose.mjs";
import { loadControlReviewSet } from "../control-authoring/review-set.mjs";
import { buildControlOverlayScaffold } from "../control-authoring/scaffold.mjs";
import { dispositionInputFromControl, validateDispositionInput } from "../dispositions.mjs";
import { compileDispositionReview, parseDispositionReview } from "../disposition-review.mjs";
import { contentDigest, evidenceDigest } from "../evidence.mjs";
import { runGateProducer } from "../gate-adapter.mjs";
import { generatedProductChecks, parseGeneratedProductsProof } from "../generated-products.mjs";
import { buildDocumentationCutoverArtifact, documentationCutoverChecks, parseDocumentationCutoverArtifact } from "../documentation-cutover.mjs";
import { parseGateResult } from "../gate-result.mjs";
import { buildInventory } from "../inventory.mjs";
import { buildInventoryDeltaProof, inventoryDeltaChecks } from "../inventory-delta.mjs";
import { issueLease, parseLease, parseLeaseRequest, validateLeaseDiff } from "../lease.mjs";
import { prepareIdentity } from "../prepare.mjs";
import { buildQueue } from "../queue.mjs";
import { sourceSnapshot } from "../snapshot.mjs";
import { parseCompletedSourceDisposition, sourceFieldBaselineDigest, sourceFieldBaselineSource } from "../source-fields.mjs";
import { prepareGateStorage, storageStatus } from "../storage-admission.mjs";
import { composeTopologyCandidate } from "../topology/compose.mjs";
import { candidateInputDigests, freezeReviewedTopology, parseReviewedTopology, reviewedTopologyView } from "../topology/freeze.mjs";
import { fullTopologyChainFixture } from "../topology/tests/full-chain-fixture.mjs";
import { cleanupRepositoryFixtures, compiledInventoryFixture, controlledFixture, gate, repositoryFixture, REVISION, topologyFixture } from "./helpers.mjs";

test.afterEach(cleanupRepositoryFixtures);

function dispositionReviewFixture(inventory) {
  return {
    schema_version: 1,
    kind: "runmat-builtin-disposition-review",
    authority: "reviewer-authored-development-input",
    baseline_inventory_digest: inventory.digest,
    groups: [{
      id: "math-basic-canonical",
      disposition: "canonical",
      identities: ["foo"],
      alias_targets: {},
      reason: null,
      evidence: ["catalog and runtime owner review"],
      review: { status: "reviewed", evidence: ["C00 disposition review"] },
    }],
    review: { status: "reviewed", evidence: ["C00 disposition review"] },
  };
}

test("inventory v2 binds lexical observations to content-derived source and disposition digests", () => {
  const repository = repositoryFixture();
  const compiledInventory = compiledInventoryFixture();
  const first = buildInventory(repository, undefined, { revision: REVISION, compiledInventory });
  const second = buildInventory(repository, undefined, { revision: REVISION, compiledInventory });
  assert.equal(first.schema_version, 2);
  assert.equal(first.digest, second.digest);
  assert.match(first.source.digest, /^sha256:[a-f0-9]{64}$/);
  assert.equal(first.source.digest, evidenceDigest({ roots: first.source.roots, files: first.source.files }));
  assert.equal(first.scanned_source_coverage.digest, evidenceDigest(first.scanned_source_coverage.paths.map((entry) => first.source.files.find((file) => file.path === entry))));
  fs.appendFileSync(path.join(repository, "crates/runmat-runtime/src/builtins/math/basic/foo.rs"), "\n// change\n");
  const changed = buildInventory(repository, undefined, { revision: REVISION, compiledInventory });
  assert.notEqual(changed.source.digest, first.source.digest);
  assert.notEqual(changed.digest, first.digest);
});

test("source dirty evidence is scoped to the frozen inventory roots", () => {
  const repository = repositoryFixture();
  fs.writeFileSync(path.join(repository, "unrelated.tmp"), "outside snapshot\n");
  assert.equal(sourceSnapshot(repository, ["Cargo.toml"]).dirty, false);
  fs.appendFileSync(path.join(repository, "Cargo.toml"), "# scoped change\n");
  assert.equal(sourceSnapshot(repository, ["Cargo.toml"]).dirty, true);
});

test("baseline prepare evidence and subject gate evidence retain distinct provenance", () => {
  const fixture = controlledFixture();
  const output = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-review-"));
  const prepared = prepareIdentity(fixture.repository, fixture.inventory, fixture.control, fixture.lease, "foo", output);
  const runtimePath = "crates/runmat-runtime/src/builtins/math/basic/foo.rs";
  fs.appendFileSync(path.join(fixture.repository, runtimePath), "\n// migrated subject\n");
  execFileSync("git", ["add", runtimePath], { cwd: fixture.repository });
  execFileSync("git", ["-c", "user.name=RunMat Test", "-c", "user.email=test@runmat.invalid", "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "subject"], { cwd: fixture.repository });
  const subject = buildInventory(fixture.repository, dispositionInputFromControl(fixture.control), { compiledInventory: fixture.compiledInventory });
  assert.notEqual(subject.source.revision, fixture.inventory.source.revision);
  assert.notEqual(subject.digest, fixture.inventory.digest);
  const gates = ["catalog-contract", "runtime-binding", "documentation-cutover", "architecture"].map((name) => gate(fixture, name, `gate-subject-${name}`, subject));
  const batch = { schema_version: 1, kind: "runmat-builtin-migration-batch", identities: ["foo"] };
  const audit = auditMigration(fixture.repository, fixture.inventory, subject, fixture.control, fixture.lease, batch, { artifact_id: "audit-subject", changed_paths: [runtimePath], prepare_results: [prepared], source_dispositions: [], gate_results: gates });
  assert.equal(audit.result, "pass");
  assert.equal(audit.source.revision, subject.source.revision);
  assert.equal(audit.baseline_inventory_digest, fixture.inventory.digest);
  assert.equal(audit.subject_inventory_digest, subject.digest);
  assert.equal(prepared.inventory_digest, fixture.inventory.digest);
});

test("inventory rejects absent, future, or tampered compiled semantic authority", () => {
  const repository = repositoryFixture();
  assert.throws(() => buildInventory(repository, undefined, { revision: REVISION }), /requires a compiled migration inventory/);
  const future = compiledInventoryFixture(); future.schema_version = 2;
  assert.throws(() => buildInventory(repository, undefined, { revision: REVISION, compiledInventory: future }), /schema_version 1/);
  const tampered = compiledInventoryFixture(); tampered.snapshot.observed.runtime_bindings[0].variant = "changed";
  assert.throws(() => buildInventory(repository, undefined, { revision: REVISION, compiledInventory: tampered }), /digest mismatch/);
  const collision = compiledInventoryFixture(); collision.snapshot.declared.legacy_documentation.push({ name: "Foo", category: null, summary: null, keywords: null, errors: null, related: null, introduced: null, status: null, examples: null });
  collision.digest.value = contentDigest(Buffer.from(JSON.stringify(collision.snapshot))).slice("sha256:".length);
  assert.throws(() => buildInventory(repository, undefined, { revision: REVISION, compiledInventory: collision }), /spellings collide case-insensitively/);
});

test("compiled constants supply public spellings only when no callable form exists", () => {
  const repository = repositoryFixture({ identity: "other" });
  const constantOnly = compiledInventoryFixture("eps");
  constantOnly.snapshot.declared.catalog_entries = [];
  constantOnly.snapshot.declared.catalog_provenance = [];
  constantOnly.snapshot.declared.constants = [{ name: "eps", kind: "real_double" }];
  constantOnly.snapshot.observed.runtime_bindings = [];
  constantOnly.snapshot.observed.implementation_provenance = [];
  constantOnly.snapshot.observed.runtime_constants = [{ name: "eps" }];
  constantOnly.digest.value = contentDigest(Buffer.from(JSON.stringify(constantOnly.snapshot))).slice("sha256:".length);
  const constantInventory = buildInventory(repository, undefined, { revision: REVISION, compiledInventory: constantOnly });
  assert.deepEqual(constantInventory.identities.find((entry) => entry.identity === "eps").spellings, ["eps"]);

  const dual = compiledInventoryFixture("inf");
  dual.snapshot.declared.constants = [
    { name: "Inf", kind: "real_double" },
    { name: "inf", kind: "real_double" },
  ];
  dual.snapshot.observed.runtime_constants = [{ name: "Inf" }, { name: "inf" }];
  dual.digest.value = contentDigest(Buffer.from(JSON.stringify(dual.snapshot))).slice("sha256:".length);
  const dualInventory = buildInventory(repository, undefined, { revision: REVISION, compiledInventory: dual });
  assert.deepEqual(dualInventory.identities.find((entry) => entry.identity === "inf").spellings, ["inf"]);
});

test("compiled authority rejects unknown or malformed nested records even with a recomputed digest", () => {
  const repository = repositoryFixture();
  const cases = [
    (value) => { value.snapshot.declared.catalog_entries[0].documentation.extra = true; },
    (value) => { value.snapshot.declared.catalog_entries[0].bindings[0].availability = "Maybe"; },
    (value) => { value.snapshot.declared.catalog_provenance[0].provenance.extra = "x"; },
    (value) => { value.snapshot.observed.runtime_bindings[0].native_symbol = "guessed"; },
    (value) => { value.snapshot.observed.implementation_provenance[0].authority = "unknown"; },
    (value) => { value.snapshot.build.crate_feature_inventory.enabled_features = ["not-known"]; },
    (value) => { value.snapshot.observed.gpu_specs = [{ key: "data.*", owner: { kind: "legacy_group", raw: "data.*", extra: true }, operation: "custom:data", supported_precisions: [], broadcast: "none", provider_hooks: [], constant_strategy: "inline_literal", residency: "inherit_inputs", nan_mode: "include", two_pass_threshold: null, workgroup_size: null, accepts_nan_mode: false, notes: "fixture" }]; },
  ];
  for (const mutate of cases) {
    const value = compiledInventoryFixture(); mutate(value);
    value.digest.value = contentDigest(Buffer.from(JSON.stringify(value.snapshot))).slice("sha256:".length);
    assert.throws(() => buildInventory(repository, undefined, { revision: REVISION, compiledInventory: value }));
  }
});

test("compiled migration findings and non-identity legacy spec keys remain explicit reviewed work", () => {
  const finding = { code: "legacy_spec_group_requires_disposition", source: "gpu_spec_registry", identity: "data.*", message: "Legacy group needs reviewed ownership" };
  const fixture = controlledFixture({ finding, legacyGroup: true });
  assert.deepEqual(fixture.inventory.migration_findings, [finding]);
  assert.equal(fixture.inventory.identities.some((entry) => entry.identity === "data.*"), false);
  assert.equal(fixture.control.migrationFindings[0].bundle_id, fixture.bundleId);
  const queue = buildQueue(fixture.inventory, fixture.control);
  assert.deepEqual(queue.rows[0].migration_findings.map((entry) => entry.identity), ["data.*"]);
  assert.ok(queue.rows[0].blockers.some((entry) => entry.startsWith("migration-finding:")));
  const missing = structuredClone(fixture.controlValue); missing.migration_findings.rows = [];
  assert.throws(() => parseFixtureControl(fixture, missing), /do not exactly cover/);
  const unknown = structuredClone(fixture.controlValue); unknown.migration_findings.rows[0].unexpected = true;
  assert.throws(() => parseFixtureControl(fixture, unknown), /fields must be exactly/);
});

test("control is closed, reviewed, reciprocal, and rejects case-fold ambiguity", () => {
  const fixture = controlledFixture();
  assert.equal(fixture.control.identities.get("foo").public_spelling, "foo");
  const future = structuredClone(fixture.controlValue); future.schema_version = 3;
  assert.throws(() => parseFixtureControl(fixture, future), /schema_version 2/);
  const extra = structuredClone(fixture.controlValue); extra.unreviewed = true;
  assert.throws(() => parseFixtureControl(fixture, extra), /fields must be exactly/);
  const collision = structuredClone(fixture.controlValue); collision.identity_controls.Foo = structuredClone(collision.identity_controls.foo);
  assert.throws(() => parseFixtureControl(fixture, collision), /collide case-insensitively|identity policy set differs/);
  const spelling = structuredClone(fixture.controlValue); spelling.identity_controls.foo.public_spelling = "bar";
  assert.throws(() => parseFixtureControl(fixture, spelling), /case-fold to the identity key/);
  const observedSpelling = structuredClone(fixture.controlValue); observedSpelling.identity_controls.foo.public_spelling = "Foo";
  assert.throws(() => parseFixtureControl(fixture, observedSpelling), /public spelling differs from the reviewed inventory/);
  const domain = structuredClone(fixture.controlValue); domain.identity_controls.foo.domain = "other";
  assert.throws(() => parseFixtureControl(fixture, domain), /fields must be exactly/);
  const disposition = structuredClone(fixture.controlValue); disposition.identity_controls.foo.disposition = { kind: "internal" };
  assert.throws(() => parseFixtureControl(fixture, disposition), /fields must be exactly/);
  const unsafeStorage = structuredClone(fixture.controlValue); unsafeStorage.storage_policy.host_profiles["fixture-host"].volume_roles.target_temp.filesystem_id = unsafeStorage.storage_policy.host_profiles["fixture-host"].volume_roles.source_worktree.filesystem_id;
  assert.throws(() => parseFixtureControl(fixture, unsafeStorage), /disjoint filesystem/);
  const noncanonicalScope = structuredClone(fixture.controlValue); noncanonicalScope.bundle_controls[fixture.bundleId].additional_authored_write_set[0].path = "crates/runmat-builtins/./src";
  assert.throws(() => parseFixtureControl(fixture, noncanonicalScope), /normalized safe repository-relative path/);
  const removal = structuredClone(fixture.controlValue); removal.identity_controls.foo.expected_removals = [{ kind: "file", path: "docs/builtins/reference/foo.json", baseline_digest: `sha256:${"a".repeat(64)}` }];
  assert.throws(() => parseFixtureControl(fixture, removal), /lacks matching baseline/);
  const incompleteInventory = structuredClone(fixture.inventory); incompleteInventory.identities.push({ identity: "bar" }); resealEvidence(incompleteInventory);
  assert.throws(() => parseFixtureControl(fixture, fixture.controlValue, incompleteInventory), /inventory digest|exactly cover|projection baseline/);
  assert.throws(() => parseFixtureControl(fixture, fixture.controlValue, fixture.inventory, { ...fixture.topology }), /deterministically validated topology view/);
  assert.throws(() => fixture.topology.bundles.set("forged", {}), /immutable/);
  assert.throws(() => fixture.control.bundles.get(fixture.bundleId).gate_plans.set("forged", {}), /immutable/);
  const forgedControl = {
    ...fixture.control,
    bundles: new Map(fixture.control.bundles),
  };
  assert.throws(() => buildQueue(fixture.inventory, forgedControl), /exact validated control manifest/);
  assert.throws(() => issueLease(fixture.lease.value.request, forgedControl), /exact validated control manifest/);
  const danglingTopology = topologyFixture(fixture.inventory, fixture.bundleId, "foo", { identity: {
    ...fixture.topology.identities.get("foo"),
    disposition: { kind: "alias", canonical: "missing", reason: null, source: "reviewed-input" },
  } });
  const danglingControl = structuredClone(fixture.controlValue);
  danglingControl.topology_digest = danglingTopology.digest;
  resealEvidence(danglingControl);
  assert.throws(() => parseFixtureControl(fixture, danglingControl, fixture.inventory, danglingTopology), /alias target is absent/);
  const missingGatePlan = structuredClone(fixture.controlValue); missingGatePlan.bundle_controls[fixture.bundleId].gate_plans = missingGatePlan.bundle_controls[fixture.bundleId].gate_plans.filter((entry) => entry.gate !== "runtime-binding");
  assert.throws(() => parseFixtureControl(fixture, missingGatePlan), /required gate runtime-binding has no reviewed gate plan/);
});

test("control overlay cannot restate topology-owned bundle or identity facts", () => {
  const fixture = controlledFixture();
  for (const [section, field, value] of [
    ["bundle_controls", "identities", ["foo"]],
    ["bundle_controls", "atomic_reason", "Copied topology authority"],
    ["identity_controls", "disposition", { kind: "canonical", target: "foo" }],
    ["identity_controls", "cohort", "C01"],
    ["identity_controls", "bundle_id", fixture.bundleId],
    ["identity_controls", "domain", "math"],
    ["identity_controls", "family", "basic"],
  ]) {
    const control = structuredClone(fixture.controlValue);
    const key = section === "bundle_controls" ? fixture.bundleId : "foo";
    control[section][key][field] = value;
    assert.throws(() => parseFixtureControl(fixture, control), /fields must be exactly/);
  }
});

test("validated controls remain bound to their exact baseline and subject target", () => {
  const fixture = controlledFixture();
  const foreign = controlledFixture({ identity: "bar" });
  const output = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-control-binding-"));
  assert.throws(() => buildQueue(foreign.inventory, fixture.control), /control baseline/);
  assert.throws(
    () => prepareIdentity(fixture.repository, foreign.inventory, fixture.control, fixture.lease, fixture.id, output),
    /control baseline/,
  );
  assert.throws(
    () => buildInventoryDeltaProof(fixture.repository, foreign.inventory, foreign.inventory, fixture.control, fixture.bundleId),
    /control baseline/,
  );
  assert.throws(() => runGateProducer({
    control: fixture.control,
    baseline_inventory: foreign.inventory,
    subject_inventory: foreign.inventory,
    bundle_id: fixture.bundleId,
    gate: "architecture",
    artifact_id: "foreign-baseline",
    inputs: null,
  }), /control baseline/);

  const subject = structuredClone(fixture.inventory);
  subject.compiled_inventory.build.architecture = "other-architecture";
  resealEvidence(subject);
  assert.throws(() => runGateProducer({
    control: fixture.control,
    baseline_inventory: fixture.inventory,
    subject_inventory: subject,
    bundle_id: fixture.bundleId,
    gate: "architecture",
    artifact_id: "foreign-target",
    inputs: null,
  }), /subject compiled target/);
});

test("subject admission accepts reviewed cross-platform targets and rechecks every disposition", () => {
  const fixture = controlledFixture({
    storageProfiles: {
      "linux-builder": {
        operating_system: "linux",
        architecture: "x86_64",
        execution_host: "runmat-linux-builder",
        volume_roles: {
          source_worktree: { role: "source-worktree", mount_path: "/workspace", filesystem_id: "posix-dev:10", minimum_free_bytes: 1, pause_below_bytes: 2, maximum_observation_age_seconds: 60 },
          target_temp: { role: "target-temp", mount_path: "/mnt/runmat-build", filesystem_id: "posix-dev:11", minimum_free_bytes: 1, pause_below_bytes: 2, maximum_observation_age_seconds: 60 },
        },
      },
    },
  });
  const compiled = structuredClone(fixture.compiledInventory);
  compiled.snapshot.build.operating_system = "linux";
  compiled.snapshot.build.architecture = "x86_64";
  compiled.digest.value = contentDigest(Buffer.from(JSON.stringify(compiled.snapshot))).slice("sha256:".length);
  const dispositions = {
    schema_version: 1,
    kind: "runmat-builtin-dispositions",
    authority: "review-input-only",
    identities: Object.fromEntries(fixture.inventory.identities.map((entry) => [
      entry.identity,
      structuredClone(entry.classification_input),
    ])),
  };
  const subject = buildInventory(fixture.repository, dispositions, {
    revision: REVISION,
    compiledInventory: compiled,
  });
  assert.equal(assertControlSubject(fixture.control, subject), subject);

  const changedDispositions = structuredClone(dispositions);
  changedDispositions.identities.foo = {
    disposition: "internal",
    canonical: null,
    domain: "math",
    family: "basic",
    reason: "Drifted subject classification",
    review: { status: "reviewed", evidence: ["fixture drift"] },
  };
  const drifted = buildInventory(fixture.repository, changedDispositions, {
    revision: REVISION,
    compiledInventory: compiled,
  });
  assert.throws(() => assertControlSubject(fixture.control, drifted), /disposition differs/);
});

test("migration operations require the exact validated authored lease", () => {
  const fixture = controlledFixture();
  const output = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-lease-binding-"));
  const forgedLease = {
    value: fixture.lease.value,
    bundle: fixture.lease.bundle,
  };
  assert.throws(
    () => prepareIdentity(fixture.repository, fixture.inventory, fixture.control, forgedLease, fixture.id, output),
    /exact validated authored lease/,
  );
  assert.throws(
    () => auditMigration(
      fixture.repository,
      fixture.inventory,
      fixture.inventory,
      fixture.control,
      forgedLease,
      { schema_version: 1, kind: "runmat-builtin-migration-batch", identities: [fixture.id] },
      { artifact_id: "forged-lease", changed_paths: [], prepare_results: [], source_dispositions: [], gate_results: [] },
    ),
    /exact validated authored lease/,
  );
});

test("control draft is deterministic, complete, and leaves review judgments unresolved", () => {
  const fixture = controlledFixture();
  const first = buildControlDraft(fixture.inventory);
  const second = buildControlDraft(fixture.inventory);
  assert.deepEqual(first, second);
  assert.equal(first.authority, "unreviewed-scaffold-only");
  assert.deepEqual(first.bundle_drafts, []);
  assert.deepEqual(first.identity_rows.map((entry) => entry.identity), fixture.inventory.identities.map((entry) => entry.identity).sort());
  assert.ok(first.identity_rows.every((entry) => entry.review.status === "unreviewed" && entry.unresolved_fields.includes("disposition") && entry.unresolved_fields.includes("maturity")));
  assert.doesNotThrow(() => parseControlDraft(first, fixture.inventory));
});

test("control draft rejects inferred facts, omissions, tampering, and review claims", () => {
  const fixture = controlledFixture();
  const mutateAndSeal = (mutate) => {
    const value = structuredClone(buildControlDraft(fixture.inventory));
    mutate(value);
    const { digest: _old, ...payload } = value;
    value.digest = evidenceDigest(payload);
    return value;
  };
  assert.throws(() => parseControlDraft(mutateAndSeal((value) => { value.bundle_drafts.push({ id: "guessed" }); }), fixture.inventory), /cannot infer or pre-populate bundles/);
  assert.throws(() => parseControlDraft(mutateAndSeal((value) => { value.identity_rows[0].disposition = "canonical"; }), fixture.inventory), /fields must be exactly/);
  assert.throws(() => parseControlDraft(mutateAndSeal((value) => { value.identity_rows.pop(); }), fixture.inventory), /nonempty array|exactly cover/);
  assert.throws(() => parseControlDraft(mutateAndSeal((value) => { value.identity_rows[0].inventory_row_digest = `sha256:${"0".repeat(64)}`; }), fixture.inventory), /differ from the inventory/);
  assert.throws(() => parseControlDraft(mutateAndSeal((value) => { value.review = { status: "reviewed", evidence: ["self claim"] }; }), fixture.inventory), /cannot claim review/);
});

test("reviewed control freeze binds the exact draft and baseline inventory", () => {
  const fixture = controlledFixture();
  assert.equal(
    freezeReviewedControl(
      fixture.draft,
      fixture.inventory,
      fixture.topology,
      fixture.reviewedControl,
    ).topology_digest,
    fixture.topology.digest,
  );
  const staleTopology = topologyFixture(fixture.inventory, fixture.bundleId, fixture.id, { controlDraftDigest: `sha256:${"0".repeat(64)}` });
  assert.throws(
    () => freezeReviewedControl(fixture.draft, fixture.inventory, staleTopology, fixture.reviewedControl),
    /does not bind the exact/,
  );
  const mutatedDraft = structuredClone(fixture.draft); mutatedDraft.identity_rows[0].compiled_authority_digest = `sha256:${"1".repeat(64)}`;
  const { digest: _old, ...payload } = mutatedDraft; mutatedDraft.digest = evidenceDigest(payload);
  assert.throws(
    () => freezeReviewedControl(mutatedDraft, fixture.inventory, fixture.topology, fixture.reviewedControl),
    /differ from the inventory/,
  );
});

test("expected file removals are bound to exact baseline bytes without a second source-item authority", () => {
  const fixture = controlledFixture({ sidecar: true });
  const value = structuredClone(fixture.controlValue);
  const source = fixture.inventory.source.files.find((entry) => entry.path === "docs/builtins/reference/foo.json");
  value.identity_controls.foo.expected_removals = [{ kind: "file", path: source.path, baseline_digest: source.content_digest }];
  value.identity_controls.foo.baseline_evidence = [{ kind: "sidecar", path: source.path, locator: null, digest: source.content_digest }];
  resealEvidence(value);
  assert.doesNotThrow(() => parseFixtureControl(fixture, value));
  value.identity_controls.foo.baseline_evidence[0].locator = { kind: "rust-item", name: "guessed" };
  assert.throws(() => parseFixtureControl(fixture, value), /source-item locators are obsolete/);
});

test("internal double-underscore identities and null runtime owners are representable", () => {
  const fixture = controlledFixture({ identity: "__register_test_classes" });
  const value = structuredClone(fixture.controlValue);
  const row = value.identity_controls.__register_test_classes;
  row.runtime_owner = null;
  row.expected_authorities.documentation = "none";
  const dispositions = {
    schema_version: 1,
    kind: "runmat-builtin-dispositions",
    authority: "review-input-only",
    identities: { __register_test_classes: {
      disposition: "internal", canonical: null, domain: "math", family: "basic",
      reason: "Generated registration helper", review: { status: "reviewed", evidence: ["fixture review"] },
    } },
  };
  const inventory = buildInventory(fixture.repository, dispositions, {
    revision: REVISION,
    compiledInventory: fixture.compiledInventory,
  });
  const topology = topologyFixture(inventory, fixture.bundleId, "__register_test_classes", { identity: {
    ...fixture.topology.identities.get("__register_test_classes"),
    disposition: { kind: "internal", canonical: null, reason: "Generated registration helper", source: "reviewed-input" },
  } });
  value.topology_digest = topology.digest;
  value.baseline_context.dispositions_digest = inventory.dispositions_digest;
  value.baseline_context.migration_findings_digest = inventory.migration_findings_digest;
  resealEvidence(value);
  assert.equal(parseFixtureControl(fixture, value, inventory, topology).identities.get("__register_test_classes").runtime_owner, null);
});

test("constant-only canonical identities require exact catalog and runtime constant authorities", () => {
  const fixture = controlledFixture();
  const value = structuredClone(fixture.controlValue);
  const row = value.identity_controls.foo;
  row.runtime_owner = null;
  row.expected_authorities.catalog_entry_count = 0;
  row.expected_authorities.catalog_constant_count = 1;
  row.expected_authorities.catalog_package = "crates/runmat-builtins/src/catalog/constant.rs";
  row.expected_authorities.runtime_bindings = [];
  row.expected_authorities.runtime_constants = ["foo"];
  row.expected_authorities.native_link = "not-applicable";
  resealEvidence(value);
  assert.doesNotThrow(() => parseFixtureControl(fixture, value));

  row.expected_authorities.runtime_constants = [];
  assert.throws(() => parseFixtureControl(fixture, value), /canonical callable identity requires a runtime owner/);
});

test("bundle graph rejects cycles, dangling edges, and authored/generated overlap", () => {
  const fixture = controlledFixture();
  const dangling = structuredClone(fixture.controlValue); dangling.bundle_controls[fixture.bundleId].prerequisites = [{ bundle_id: "missing", kind: "semantic" }];
  assert.throws(() => parseFixtureControl(fixture, dangling), /dangling prerequisite/);
  const overlap = structuredClone(fixture.controlValue); overlap.bundle_controls[fixture.bundleId].additional_authored_write_set.push({ kind: "file", path: "crates/runmat-runtime/src/builtins/generated_wasm_registry.rs" });
  assert.throws(() => parseFixtureControl(fixture, overlap), /overlaps authored write scope/);
});

test("disposition v1 is closed and supports reviewed internal double-underscore identities", () => {
  const value = { schema_version: 1, kind: "runmat-builtin-dispositions", authority: "review-input-only", identities: { __helper: { disposition: "internal", canonical: null, domain: "internal", family: "registration", reason: "Generated helper", review: { status: "reviewed", evidence: ["owner review"] } } } };
  assert.doesNotThrow(() => validateDispositionInput(value));
  value.identities.__helper.extra = true;
  assert.throws(() => validateDispositionInput(value), /fields must be exactly/);
});

test("disposition review expands exact reviewed groups without inferred selectors", () => {
  const fixture = controlledFixture();
  const review = dispositionReviewFixture(fixture.inventory);
  const output = compileDispositionReview(review, fixture.inventory);
  assert.deepEqual(Object.keys(output.identities), ["foo"]);
  assert.deepEqual(output.identities.foo, {
    disposition: "canonical",
    canonical: null,
    domain: null,
    family: null,
    reason: null,
    review: {
      status: "reviewed",
      evidence: ["C00 disposition review", "catalog and runtime owner review"],
    },
  });
  assert.doesNotThrow(() => validateDispositionInput(output));

  const inventory = buildInventory(fixture.repository, output, {
    revision: REVISION,
    compiledInventory: fixture.compiledInventory,
  });
  const control = structuredClone(fixture.controlValue);
  const topology = topologyFixture(inventory, fixture.bundleId, "foo", { identity: {
    ...fixture.topology.identities.get("foo"), domain: "reviewed-target",
  } });
  control.topology_digest = topology.digest;
  control.baseline_context.dispositions_digest = inventory.dispositions_digest;
  resealEvidence(control);
  assert.equal(parseFixtureControl(fixture, control, inventory, topology).identities.get("foo").domain, "reviewed-target");
});

test("disposition review rejects omissions, overlap, guesses, and stale baselines", () => {
  const fixture = controlledFixture();
  const omitted = dispositionReviewFixture(fixture.inventory);
  omitted.groups[0].identities = [];
  assert.throws(() => parseDispositionReview(omitted, fixture.inventory), /nonempty array/);

  const duplicate = dispositionReviewFixture(fixture.inventory);
  duplicate.groups.push({ ...structuredClone(duplicate.groups[0]), id: "math-basic-second" });
  assert.throws(() => parseDispositionReview(duplicate, fixture.inventory), /assigned to both/);

  const wildcard = dispositionReviewFixture(fixture.inventory);
  wildcard.groups[0].identities = ["foo.*"];
  assert.throws(() => parseDispositionReview(wildcard, fixture.inventory), /unknown inventory identity/);

  const stale = dispositionReviewFixture(fixture.inventory);
  stale.baseline_inventory_digest = `sha256:${"0".repeat(64)}`;
  assert.throws(() => parseDispositionReview(stale, fixture.inventory), /exact baseline inventory/);
});

test("reviewed target disposition may replace an existing catalog authority", () => {
  const fixture = controlledFixture();
  const review = dispositionReviewFixture(fixture.inventory);
  review.groups[0].disposition = "internal";
  review.groups[0].reason = "Reviewed target removes an accidentally public development binding";
  const dispositions = compileDispositionReview(review, fixture.inventory);
  const subject = buildInventory(fixture.repository, dispositions, {
    revision: REVISION,
    compiledInventory: fixture.compiledInventory,
  });
  assert.equal(subject.identities[0].disposition.kind, "internal");
  assert.equal(subject.diagnostics.length, 0);
});

test("compile-dispositions CLI expands only a baseline-bound reviewed input", () => {
  const fixture = controlledFixture();
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-disposition-review-"));
  const inventoryPath = path.join(directory, "inventory.json");
  const reviewPath = path.join(directory, "review.json");
  const outputPath = path.join(directory, "dispositions.json");
  fs.writeFileSync(inventoryPath, JSON.stringify(fixture.inventory));
  fs.writeFileSync(reviewPath, JSON.stringify(dispositionReviewFixture(fixture.inventory)));
  const cli = path.resolve("scripts/development/builtin-migration-factory.mjs");
  const result = spawnSync(process.execPath, [
    cli, "compile-dispositions", "--review", reviewPath,
    "--baseline-inventory", inventoryPath, "--output", outputPath,
  ], { encoding: "utf8" });
  assert.equal(result.status, 0, result.stderr);
  assert.doesNotThrow(() => validateDispositionInput(JSON.parse(fs.readFileSync(outputPath))));

  const missingReview = spawnSync(process.execPath, [
    cli, "compile-dispositions", "--baseline-inventory", inventoryPath,
  ], { encoding: "utf8" });
  assert.equal(missingReview.status, 2);
  assert.match(missingReview.stderr, /requires --review/);
});

test("gate producer requests cannot inject commands, results, checks, or storage", () => {
  const fixture = controlledFixture();
  const request = { control: fixture.controlValue, baseline_inventory: fixture.inventory, subject_inventory: fixture.inventory, bundle_id: fixture.bundleId, gate: "architecture", artifact_id: "forged", inputs: null };
  assert.throws(() => runGateProducer({ ...request, command: { executable: "/private/tmp/fake", arguments: [], cwd: "/private/tmp" } }), /fields must be exactly/);
  assert.throws(() => runGateProducer({ ...request, control: { value: fixture.controlValue, digest: fixture.control.digest, bundles: new Map() } }), /exact validated control manifest/);
  const fake = structuredClone(fixture.controlValue);
  fake.bundle_controls[fixture.bundleId].gate_plans[0].program.path = "scripts/fake.mjs";
  assert.throws(() => parseFixtureControl(fixture, fake), /absent from or differs from the frozen source snapshot/);
  fs.writeFileSync(path.join(fixture.repository, "nearby-documentation-token.json"), "{\"result\":\"pass\"}\n");
});

test("generated product proof requires two equal runs, checked-in equality, and exact reviewed outputs", () => {
  const fixture = controlledFixture();
  const productPath = "crates/runmat-runtime/src/builtins/generated_wasm_registry.rs";
  const generatorPath = "scripts/regenerate-wasm-registry.mjs";
  const observed = fs.readFileSync(path.join(fixture.repository, productPath));
  const generator = fs.readFileSync(path.join(fixture.repository, generatorPath));
  const observation = { byte_length: observed.length, content_digest: contentDigest(observed) };
  const value = {
    schema_version: 1,
    kind: "runmat-builtin-generated-products-proof",
    authority: "machine-derived-integration-evidence",
    products: [{
      product_id: "wasm-registry",
      path: productPath,
      generator: { path: generatorPath, content_digest: contentDigest(generator) },
      checked_in: observation,
      first: observation,
      second: observation,
      deterministic: true,
      synchronized: true,
    }],
    result: "pass",
  };
  const expected = {
    integration_outputs: fixture.control.bundles.get(fixture.bundleId).integration_outputs,
    source_files: fixture.inventory.source.files,
  };
  const parsed = parseGeneratedProductsProof(value, expected);
  assert.equal(generatedProductChecks(parsed, [fixture.id])[0].result, "pass");
  const stale = structuredClone(value);
  stale.products[0].checked_in = { ...observation, content_digest: `sha256:${"0".repeat(64)}` };
  stale.products[0].synchronized = false;
  stale.result = "fail";
  assert.equal(parseGeneratedProductsProof(stale, expected).result, "fail");
  const forged = structuredClone(stale);
  forged.products[0].synchronized = true;
  assert.throws(() => parseGeneratedProductsProof(forged, expected), /conflicts with observed digests/);
  const extra = structuredClone(value);
  extra.products.push({ ...structuredClone(extra.products[0]), product_id: "invented" });
  extra.products.sort((left, right) => left.product_id.localeCompare(right.product_id));
  assert.throws(() => parseGeneratedProductsProof(extra, expected), /reviewed integration outputs/);
});

test("inventory delta proof admits only the reviewed bundle authority and path transition", () => {
  const fixture = controlledFixture();
  const dispositions = {
    schema_version: 1,
    kind: "runmat-builtin-dispositions",
    authority: "review-input-only",
    identities: Object.fromEntries(fixture.inventory.identities.map((entry) => [
      entry.identity,
      structuredClone(entry.classification_input),
    ])),
  };
  const proof = buildInventoryDeltaProof(fixture.repository, fixture.inventory, fixture.inventory, fixture.control, fixture.bundleId);
  assert.equal(proof.result, "pass");
  assert.equal(inventoryDeltaChecks(proof)[0].result, "pass");

  const wrongBinding = structuredClone(fixture.compiledInventory);
  wrongBinding.snapshot.observed.implementation_provenance[0].source_file = "crates/runmat-runtime/src/builtins/math/basic/other.rs";
  wrongBinding.digest.value = contentDigest(Buffer.from(JSON.stringify(wrongBinding.snapshot))).slice("sha256:".length);
  const wrongBindingInventory = buildInventory(fixture.repository, dispositions, { revision: REVISION, compiledInventory: wrongBinding });
  const wrongBindingProof = buildInventoryDeltaProof(fixture.repository, fixture.inventory, wrongBindingInventory, fixture.control, fixture.bundleId);
  assert.equal(wrongBindingProof.result, "fail");
  assert.match(wrongBindingProof.identities[0].failures.join("\n"), /binding provenance differs/);

  fs.appendFileSync(path.join(fixture.repository, "Cargo.toml"), "\n# unreviewed\n");
  const escapedInventory = buildInventory(fixture.repository, dispositions, { revision: REVISION, compiledInventory: fixture.compiledInventory });
  const escaped = buildInventoryDeltaProof(fixture.repository, fixture.inventory, escapedInventory, fixture.control, fixture.bundleId);
  assert.equal(escaped.result, "fail");
  assert.match(escaped.failures.join("\n"), /source changed outside the reviewed bundle scopes/);
});

test("validate-control CLI requires the full reproducible topology chain", () => {
  const fixture = controlledFixture();
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-control-cli-"));
  const controlPath = path.join(directory, "control.json");
  const inventoryPath = path.join(directory, "inventory.json");
  fs.writeFileSync(controlPath, JSON.stringify(fixture.controlValue));
  fs.writeFileSync(inventoryPath, JSON.stringify(fixture.inventory));
  const cli = path.resolve("scripts/development/builtin-migration-factory.mjs");
  const missing = spawnSync(process.execPath, [cli, "validate-control", "--control", controlPath], { encoding: "utf8" });
  assert.equal(missing.status, 2);
  assert.match(missing.stderr, /requires --baseline-inventory/);
  const missingTopology = spawnSync(process.execPath, [cli, "validate-control", "--control", controlPath, "--baseline-inventory", inventoryPath], { encoding: "utf8" });
  assert.equal(missingTopology.status, 2);
  assert.match(missingTopology.stderr, /requires --component-graph/);
  const tampered = structuredClone(fixture.inventory); tampered.source.digest = `sha256:${"0".repeat(64)}`;
  const tamperedPath = path.join(directory, "tampered.json"); fs.writeFileSync(tamperedPath, JSON.stringify(tampered));
  const rejected = spawnSync(process.execPath, [cli, "validate-control", "--control", controlPath, "--baseline-inventory", tamperedPath], { encoding: "utf8" });
  assert.equal(rejected.status, 2);
});

test("control draft CLI remains separate while control freeze requires reviewed topology", () => {
  const fixture = controlledFixture();
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-control-workflow-"));
  const inventoryPath = path.join(directory, "inventory.json");
  const draftPath = path.join(directory, "draft.json");
  const reviewedPath = path.join(directory, "reviewed.json");
  fs.writeFileSync(inventoryPath, JSON.stringify(fixture.inventory));
  const cli = path.resolve("scripts/development/builtin-migration-factory.mjs");
  const draftResult = spawnSync(process.execPath, [cli, "draft-control", "--baseline-inventory", inventoryPath, "--output", draftPath], { encoding: "utf8" });
  assert.equal(draftResult.status, 0, draftResult.stderr);
  const draft = JSON.parse(fs.readFileSync(draftPath));
  const reviewed = structuredClone(fixture.controlValue);
  fs.writeFileSync(reviewedPath, JSON.stringify(reviewed));
  const freezeResult = spawnSync(process.execPath, [cli, "freeze-control", "--draft", draftPath, "--control", reviewedPath, "--baseline-inventory", inventoryPath], { encoding: "utf8" });
  assert.equal(freezeResult.status, 2);
  assert.match(freezeResult.stderr, /requires --component-graph/);
});

test("control CLI reconstructs the complete reviewed topology chain before freeze, validation, and lease issuance", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-full-control-workflow-"));
  const workflow = writeFullControlWorkflow(directory);
  const cli = path.resolve("scripts/development/builtin-migration-factory.mjs");
  const frozenPath = path.join(directory, "frozen-control.json");
  const leasePath = path.join(directory, "lease.json");
  const scaffoldPath = path.join(directory, "cli-scaffold.json");
  const candidatePath = path.join(directory, "cli-control-candidate.json");
  const initializedReviewDirectory = path.join(directory, "cli-review-templates");
  const authoredReviewDirectory = path.join(directory, "cli-authored-reviews");
  const indexedReviewDirectory = path.join(directory, "cli-indexed-reviews");
  const attestationTemplatePath = path.join(directory, "cli-control-attestation-template.json");
  const attestationReviewPath = path.join(directory, "cli-control-attestation-review.json");
  const attestationPath = path.join(directory, "cli-control-attestation.json");
  const scaffold = spawnSync(process.execPath, [
    cli, "scaffold-control", "--baseline-inventory", workflow.paths.inventory,
    ...workflow.topologyArguments, "--output", scaffoldPath,
  ], { encoding: "utf8" });
  assert.equal(scaffold.status, 0, scaffold.stderr);
  assert.deepEqual(JSON.parse(fs.readFileSync(scaffoldPath)), workflow.values.scaffold);
  const initialize = spawnSync(process.execPath, [
    cli, "init-control-reviews", "--baseline-inventory", workflow.paths.inventory,
    ...workflow.topologyArguments, "--control-scaffold", scaffoldPath,
    "--review-directory", initializedReviewDirectory,
  ], { encoding: "utf8" });
  assert.equal(initialize.status, 0, initialize.stderr);
  assert.equal(JSON.parse(initialize.stdout).bundle_templates, 7);
  copyReviewSetAsAuthoringFiles(workflow.paths.controlReviewSet, authoredReviewDirectory);
  const index = spawnSync(process.execPath, [
    cli, "index-control-reviews", "--baseline-inventory", workflow.paths.inventory,
    ...workflow.topologyArguments, "--control-scaffold", scaffoldPath,
    "--review-directory", authoredReviewDirectory,
    "--review-set-directory", indexedReviewDirectory,
  ], { encoding: "utf8" });
  assert.equal(index.status, 0, index.stderr);
  const indexedManifest = JSON.parse(index.stdout).manifest;
  const compose = spawnSync(process.execPath, [
    cli, "compose-control", "--baseline-inventory", workflow.paths.inventory,
    ...workflow.topologyArguments,
    "--control-scaffold", scaffoldPath,
    "--control-review-set", indexedManifest,
    "--output", candidatePath,
  ], { encoding: "utf8" });
  assert.equal(compose.status, 0, compose.stderr);
  assert.deepEqual(JSON.parse(fs.readFileSync(candidatePath)), workflow.values.controlCandidate);
  const attestationTemplate = spawnSync(process.execPath, [
    cli, "scaffold-control-attestation", "--baseline-inventory", workflow.paths.inventory,
    ...workflow.topologyArguments,
    "--control-scaffold", scaffoldPath, "--control-review-set", indexedManifest,
    "--control-candidate", candidatePath, "--output", attestationTemplatePath,
  ], { encoding: "utf8" });
  assert.equal(attestationTemplate.status, 0, attestationTemplate.stderr);
  const attestationReview = JSON.parse(fs.readFileSync(attestationTemplatePath));
  attestationReview.review = reviewed("full-chain control attestation");
  fs.writeFileSync(attestationReviewPath, `${JSON.stringify(attestationReview, null, 2)}\n`);
  const sealAttestation = spawnSync(process.execPath, [
    cli, "seal-control-attestation", "--baseline-inventory", workflow.paths.inventory,
    ...workflow.topologyArguments,
    "--control-scaffold", scaffoldPath, "--control-review-set", indexedManifest,
    "--control-candidate", candidatePath, "--attestation-review", attestationReviewPath,
    "--output", attestationPath,
  ], { encoding: "utf8" });
  assert.equal(sealAttestation.status, 0, sealAttestation.stderr);
  assert.deepEqual(JSON.parse(fs.readFileSync(attestationPath)), workflow.values.controlAttestation);
  const controlArguments = [
    "--control-scaffold", scaffoldPath,
    "--control-review-set", indexedManifest,
    "--control-candidate", candidatePath,
    "--control-attestation", attestationPath,
  ];
  const freeze = spawnSync(process.execPath, [
    cli, "freeze-control",
    "--baseline-inventory", workflow.paths.inventory,
    ...workflow.topologyArguments, ...controlArguments, "--output", frozenPath,
  ], { encoding: "utf8" });
  assert.equal(freeze.status, 0, freeze.stderr);
  assert.deepEqual(JSON.parse(fs.readFileSync(frozenPath)), workflow.control);

  const validate = spawnSync(process.execPath, [
    cli, "validate-control", "--control", frozenPath,
    "--baseline-inventory", workflow.paths.inventory,
    ...workflow.topologyArguments, ...controlArguments,
  ], { encoding: "utf8" });
  assert.equal(validate.status, 0, validate.stderr);
  assert.deepEqual(JSON.parse(validate.stdout), workflow.control);

  const issue = spawnSync(process.execPath, [
    cli, "issue-lease", "--request", workflow.paths.leaseRequest,
    "--control", frozenPath, "--baseline-inventory", workflow.paths.inventory,
    ...workflow.topologyArguments, ...controlArguments, "--output", leasePath,
  ], { encoding: "utf8" });
  assert.equal(issue.status, 0, issue.stderr);
  const lease = JSON.parse(fs.readFileSync(leasePath));
  assert.equal(lease.control_manifest_digest, workflow.control.digest);
  assert.equal(lease.bundle_id, workflow.bundleId);
  assert.match(lease.digest, /^sha256:[a-f0-9]{64}$/);

  for (const [field, mutate] of [
    ["candidate", (value) => { value.identities.alpha.family = "tampered"; resealEvidence(value); }],
    ["attestation", (value) => { value.review.evidence = ["tampered attestation"]; }],
    ["c01C03Review", (value) => { value.review.evidence = ["tampered cohort review"]; }],
    ["scaffold", (value) => { value.identity_rows[0].review.status = "reviewed"; resealEvidence(value); }],
    ["controlCandidate", (value) => { value.identity_controls.alpha.owner = "tampered"; resealEvidence(value); }],
    ["controlAttestation", (value) => { value.candidate_digest = `sha256:${"0".repeat(64)}`; resealEvidence(value); }],
  ]) {
    const tamperedPath = path.join(directory, `tampered-${field}.json`);
    const tampered = structuredClone(workflow.values[field]);
    mutate(tampered);
    fs.writeFileSync(tamperedPath, JSON.stringify(tampered));
    const arguments_ = [...workflow.topologyArguments, ...workflow.controlArguments]
      .map((entry) => entry === workflow.paths[field] ? tamperedPath : entry);
    const rejected = spawnSync(process.execPath, [
      cli, "validate-control", "--control", frozenPath,
      "--baseline-inventory", workflow.paths.inventory, ...arguments_,
    ], { encoding: "utf8" });
    assert.equal(rejected.status, 2, `${field} tamper unexpectedly passed:\n${rejected.stdout}\n${rejected.stderr}`);
  }
});

test("gate evidence binds the subject execution target through production and later sealing", () => {
  const fixture = controlledFixture();
  const expected = { source_revision: fixture.inventory.source.revision, source_digest: fixture.inventory.source.digest, baseline_source_revision: fixture.inventory.source.revision, baseline_inventory_digest: fixture.inventory.digest, subject_inventory_digest: fixture.inventory.digest, control_manifest_digest: fixture.control.digest, bundle_id: fixture.bundleId, storage_policy: fixture.control.value.storage_policy, compiled_build: fixture.inventory.compiled_inventory.build, execution_targets: fixture.control.executionTargets };
  const stale = gate(fixture, "architecture"); stale.storage_admission.observed_at = "2026-09-10T23:00:00.000Z";
  assert.throws(() => parseGateResult(stale, expected), /time bound/);
  const filesystem = gate(fixture, "architecture"); filesystem.storage_admission.volumes[0].filesystem_id = "posix-dev:3";
  assert.throws(() => parseGateResult(filesystem, expected), /differs from reviewed/);
  const repositoryVolume = gate(fixture, "architecture");
  repositoryVolume.storage_admission.path_bindings.repository.filesystem_id = "posix-dev:2";
  assert.throws(() => parseGateResult(repositoryVolume, expected), /source-worktree volume/);
  const targetVolume = gate(fixture, "architecture");
  targetVolume.storage_admission.path_bindings.cargo_target.filesystem_id = "posix-dev:1";
  assert.throws(() => parseGateResult(targetVolume, expected), /target-temp volume/);
  const environment = gate(fixture, "architecture");
  environment.producer_evidence.invocation.environment.CARGO_TARGET_DIR = "/private/tmp/unreviewed-target";
  assert.throws(() => parseGateResult(environment, expected), /environment differs/);
  const primaryTool = gate(fixture, "architecture");
  primaryTool.producer_evidence.contract.primary_tool = "git";
  assert.throws(() => parseGateResult(primaryTool, expected), /executable differs from the primary reviewed tool/);
  const toolPath = gate(fixture, "architecture");
  toolPath.producer_evidence.invocation.tools.find((entry) => entry.role === "node").path = toolPath.producer_evidence.invocation.tools.find((entry) => entry.role === "git").path;
  assert.throws(() => parseGateResult(toolPath, expected), /node: producer tool bytes differ/);
  const toolEnvironment = gate(fixture, "architecture");
  toolEnvironment.producer_evidence.invocation.tool_environment.removed = [];
  assert.throws(() => parseGateResult(toolEnvironment, expected), /does not enforce the reviewed tool selection/);
  const paused = gate(fixture, "architecture");
  paused.storage_admission.volumes[0].available_bytes = 1;
  paused.storage_admission.volumes[0].status = "paused";
  assert.throws(() => parseGateResult(paused, expected), /cannot pass below/);
  const rejected = gate(fixture, "architecture");
  rejected.storage_admission.volumes[0].available_bytes = 0;
  rejected.storage_admission.volumes[0].status = "rejected";
  assert.throws(() => parseGateResult(rejected, expected), /cannot pass below/);
  const foreignPolicy = structuredClone(fixture.control.value.storage_policy);
  foreignPolicy.host_profiles["linux-host"] = {
    operating_system: "linux", architecture: "x86_64", execution_host: "linux-builder",
    volume_roles: structuredClone(foreignPolicy.host_profiles["fixture-host"].volume_roles),
  };
  const foreign = gate(fixture, "architecture");
  foreign.execution_target = { operating_system: "linux", architecture: "x86_64" };
  foreign.storage_admission.profile_id = "linux-host";
  foreign.storage_admission.execution_host = "linux-builder";
  assert.throws(() => parseGateResult(foreign, { ...expected, storage_policy: foreignPolicy }), /execution target differs from the subject build target/);

  const sealPlans = new Map([...fixture.control.bundles.get(fixture.bundleId).gate_plans].map(([gateName, plan]) => [
    gateName,
    {
      ...plan,
      program: {
        ...plan.program,
        approved_toolchains: [
          { operating_system: "linux", architecture: "x86_64", tools: plan.program.approved_toolchains[0].tools },
          ...plan.program.approved_toolchains,
        ],
      },
    },
  ]));
  const sealExpected = {
    ...expected,
    compiled_build: undefined,
    execution_targets: [
      { operating_system: "linux", architecture: "x86_64" },
      ...fixture.control.executionTargets,
    ],
    storage_policy: foreignPolicy,
    gate_plans: sealPlans,
    repository: fixture.repository,
  };
  assert.equal(parseGateResult(foreign, sealExpected).execution_target.operating_system, "linux");
  const unreviewedTarget = structuredClone(foreign);
  unreviewedTarget.execution_target = { operating_system: "windows", architecture: "x86_64" };
  assert.throws(() => parseGateResult(unreviewedTarget, sealExpected), /absent from the reviewed control target set/);
  const samePlatformPolicy = structuredClone(fixture.control.value.storage_policy);
  samePlatformPolicy.host_profiles["other-macos-host"] = {
    operating_system: fixture.inventory.compiled_inventory.build.operating_system,
    architecture: fixture.inventory.compiled_inventory.build.architecture,
    execution_host: "other-macos-builder",
    volume_roles: structuredClone(samePlatformPolicy.host_profiles["fixture-host"].volume_roles),
  };
  const wrongHost = gate(fixture, "architecture"); wrongHost.storage_admission.profile_id = "other-macos-host";
  assert.throws(
    () => parseGateResult(wrongHost, { ...expected, storage_policy: samePlatformPolicy }),
    /execution host differs from reviewed profile/,
  );
  const forged = gate(fixture, "architecture"); forged.producer_evidence.process.exit_code = 1;
  assert.throws(() => parseGateResult(forged, expected), /conflicts|inconsistent/);
  const artifactExpected = { ...expected, gate_plans: fixture.control.bundles.get(fixture.bundleId).gate_plans, compiled_build: fixture.inventory.compiled_inventory.build, repository: fixture.repository };
  const missing = gate(fixture, "catalog-contract");
  missing.artifacts = [];
  missing.storage_admission.path_bindings.artifacts = [];
  assert.throws(() => parseGateResult(missing, artifactExpected), /does not cover the reviewed artifact roles/);
  const unreviewed = gate(fixture, "architecture");
  const unreviewedPath = path.join(fixture.repository, "..", "unreviewed-artifact.json");
  const unreviewedBytes = Buffer.from("{}\n"); fs.writeFileSync(unreviewedPath, unreviewedBytes);
  unreviewed.artifacts.push({ role: "invented", path: unreviewedPath, byte_length: unreviewedBytes.length, content_digest: contentDigest(unreviewedBytes) });
  unreviewed.storage_admission.path_bindings.artifacts.push({ role: "invented", path: unreviewedPath, filesystem_id: "posix-dev:2" });
  assert.throws(() => parseGateResult(unreviewed, artifactExpected), /unreviewed artifact role/);
  const tampered = gate(fixture, "runtime-binding"); fs.appendFileSync(tampered.artifacts[0].path, "tamper");
  assert.throws(() => parseGateResult(tampered, artifactExpected), /artifact bytes differ/);
});

test("storage admission binds the real repository, build, temporary, and artifact paths before execution", () => {
  const repository = repositoryFixture();
  const targetRoot = path.join(path.dirname(repository), "target-volume");
  fs.mkdirSync(targetRoot);
  const stats = fs.statSync(repository, { bigint: true });
  const filesystemId = process.platform === "win32"
    ? `windows-volume:${stats.dev.toString(16).padStart(8, "0")}`
    : `posix-dev:${stats.dev}`;
  const volume = (role, mountPath) => ({
    role,
    mount_path: fs.realpathSync(mountPath),
    filesystem_id: filesystemId,
    minimum_free_bytes: 1,
    pause_below_bytes: 2,
    maximum_observation_age_seconds: 60,
  });
  const policy = { host_profiles: { fixture: {
    operating_system: "fixture-os",
    architecture: "fixture-arch",
    execution_host: os.hostname(),
    volume_roles: {
      source_worktree: volume("source-worktree", repository),
      target_temp: volume("target-temp", targetRoot),
    },
  } } };
  const artifactOutput = path.join(targetRoot, "evidence", "architecture.json");
  const prepared = prepareGateStorage(
    policy,
    { operating_system: "fixture-os", architecture: "fixture-arch" },
    repository,
    [{ role: "architecture-proof", output: artifactOutput }],
    { control_digest: `sha256:${"a".repeat(64)}`, bundle_id: "fixture-bundle", artifact_id: "fixture-artifact" },
  );
  assert.equal(prepared.admission.path_bindings.repository.path, fs.realpathSync(repository));
  assert.equal(
    prepared.admission.path_bindings.artifacts[0].path,
    path.join(fs.realpathSync(targetRoot), "evidence", "architecture.json"),
  );
  assert.equal(prepared.environment.CARGO_TARGET_DIR, prepared.admission.path_bindings.cargo_target.path);
  assert.equal(prepared.environment.TMPDIR, prepared.admission.path_bindings.temporary.path);
  assert.ok(fs.statSync(prepared.environment.CARGO_TARGET_DIR).isDirectory());
  assert.ok(fs.statSync(prepared.environment.TMPDIR).isDirectory());
  assert.equal(storageStatus(0, 1, 2), "rejected");
  assert.equal(storageStatus(1, 1, 2), "paused");
  assert.equal(storageStatus(2, 1, 2), "admitted");
});

test("queue is derived by bundle and carries prerequisite, scope, and maturity facts", () => {
  const fixture = controlledFixture();
  const queue = buildQueue(fixture.inventory, fixture.control);
  assert.equal(queue.schema_version, 2);
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
  assert.throws(() => parseLease(extra, fixture.control), /fields must be exactly/);
});

test("lease issuance derives immutable scopes from a reviewed request and control", () => {
  const fixture = controlledFixture();
  const request = {
    schema_version: 1, kind: "runmat-builtin-migration-lease-request", authority: "reviewed-development-request",
    control_manifest_digest: fixture.control.digest, bundle_id: fixture.bundleId, lease_id: "lane-17",
    owner: "builtin-migrator", issued_at: "2026-09-11T01:00:00.000Z", expires_at: "2026-09-11T05:00:00.000Z",
    review: { status: "reviewed", evidence: ["C00 assignment review"] },
  };
  assert.doesNotThrow(() => parseLeaseRequest(request, fixture.control));
  const first = issueLease(request, fixture.control);
  const second = issueLease(request, fixture.control);
  assert.deepEqual(first, second);
  assert.deepEqual(first.authored_write_set, fixture.control.bundles.get(fixture.bundleId).authored_write_set);
  assert.doesNotThrow(() => parseLease(first, fixture.control));

  const injected = structuredClone(request); injected.authored_write_set = [{ kind: "tree", path: "." }];
  assert.throws(() => issueLease(injected, fixture.control), /fields must be exactly/);
  const unreviewed = structuredClone(request); unreviewed.review = { status: "unreviewed", evidence: [] };
  assert.throws(() => issueLease(unreviewed, fixture.control), /must be reviewed/);
  const widened = structuredClone(first); widened.authored_write_set.push({ kind: "tree", path: "crates" });
  const { digest: _old, ...payload } = widened; widened.digest = evidenceDigest(payload);
  assert.throws(() => parseLease(widened, fixture.control), /differs from reviewed bundle scope/);
});

test("prepare v2 is source-neutral and inventories every legacy JSON leaf", () => {
  const fixture = controlledFixture({ sidecar: true });
  const output = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-review-"));
  const result = prepareIdentity(fixture.repository, fixture.inventory, fixture.control, fixture.lease, "foo", output);
  assert.equal(result.schema_version, 2);
  assert.equal(result.source_changes, false);
  const checklistPath = path.join(result.workspace, "source-field-disposition.json");
  const checklist = JSON.parse(fs.readFileSync(checklistPath));
  assert.ok(checklist.sources[0].leaves.length >= 3);
  assert.ok(checklist.sources[0].leaves.every((leaf) => leaf.disposition === "pending"));
  assert.throws(() => parseCompletedSourceDisposition(checklist, "foo", result.checklist_baseline_digest), /not reviewed/);
});

test("completed field dispositions require exact prepared leaves and review evidence", () => {
  const fixture = controlledFixture({ sidecar: true });
  const output = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-review-"));
  const result = prepareIdentity(fixture.repository, fixture.inventory, fixture.control, fixture.lease, "foo", output);
  const checklist = JSON.parse(fs.readFileSync(path.join(result.workspace, "source-field-disposition.json")));
  checklist.review = { status: "reviewed", evidence: ["review"] };
  for (const source of checklist.sources) for (const leaf of source.leaves) { leaf.disposition = "preserved"; leaf.destination = { kind: "catalog-documentation", catalog_identity: "foo", pointer: leaf.pointer, value_digest: leaf.value_digest }; }
  assert.doesNotThrow(() => parseCompletedSourceDisposition(checklist, "foo", result.checklist_baseline_digest));
  checklist.sources[0].leaves.pop();
  assert.throws(() => parseCompletedSourceDisposition(checklist, "foo", result.checklist_baseline_digest), /prepared baseline/);
});

test("audit v4 cannot pass on file presence or example tokens without exact gate evidence", () => {
  const fixture = controlledFixture();
  const output = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-review-"));
  const prepared = prepareIdentity(fixture.repository, fixture.inventory, fixture.control, fixture.lease, "foo", output);
  const batch = { schema_version: 1, kind: "runmat-builtin-migration-batch", identities: ["foo"] };
  const absent = auditMigration(fixture.repository, fixture.inventory, fixture.inventory, fixture.control, fixture.lease, batch, { artifact_id: "audit-foo", changed_paths: [], prepare_results: [prepared], source_dispositions: [], gate_results: [] });
  assert.equal(absent.result, "fail");
  assert.ok(absent.identities[0].failures.some((entry) => entry.code === "required-gate-missing"));
  const gates = ["catalog-contract", "runtime-binding", "documentation-cutover", "architecture"].map((name) => gate(fixture, name));
  const passed = auditMigration(fixture.repository, fixture.inventory, fixture.inventory, fixture.control, fixture.lease, batch, { artifact_id: "audit-foo", changed_paths: [], prepare_results: [prepared], source_dispositions: [], gate_results: gates });
  assert.equal(passed.result, "pass");
  const stale = structuredClone(gates); stale[0].source_digest = `sha256:${"b".repeat(64)}`;
  assert.equal(auditMigration(fixture.repository, fixture.inventory, fixture.inventory, fixture.control, fixture.lease, batch, { artifact_id: "audit-foo", changed_paths: [], prepare_results: [prepared], source_dispositions: [], gate_results: stale }).result, "fail");
  const impersonated = structuredClone(gates); impersonated[0].producer_evidence.contract.producer_source_digest = `sha256:${"e".repeat(64)}`;
  const rejected = auditMigration(fixture.repository, fixture.inventory, fixture.inventory, fixture.control, fixture.lease, batch, { artifact_id: "audit-foo", changed_paths: [], prepare_results: [prepared], source_dispositions: [], gate_results: impersonated });
  assert.ok(rejected.global_failures.some((entry) => entry.detail.includes("differs from the reviewed gate plan")));
});

test("source-field destinations require value-digest reconciliation from the documentation producer", () => {
  const fixture = controlledFixture({ sidecar: true });
  const output = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-review-"));
  const prepared = prepareIdentity(fixture.repository, fixture.inventory, fixture.control, fixture.lease, "foo", output);
  const disposition = JSON.parse(fs.readFileSync(path.join(prepared.workspace, "source-field-disposition.json")));
  disposition.review = { status: "reviewed", evidence: ["review"] };
  for (const source of disposition.sources) for (const leaf of source.leaves) { leaf.disposition = "preserved"; leaf.destination = { kind: "catalog-documentation", catalog_identity: "foo", pointer: leaf.pointer, value_digest: leaf.value_digest }; }
  const gates = ["catalog-contract", "runtime-binding", "documentation-cutover", "architecture"].map((name) => gate(fixture, name));
  const batch = { schema_version: 1, kind: "runmat-builtin-migration-batch", identities: ["foo"] };
  const missing = auditMigration(fixture.repository, fixture.inventory, fixture.inventory, fixture.control, fixture.lease, batch, { artifact_id: "audit-destination", changed_paths: [], prepare_results: [prepared], source_dispositions: [disposition], gate_results: gates });
  assert.ok(missing.identities[0].failures.some((entry) => entry.code === "destination-proof-missing"));
  const documentation = gates.find((entry) => entry.gate === "documentation-cutover");
  for (const source of disposition.sources) for (const leaf of source.leaves) documentation.checks.push({ id: `destination:foo:${source.path}:${leaf.pointer}:${leaf.destination.kind}:${leaf.destination.pointer}`, result: "pass", evidence_digest: leaf.value_digest });
  const reconciled = auditMigration(fixture.repository, fixture.inventory, fixture.inventory, fixture.control, fixture.lease, batch, { artifact_id: "audit-destination", changed_paths: [], prepare_results: [prepared], source_dispositions: [disposition], gate_results: gates });
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
  assert.throws(() => parseDocumentationCutoverArtifact(alteredArtifact, expected), /exactly cover/);
});

test("batch schema rejects unsafe and case-colliding identities", () => {
  assert.deepEqual(parseBatch({ schema_version: 1, kind: "runmat-builtin-migration-batch", identities: ["__helper"] }), ["__helper"]);
  assert.throws(() => parseBatch({ schema_version: 1, kind: "runmat-builtin-migration-batch", identities: ["foo", "Foo"] }), /unique/);
  assert.throws(() => parseBatch({ schema_version: 1, kind: "runmat-builtin-migration-batch", identities: ["../foo"] }), /invalid/);
});

function writeFullControlWorkflow(directory) {
  const input = fullTopologyChainFixture();
  const candidate = composeTopologyCandidate(input);
  const attestation = {
    schema_version: 1,
    kind: "runmat-builtin-topology-attestation",
    authority: "reviewer-authored-development-input",
    program: "RM-1064/C00-C07",
    candidate_digest: candidate.digest,
    input_digests: candidateInputDigests(candidate),
    review: reviewed("full-chain fixture attestation"),
  };
  const topology = freezeReviewedTopology(candidate, attestation, candidate);
  const topologyView = reviewedTopologyView(parseReviewedTopology(topology, candidate, attestation, candidate));
  const scaffold = buildControlOverlayScaffold(input.baselineInventory, input.controlDraft, topologyView);
  const policies = fullChainControl(input.baselineInventory, topology);
  const controlReviewSet = writeFullControlReviewSet(
    directory,
    input.baselineInventory,
    topologyView,
    scaffold,
    policies,
  );
  const reviewSet = loadControlReviewSet(controlReviewSet, {
    scaffold,
    topology: topologyView,
    inventory: input.baselineInventory,
  });
  const controlCandidate = composeControlCandidate({
    inventory: input.baselineInventory,
    topology: topologyView,
    scaffold,
    reviewSet,
  });
  const controlAttestationPayload = {
    schema_version: 1,
    kind: "runmat-builtin-migration-control-attestation",
    authority: "reviewer-authored-development-input",
    program: "RM-1064/C00-C07",
    candidate_digest: controlCandidate.digest,
    input_digests: controlCandidateInputDigests(controlCandidate),
    review: reviewed("full-chain control attestation"),
  };
  const controlAttestation = {
    ...controlAttestationPayload,
    digest: evidenceDigest(controlAttestationPayload),
  };
  const control = validateControlReviewChain(
    controlCandidate,
    controlAttestation,
    controlCandidate,
  ).controlValue;
  const bundleId = Object.keys(topology.bundles).sort()[0];
  const leaseRequest = {
    schema_version: 1,
    kind: "runmat-builtin-migration-lease-request",
    authority: "reviewed-development-request",
    control_manifest_digest: control.digest,
    bundle_id: bundleId,
    lease_id: "full-chain-cli-lease",
    owner: "fixture-migrator",
    issued_at: "2026-09-11T01:00:00.000Z",
    expires_at: "2026-09-11T02:00:00.000Z",
    review: reviewed("full-chain fixture lease review"),
  };
  const values = {
    inventory: input.baselineInventory,
    componentGraph: input.componentGraph,
    draft: input.controlDraft,
    c01C03Review: input.reviewValues.get("c01_c03"),
    c04C05Review: input.reviewValues.get("c04_c05"),
    c06C07Review: input.reviewValues.get("c06_c07"),
    reconciliation: input.reconciliationValue,
    stabilityCorrections: input.stabilityCorrectionsValue,
    candidate,
    attestation,
    topology,
    scaffold,
    controlCandidate,
    controlAttestation,
    control,
    leaseRequest,
  };
  const paths = Object.fromEntries(Object.entries(values).map(([name, value]) => {
    const target = path.join(directory, `${name}.json`);
    fs.writeFileSync(target, `${JSON.stringify(value, null, 2)}\n`);
    return [name, target];
  }));
  const topologyArguments = [
    "--topology", paths.topology,
    "--candidate", paths.candidate,
    "--attestation", paths.attestation,
    "--component-graph", paths.componentGraph,
    "--draft", paths.draft,
    "--c01-c03-review", paths.c01C03Review,
    "--c04-c05-review", paths.c04C05Review,
    "--c06-c07-review", paths.c06C07Review,
    "--reconciliation", paths.reconciliation,
    "--stability-corrections", paths.stabilityCorrections,
  ];
  const controlArguments = [
    "--control-scaffold", paths.scaffold,
    "--control-review-set", controlReviewSet,
    "--control-candidate", paths.controlCandidate,
    "--control-attestation", paths.controlAttestation,
  ];
  return { values, paths: { ...paths, controlReviewSet }, topologyArguments, controlArguments, control, bundleId };
}

function fullChainControl(inventory, topology) {
  const review = reviewed("full-chain control review");
  const sourceDigest = inventory.source.files[0].content_digest;
  const approvedToolchains = [{
    operating_system: inventory.compiled_inventory.build.operating_system,
    architecture: inventory.compiled_inventory.build.architecture,
    tools: [{ role: "node", content_digest: `sha256:${"7".repeat(64)}` }],
  }];
  const gatePlans = [
    fixtureGatePlan("architecture", "exit_status", [], sourceDigest, approvedToolchains),
    fixtureGatePlan("catalog-contract", "compiled_inventory", ["compiled-inventory"], sourceDigest, approvedToolchains),
  ];
  const bundleControls = Object.fromEntries(Object.keys(topology.bundles).sort().map((bundleId) => [bundleId, {
    prerequisites: [],
    additional_authored_write_set: [{ kind: "file", path: `crates/runmat-runtime/src/builtins/fixture/control/${bundleId}.rs` }],
    integration_outputs: [],
    gate_plans: gatePlans,
    owner_role: "fixture-migrator",
    complexity: { class: "low", weight: topology.bundles[bundleId].identities.length, basis: ["full-chain fixture review"] },
    review,
  }]));
  const inventoryByIdentity = new Map(inventory.identities.map((entry) => [entry.identity, entry]));
  const identityControls = Object.fromEntries(Object.keys(topology.identities).sort().map((identity) => {
    const row = inventoryByIdentity.get(identity);
    const maturity = Object.fromEntries(MATURITY_GATES.map((gate) => [gate, gate === "identity"
      ? { applicability: "required", reason: null, evidence: [] }
      : { applicability: "not-applicable", reason: "Outside this control-workflow fixture", evidence: ["full-chain fixture review"] }]));
    return [identity, {
      public_spelling: identity,
      runtime_owner: row.ownership.runtime[0],
      shared_dependencies: [],
      complexity: { class: "low", weight: 1, basis: ["full-chain fixture review"] },
      maturity,
      expected_authorities: {
        catalog_package: null,
        catalog_entry_count: 0,
        catalog_constant_count: 0,
        documentation: "none",
        runtime_bindings: [],
        runtime_constants: [],
        native_link: "not-applicable",
        wasm_registry: "not-applicable",
      },
      expected_removals: [],
      baseline_evidence: [],
      owner: "fixture-migrator",
      review,
    }];
  }));
  const payload = {
    schema_version: 2,
    kind: "runmat-builtin-migration-control-manifest",
    authority: "reviewed-development-control",
    program: "RM-1064/C00-C07",
    topology_digest: topology.digest,
    baseline_context: {
      source_digest: inventory.source.digest,
      dispositions_digest: inventory.dispositions_digest,
      migration_findings_digest: inventory.migration_findings_digest,
      compiled_target: structuredClone(inventory.compiled_inventory.build),
    },
    cohorts: ["prerequisite", "A", "B", "C", "D", "E", "F", "G"].map((semantic, order) => ({ id: `C0${order}`, semantic, order })),
    bundle_controls: bundleControls,
    identity_controls: identityControls,
    migration_findings: {
      schema_version: 1,
      kind: "runmat-builtin-migration-finding-dispositions",
      rows: [],
      review,
    },
    exception_manifest: { entries: [], review },
    execution_targets: [{
      operating_system: inventory.compiled_inventory.build.operating_system,
      architecture: inventory.compiled_inventory.build.architecture,
    }],
    storage_policy: {
      host_profiles: { "fixture-host": {
        operating_system: inventory.compiled_inventory.build.operating_system,
        architecture: inventory.compiled_inventory.build.architecture,
        execution_host: os.hostname(),
        volume_roles: {
          source_worktree: { role: "source-worktree", mount_path: "/source", filesystem_id: "posix-dev:1", minimum_free_bytes: 1, pause_below_bytes: 2, maximum_observation_age_seconds: 60 },
          target_temp: { role: "target-temp", mount_path: "/target", filesystem_id: "posix-dev:2", minimum_free_bytes: 1, pause_below_bytes: 2, maximum_observation_age_seconds: 60 },
        },
      } },
      targets_must_be_disjoint: true,
      occt_default: "disabled-unless-affected",
    },
    review,
  };
  return { ...payload, digest: evidenceDigest(payload) };
}

function writeFullControlReviewSet(directory, inventory, topology, scaffold, policies) {
  const root = path.join(directory, "control-review-set");
  fs.mkdirSync(path.join(root, "bundles"), { recursive: true });
  const programs = new Map();
  for (const control of Object.values(policies.bundle_controls)) {
    for (const plan of control.gate_plans) programs.set(JSON.stringify(plan.program), plan.program);
  }
  const orderedPrograms = [...programs.entries()].sort(([left], [right]) => left.localeCompare(right));
  const profileIds = new Map(orderedPrograms.map(([key], index) => [key, `program-${index + 1}`]));
  const programProfiles = Object.fromEntries(orderedPrograms.map(([, program], index) => [`program-${index + 1}`, {
    program,
    review: reviewed("full-chain executable review"),
  }]));
  const globalPayload = {
    schema_version: 1,
    kind: "runmat-builtin-migration-global-control-review",
    authority: "reviewer-authored-development-input",
    program: "RM-1064/C00-C07",
    bindings: {
      scaffold_digest: scaffold.digest,
      topology_digest: topology.digest,
      migration_finding_rows_digest: evidenceDigest(scaffold.migration_finding_rows),
    },
    program_profiles: programProfiles,
    migration_findings: policies.migration_findings,
    exception_manifest: policies.exception_manifest,
    execution_targets: policies.execution_targets,
    storage_policy: policies.storage_policy,
    review: reviewed("full-chain global control review"),
  };
  const global = { ...globalPayload, digest: evidenceDigest(globalPayload) };
  const globalBytes = Buffer.from(`${JSON.stringify(global, null, 2)}\n`);
  fs.writeFileSync(path.join(root, "global.json"), globalBytes);

  const references = [];
  for (const bundleId of Object.keys(policies.bundle_controls).sort()) {
    const topologyBundle = topology.bundles.get(bundleId);
    const scaffoldBundle = scaffold.bundle_rows.find((row) => row.bundle_id === bundleId);
    const control = policies.bundle_controls[bundleId];
    const bundleControl = {
      ...structuredClone(control),
      gate_plans: control.gate_plans.map(({ program, ...plan }) => ({
        gate: plan.gate,
        program_profile_id: profileIds.get(JSON.stringify(program)),
        arguments: [...plan.arguments],
        working_directory: plan.working_directory,
        parser: plan.parser,
        expected_artifact_roles: [...plan.expected_artifact_roles],
      })),
    };
    const identityRows = topologyBundle.identities.map((identity) => ({
      identity,
      scaffold_identity_row_digest: evidenceDigest(scaffold.identity_rows.find((row) => row.identity === identity)),
      topology_identity_digest: evidenceDigest(topology.identities.get(identity)),
    }));
    const bundlePayload = {
      schema_version: 1,
      kind: "runmat-builtin-migration-bundle-control-review",
      authority: "reviewer-authored-development-input",
      program: "RM-1064/C00-C07",
      bindings: {
        scaffold_digest: scaffold.digest,
        topology_digest: topology.digest,
        bundle_id: bundleId,
        scaffold_bundle_row_digest: evidenceDigest(scaffoldBundle),
        topology_bundle_row_digest: evidenceDigest(topologyBundle),
        identity_rows: identityRows,
      },
      bundle_control: bundleControl,
      identity_controls: Object.fromEntries(topologyBundle.identities.map((identity) => [identity, policies.identity_controls[identity]])),
      review: reviewed("full-chain bundle control review"),
    };
    const bundle = { ...bundlePayload, digest: evidenceDigest(bundlePayload) };
    const relative = `bundles/${bundleId}.json`;
    const bytes = Buffer.from(`${JSON.stringify(bundle, null, 2)}\n`);
    fs.writeFileSync(path.join(root, relative), bytes);
    references.push({ bundle_id: bundleId, path: relative, content_digest: contentDigest(bytes) });
  }
  const manifestPayload = {
    schema_version: 1,
    kind: "runmat-builtin-migration-control-review-set",
    authority: "content-addressed-review-index-only",
    program: "RM-1064/C00-C07",
    bindings: {
      scaffold_digest: scaffold.digest,
      topology_digest: topology.digest,
      inventory_digest: inventory.digest,
    },
    global_review: { path: "global.json", content_digest: contentDigest(globalBytes) },
    bundle_reviews: references,
  };
  const manifest = { ...manifestPayload, digest: evidenceDigest(manifestPayload) };
  const manifestPath = path.join(root, "review-set.json");
  fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  return manifestPath;
}

function copyReviewSetAsAuthoringFiles(manifestPath, target) {
  const root = path.dirname(manifestPath);
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  fs.mkdirSync(path.join(target, "bundles"), { recursive: true });
  for (const relative of [manifest.global_review.path, ...manifest.bundle_reviews.map((entry) => entry.path)]) {
    const value = JSON.parse(fs.readFileSync(path.join(root, relative), "utf8"));
    delete value.digest;
    fs.writeFileSync(path.join(target, relative), `${JSON.stringify(value, null, 2)}\n`);
  }
}

function fixtureGatePlan(gate, parser, expectedArtifactRoles, sourceDigest, approvedToolchains) {
  return {
    gate,
    program: {
      kind: "repository_script",
      path: "scripts/development/check-architecture-boundaries.mjs",
      content_digest: sourceDigest,
      approved_toolchains: approvedToolchains,
    },
    arguments: [],
    working_directory: "repository",
    parser,
    expected_artifact_roles: expectedArtifactRoles,
  };
}

function reviewed(evidence) {
  return { status: "reviewed", evidence: [evidence] };
}

function parseFixtureControl(fixture, value = fixture.controlValue, inventory = fixture.inventory, topology = fixture.topology) {
  const candidate = structuredClone(value);
  if (topology !== fixture.topology) {
    candidate.topology_digest = topology.digest;
    candidate.inputs.baseline_inventory_digest = topology.baseline.inventory_digest;
    candidate.inputs.control_draft_digest = topology.baseline.control_draft_digest;
    candidate.inputs.reviewed_topology_digest = topology.digest;
    resealEvidence(candidate);
  }
  return validateControlManifestStructure(candidate, { inventory, reviewedTopology: topology });
}

function resealEvidence(value) {
  delete value.digest;
  value.digest = evidenceDigest(value);
  return value;
}
