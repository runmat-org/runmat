import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { auditMigration, parseBatch } from "../audit.mjs";
import { parseControlManifest } from "../control.mjs";
import { buildControlDraft, freezeReviewedControl, parseControlDraft } from "../control-draft.mjs";
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
import { cleanupRepositoryFixtures, compiledInventoryFixture, controlledFixture, gate, repositoryFixture, REVISION } from "./helpers.mjs";

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
      domain: "math",
      family: "basic",
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
  assert.throws(() => parseControlManifest(missing, fixture.inventory), /do not exactly cover/);
  const unknown = structuredClone(fixture.controlValue); unknown.migration_findings.rows[0].unexpected = true;
  assert.throws(() => parseControlManifest(unknown, fixture.inventory), /fields must be exactly/);
});

test("control is closed, reviewed, reciprocal, and rejects case-fold ambiguity", () => {
  const fixture = controlledFixture();
  assert.equal(fixture.control.identities.get("foo").public_spelling, "foo");
  const future = structuredClone(fixture.controlValue); future.schema_version = 2;
  assert.throws(() => parseControlManifest(future), /schema_version 1/);
  const extra = structuredClone(fixture.controlValue); extra.unreviewed = true;
  assert.throws(() => parseControlManifest(extra), /fields must be exactly/);
  const collision = structuredClone(fixture.controlValue); collision.identities.Foo = structuredClone(collision.identities.foo); collision.identities.Foo.identity = "foo";
  assert.throws(() => parseControlManifest(collision), /collide case-insensitively/);
  const spelling = structuredClone(fixture.controlValue); spelling.identities.foo.public_spelling = "bar";
  assert.throws(() => parseControlManifest(spelling), /case-fold to the identity key/);
  const observedSpelling = structuredClone(fixture.controlValue); observedSpelling.identities.foo.public_spelling = "Foo";
  assert.throws(() => parseControlManifest(observedSpelling, fixture.inventory), /public spelling differs from the reviewed inventory/);
  const domain = structuredClone(fixture.controlValue); domain.bundles[fixture.bundleId].domain = "other"; domain.identities.foo.domain = "other";
  assert.throws(() => parseControlManifest(domain, fixture.inventory), /domain or family differs from the reviewed inventory/);
  const disposition = structuredClone(fixture.controlValue); disposition.identities.foo.disposition = { kind: "internal", reason: "Changed after disposition review", evidence: ["late control edit"] }; disposition.identities.foo.runtime_owner = null;
  assert.throws(() => parseControlManifest(disposition, fixture.inventory), /disposition differs from the reviewed inventory/);
  const unsafeStorage = structuredClone(fixture.controlValue); unsafeStorage.storage_policy.volume_roles.target_temp.filesystem_id = unsafeStorage.storage_policy.volume_roles.source_worktree.filesystem_id;
  assert.throws(() => parseControlManifest(unsafeStorage), /disjoint filesystem/);
  const removal = structuredClone(fixture.controlValue); removal.identities.foo.expected_removals = [{ kind: "file", path: "docs/builtins/reference/foo.json", baseline_digest: `sha256:${"a".repeat(64)}` }];
  assert.throws(() => parseControlManifest(removal), /lacks matching baseline/);
  const incompleteInventory = structuredClone(fixture.inventory); incompleteInventory.identities.push({ identity: "bar" });
  assert.throws(() => parseControlManifest(fixture.controlValue, incompleteInventory), /exactly cover/);
  const danglingAlias = structuredClone(fixture.controlValue); danglingAlias.identities.foo.disposition = { kind: "alias", target: "missing" }; danglingAlias.identities.foo.runtime_owner = null;
  assert.throws(() => parseControlManifest(danglingAlias), /alias target is absent/);
  const missingGatePlan = structuredClone(fixture.controlValue); missingGatePlan.bundles[fixture.bundleId].gate_plans = missingGatePlan.bundles[fixture.bundleId].gate_plans.filter((entry) => entry.gate !== "runtime-binding");
  assert.throws(() => parseControlManifest(missingGatePlan, fixture.inventory), /required gate runtime-binding has no reviewed gate plan/);
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
  const draft = buildControlDraft(fixture.inventory);
  const reviewed = structuredClone(fixture.controlValue);
  reviewed.control_draft_digest = draft.digest;
  assert.equal(freezeReviewedControl(draft, reviewed, fixture.inventory).value.control_draft_digest, draft.digest);
  const stale = structuredClone(reviewed); stale.control_draft_digest = `sha256:${"0".repeat(64)}`;
  assert.throws(() => freezeReviewedControl(draft, stale, fixture.inventory), /does not cite the exact/);
  const mutatedDraft = structuredClone(draft); mutatedDraft.identity_rows[0].compiled_authority_digest = `sha256:${"1".repeat(64)}`;
  const { digest: _old, ...payload } = mutatedDraft; mutatedDraft.digest = evidenceDigest(payload);
  assert.throws(() => freezeReviewedControl(mutatedDraft, reviewed, fixture.inventory), /differ from the inventory/);
});

test("expected file removals are bound to exact baseline bytes without a second source-item authority", () => {
  const fixture = controlledFixture({ sidecar: true });
  const value = structuredClone(fixture.controlValue);
  const source = fixture.inventory.source.files.find((entry) => entry.path === "docs/builtins/reference/foo.json");
  value.identities.foo.expected_removals = [{ kind: "file", path: source.path, baseline_digest: source.content_digest }];
  value.identities.foo.baseline_evidence = [{ kind: "sidecar", path: source.path, locator: null, digest: source.content_digest }];
  assert.doesNotThrow(() => parseControlManifest(value, fixture.inventory));
  value.identities.foo.baseline_evidence[0].locator = { kind: "rust-item", name: "guessed" };
  assert.throws(() => parseControlManifest(value, fixture.inventory), /source-item locators are obsolete/);
});

test("internal double-underscore identities and null runtime owners are representable", () => {
  const fixture = controlledFixture();
  const value = structuredClone(fixture.controlValue);
  const row = value.identities.foo;
  delete value.identities.foo;
  row.identity = "__register_test_classes"; row.public_spelling = "__register_test_classes";
  row.disposition = { kind: "internal", reason: "Generated registration helper", evidence: ["review"] };
  row.runtime_owner = null; row.expected_authorities.catalog_entry_count = 0; row.expected_authorities.catalog_package = null; row.expected_authorities.documentation = "none";
  value.identities.__register_test_classes = row;
  value.bundles[fixture.bundleId].identities = ["__register_test_classes"];
  assert.equal(parseControlManifest(value).identities.get("__register_test_classes").runtime_owner, null);
});

test("bundle graph rejects cycles, dangling edges, and authored/generated overlap", () => {
  const fixture = controlledFixture();
  const dangling = structuredClone(fixture.controlValue); dangling.bundles[fixture.bundleId].prerequisites = [{ bundle_id: "missing", kind: "semantic" }];
  assert.throws(() => parseControlManifest(dangling), /dangling prerequisite/);
  const overlap = structuredClone(fixture.controlValue); overlap.bundles[fixture.bundleId].authored_write_set.push({ kind: "file", path: "crates/runmat-runtime/src/builtins/generated_wasm_registry.rs" });
  assert.throws(() => parseControlManifest(overlap), /overlaps authored write scope/);
  const cross = structuredClone(fixture.controlValue);
  cross.bundles.other = { ...structuredClone(cross.bundles[fixture.bundleId]), id: "other", identities: ["bar"], authored_write_set: [{ kind: "file", path: "crates/runmat-runtime/src/builtins/generated_wasm_registry.rs" }], integration_outputs: [] };
  cross.identities.bar = { ...structuredClone(cross.identities.foo), identity: "bar", public_spelling: "bar", bundle_id: "other", disposition: { kind: "canonical", target: "bar" }, runtime_owner: "crates/runmat-runtime/src/builtins/bar.rs" };
  assert.throws(() => parseControlManifest(cross), /overlaps .* integration output/);
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
    domain: "math",
    family: "basic",
    reason: null,
    review: {
      status: "reviewed",
      evidence: ["C00 disposition review", "catalog and runtime owner review"],
    },
  });
  assert.doesNotThrow(() => validateDispositionInput(output));
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
  const fake = structuredClone(fixture.controlValue);
  fake.bundles[fixture.bundleId].gate_plans[0].program.path = "scripts/fake.mjs";
  assert.throws(() => parseControlManifest(fake, fixture.inventory), /absent from or differs from the frozen source snapshot/);
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
  const control = { ...fixture.control, active_bundle_id: fixture.bundleId };
  const proof = buildInventoryDeltaProof(fixture.repository, fixture.inventory, fixture.inventory, control);
  assert.equal(proof.result, "pass");
  assert.equal(inventoryDeltaChecks(proof)[0].result, "pass");

  const wrongBinding = structuredClone(fixture.compiledInventory);
  wrongBinding.snapshot.observed.implementation_provenance[0].source_file = "crates/runmat-runtime/src/builtins/math/basic/other.rs";
  wrongBinding.digest.value = contentDigest(Buffer.from(JSON.stringify(wrongBinding.snapshot))).slice("sha256:".length);
  const wrongBindingInventory = buildInventory(fixture.repository, undefined, { revision: REVISION, compiledInventory: wrongBinding });
  const wrongBindingProof = buildInventoryDeltaProof(fixture.repository, fixture.inventory, wrongBindingInventory, control);
  assert.equal(wrongBindingProof.result, "fail");
  assert.match(wrongBindingProof.identities[0].failures.join("\n"), /binding provenance differs/);

  fs.appendFileSync(path.join(fixture.repository, "Cargo.toml"), "\n# unreviewed\n");
  const escapedInventory = buildInventory(fixture.repository, undefined, { revision: REVISION, compiledInventory: fixture.compiledInventory });
  const escaped = buildInventoryDeltaProof(fixture.repository, fixture.inventory, escapedInventory, control);
  assert.equal(escaped.result, "fail");
  assert.match(escaped.failures.join("\n"), /source changed outside the reviewed bundle scopes/);
});

test("validate-control CLI requires and verifies the frozen baseline inventory", () => {
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
  const valid = spawnSync(process.execPath, [cli, "validate-control", "--control", controlPath, "--baseline-inventory", inventoryPath], { encoding: "utf8" });
  assert.equal(valid.status, 0, valid.stderr);
  const tampered = structuredClone(fixture.inventory); tampered.source.digest = `sha256:${"0".repeat(64)}`;
  const tamperedPath = path.join(directory, "tampered.json"); fs.writeFileSync(tamperedPath, JSON.stringify(tampered));
  const rejected = spawnSync(process.execPath, [cli, "validate-control", "--control", controlPath, "--baseline-inventory", tamperedPath], { encoding: "utf8" });
  assert.equal(rejected.status, 2);
});

test("control draft, freeze, and lease issuance CLI keep review and derivation separate", () => {
  const fixture = controlledFixture();
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-control-workflow-"));
  const inventoryPath = path.join(directory, "inventory.json");
  const draftPath = path.join(directory, "draft.json");
  const reviewedPath = path.join(directory, "reviewed.json");
  const frozenPath = path.join(directory, "frozen.json");
  const requestPath = path.join(directory, "lease-request.json");
  const leasePath = path.join(directory, "lease.json");
  fs.writeFileSync(inventoryPath, JSON.stringify(fixture.inventory));
  const cli = path.resolve("scripts/development/builtin-migration-factory.mjs");
  const draftResult = spawnSync(process.execPath, [cli, "draft-control", "--baseline-inventory", inventoryPath, "--output", draftPath], { encoding: "utf8" });
  assert.equal(draftResult.status, 0, draftResult.stderr);
  const draft = JSON.parse(fs.readFileSync(draftPath));
  const reviewed = structuredClone(fixture.controlValue); reviewed.control_draft_digest = draft.digest;
  fs.writeFileSync(reviewedPath, JSON.stringify(reviewed));
  const freezeResult = spawnSync(process.execPath, [cli, "freeze-control", "--draft", draftPath, "--control", reviewedPath, "--baseline-inventory", inventoryPath, "--output", frozenPath], { encoding: "utf8" });
  assert.equal(freezeResult.status, 0, freezeResult.stderr);
  const control = parseControlManifest(JSON.parse(fs.readFileSync(frozenPath)), fixture.inventory);
  const request = { schema_version: 1, kind: "runmat-builtin-migration-lease-request", authority: "reviewed-development-request", control_manifest_digest: control.digest, bundle_id: fixture.bundleId, lease_id: "cli-lane", owner: "builtin-migrator", issued_at: "2026-09-11T01:00:00.000Z", expires_at: "2026-09-11T02:00:00.000Z", review: { status: "reviewed", evidence: ["assignment review"] } };
  fs.writeFileSync(requestPath, JSON.stringify(request));
  const leaseResult = spawnSync(process.execPath, [cli, "issue-lease", "--request", requestPath, "--control", frozenPath, "--baseline-inventory", inventoryPath, "--output", leasePath], { encoding: "utf8" });
  assert.equal(leaseResult.status, 0, leaseResult.stderr);
  assert.doesNotThrow(() => parseLease(JSON.parse(fs.readFileSync(leasePath)), control));
});

test("gate evidence rejects stale storage, filesystem identity, and forged process status", () => {
  const fixture = controlledFixture();
  const expected = { source_revision: fixture.inventory.source.revision, source_digest: fixture.inventory.source.digest, baseline_source_revision: fixture.inventory.source.revision, baseline_inventory_digest: fixture.inventory.digest, subject_inventory_digest: fixture.inventory.digest, control_manifest_digest: fixture.control.digest, bundle_id: fixture.bundleId, storage_policy: fixture.control.value.storage_policy };
  const stale = gate(fixture, "architecture"); stale.storage_admission.observed_at = "2026-09-10T23:00:00.000Z";
  assert.throws(() => parseGateResult(stale, expected), /time bound/);
  const filesystem = gate(fixture, "architecture"); filesystem.storage_admission.volumes[0].filesystem_id = "posix-dev:3";
  assert.throws(() => parseGateResult(filesystem, expected), /differs from reviewed/);
  const forged = gate(fixture, "architecture"); forged.producer_evidence.process.exit_code = 1;
  assert.throws(() => parseGateResult(forged, expected), /conflicts|inconsistent/);
  const artifactExpected = { ...expected, gate_plans: fixture.control.bundles.get(fixture.bundleId).gate_plans, compiled_build: fixture.inventory.compiled_inventory.build, repository: fixture.repository };
  const missing = gate(fixture, "catalog-contract"); missing.artifacts = [];
  assert.throws(() => parseGateResult(missing, artifactExpected), /does not cover the reviewed artifact roles/);
  const unreviewed = gate(fixture, "architecture");
  const unreviewedPath = path.join(fixture.repository, "..", "unreviewed-artifact.json");
  const unreviewedBytes = Buffer.from("{}\n"); fs.writeFileSync(unreviewedPath, unreviewedBytes);
  unreviewed.artifacts.push({ role: "invented", path: unreviewedPath, byte_length: unreviewedBytes.length, content_digest: contentDigest(unreviewedBytes) });
  assert.throws(() => parseGateResult(unreviewed, artifactExpected), /unreviewed artifact role/);
  const tampered = gate(fixture, "runtime-binding"); fs.appendFileSync(tampered.artifacts[0].path, "tamper");
  assert.throws(() => parseGateResult(tampered, artifactExpected), /artifact bytes differ/);
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

test("lease rejects integration output authorship and paths outside reviewed scope", () => {
  const fixture = controlledFixture();
  assert.throws(() => validateLeaseDiff(fixture.lease.bundle, ["README.md"]), /lease violation/);
  assert.throws(() => validateLeaseDiff(fixture.lease.bundle, ["crates/runmat-runtime/src/builtins/generated_wasm_registry.rs"]), /lease violation/);
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
