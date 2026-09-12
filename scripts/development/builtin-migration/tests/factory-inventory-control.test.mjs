import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { auditMigration } from "../audit.mjs";
import { assertControlSubject } from "../control.mjs";
import { buildControlDraft, freezeReviewedControl, parseControlDraft } from "../control-draft.mjs";
import { dispositionInputFromControl, validateDispositionInput } from "../dispositions.mjs";
import { compileDispositionReview, parseDispositionReview } from "../disposition-review.mjs";
import { contentDigest, evidenceDigest } from "../evidence.mjs";
import { runGateProducer } from "../gate-adapter.mjs";
import { buildInventory } from "../inventory.mjs";
import { buildInventoryDeltaProof, finalIdentityAuthorityFailures } from "../inventory-delta.mjs";
import { parseIdentityControlPolicy } from "../control-authoring/policy-schema.mjs";
import { identityControlAuthorityTemplate } from "../control-authoring/identity-template.mjs";
import { parseImplementationAuthority, validateIdentityAuthorityGraph } from "../identity-authority.mjs";
import { issueLease } from "../lease.mjs";
import { prepareIdentity } from "../prepare.mjs";
import { buildQueue } from "../queue.mjs";
import { sourceSnapshot } from "../snapshot.mjs";
import {
  cleanupRepositoryFixtures, compiledInventoryFixture, controlledFixture, gate,
  repositoryFixture, REVISION, topologyFixture,
} from "./helpers.mjs";
import {
  parseFixtureControl, resealEvidence, reviewed, runtimeConstant, setRegistrationManifest,
} from "./factory-workflow-fixture.mjs";
import { createTemporaryDirectory } from "./temporary-directories.mjs";

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

function catalogConstant(name) {
  return {
    name,
    kind: "real_double",
    provenance: {
      source_file: "crates/runmat-builtins/src/catalog/constant.rs",
      module_path: "catalog::constant",
    },
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
  const output = createTemporaryDirectory("runmat-review-");
  const prepared = prepareIdentity(fixture.repository, fixture.inventory, fixture.control, fixture.lease, "foo", output);
  const runtimePath = "crates/runmat-runtime/src/builtins/math/basic/foo.rs";
  fs.appendFileSync(path.join(fixture.repository, runtimePath), "\n// migrated subject\n");
  execFileSync("git", ["add", runtimePath], { cwd: fixture.repository });
  execFileSync("git", ["-c", "user.name=RunMat Test", "-c", "user.email=test@runmat.invalid", "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "subject"], { cwd: fixture.repository });
  const subject = buildInventory(fixture.repository, dispositionInputFromControl(fixture.control), { compiledInventory: fixture.compiledInventory });
  assert.notEqual(subject.source.revision, fixture.inventory.source.revision);
  assert.notEqual(subject.digest, fixture.inventory.digest);
  const gates = ["catalog-contract", "runtime-binding", "documentation-cutover", "native-link", "architecture", "focused-tests", "format-diff", "strict-clippy"].map((name) => gate(fixture, name, `gate-subject-${name}`, subject));
  const batch = { schema_version: 1, kind: "runmat-builtin-migration-batch", identities: ["foo"] };
  const audit = auditMigration(fixture.repository, fixture.inventory, fixture.inventory, subject, fixture.control, fixture.lease, batch, { artifact_id: "audit-subject", authored_revision: subject.source.revision, prepare_results: [prepared], source_dispositions: [], gate_results: gates });
  assert.equal(audit.result, "pass");
  assert.equal(audit.source.revision, subject.source.revision);
  assert.equal(audit.control_baseline_inventory_digest, fixture.inventory.digest);
  assert.equal(audit.lease_base_inventory_digest, fixture.inventory.digest);
  assert.equal(audit.subject_inventory_digest, subject.digest);
  assert.equal(prepared.inventory_digest, fixture.inventory.digest);
});

test("inventory rejects absent, future, or tampered compiled semantic authority", () => {
  const repository = repositoryFixture();
  assert.throws(() => buildInventory(repository, undefined, { revision: REVISION }), /requires a compiled migration inventory/);
  const future = compiledInventoryFixture(); future.schema_version = 4;
  assert.throws(() => buildInventory(repository, undefined, { revision: REVISION, compiledInventory: future }), /schema_version 3/);
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
  constantOnly.snapshot.declared.constants = [catalogConstant("eps")];
  constantOnly.snapshot.observed.runtime_bindings = [];
  constantOnly.snapshot.observed.implementation_provenance = [];
  constantOnly.snapshot.observed.runtime_constants = [runtimeConstant("eps")];
  setRegistrationManifest(constantOnly.snapshot, [{ kind: "constant", declaration: "eps", variant: null, builtin_path: "builtins::constants" }]);
  constantOnly.digest.value = contentDigest(Buffer.from(JSON.stringify(constantOnly.snapshot))).slice("sha256:".length);
  const constantInventory = buildInventory(repository, undefined, { revision: REVISION, compiledInventory: constantOnly });
  assert.deepEqual(constantInventory.identities.find((entry) => entry.identity === "eps").spellings, ["eps"]);

  const dual = compiledInventoryFixture("inf");
  dual.snapshot.declared.constants = [
    catalogConstant("Inf"),
    catalogConstant("inf"),
  ];
  dual.snapshot.observed.runtime_constants = [runtimeConstant("Inf"), runtimeConstant("inf")];
  setRegistrationManifest(dual.snapshot, [
    { kind: "builtin", declaration: "inf", variant: "default", builtin_path: "builtins::inf" },
    { kind: "constant", declaration: "Inf", variant: null, builtin_path: "builtins::constants" },
    { kind: "constant", declaration: "inf", variant: null, builtin_path: "builtins::constants" },
  ]);
  dual.digest.value = contentDigest(Buffer.from(JSON.stringify(dual.snapshot))).slice("sha256:".length);
  const dualInventory = buildInventory(repository, undefined, { revision: REVISION, compiledInventory: dual });
  assert.deepEqual(dualInventory.identities.find((entry) => entry.identity === "inf").spellings, ["inf"]);
});

test("compiled catalog aliases are typed edges to canonical entries", () => {
  const repository = repositoryFixture();
  const compiled = compiledInventoryFixture();
  compiled.snapshot.declared.catalog_aliases = [{
    alias: { name: "foalias" }, canonical: { name: "foo" },
    provenance: {
      source_file: "crates/runmat-builtins/src/catalog/aliases/math.rs",
      module_path: "catalog::aliases::math",
    },
  }];
  compiled.digest.value = contentDigest(Buffer.from(JSON.stringify(compiled.snapshot))).slice("sha256:".length);
  const inventory = buildInventory(repository, undefined, {
    revision: REVISION, compiledInventory: compiled,
  });
  const alias = inventory.identities.find((entry) => entry.identity === "foalias");
  assert.deepEqual(alias.spellings, ["foalias"]);
  assert.equal(alias.semantic_authority.catalog_aliases[0].canonical.name, "foo");

  compiled.snapshot.declared.catalog_aliases[0].canonical.name = "missing";
  compiled.digest.value = contentDigest(Buffer.from(JSON.stringify(compiled.snapshot))).slice("sha256:".length);
  assert.throws(
    () => buildInventory(repository, undefined, { revision: REVISION, compiledInventory: compiled }),
    /catalog alias target is not canonical/,
  );
});

test("compiled authority rejects unknown or malformed nested records even with a recomputed digest", () => {
  const repository = repositoryFixture();
  const cases = [
    (value) => { value.snapshot.declared.catalog_entries[0].documentation.extra = true; },
    (value) => { value.snapshot.declared.catalog_entries[0].bindings[0].availability = "Maybe"; },
    (value) => { value.snapshot.declared.catalog_provenance[0].provenance.extra = "x"; },
    (value) => { value.snapshot.observed.runtime_bindings[0].native_symbol = "guessed"; },
    (value) => { value.snapshot.observed.runtime_constants = [{ name: "pi" }]; },
    (value) => { value.snapshot.observed.implementation_provenance[0].authority = "unknown"; },
    (value) => { value.snapshot.build.crate_feature_inventory.enabled_features = ["not-known"]; },
    (value) => { value.snapshot.observed.gpu_specs = [{ key: "data.*", builtin_path: "builtins::foo", owner: { kind: "legacy_group", raw: "data.*", extra: true }, operation: "custom:data", supported_precisions: [], broadcast: "none", provider_hooks: [], constant_strategy: "inline_literal", residency: "inherit_inputs", nan_mode: "include", two_pass_threshold: null, workgroup_size: null, accepts_nan_mode: false, notes: "fixture" }]; },
  ];
  for (const mutate of cases) {
    const value = compiledInventoryFixture(); mutate(value);
    value.digest.value = contentDigest(Buffer.from(JSON.stringify(value.snapshot))).slice("sha256:".length);
    assert.throws(() => buildInventory(repository, undefined, { revision: REVISION, compiledInventory: value }));
  }
});

test("compiled grouped owners require exact typed membership in the compiled identity set", () => {
  const repository = repositoryFixture();
  const empty = compiledInventoryFixture("foo", { legacyGroup: true });
  empty.snapshot.observed.gpu_specs[0].owner.affected_identities = [];
  empty.snapshot.validation.migration_readiness = groupedOwnerReadiness(
    empty.snapshot.observed.gpu_specs[0].owner,
  );
  empty.digest.value = contentDigest(Buffer.from(JSON.stringify(empty.snapshot))).slice("sha256:".length);
  assert.throws(
    () => buildInventory(repository, undefined, { revision: REVISION, compiledInventory: empty }),
    /must be a nonempty array/,
  );

  const unknown = compiledInventoryFixture("foo", { legacyGroup: true });
  unknown.snapshot.observed.gpu_specs[0].owner.affected_identities = [{ name: "invented" }];
  unknown.snapshot.validation.migration_readiness = groupedOwnerReadiness(
    unknown.snapshot.observed.gpu_specs[0].owner,
  );
  unknown.digest.value = contentDigest(Buffer.from(JSON.stringify(unknown.snapshot))).slice("sha256:".length);
  assert.throws(
    () => buildInventory(repository, undefined, { revision: REVISION, compiledInventory: unknown }),
    /grouped provider ownership differs from compiled builtin-path provenance/,
  );

  const mismatched = compiledInventoryFixture("foo", { legacyGroup: true });
  mismatched.snapshot.declared.constants = [catalogConstant("other")];
  mismatched.snapshot.observed.runtime_constants = [runtimeConstant("other")];
  setRegistrationManifest(mismatched.snapshot, [
    ...mismatched.snapshot.observed.registration_manifest.entries,
    { kind: "constant", declaration: "other", variant: null, builtin_path: "builtins::constants" },
  ]);
  mismatched.snapshot.validation.migration_readiness = groupedOwnerReadiness({
    kind: "legacy_group",
    raw: "data.*",
    affected_identities: [{ name: "other" }],
  });
  mismatched.digest.value = contentDigest(Buffer.from(JSON.stringify(mismatched.snapshot))).slice("sha256:".length);
  assert.throws(
    () => buildInventory(repository, undefined, { revision: REVISION, compiledInventory: mismatched }),
    /grouped provider ownership differs|do not exactly preserve compiled provider ownership/,
  );
});

test("compiled exact spec owners bind their typed identity to registration provenance", () => {
  const repository = repositoryFixture();
  const mismatched = compiledInventoryFixture("foo", { legacyGroup: true });
  mismatched.snapshot.observed.gpu_specs[0].key = "foo";
  mismatched.snapshot.observed.gpu_specs[0].builtin_path = "builtins::other";
  mismatched.snapshot.observed.gpu_specs[0].owner = {
    kind: "exact_builtin", identity: { name: "foo" },
  };
  mismatched.snapshot.validation = {
    status: "invalid",
    errors: [{
      source: "gpu_spec_registry",
      identity: "foo",
      message: "Exact owner has no implementation provenance at builtins::other",
    }],
    migration_readiness: { status: "ready", findings: [] },
  };
  mismatched.digest.value = contentDigest(Buffer.from(JSON.stringify(mismatched.snapshot))).slice("sha256:".length);
  assert.throws(
    () => buildInventory(repository, undefined, { revision: REVISION, compiledInventory: mismatched }),
    /validation is not clean|differs from its compiler module/,
  );

  const canonicalPrefix = structuredClone(mismatched);
  canonicalPrefix.snapshot.observed.gpu_specs[0].builtin_path = "crate::builtins::foo";
  canonicalPrefix.snapshot.validation = {
    status: "valid", errors: [], migration_readiness: { status: "ready", findings: [] },
  };
  canonicalPrefix.digest.value = contentDigest(Buffer.from(JSON.stringify(canonicalPrefix.snapshot))).slice("sha256:".length);
  assert.doesNotThrow(
    () => buildInventory(repository, undefined, { revision: REVISION, compiledInventory: canonicalPrefix }),
  );
});

function groupedOwnerReadiness(owner) {
  return {
    status: "incomplete",
    findings: [{
      code: "legacy_spec_group_requires_disposition",
      source: "gpu_spec_registry",
      affected: { kind: "owner", owner: structuredClone(owner) },
      message: "Legacy group needs reviewed ownership",
    }],
  };
}

test("compiled migration findings and non-identity legacy spec keys remain explicit reviewed work", () => {
  const finding = { code: "legacy_spec_group_requires_disposition", source: "gpu_spec_registry", affected: { kind: "owner", owner: { kind: "legacy_group", raw: "data.*", affected_identities: [{ name: "foo" }] } }, message: "Legacy group needs reviewed ownership" };
  const fixture = controlledFixture({ finding, legacyGroup: true });
  assert.deepEqual(fixture.inventory.migration_findings, [finding]);
  assert.equal(fixture.inventory.identities.some((entry) => entry.identity === "data.*"), false);
  assert.equal(fixture.control.migrationFindings[0].bundle_id, fixture.bundleId);
  const queue = buildQueue(fixture.inventory, fixture.control);
  assert.deepEqual(queue.rows[0].migration_findings.map((entry) => entry.affected.owner.raw), ["data.*"]);
  assert.deepEqual(queue.rows[0].finding_work_items, [evidenceDigest(finding)]);
  assert.equal(queue.rows[0].blockers.some((entry) => entry.startsWith("migration-finding:")), false);
  assert.equal(queue.rows[0].migration_state, "ready");
  const missing = structuredClone(fixture.controlValue); missing.migration_findings.rows = [];
  assert.throws(() => parseFixtureControl(fixture, missing), /do not exactly cover/);
  const unknown = structuredClone(fixture.controlValue); unknown.migration_findings.rows[0].unexpected = true;
  assert.throws(() => parseFixtureControl(fixture, unknown), /fields must be exactly/);
});

test("control is closed, reviewed, reciprocal, and rejects case-fold ambiguity", () => {
  const fixture = controlledFixture();
  assert.equal(fixture.control.identities.get("foo").public_identity.primary_spelling.spelling, "foo");
  const legacy = structuredClone(fixture.controlValue); legacy.schema_version = 4;
  assert.throws(() => parseFixtureControl(fixture, legacy), /schema_version 5/);
  const future = structuredClone(fixture.controlValue); future.schema_version = 6;
  assert.throws(() => parseFixtureControl(fixture, future), /schema_version 5/);
  const extra = structuredClone(fixture.controlValue); extra.unreviewed = true;
  assert.throws(() => parseFixtureControl(fixture, extra), /fields must be exactly/);
  const collision = structuredClone(fixture.controlValue); collision.identity_controls.Foo = structuredClone(collision.identity_controls.foo);
  assert.throws(() => parseFixtureControl(fixture, collision), /collide case-insensitively|identity policy set differs/);
  const spelling = structuredClone(fixture.controlValue); spelling.identity_controls.foo.public_identity.primary_spelling.spelling = "bar";
  assert.throws(() => parseFixtureControl(fixture, spelling), /case-fold to its identity/);
  const observedSpelling = structuredClone(fixture.controlValue); observedSpelling.identity_controls.foo.public_identity.primary_spelling.spelling = "Foo";
  assert.throws(() => parseFixtureControl(fixture, observedSpelling), /public identity spelling differs from the reviewed inventory/);
  const domain = structuredClone(fixture.controlValue); domain.identity_controls.foo.domain = "other";
  assert.throws(() => parseFixtureControl(fixture, domain), /fields must be exactly/);
  const disposition = structuredClone(fixture.controlValue); disposition.identity_controls.foo.disposition = { kind: "internal" };
  assert.throws(() => parseFixtureControl(fixture, disposition), /fields must be exactly/);
  const unsafeStorage = structuredClone(fixture.controlValue); unsafeStorage.storage_policy.host_profiles["fixture-host"].volume_roles.target_temp.filesystem_id = unsafeStorage.storage_policy.host_profiles["fixture-host"].volume_roles.source_worktree.filesystem_id;
  assert.throws(() => parseFixtureControl(fixture, unsafeStorage), /disjoint filesystem/);
  const noncanonicalScope = structuredClone(fixture.controlValue); noncanonicalScope.bundle_controls[fixture.bundleId].additional_authored_write_set[0].path = "crates/runmat-builtins/./src";
  assert.throws(() => parseFixtureControl(fixture, noncanonicalScope), /normalized safe repository-relative path/);
  const removal = structuredClone(fixture.controlValue); removal.bundle_controls[fixture.bundleId].expected_removals = [{ kind: "file", path: "docs/builtins/reference/foo.json", baseline_digest: `sha256:${"a".repeat(64)}`, affected_identities: ["foo"] }];
  assert.throws(() => parseFixtureControl(fixture, removal), /typed baseline evidence ownership/);
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
  danglingControl.identity_controls.foo.public_identity = {
    kind: "alias",
    alias_spelling: { identity: "foo", spelling: "foo" },
    canonical_identity: "missing",
  };
  danglingControl.identity_controls.foo.implementation = {
    callable: { kind: "none", reason: "alias-resolution" },
    constant: { kind: "none", reason: "alias-resolution" },
  };
  danglingControl.identity_controls.foo.expected_authorities = {
    catalog_package: null,
    catalog_alias_package: "crates/runmat-builtins/src/catalog/aliases/math.rs",
    catalog_constant_package: null,
    catalog_entry_count: 0,
    catalog_constant_count: 0,
    documentation: "alias",
    native_link: "not-applicable",
    wasm_registry: "not-applicable",
  };
  for (const gate of ["runtime-binding", "link-reachability"]) {
    danglingControl.identity_controls.foo.maturity[gate] = {
      applicability: "not-applicable", reason: "Alias resolution owns no implementation", evidence: ["fixture review"],
    };
  }
  danglingControl.topology_digest = danglingTopology.digest;
  resealEvidence(danglingControl);
  assert.throws(
    () => parseFixtureControl(fixture, danglingControl, fixture.inventory, danglingTopology),
    /alias target must be a present primary public identity/,
  );
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
  const output = createTemporaryDirectory("runmat-control-binding-");
  assert.throws(() => buildQueue(foreign.inventory, fixture.control), /control baseline/);
  assert.throws(
    () => prepareIdentity(fixture.repository, foreign.inventory, fixture.control, fixture.lease, fixture.id, output),
    /control baseline/,
  );
  assert.throws(
    () => buildInventoryDeltaProof(fixture.repository, foreign.inventory, foreign.inventory, fixture.control, fixture.bundleId),
    /reviewed control identity set/,
  );
  assert.throws(() => runGateProducer({
    control: fixture.control,
    lease: fixture.lease,
    queue_state: fixture.queueState,
    queue_checkpoint: fixture.queueCheckpoint,
    control_baseline_inventory: foreign.inventory,
    lease_base_inventory: foreign.inventory,
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
    lease: fixture.lease,
    queue_state: fixture.queueState,
    queue_checkpoint: fixture.queueCheckpoint,
    control_baseline_inventory: fixture.inventory,
    lease_base_inventory: fixture.inventory,
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
  assert.throws(
    () => assertControlSubject(fixture.control, drifted),
    /internal identity differs from the reviewed inventory/,
  );
});

test("migration operations require the exact validated authored lease", () => {
  const fixture = controlledFixture();
  const output = createTemporaryDirectory("runmat-lease-binding-");
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
      fixture.inventory,
      fixture.control,
      forgedLease,
      { schema_version: 1, kind: "runmat-builtin-migration-batch", identities: [fixture.id] },
      { artifact_id: "forged-lease", authored_revision: fixture.inventory.source.revision, prepare_results: [], source_dispositions: [], gate_results: [] },
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
  assert.ok(first.identity_rows.every((entry) => entry.review.status === "unreviewed" && entry.unresolved_fields.includes("topology_disposition") && entry.unresolved_fields.includes("maturity")));
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
  value.bundle_controls[fixture.bundleId].expected_removals = [{ kind: "file", path: source.path, baseline_digest: source.content_digest, affected_identities: ["foo"] }];
  resealEvidence(value);
  assert.doesNotThrow(() => parseFixtureControl(fixture, value));
  value.bundle_controls[fixture.bundleId].baseline_evidence.find((entry) => entry.path === source.path).kind = "runtime-owner";
  assert.throws(() => parseFixtureControl(fixture, value), /typed path evidence|canonically ordered/);
});

test("internal double-underscore identities retain typed implementation authority", () => {
  const fixture = controlledFixture({ identity: "__register_test_classes" });
  const value = structuredClone(fixture.controlValue);
  const row = value.identity_controls.__register_test_classes;
  row.public_identity = {
    kind: "internal",
    reason: "Generated registration helper",
    evidence: ["fixture review"],
  };
  row.expected_authorities.documentation = "none";
  row.maturity.documentation = {
    applicability: "not-applicable", reason: "Internal entries are not public documentation", evidence: ["fixture review"],
  };
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
  const parsed = parseFixtureControl(fixture, value, inventory, topology)
    .identities.get("__register_test_classes");
  assert.equal(parsed.public_identity.kind, "internal");
  assert.equal(parsed.implementation.callable.kind, "owned");
});

test("final authority checks internal executable bindings and strips alias implementations", () => {
  const fixture = controlledFixture({ identity: "__helper" });
  const internal = structuredClone(fixture.controlValue.identity_controls.__helper);
  internal.public_identity = { kind: "internal", reason: "Runtime helper", evidence: ["review"] };
  internal.expected_authorities = {
    ...internal.expected_authorities,
    documentation: "none",
  };
  const current = structuredClone(fixture.inventory.identities[0]);
  current.semantic_authority.catalog_entries[0].descriptor.completion_policy = "HiddenInternal";
  assert.deepEqual(finalIdentityAuthorityFailures("__helper", current, internal), []);
  current.semantic_authority.implementation_provenance[0].function = "wrong_helper";
  assert.match(
    finalIdentityAuthorityFailures("__helper", current, internal).join("\n"),
    /canonical runtime binding provenance differs from review/,
  );

  const alias = structuredClone(internal);
  alias.public_identity = {
    kind: "alias",
    alias_spelling: { identity: "__helper", spelling: "__helper" },
    canonical_identity: "foo",
  };
  alias.forms = { kind: "unobserved", callable_spellings: [], constant_spellings: [] };
  alias.implementation = {
    callable: { kind: "none", reason: "alias-resolution" },
    constant: { kind: "none", reason: "alias-resolution" },
  };
  alias.expected_authorities = {
    catalog_package: null,
    catalog_alias_package: "crates/runmat-builtins/src/catalog/aliases/internal.rs",
    catalog_constant_package: null,
    catalog_entry_count: 0,
    catalog_constant_count: 0,
    documentation: "alias",
    native_link: "not-applicable",
    wasm_registry: "not-applicable",
  };
  assert.match(
    finalIdentityAuthorityFailures("__helper", current, alias).join("\n"),
    /alias retains copied authority/,
  );

  current.semantic_authority.catalog_aliases = [{
    alias: { name: "__helper" }, canonical: { name: "foo" },
    provenance: {
      source_file: "crates/runmat-builtins/src/catalog/aliases/internal.rs",
      module_path: "catalog::aliases::internal",
    },
  }];
  current.semantic_authority.catalog_entries = [];
  current.semantic_authority.catalog_provenance = [];
  current.semantic_authority.runtime_bindings = [];
  current.semantic_authority.implementation_provenance = [];
  current.registrations.wasm = [];
  current.ownership.catalog = [];
  current.ownership.catalog_documentation = [];
  assert.deepEqual(finalIdentityAuthorityFailures("__helper", current, alias), []);
  current.semantic_authority.catalog_aliases[0].canonical.name = "wrong";
  assert.match(
    finalIdentityAuthorityFailures("__helper", current, alias).join("\n"),
    /compiled catalog alias edge differs from review/,
  );
});

test("review templates leave legacy binding cutovers for explicit review", () => {
  const scaffold = { authority_proposals: { identity_rows: [{
    identity: "foo",
    proposal: {
      public_identity: {
        kind: "primary", primary_spelling: { identity: "foo", spelling: "foo" },
      },
      forms: { kind: "callable", callable_spellings: ["foo"], constant_spellings: [] },
      implementation: {
        callable: {
          proposed_owner_path: "crates/runmat-runtime/src/builtins/math/foo.rs",
          observed_bindings: [{
            authority: "legacy_function", binding_variant: null,
            function: "foo_builtin", builtin_path: "builtins::math::foo",
          }],
        },
        constant: { proposed_owner_path: null, observed_bindings: [] },
      },
    },
  }] } };
  const template = identityControlAuthorityTemplate(scaffold, "foo");
  assert.equal(template.implementation.callable, null);
  const legacy = {
    callable: { kind: "owned", owner_path: "crates/runmat-runtime/src/builtins/math/foo.rs", bindings: [{
    kind: "legacy_function", function: "foo_builtin", builtin_path: "builtins::math::foo",
    }] },
    constant: { kind: "none", reason: "no-constant-form" },
  };
  assert.throws(
    () => parseImplementationAuthority(legacy, "foo"),
    /unsupported kind/,
  );
});

test("constant-only canonical identities require exact catalog and runtime constant authorities", () => {
  const fixture = controlledFixture();
  const row = structuredClone(fixture.controlValue.identity_controls.foo);
  row.forms = { kind: "constant", callable_spellings: [], constant_spellings: ["foo"] };
  row.implementation = {
    callable: { kind: "none", reason: "no-callable-form" },
    constant: {
      kind: "owned",
      owner_path: "crates/runmat-runtime/src/builtins/constants/mod.rs",
      bindings: [{
        kind: "constant_registration", constant: "foo", builtin_path: "builtins::constants",
      }],
    },
  };
  row.expected_authorities.catalog_entry_count = 0;
  row.expected_authorities.catalog_constant_count = 1;
  row.expected_authorities.catalog_package = null;
  row.expected_authorities.catalog_constant_package = "crates/runmat-builtins/src/catalog/constant.rs";
  row.expected_authorities.documentation = "none";
  row.maturity.documentation = {
    applicability: "not-applicable", reason: "constant fixture has no public documentation entry",
    evidence: ["fixture review"],
  };
  row.expected_authorities.native_link = "not-applicable";
  row.maturity["link-reachability"] = {
    applicability: "not-applicable", reason: "constant registration has no callable native symbol",
    evidence: ["fixture review"],
  };
  assert.doesNotThrow(() => parseIdentityControlPolicy(row, "foo"));
  assert.doesNotThrow(() => validateIdentityAuthorityGraph(new Map([["foo", row]])));

  const current = structuredClone(fixture.inventory.identities[0]);
  current.semantic_authority.catalog_entries = [];
  current.semantic_authority.catalog_provenance = [];
  current.semantic_authority.constants = [catalogConstant("foo")];
  current.semantic_authority.implementation_provenance = [];
  current.semantic_authority.runtime_bindings = [];
  current.semantic_authority.runtime_constants = [runtimeConstant("foo")];
  assert.deepEqual(finalIdentityAuthorityFailures("foo", current, row), []);

  row.implementation.constant.bindings = [];
  assert.match(
    finalIdentityAuthorityFailures("foo", current, row).join("\n"),
    /runtime constant provenance differs from review/,
  );
});

test("bundle graph rejects cycles, dangling edges, and authored/generated overlap", () => {
  const fixture = controlledFixture();
  const dangling = structuredClone(fixture.controlValue); dangling.bundle_controls[fixture.bundleId].prerequisites = [{ bundle_id: "missing", kind: "semantic" }];
  assert.throws(() => parseFixtureControl(fixture, dangling), /dangling prerequisite/);
  const overlap = structuredClone(fixture.controlValue); overlap.bundle_controls[fixture.bundleId].additional_authored_write_set.push({ kind: "file", path: "crates/runmat-runtime/src/builtins/generated_wasm_registry.rs" });
  assert.throws(
    () => parseFixtureControl(fixture, overlap),
    /authored scope overlaps integration product/,
  );
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
  const directory = createTemporaryDirectory("runmat-disposition-review-");
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
