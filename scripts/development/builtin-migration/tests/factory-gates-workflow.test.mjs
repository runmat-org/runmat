import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { contentDigest } from "../evidence.mjs";
import { runGateProducer } from "../gate-adapter.mjs";
import { generatedProductChecks, parseGeneratedProductsProof } from "../generated-products.mjs";
import { parseGateResult } from "../gate-result.mjs";
import { buildInventory } from "../inventory.mjs";
import { buildInventoryDeltaProof, inventoryDeltaChecks } from "../inventory-delta.mjs";
import { prepareGateStorage, storageStatus } from "../storage-admission.mjs";
import {
  cleanupRepositoryFixtures, controlledFixture, gate, repositoryFixture, REVISION,
} from "./helpers.mjs";
import {
  cleanFactoryCliRepository, copyReviewSetAsAuthoringFiles, parseFixtureControl,
  resealEvidence, reviewed, writeFullControlWorkflow,
} from "./factory-workflow-fixture.mjs";
import { createTemporaryDirectory } from "./temporary-directories.mjs";

test.afterEach(cleanupRepositoryFixtures);

test("gate producer requests cannot inject commands, results, checks, or storage", () => {
  const fixture = controlledFixture();
  const request = { control: fixture.controlValue, lease: fixture.lease, control_baseline_inventory: fixture.inventory, lease_base_inventory: fixture.inventory, subject_inventory: fixture.inventory, bundle_id: fixture.bundleId, gate: "architecture", artifact_id: "forged", inputs: null };
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
  const nativeManifest = fixture.compiledInventory.snapshot.observed.registration_manifest;
  const manifestIdentity = { schema_version: 1, digest: nativeManifest.digest, counts: structuredClone(nativeManifest.counts) };
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
      verification: { kind: "native_wasm_registration_manifest", generated_manifest: manifestIdentity, native_manifest: manifestIdentity, result: "pass" },
    }],
    result: "pass",
  };
  const integrationProducts = Object.entries(fixture.control.value.integration_products)
    .map(([product_id, definition]) => ({ product_id, ...definition }))
    .sort((left, right) => left.product_id.localeCompare(right.product_id));
  const expected = {
    integration_products: integrationProducts,
    source_files: fixture.inventory.source.files,
    native_registration_manifest: nativeManifest,
  };
  const parsed = parseGeneratedProductsProof(value, expected);
  assert.equal(generatedProductChecks(parsed, [fixture.id])[0].result, "pass");
  const empty = { ...value, products: [], result: "pass" };
  assert.equal(parseGeneratedProductsProof(empty, { ...expected, integration_products: [] }).result, "pass");
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
  assert.throws(() => parseGeneratedProductsProof(extra, expected), /not globally reviewed|reviewed integration outputs/);
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
  const directory = createTemporaryDirectory("runmat-control-cli-");
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
  const directory = createTemporaryDirectory("runmat-control-workflow-");
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
  const directory = createTemporaryDirectory("runmat-full-control-workflow-");
  const cliRepository = cleanFactoryCliRepository();
  const revision = `git:${execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: cliRepository,
    encoding: "utf8",
  }).trim()}`;
  const workflow = writeFullControlWorkflow(
    directory,
    revision,
    contentDigest(fs.readFileSync(path.join(cliRepository, "scripts/development/check-architecture-boundaries.mjs"))),
    fs.statSync(path.join(cliRepository, "scripts/development/check-architecture-boundaries.mjs")).mode & 0o777,
  );
  const cli = path.join(cliRepository, "scripts/development/builtin-migration-factory.mjs");
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
    "--lease-base-inventory", workflow.paths.inventory, "--state", workflow.paths.queueState,
    "--queue-checkpoint", workflow.paths.queueCheckpoint,
    "--trusted-queue-checkpoint-digest", workflow.values.queueCheckpoint.digest,
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
  const expected = { source_revision: fixture.inventory.source.revision, source_digest: fixture.inventory.source.digest, control_baseline_source_revision: fixture.inventory.source.revision, control_baseline_inventory_digest: fixture.inventory.digest, lease_base_inventory_digest: fixture.inventory.digest, subject_inventory_digest: fixture.inventory.digest, control_manifest_digest: fixture.control.digest, bundle_id: fixture.bundleId, lease_id: fixture.lease.value.lease_id, lease_digest: fixture.lease.value.digest, storage_policy: fixture.control.value.storage_policy, compiled_build: fixture.inventory.compiled_inventory.build, execution_targets: fixture.control.executionTargets };
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
