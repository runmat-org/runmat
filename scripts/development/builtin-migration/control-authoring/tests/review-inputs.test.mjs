import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test, { afterEach } from "node:test";

import { buildControlDraft } from "../../control-draft.mjs";
import { bundleBaselineEvidence } from "../../baseline-evidence.mjs";
import { parseControlManifest } from "../../control.mjs";
import { contentDigest, evidenceDigest } from "../../evidence.mjs";
import { buildInventory } from "../../inventory.mjs";
import {
  REVISION, cleanupRepositoryFixtures, compiledInventoryFixture, fixtureIntegrationProducts,
  fixtureCatalogCompositionChild, fixtureCompositionChild, fixtureGatePlans, fixtureModuleCompositionBaseline,
  fixtureRootCompositionChild, fixtureTargetPolicy, repositoryFixture,
  fixtureReviewedModuleCompositionBaseline,
  topologyFixture,
} from "../../tests/helpers.mjs";
import { reviewedPilotPolicy } from "../../tests/pilot-policy-fixture.mjs";
import { parseBundleControlReview } from "../bundle-review.mjs";
import { validateControlReviewChain } from "../authority.mjs";
import { buildControlAttestationTemplate, sealControlAttestation } from "../attestation-template.mjs";
import { composeControlCandidate, controlCandidateInputDigests, parseControlCandidate } from "../compose.mjs";
import { parseGlobalControlReview } from "../global-review.mjs";
import { MATURITY_GATES, parseStoragePolicy } from "../policy-schema.mjs";
import { assertValidatedControlReviewSet, loadControlReviewSet } from "../review-set.mjs";
import { buildControlOverlayScaffold } from "../scaffold.mjs";
import { indexControlReviews, initializeControlReviewTemplates } from "../templates.mjs";

const temporaryDirectories = new Set();

afterEach(() => {
  cleanupRepositoryFixtures();
  for (const directory of temporaryDirectories) fs.rmSync(directory, { recursive: true, force: true });
  temporaryDirectories.clear();
});

test("bundle and global reviews bind exact scaffold and topology rows", () => {
  const fixture = reviewFixture();
  const global = parseGlobalControlReview(
    fixture.globalReview, fixture.scaffold, fixture.topology, fixture.inventory,
  );
  const bundle = parseBundleControlReview(
    fixture.bundleReview, fixture.scaffold, fixture.topology, fixture.inventory, global,
  );
  assert.equal(bundle.bundleId, fixture.bundleId);
  assert.deepEqual([...bundle.identityControls.keys()], [fixture.id]);
  assert.deepEqual([...global.programProfiles.keys()], [...new Set(fixture.bundleReview.bundle_control.gate_plans.map((entry) => entry.program_profile_id))]);
  assert.throws(() => bundle.identityControls.set("bar", {}), /immutable/);

  for (const version of [4, 5, 7]) {
    const wrongVersion = structuredClone(fixture.globalReview);
    wrongVersion.schema_version = version;
    resign(wrongVersion);
    assert.throws(
      () => parseGlobalControlReview(
        wrongVersion, fixture.scaffold, fixture.topology, fixture.inventory,
      ),
      /schema_version 6/,
    );
  }
  for (const version of [4, 5, 7]) {
    const wrongVersion = structuredClone(fixture.bundleReview);
    wrongVersion.schema_version = version;
    resign(wrongVersion);
    assert.throws(
      () => parseBundleControlReview(
        wrongVersion, fixture.scaffold, fixture.topology, fixture.inventory, global,
      ),
      /schema_version 6/,
    );
  }

  const drift = structuredClone(fixture.bundleReview);
  drift.bindings.scaffold_bundle_row_digest = `sha256:${"0".repeat(64)}`;
  resign(drift);
  assert.throws(
    () => parseBundleControlReview(
      drift, fixture.scaffold, fixture.topology, fixture.inventory, global,
    ),
    /scaffold row digest mismatch/,
  );

  const proposalDrift = structuredClone(fixture.bundleReview);
  proposalDrift.bindings.identity_rows[0].authority_proposal_digest = `sha256:${"0".repeat(64)}`;
  resign(proposalDrift);
  assert.throws(
    () => parseBundleControlReview(
      proposalDrift, fixture.scaffold, fixture.topology, fixture.inventory, global,
    ),
    /authority proposal digest mismatch/,
  );

  const uncoveredTarget = structuredClone(fixture.globalReview);
  uncoveredTarget.storage_policy.host_profiles["linux-host"] = {
    operating_system: "linux",
    architecture: "x86_64",
    execution_host: "runmat-linux-builder",
    volume_roles: structuredClone(
      uncoveredTarget.storage_policy.host_profiles["fixture-host"].volume_roles,
    ),
  };
  resign(uncoveredTarget);
  assert.throws(
    () => parseGlobalControlReview(
      uncoveredTarget, fixture.scaffold, fixture.topology, fixture.inventory,
    ),
    /must exactly cover every reviewed execution target/,
  );
});

test("production global review cannot bypass the fixed composition registry", () => {
  const fixture = reviewFixture();
  assert.ok(parseGlobalControlReview(
    fixture.globalReview, fixture.scaffold, fixture.topology, fixture.inventory,
  ).integrationProducts.has("wasm-registry"), "unrelated reviewed products remain admitted");

  const rawProjection = structuredClone(fixture.globalReview);
  rawProjection.module_composition_baseline = rawProjection.module_composition_baseline.projection;
  rawProjection.bindings.module_composition_reviewed_baseline_digest = evidenceDigest(rawProjection.module_composition_baseline);
  resign(rawProjection);
  assert.throws(
    () => parseGlobalControlReview(
      rawProjection, fixture.scaffold, fixture.topology, fixture.inventory,
    ),
    /reviewed module composition baseline/,
  );

  const omitted = structuredClone(fixture.globalReview);
  delete omitted.integration_products["catalog-aliases"];
  omitted.module_composition_baseline.projection.products = omitted.module_composition_baseline.projection.products
    .filter((entry) => entry.product_id !== "catalog-aliases");
  resignCompositionBaseline(omitted);
  assert.throws(
    () => parseGlobalControlReview(
      omitted, fixture.scaffold, fixture.topology, fixture.inventory,
    ),
    /module composition projection.*missing catalog-aliases/,
  );

  const aggregationDrift = structuredClone(fixture.globalReview);
  aggregationDrift.module_composition_baseline.projection.products
    .find((entry) => entry.product_id === "catalog-constants")
    .aggregations = ["entries", "constants"];
  resignCompositionBaseline(aggregationDrift);
  assert.throws(
    () => parseGlobalControlReview(
      aggregationDrift, fixture.scaffold, fixture.topology, fixture.inventory,
    ),
    /catalog-constants: projection aggregation roles do not match/,
  );

  const digestMismatch = structuredClone(fixture.globalReview);
  digestMismatch.bindings.module_composition_reviewed_baseline_digest = `sha256:${"0".repeat(64)}`;
  resign(digestMismatch);
  assert.throws(
    () => parseGlobalControlReview(
      digestMismatch, fixture.scaffold, fixture.topology, fixture.inventory,
    ),
    /trusted digest/,
  );
});

test("review-set loader verifies exact bytes, coverage, and global profile references", () => {
  const fixture = reviewFixture();
  const directory = writeReviewSet(fixture);
  const loaded = loadControlReviewSet(path.join(directory, "manifest.json"), fixture);
  assert.equal(loaded.bundleReviews.size, 1);
  assert.equal(loaded.globalReview.digest, fixture.globalReview.digest);
  assert.equal(assertValidatedControlReviewSet(loaded), loaded);
  assert.throws(() => assertValidatedControlReviewSet({
    value: loaded.value,
    bundleReviews: loaded.bundleReviews,
    globalReview: loaded.globalReview,
  }), /exact validated control review set/);

  const badBundle = structuredClone(fixture.bundleReview);
  badBundle.bundle_control.gate_plans[0].program_profile_id = "missing-profile";
  resign(badBundle);
  writeJson(path.join(directory, "bundle.json"), badBundle);
  rewriteManifestDigests(directory);
  assert.throws(() => loadControlReviewSet(path.join(directory, "manifest.json"), fixture), /unknown global program profile/);

  writeJson(path.join(directory, "bundle.json"), fixture.bundleReview);
  rewriteManifestDigests(directory);
  const manifest = readJson(path.join(directory, "manifest.json"));
  manifest.digest = `sha256:${"0".repeat(64)}`;
  writeJson(path.join(directory, "manifest.json"), manifest);
  assert.throws(() => loadControlReviewSet(path.join(directory, "manifest.json"), fixture), /manifest digest mismatch/);
});

test("review-set loader rejects byte drift and paths escaping through symlinks", () => {
  const fixture = reviewFixture();
  const directory = writeReviewSet(fixture);
  fs.appendFileSync(path.join(directory, "bundle.json"), "\n");
  assert.throws(() => loadControlReviewSet(path.join(directory, "manifest.json"), fixture), /exact byte content digest mismatch/);

  writeJson(path.join(directory, "bundle.json"), fixture.bundleReview);
  const outside = path.join(path.dirname(directory), `${path.basename(directory)}-outside.json`);
  writeJson(outside, fixture.bundleReview);
  fs.symlinkSync(outside, path.join(directory, "escape.json"));
  const manifest = readJson(path.join(directory, "manifest.json"));
  manifest.bundle_reviews[0].path = "escape.json";
  manifest.bundle_reviews[0].content_digest = contentDigest(fs.readFileSync(outside));
  resign(manifest);
  writeJson(path.join(directory, "manifest.json"), manifest);
  assert.throws(() => loadControlReviewSet(path.join(directory, "manifest.json"), fixture), /must resolve within/);
  fs.rmSync(outside);
});

test("control composition and attestation are deterministic capabilities, not review labels", () => {
  const fixture = reviewFixture();
  const directory = writeReviewSet(fixture);
  const reviewSet = loadControlReviewSet(path.join(directory, "manifest.json"), fixture);
  const input = {
    inventory: fixture.inventory,
    topology: fixture.topology,
    scaffold: fixture.scaffold,
    reviewSet,
  };
  const candidate = composeControlCandidate(input);
  assert.deepEqual(candidate, composeControlCandidate(input));
  assert.equal(candidate.schema_version, 7);
  for (const version of [5, 6, 8]) {
    const wrongVersion = structuredClone(candidate);
    wrongVersion.schema_version = version;
    assert.throws(() => parseControlCandidate(wrongVersion, candidate), /schema_version 7/);
  }
  assert.throws(
    () => composeControlCandidate({ ...input, reviewSet: { ...reviewSet } }),
    /exact validated control review set/,
  );
  assert.throws(
    () => parseControlCandidate(candidate, structuredClone(candidate)),
    /exact deterministically composed control candidate/,
  );

  const attestationPayload = {
    schema_version: 1,
    kind: "runmat-builtin-migration-control-attestation",
    authority: "reviewer-authored-development-input",
    program: "RM-1064/C00-C07",
    candidate_digest: candidate.digest,
    input_digests: controlCandidateInputDigests(candidate),
    review: {
      status: "reviewed",
      evidence: ["review:zeta independent fixture review", "review:alpha independent fixture review"],
    },
  };
  const attestation = { ...attestationPayload, digest: evidenceDigest(attestationPayload) };
  const reviewed = validateControlReviewChain(candidate, attestation, candidate);
  assert.equal(reviewed.controlValue.schema_version, 8);
  assert.doesNotThrow(() => parseControlManifest(reviewed.controlValue, {
    inventory: fixture.inventory,
    reviewedTopology: fixture.topology,
    reviewedControl: reviewed,
  }));
  assert.throws(() => parseControlManifest(reviewed.controlValue, {
    inventory: fixture.inventory,
    reviewedTopology: fixture.topology,
  }), /validated control review chain/);
  assert.throws(() => parseControlManifest(reviewed.controlValue, {
    inventory: fixture.inventory,
    reviewedTopology: fixture.topology,
    reviewedControl: { ...reviewed },
  }), /exact deterministically validated control review chain/);

  const drift = structuredClone(attestation);
  drift.candidate_digest = `sha256:${"0".repeat(64)}`;
  resign(drift);
  assert.throws(
    () => validateControlReviewChain(candidate, drift, candidate),
    /attestation|candidate|differs/,
  );
  const forgedCandidate = structuredClone(candidate);
  forgedCandidate.identity_controls[fixture.id].owner = "self-authored replacement";
  resign(forgedCandidate);
  const forgedAttestation = structuredClone(attestation);
  forgedAttestation.candidate_digest = forgedCandidate.digest;
  forgedAttestation.input_digests = controlCandidateInputDigests(forgedCandidate);
  resign(forgedAttestation);
  assert.throws(
    () => validateControlReviewChain(forgedCandidate, forgedAttestation, candidate),
    /deterministic recomposition/,
  );
});

test("nonempty source migrations survive the complete reviewed control chain exactly", () => {
  const fixture = reviewFixture({ sourceMigration: true });
  const expected = fixture.bundleReview.bundle_control.source_migrations;
  const directory = writeReviewSet(fixture);
  const reviewSet = loadControlReviewSet(path.join(directory, "manifest.json"), fixture);
  const reviewedBundle = reviewSet.bundleReviews.get(fixture.bundleId).bundleControl;
  assert.deepEqual(reviewedBundle.source_migrations, expected);

  const candidate = composeControlCandidate({ ...fixture, reviewSet });
  assert.deepEqual(candidate.bundle_controls[fixture.bundleId].source_migrations, expected);
  const attestationPayload = {
    schema_version: 1,
    kind: "runmat-builtin-migration-control-attestation",
    authority: "reviewer-authored-development-input",
    program: "RM-1064/C00-C07",
    candidate_digest: candidate.digest,
    input_digests: controlCandidateInputDigests(candidate),
    review: { status: "reviewed", evidence: ["independent source-migration review"] },
  };
  const attestation = { ...attestationPayload, digest: evidenceDigest(attestationPayload) };
  const reviewed = validateControlReviewChain(candidate, attestation, candidate);
  assert.deepEqual(
    reviewed.controlValue.bundle_controls[fixture.bundleId].source_migrations,
    expected,
  );
  const control = parseControlManifest(reviewed.controlValue, {
    inventory: fixture.inventory,
    reviewedTopology: fixture.topology,
    reviewedControl: reviewed,
  });
  assert.deepEqual(control.bundles.get(fixture.bundleId).source_migrations, expected);

  const dropped = structuredClone(candidate);
  dropped.bundle_controls[fixture.bundleId].source_migrations = [];
  resign(dropped);
  const forgedAttestation = {
    ...structuredClone(attestation),
    candidate_digest: dropped.digest,
    input_digests: controlCandidateInputDigests(dropped),
  };
  resign(forgedAttestation);
  assert.throws(
    () => validateControlReviewChain(dropped, forgedAttestation, candidate),
    /deterministic recomposition/,
  );
});

test("existing control reviews retain uniqueness-only evidence ordering", () => {
  const fixture = reviewFixture();
  const global = structuredClone(fixture.globalReview);
  global.review.evidence = ["review:zeta", "review:alpha"];
  resign(global);
  assert.doesNotThrow(() => parseGlobalControlReview(
    global, fixture.scaffold, fixture.topology, fixture.inventory,
  ));
});

test("reviewed pilot policy survives the complete control authority chain exactly", () => {
  const fixture = reviewFixture();
  const directory = writeReviewSet(fixture);
  const reviewSet = loadControlReviewSet(path.join(directory, "manifest.json"), fixture);
  const candidate = composeControlCandidate({ ...fixture, reviewSet });
  const attestationPayload = {
    schema_version: 1,
    kind: "runmat-builtin-migration-control-attestation",
    authority: "reviewer-authored-development-input",
    program: "RM-1064/C00-C07",
    candidate_digest: candidate.digest,
    input_digests: controlCandidateInputDigests(candidate),
    review: { status: "reviewed", evidence: ["independent pilot authority review"] },
  };
  const attestation = { ...attestationPayload, digest: evidenceDigest(attestationPayload) };
  const reviewed = validateControlReviewChain(candidate, attestation, candidate);
  const control = parseControlManifest(reviewed.controlValue, {
    inventory: fixture.inventory,
    reviewedTopology: fixture.topology,
    reviewedControl: reviewed,
  });
  assert.deepEqual(reviewSet.pilotPolicy.value, fixture.globalReview.pilot_policy);
  assert.deepEqual(candidate.pilot_policy, fixture.globalReview.pilot_policy);
  assert.deepEqual(reviewed.controlValue.pilot_policy, fixture.globalReview.pilot_policy);
  assert.deepEqual(control.pilotPolicy.value, fixture.globalReview.pilot_policy);
  assert.equal(control.pilotPolicyDigest, evidenceDigest(fixture.globalReview.pilot_policy));
  assert.throws(() => control.pilotPolicy.waveByBundle.set("invented", 1), /immutable/);

  const forgedCandidate = structuredClone(candidate);
  forgedCandidate.pilot_policy.admission.maximum_elapsed_milliseconds += 1;
  resign(forgedCandidate);
  assert.throws(
    () => parseControlCandidate(forgedCandidate, candidate),
    /differs from deterministic recomposition/,
  );
});

test("control authoring rejects pilot policy count, membership, and digest drift", () => {
  for (const [name, mutate, expected] of [
    ["count", (global) => { global.pilot_policy.derived_counts.identities += 1; }, /derived identities does not match topology/],
    ["membership", (global) => { global.pilot_policy.waves[0].bundle_ids = ["missing-bundle"]; }, /unknown topology bundle/],
    ["review", (global) => { global.pilot_policy.review.status = "unreviewed"; }, /status must be reviewed/],
  ]) {
    const fixture = reviewFixture();
    const directory = writeReviewSet(fixture);
    const global = readJson(path.join(directory, "global.json"));
    mutate(global);
    resign(global);
    writeJson(path.join(directory, "global.json"), global);
    rewriteManifestDigests(directory);
    assert.throws(
      () => loadControlReviewSet(path.join(directory, "manifest.json"), fixture),
      expected,
      name,
    );
  }

  const fixture = reviewFixture();
  const directory = writeReviewSet(fixture);
  const global = readJson(path.join(directory, "global.json"));
  global.pilot_policy.admission.maximum_elapsed_milliseconds += 1;
  writeJson(path.join(directory, "global.json"), global);
  rewriteManifestDigests(directory);
  assert.throws(
    () => loadControlReviewSet(path.join(directory, "manifest.json"), fixture),
    /global control review digest mismatch/,
  );
});

test("review templates, indexing, and attestation sealing form a non-overwriting authoring workflow", () => {
  const fixture = reviewFixture();
  const parent = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-control-authoring-"));
  temporaryDirectories.add(parent);
  const initialized = initializeControlReviewTemplates(path.join(parent, "templates"), fixture.scaffold, fixture.topology);
  assert.equal(initialized.bundle_templates, 1);
  assert.ok(fs.existsSync(path.join(parent, "templates", "bundles", `${fixture.bundleId}.json`)));
  const globalTemplate = readJson(path.join(parent, "templates", "global.json"));
  assert.equal(globalTemplate.schema_version, 6);
  assert.equal(globalTemplate.pilot_policy, null);
  assert.throws(
    () => initializeControlReviewTemplates(path.join(parent, "templates"), fixture.scaffold, fixture.topology),
    /exist|EEXIST/,
  );
  const rejectedOutput = path.join(parent, "rejected-index");
  assert.throws(
    () => indexControlReviews(path.join(parent, "templates"), rejectedOutput, fixture),
    /must be reviewed|must be an array|must be an object|must exactly cover the fixed registry/,
  );
  assert.equal(fs.existsSync(rejectedOutput), false);

  const authored = path.join(parent, "authored");
  fs.mkdirSync(path.join(authored, "bundles"), { recursive: true });
  const global = structuredClone(fixture.globalReview); delete global.digest;
  const bundle = structuredClone(fixture.bundleReview); delete bundle.digest;
  writeJson(path.join(authored, "global.json"), global);
  writeJson(path.join(authored, "bundles", `${fixture.bundleId}.json`), bundle);
  const indexed = indexControlReviews(authored, path.join(parent, "indexed"), fixture);
  const reviewSet = loadControlReviewSet(indexed.manifest, fixture);
  const candidate = composeControlCandidate({ ...fixture, reviewSet });
  const template = buildControlAttestationTemplate(candidate);
  assert.equal(template.review.status, "unreviewed");
  template.review = { status: "reviewed", evidence: ["independent fixture review"] };
  const attestation = sealControlAttestation(template, candidate);
  assert.equal(validateControlReviewChain(candidate, attestation, candidate).candidate.digest, candidate.digest);
  assert.throws(() => sealControlAttestation(attestation, candidate), /must not supply its own digest/);
});

test("candidate composition rejects cross-row defects before independent attestation", () => {
  const fixture = reviewFixture();
  for (const [label, mutate, message] of [
    ["self prerequisite", (bundle) => { bundle.bundle_control.prerequisites = [{ bundle_id: fixture.bundleId, kind: "semantic" }]; }, /cannot depend on itself|pilot prerequisite cannot be self-referential/],
    ["scope overlap", (bundle) => {
      bundle.bundle_control.additional_authored_write_set.push({
        kind: "file",
        path: "crates/runmat-runtime/src/builtins/generated_wasm_registry.rs",
      });
    }, /authored scope overlaps integration product/],
    ["missing required gate", (bundle) => { bundle.bundle_control.gate_plans = bundle.bundle_control.gate_plans.filter((plan) => plan.gate !== "catalog-contract"); }, /required gate catalog-contract/],
    ["baseline spelling drift", (bundle) => { bundle.identity_controls[fixture.id].public_identity.primary_spelling.spelling = "Foo"; }, /public identity spelling differs/],
  ]) {
    const directory = writeReviewSet(fixture);
    const bundle = readJson(path.join(directory, "bundle.json"));
    mutate(bundle);
    resign(bundle);
    writeJson(path.join(directory, "bundle.json"), bundle);
    rewriteManifestDigests(directory);
    assert.throws(
      () => {
        const reviewSet = loadControlReviewSet(path.join(directory, "manifest.json"), fixture);
        composeControlCandidate({ ...fixture, reviewSet });
      },
      message,
      label,
    );
  }
});

test("global review rejects duplicate program authority under different profile names", () => {
  const fixture = reviewFixture();
  const duplicate = structuredClone(fixture.globalReview);
  duplicate.program_profiles["profile-duplicate"] = structuredClone(
    duplicate.program_profiles["profile-architecture"],
  );
  duplicate.program_profiles = Object.fromEntries(
    Object.entries(duplicate.program_profiles).sort(([left], [right]) => left.localeCompare(right)),
  );
  resign(duplicate);
  assert.throws(
    () => parseGlobalControlReview(
      duplicate, fixture.scaffold, fixture.topology, fixture.inventory,
    ),
    /program payload duplicates reviewed profile/,
  );
});

test("storage policy supports distinct host profiles and rejects ambiguous selectors", () => {
  const fixture = reviewFixture();
  const policy = structuredClone(fixture.globalReview.storage_policy);
  policy.host_profiles["linux-builder"] = {
    operating_system: "linux",
    architecture: "x86_64",
    execution_host: "runmat-linux-builder",
    volume_roles: {
      source_worktree: {
        role: "source-worktree",
        mount_path: "/workspace",
        filesystem_id: "posix-dev:10",
        minimum_free_bytes: 1,
        pause_below_bytes: 2,
        maximum_observation_age_seconds: 60,
      },
      target_temp: {
        role: "target-temp",
        mount_path: "/mnt/runmat-build",
        filesystem_id: "posix-dev:11",
        minimum_free_bytes: 1,
        pause_below_bytes: 2,
        maximum_observation_age_seconds: 60,
      },
    },
  };
  assert.equal(parseStoragePolicy(policy), policy);

  policy.host_profiles["duplicate-selector"] = structuredClone(
    policy.host_profiles["linux-builder"],
  );
  policy.host_profiles = Object.fromEntries(
    Object.entries(policy.host_profiles).sort(([left], [right]) => left.localeCompare(right)),
  );
  assert.throws(() => parseStoragePolicy(policy), /selectors must be unique/);
});

function reviewFixture({ sourceMigration = false } = {}) {
  const id = "foo";
  const bundleId = "math-basic-foo";
  const repository = repositoryFixture({ identity: id });
  const migrationSource = "crates/runmat-runtime/src/builtins/math/legacy.rs";
  const migrationTarget = "crates/runmat-runtime/src/builtins/math/legacy/mod.rs";
  const migrationSupport = "crates/runmat-runtime/src/builtins/math/legacy/support.rs";
  if (sourceMigration) {
    fs.writeFileSync(path.join(repository, migrationSource), "pub(crate) fn legacy_support() {}\n");
    execFileSync("git", ["add", migrationSource], { cwd: repository });
    execFileSync("git", [
      "-c", "user.name=RunMat Test", "-c", "user.email=test@runmat.invalid",
      "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "add legacy source",
    ], { cwd: repository });
  }
  const compiledInventory = compiledInventoryFixture(id);
  const dispositions = { schema_version: 1, kind: "runmat-builtin-dispositions", authority: "review-input-only", identities: {
    [id]: { disposition: "canonical", canonical: null, domain: "math", family: "basic", reason: null, review: { status: "reviewed", evidence: ["fixture review"] } },
  } };
  const inventory = buildInventory(repository, dispositions, { revision: REVISION, compiledInventory });
  const draft = buildControlDraft(inventory);
  const topology = topologyFixture(inventory, bundleId, id, { controlDraftDigest: draft.digest });
  const scaffold = buildControlOverlayScaffold(inventory, draft, topology);
  const gatePolicy = reviewedGatePolicy(inventory);
  const requiredMaturity = new Set([
    "identity", "disposition", "catalog-contract", "runtime-binding",
    "link-reachability", "documentation",
  ]);
  const maturity = Object.fromEntries(MATURITY_GATES.map((gate) => [gate, requiredMaturity.has(gate)
    ? { applicability: "required", reason: null, evidence: [] }
    : { applicability: "not-applicable", reason: "Outside the review-input fixture", evidence: ["fixture review"] }]));
  const sourceMigrations = sourceMigration ? [{
    strategy: "module-support-reparent",
    source_path: migrationSource,
    source_baseline_digest: inventory.source.files
      .find((entry) => entry.path === migrationSource).content_digest,
    promoted_target: { kind: "authored", path: migrationTarget },
    destination_paths: [migrationTarget, migrationSupport],
    support_destinations: [{ destination_path: migrationSupport, reexports: [] }],
    identity_destinations: [],
    reason: "Reparent shared legacy support beneath its authored module",
    review: { status: "reviewed", evidence: ["fixture source-migration review"] },
  }] : [];
  const legacyBefore = {
    module: "legacy", source_kind: "file", source_path: migrationSource,
    role: "support", visibility: "private",
    declaration_condition: { kind: "always" }, declaration_order: 1,
    macro_use: false, reexports: [], aggregation_sources: [],
  };
  const legacyAfter = {
    ...legacyBefore, source_kind: "directory", source_path: migrationTarget,
  };
  const bundleControl = {
    prerequisites: [],
    additional_authored_write_set: [
      { kind: "tree", path: `crates/runmat-builtins/src/catalog/entries/math/basic/${id}` },
      { kind: "file", path: `crates/runmat-runtime/src/builtins/math/basic/${id}.rs` },
      ...(sourceMigration ? [
        { kind: "file", path: migrationSource },
        { kind: "tree", path: path.posix.dirname(migrationTarget) },
      ] : []),
    ],
    integration_product_refs: sourceMigration
      ? ["runtime-math", "wasm-registry"] : ["wasm-registry"],
    module_composition_transition: sourceMigration ? {
      schema_version: 5,
      kind: "runmat-builtin-module-composition-transition",
      transition_id: bundleId,
      product_states: [{
        product_id: "runtime-math", before_state: "present", after_state: "present",
      }],
      changes: [{
        product_id: "runtime-math", operation: "replace",
        before: legacyBefore, after: legacyAfter,
      }],
    } : null,
    source_migrations: sourceMigrations,
    expected_removals: [],
    baseline_evidence: bundleBaselineEvidence(inventory, [id]),
    gate_plans: gatePolicy.gatePlans,
    owner_role: "builtin-migrator", complexity: { class: "low", weight: 1, basis: ["single identity"] },
    review: { status: "reviewed", evidence: ["fixture review"] },
  };
  const runtimeOwner = `crates/runmat-runtime/src/builtins/math/basic/${id}.rs`;
  const identityControls = { [id]: {
    public_identity: { kind: "primary", primary_spelling: { identity: id, spelling: id } },
    forms: { kind: "callable", callable_spellings: [id], constant_spellings: [] },
    implementation: {
      callable: { kind: "owned", owner_path: runtimeOwner, bindings: [{
        kind: "canonical_binding", function: `${id}_builtin`,
        variant: "default", builtin_path: `builtins::${id}`,
        native_symbol: `runmat_builtin_binding_v1_${Buffer.from(id).toString("hex")}_${Buffer.from("default").toString("hex")}`,
      }] },
      constant: { kind: "none", reason: "no-constant-form" },
    },
    shared_dependencies: [],
    complexity: { class: "low", weight: 1, basis: ["single identity"] },
    maturity,
    expected_authorities: {
      catalog_package: `crates/runmat-builtins/src/catalog/entries/math/basic/${id}/mod.rs`,
      catalog_alias_package: null,
      catalog_constant_package: null,
      catalog_entry_count: 1,
      catalog_constant_count: 0,
      documentation: "catalog",
      native_link: "required", wasm_registry: "not-applicable",
    },
    owner: "fixture",
    review: { status: "reviewed", evidence: ["fixture review"] },
  } };
  const scaffoldBundle = scaffold.bundle_rows.find((entry) => entry.bundle_id === bundleId);
  const scaffoldIdentity = scaffold.identity_rows.find((entry) => entry.identity === id);
  const bundlePayload = {
    schema_version: 6,
    kind: "runmat-builtin-migration-bundle-control-review",
    authority: "reviewer-authored-development-input",
    program: "RM-1064/C00-C07",
    bindings: {
      scaffold_digest: scaffold.digest,
      topology_digest: topology.digest,
      bundle_id: bundleId,
      scaffold_bundle_row_digest: evidenceDigest(scaffoldBundle),
      topology_bundle_row_digest: evidenceDigest(topology.bundles.get(bundleId)),
      identity_rows: [{
        identity: id,
        scaffold_identity_row_digest: evidenceDigest(scaffoldIdentity),
        topology_identity_digest: evidenceDigest(topology.identities.get(id)),
        authority_proposal_digest: scaffold.authority_proposals.identity_rows
          .find((entry) => entry.identity === id).proposal_digest,
      }],
    },
    bundle_control: bundleControl,
    identity_controls: identityControls,
    review: { status: "reviewed", evidence: ["fixture bundle review"] },
  };
  const bundleReview = { ...bundlePayload, digest: evidenceDigest(bundlePayload) };
  const moduleCompositionBaseline = fixtureReviewedModuleCompositionBaseline(
    fixtureModuleCompositionBaseline(
      new Map([
        ["catalog-root", [fixtureRootCompositionChild("catalog")]],
        ["catalog-math", [fixtureCatalogCompositionChild()]],
        ["runtime-root", [fixtureRootCompositionChild("runtime")]],
        ["runtime-math", sourceMigration
          ? [fixtureCompositionChild(), legacyBefore]
          : [fixtureCompositionChild()]],
      ]),
      new Set(["catalog-math", "catalog-root", "runtime-math", "runtime-root"]),
    ),
  );
  const globalPayload = {
    schema_version: 6,
    kind: "runmat-builtin-migration-global-control-review",
    authority: "reviewer-authored-development-input",
    program: "RM-1064/C00-C07",
    bindings: { scaffold_digest: scaffold.digest, topology_digest: topology.digest, migration_finding_rows_digest: evidenceDigest(scaffold.migration_finding_rows), module_composition_reviewed_baseline_digest: moduleCompositionBaseline.digest },
    program_profiles: gatePolicy.programProfiles,
    integration_products: fixtureIntegrationProducts(
      inventory, sourceMigration, "runtime-math",
    ),
    module_composition_baseline: moduleCompositionBaseline,
    migration_findings: { schema_version: 1, kind: "runmat-builtin-migration-finding-dispositions", rows: [], review: { status: "reviewed", evidence: ["fixture review"] } },
    exception_manifest: { entries: [], review: { status: "reviewed", evidence: ["fixture review"] } },
    target_policy: fixtureTargetPolicy([{
      operating_system: inventory.compiled_inventory.build.operating_system,
      architecture: inventory.compiled_inventory.build.architecture,
    }]),
    storage_policy: { host_profiles: { "fixture-host": {
      operating_system: inventory.compiled_inventory.build.operating_system,
      architecture: inventory.compiled_inventory.build.architecture,
      execution_host: os.hostname(),
      volume_roles: {
        source_worktree: { role: "source-worktree", mount_path: "/System/Volumes/Data", filesystem_id: "posix-dev:1", minimum_free_bytes: 1, pause_below_bytes: 2, maximum_observation_age_seconds: 60 },
        target_temp: { role: "target-temp", mount_path: "/private/tmp/runmat-integration-tmp", filesystem_id: "posix-dev:2", minimum_free_bytes: 1, pause_below_bytes: 2, maximum_observation_age_seconds: 60 },
      },
    } }, targets_must_be_disjoint: true, occt_default: "disabled-unless-affected" },
    pilot_policy: reviewedPilotPolicy(
      topology, new Map([[bundleId, bundleControl.prerequisites]]),
    ),
    review: { status: "reviewed", evidence: ["fixture global review"] },
  };
  const globalReview = { ...globalPayload, digest: evidenceDigest(globalPayload) };
  return { repository, inventory, compiledInventory, id, bundleId, draft, topology, scaffold, bundleReview, globalReview };
}

function reviewedGatePolicy(inventory) {
  const profiles = new Map();
  const gatePlans = fixtureGatePlans(inventory).map(({ program, ...plan }) => {
    const key = JSON.stringify(program);
    if (!profiles.has(key)) profiles.set(key, `profile-${plan.gate}`);
    return { ...plan, program_profile_id: profiles.get(key) };
  });
  const programs = new Map(fixtureGatePlans(inventory).map((plan) => [
    JSON.stringify(plan.program),
    plan.program,
  ]));
  const programProfiles = Object.fromEntries([...profiles]
    .map(([key, id]) => [id, {
      program: programs.get(key),
      review: { status: "reviewed", evidence: ["fixture program review"] },
    }])
    .sort(([left], [right]) => left.localeCompare(right)));
  return { gatePlans, programProfiles };
}

function writeReviewSet(fixture) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-control-review-set-"));
  temporaryDirectories.add(directory);
  writeJson(path.join(directory, "bundle.json"), fixture.bundleReview);
  writeJson(path.join(directory, "global.json"), fixture.globalReview);
  const payload = {
    schema_version: 1,
    kind: "runmat-builtin-migration-control-review-set",
    authority: "content-addressed-review-index-only",
    program: "RM-1064/C00-C07",
    bindings: { scaffold_digest: fixture.scaffold.digest, topology_digest: fixture.topology.digest, inventory_digest: fixture.inventory.digest },
    global_review: { path: "global.json", content_digest: contentDigest(fs.readFileSync(path.join(directory, "global.json"))) },
    bundle_reviews: [{ bundle_id: fixture.bundleId, path: "bundle.json", content_digest: contentDigest(fs.readFileSync(path.join(directory, "bundle.json"))) }],
  };
  writeJson(path.join(directory, "manifest.json"), { ...payload, digest: evidenceDigest(payload) });
  return directory;
}

function rewriteManifestDigests(directory) {
  const manifest = readJson(path.join(directory, "manifest.json"));
  manifest.global_review.content_digest = contentDigest(fs.readFileSync(path.join(directory, manifest.global_review.path)));
  for (const reference of manifest.bundle_reviews) reference.content_digest = contentDigest(fs.readFileSync(path.join(directory, reference.path)));
  resign(manifest);
  writeJson(path.join(directory, "manifest.json"), manifest);
}

function resign(value) { const { digest: _ignored, ...payload } = value; value.digest = evidenceDigest(payload); }
function resignCompositionBaseline(globalReview) {
  resign(globalReview.module_composition_baseline);
  globalReview.bindings.module_composition_reviewed_baseline_digest = globalReview.module_composition_baseline.digest;
  resign(globalReview);
}
function writeJson(target, value) { fs.writeFileSync(target, `${JSON.stringify(value, null, 2)}\n`); }
function readJson(target) { return JSON.parse(fs.readFileSync(target, "utf8")); }
