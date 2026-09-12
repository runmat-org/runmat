import fs from "node:fs";
import { execFileSync } from "node:child_process";
import os from "node:os";
import path from "node:path";
import { MATURITY_GATES, parseControlManifest } from "../control.mjs";
import { bundleBaselineEvidence } from "../baseline-evidence.mjs";
import { buildControlDraft } from "../control-draft.mjs";
import { validateControlReviewChain } from "../control-authoring/authority.mjs";
import { composeControlCandidate, controlCandidateInputDigests } from "../control-authoring/compose.mjs";
import { loadControlReviewSet } from "../control-authoring/review-set.mjs";
import { buildControlOverlayScaffold } from "../control-authoring/scaffold.mjs";
import { contentDigest, evidenceDigest } from "../evidence.mjs";
import { buildDocumentationCutoverArtifact, documentationCutoverChecks } from "../documentation-cutover.mjs";
import { gatePlanEvidence } from "../gate-plan.mjs";
import { buildInventory } from "../inventory.mjs";
import { issueLease, parseLease } from "../lease.mjs";
import { acceptedSealSet, barrierSealSet, emptyQueueState, validateQueueState } from "../queue.mjs";
import { validateQueueCheckpoint } from "../queue-checkpoint.mjs";
import { R31_REQUIRED_LANES } from "../target-policy.mjs";
import {
  candidateInputDigests, freezeReviewedTopology, parseReviewedTopology, reviewedTopologyView,
} from "../topology/freeze.mjs";
import { cleanupTemporaryDirectories, createTemporaryDirectory } from "./temporary-directories.mjs";

export const REVISION = `git:${"1".repeat(40)}`;

export function repositoryFixture({ sidecar = false, identity = "foo", composition = false } = {}) {
  const fixtureRoot = createTemporaryDirectory("runmat-migration-factory-");
  const root = path.join(fixtureRoot, "repository");
  fs.mkdirSync(root);
  write(root, "Cargo.toml", "[workspace]\nresolver = \"2\"\n");
  write(root, "scripts/development/check-architecture-boundaries.mjs", "process.exit(0);\n");
  write(root, "scripts/development/builtin-migration/documentation-export-cli.mjs", "process.stdout.write('{}\\n');\n");
  write(root, "scripts/regenerate-wasm-registry.mjs", "// fixture generator identity\n");
  if (composition) {
    write(root, "crates/runmat-runtime/src/builtins/math/mod.rs", "// fixture generated parent\n");
    write(root, "crates/runmat-runtime/src/builtins/math/basic/mod.rs", "pub mod foo;\n");
  }
  write(root, "scripts/development/verify-builtin-generated-products.mjs", `
import fs from "node:fs"; import crypto from "node:crypto";
const digest = (bytes) => "sha256:" + crypto.createHash("sha256").update(bytes).digest("hex");
const input = JSON.parse(fs.readFileSync(0, "utf8"));
const observed = fs.readFileSync("crates/runmat-runtime/src/builtins/generated_wasm_registry.rs");
const generator = fs.readFileSync("scripts/regenerate-wasm-registry.mjs");
const manifest = input.native_registration_manifest;
const value = { schema_version: 2, kind: "runmat-builtin-generated-products-proof", authority: "machine-derived-integration-evidence", products: [{ product_id: "wasm-registry", path: "crates/runmat-runtime/src/builtins/generated_wasm_registry.rs", generator: { path: "scripts/regenerate-wasm-registry.mjs", content_digest: digest(generator) }, checked_in: { byte_length: observed.length, content_digest: digest(observed) }, first: { byte_length: observed.length, content_digest: digest(observed) }, second: { byte_length: observed.length, content_digest: digest(observed) }, deterministic: true, synchronized: true, verification: { kind: "native_wasm_registration_manifest", generated_manifest: manifest, native_manifest: manifest, result: "pass" } }], result: "pass" };
process.stdout.write(JSON.stringify(value) + "\\n");
`);
  write(root, `crates/runmat-runtime/src/builtins/math/basic/${identity}.rs`, `
#[runtime_builtin(name = "${identity}", category = "math/basic", builtin_path = "crate::builtins::math::basic::${identity}")]
fn ${identity.replaceAll(".", "_")}_builtin() {}
#[cfg(test)] mod tests {}
`);
  write(root, "crates/runmat-runtime/src/builtins/generated_wasm_registry.rs", "// no fixture WASM registrations\n");
  write(root, `crates/runmat-builtins/src/catalog/entries/math/basic/${identity}/mod.rs`, `
pub const ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
 identity: BuiltinCatalogIdentity { name: "${identity}" },
 link: BuiltinLinkContract {},
 contract: BuiltinContractDeclaration { inference_rule: BuiltinInferenceRule::Math(Rule::Foo) },
 documentation: BuiltinDocumentation { summary: "Foo", examples: &["${identity}(1)"] },
};`);
  if (sidecar) write(root, `docs/builtins/reference/${identity}.json`, JSON.stringify({ name: identity, summary: "Foo", examples: [{ input: `${identity}(1)` }] }));
  execFileSync("git", ["init", "--quiet"], { cwd: root });
  execFileSync("git", ["add", "."], { cwd: root });
  execFileSync("git", ["-c", "user.name=RunMat Test", "-c", "user.email=test@runmat.invalid", "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "fixture"], { cwd: root });
  return root;
}

export function cleanupRepositoryFixtures() {
  cleanupTemporaryDirectories();
}

export function controlledFixture(options = {}) {
  const repository = repositoryFixture(options);
  const revision = `git:${execFileSync("git", ["rev-parse", "HEAD"], { cwd: repository, encoding: "utf8" }).trim()}`;
  const id = options.identity ?? "foo";
  const compiledInventory = compiledInventoryFixture(id, options);
  const seedInventory = buildInventory(repository, undefined, { revision, compiledInventory });
  const dispositions = {
    schema_version: 1,
    kind: "runmat-builtin-dispositions",
    authority: "review-input-only",
    identities: {
      [id]: {
        disposition: "canonical",
        canonical: null,
        domain: "math",
        family: "basic",
        reason: null,
        review: { status: "reviewed", evidence: ["fixture review"] },
      },
    },
  };
  const inventory = buildInventory(repository, dispositions, { revision, compiledInventory });
  if (options.unresolved) {
    inventory.identities[0].unresolved = [...options.unresolved];
    const { digest: _ignored, ...payload } = inventory;
    inventory.digest = evidenceDigest(payload);
  }
  const bundleId = "math-basic-foo";
  const maturity = Object.fromEntries(MATURITY_GATES.map((gate) => [gate, gate === "identity" || gate === "disposition" || gate === "catalog-contract" || gate === "runtime-binding" || gate === "documentation" || gate === "link-reachability"
    ? { applicability: "required", reason: null, evidence: [] }
    : { applicability: "not-applicable", reason: "Reviewed as outside this fixture's behavior", evidence: ["fixture review"] }]));
  const draft = buildControlDraft(inventory);
  const topology = topologyFixture(inventory, bundleId, id, { controlDraftDigest: draft.digest });
  const compositionProduct = "runtime-math-parent";
  const compositionChild = {
    module: "basic", source_kind: "directory",
    source_path: "crates/runmat-runtime/src/builtins/math/basic/mod.rs",
    role: "group", visibility: "public", feature_policy: { kind: "always" },
    macro_use: false, reexport: { kind: "none" }, aggregation_sources: [],
  };
  const bundleControl = { prerequisites: [], additional_authored_write_set: [{ kind: "tree", path: `crates/runmat-builtins/src/catalog/entries/math/basic/${id}` }, { kind: "file", path: `crates/runmat-runtime/src/builtins/math/basic/${id}.rs` }, ...(options.composition ? [{ kind: "file", path: compositionChild.source_path }] : []), ...(options.sidecar ? [{ kind: "file", path: `docs/builtins/reference/${id}.json` }] : [])], integration_product_refs: options.composition ? [compositionProduct, "wasm-registry"] : ["wasm-registry"], module_composition_transition: options.composition ? { schema_version: 2, kind: "runmat-builtin-module-composition-transition", transition_id: bundleId, changes: [{ product_id: compositionProduct, operation: "add", before: null, after: compositionChild }] } : null, expected_removals: [], baseline_evidence: bundleBaselineEvidence(inventory, [id]), gate_plans: fixtureGatePlans(inventory, Object.values(options.storageProfiles ?? {})), owner_role: "builtin-migrator", complexity: { class: "low", weight: 1, basis: ["single identity"] }, review: { status: "reviewed", evidence: ["fixture review"] } };
  const runtimeOwner = `crates/runmat-runtime/src/builtins/math/basic/${id}.rs`;
  const identityControl = {
    public_identity: {
      kind: "primary", primary_spelling: { identity: id, spelling: id },
    },
    forms: { kind: "callable", callable_spellings: [id], constant_spellings: [] },
    implementation: {
      callable: {
        kind: "owned", owner_path: runtimeOwner, bindings: [{
          kind: "canonical_binding", function: `${id}_builtin`,
          variant: "default", builtin_path: `builtins::${id}`,
          native_symbol: nativeSymbol(id, "default"),
        }],
      },
      constant: { kind: "none", reason: "no-constant-form" },
    },
    shared_dependencies: [],
    complexity: { class: "low", weight: 1, basis: ["single identity"] },
    maturity,
    expected_authorities: { catalog_package: `crates/runmat-builtins/src/catalog/entries/math/basic/${id}/mod.rs`, catalog_alias_package: null, catalog_constant_package: null, catalog_entry_count: 1, catalog_constant_count: 0, documentation: "catalog", native_link: "required", wasm_registry: "not-applicable" },
    owner: "fixture", review: { status: "reviewed", evidence: ["fixture review"] },
  };
  const migrationFindings = { schema_version: 1, kind: "runmat-builtin-migration-finding-dispositions", rows: inventory.migration_findings.map((finding) => ({ finding_digest: evidenceDigest(finding), ...finding, disposition: "bundle-work", bundle_id: bundleId, reason: "Fixture migration work", evidence: ["fixture review"] })), review: { status: "reviewed", evidence: ["fixture review"] } };
  const exceptionManifest = { entries: [], review: { status: "reviewed", evidence: ["fixture review"] } };
  const hostProfiles = { "fixture-host": {
      operating_system: inventory.compiled_inventory.build.operating_system,
      architecture: inventory.compiled_inventory.build.architecture,
      execution_host: os.hostname(),
      volume_roles: {
        source_worktree: { role: "source-worktree", mount_path: "/System/Volumes/Data", filesystem_id: "posix-dev:1", minimum_free_bytes: 1, pause_below_bytes: 2, maximum_observation_age_seconds: 60 },
        target_temp: { role: "target-temp", mount_path: "/private/tmp/runmat-integration-tmp", filesystem_id: "posix-dev:2", minimum_free_bytes: 1, pause_below_bytes: 2, maximum_observation_age_seconds: 60 },
      },
    }, ...(options.storageProfiles ?? {}) };
  const storagePolicy = {
    host_profiles: Object.fromEntries(Object.entries(hostProfiles).sort(([left], [right]) => left.localeCompare(right))),
    targets_must_be_disjoint: true,
    occt_default: "disabled-unless-affected",
  };
  const scaffold = buildControlOverlayScaffold(inventory, draft, topology);
  const { reviewSet, manifestPath: controlReviewSetPath } = fixtureControlReviewSet(repository, inventory, topology, scaffold, bundleId, bundleControl, id, identityControl, migrationFindings, exceptionManifest, storagePolicy, options);
  const candidate = composeControlCandidate({ inventory, topology, scaffold, reviewSet });
  const attestationPayload = { schema_version: 1, kind: "runmat-builtin-migration-control-attestation", authority: "reviewer-authored-development-input", program: "RM-1064/C00-C07", candidate_digest: candidate.digest, input_digests: controlCandidateInputDigests(candidate), review: { status: "reviewed", evidence: ["fixture independent review"] } };
  const attestation = { ...attestationPayload, digest: evidenceDigest(attestationPayload) };
  const reviewedControl = validateControlReviewChain(candidate, attestation, candidate);
  const controlValue = reviewedControl.controlValue;
  const control = parseControlManifest(controlValue, { inventory, reviewedTopology: topology, reviewedControl });
  const queueState = validateQueueState(emptyQueueState(control), control, () => null);
  const queueCheckpointValue = initialQueueCheckpointValue(control, queueState);
  const queueCheckpoint = validateQueueCheckpoint(
    queueCheckpointValue, queueCheckpointValue.digest, queueState, control,
  );
  const accepted = acceptedSealSet(queueState, control);
  const barriers = barrierSealSet(queueState, control, bundleId);
  const leaseRequest = { schema_version: 4, kind: "runmat-builtin-migration-lease-request", authority: "reviewed-development-request", control_manifest_digest: control.digest, bundle_id: bundleId, lease_id: "lease-foo", owner: "fixture", base_revision: revision, lease_base_inventory: leaseBaseInventoryBinding(inventory), queue_checkpoint_digest: queueCheckpoint.digest, accepted_seals: accepted.value.seals, accepted_seal_set_digest: accepted.value.digest, barrier_seals: barriers.value.seals, barrier_seal_set_digest: barriers.value.digest, issued_at: "2020-01-01T00:00:00.000Z", expires_at: "2099-01-01T00:00:00.000Z", review: { status: "reviewed", evidence: ["fixture review"] } };
  const leaseValue = issueLease(
    leaseRequest, control, repository, inventory, queueState, queueCheckpoint,
  );
  const lease = parseLease(leaseValue, control, repository);
  return { repository, inventory, compiledInventory, controlValue, control, topology, draft, scaffold, reviewSet, controlReviewSetPath, candidate, attestation, reviewedControl, queueState, queueCheckpointValue, queueCheckpoint, leaseRequest, leaseValue, lease, id, bundleId };
}

export function initialQueueCheckpointValue(control, queueState) {
  const accepted = acceptedSealSet(queueState, control);
  const payload = {
    schema_version: 1,
    kind: "runmat-builtin-migration-queue-checkpoint",
    authority: "reviewer-authored-current-queue-checkpoint",
    control_manifest_digest: control.digest,
    queue_state_digest: queueState.stateDigest,
    predecessor_checkpoint_digest: null,
    head_seal: null,
    source_revision: control.baseline.revision,
    source_digest: control.baseline.source_digest,
    accepted_seal_set_digest: accepted.value.digest,
    review: { status: "reviewed", evidence: ["fixture current queue checkpoint review"] },
  };
  return { ...payload, digest: evidenceDigest(payload) };
}

export function leaseBaseInventoryBinding(inventory) {
  return {
    source_revision: inventory.source.revision,
    source_digest: inventory.source.digest,
    inventory_digest: inventory.digest,
    compiled_inventory_digest: inventory.compiled_inventory.digest,
  };
}

function fixtureControlReviewSet(repository, inventory, topology, scaffold, bundleId, bundleControl, id, identityControl, migrationFindings, exceptionManifest, storagePolicy, options) {
  const root = path.join(path.dirname(repository), "control-review");
  fs.mkdirSync(path.join(root, "bundles"), { recursive: true });
  const programs = new Map();
  for (const plan of bundleControl.gate_plans) programs.set(JSON.stringify(plan.program), plan.program);
  const orderedPrograms = [...programs.entries()].sort(([left], [right]) => left.localeCompare(right));
  const profileByProgram = new Map(orderedPrograms.map(([key], index) => [key, `program-${index + 1}`]));
  const programProfiles = Object.fromEntries(orderedPrograms.map(([key, program], index) => [`program-${index + 1}`, {
    program,
    review: { status: "reviewed", evidence: ["fixture executable review"] },
  }]));
  const globalPayload = {
    schema_version: 3,
    kind: "runmat-builtin-migration-global-control-review",
    authority: "reviewer-authored-development-input",
    program: "RM-1064/C00-C07",
    bindings: {
      scaffold_digest: scaffold.digest,
      topology_digest: topology.digest,
      migration_finding_rows_digest: evidenceDigest(scaffold.migration_finding_rows),
    },
    program_profiles: programProfiles,
    integration_products: fixtureIntegrationProducts(inventory, options.composition),
    module_composition_baseline: options.composition ? {
      schema_version: 2,
      kind: "runmat-builtin-module-composition-projection",
      products: [{
        product_id: "runtime-math-parent", crate_role: "runtime",
        path: "crates/runmat-runtime/src/builtins/math/mod.rs",
        module_path: "crate::builtins::math", aggregations: [], children: [],
      }],
    } : null,
    migration_findings: migrationFindings,
    exception_manifest: exceptionManifest,
    target_policy: fixtureTargetPolicy([...new Map(Object.values(storagePolicy.host_profiles).map((profile) => [
      `${profile.operating_system}\0${profile.architecture}`,
      { operating_system: profile.operating_system, architecture: profile.architecture },
    ])).entries()].sort(([left], [right]) => left.localeCompare(right)).map(([, target]) => target)),
    storage_policy: storagePolicy,
    review: { status: "reviewed", evidence: ["fixture global review"] },
  };
  const globalValue = { ...globalPayload, digest: evidenceDigest(globalPayload) };
  const globalBytes = writeEvidenceJson(path.join(root, "global.json"), globalValue);

  const scaffoldBundle = scaffold.bundle_rows.find((entry) => entry.bundle_id === bundleId);
  const scaffoldIdentity = scaffold.identity_rows.find((entry) => entry.identity === id);
  const bundleReviewControl = {
    ...structuredClone(bundleControl),
    gate_plans: bundleControl.gate_plans.map(({ program, ...plan }) => ({
      gate: plan.gate,
      program_profile_id: profileByProgram.get(JSON.stringify(program)),
      arguments: [...plan.arguments],
      working_directory: plan.working_directory,
      parser: plan.parser,
      expected_artifact_roles: [...plan.expected_artifact_roles],
    })),
  };
  const bundlePayload = {
    schema_version: 4,
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
    bundle_control: bundleReviewControl,
    identity_controls: { [id]: identityControl },
    review: { status: "reviewed", evidence: ["fixture bundle review"] },
  };
  const bundleValue = { ...bundlePayload, digest: evidenceDigest(bundlePayload) };
  const bundlePath = `bundles/${bundleId}.json`;
  const bundleBytes = writeEvidenceJson(path.join(root, bundlePath), bundleValue);
  const manifestPayload = {
    schema_version: 1,
    kind: "runmat-builtin-migration-control-review-set",
    authority: "content-addressed-review-index-only",
    program: "RM-1064/C00-C07",
    bindings: { scaffold_digest: scaffold.digest, topology_digest: topology.digest, inventory_digest: inventory.digest },
    global_review: { path: "global.json", content_digest: contentDigest(globalBytes) },
    bundle_reviews: [{ bundle_id: bundleId, path: bundlePath, content_digest: contentDigest(bundleBytes) }],
  };
  const manifestValue = { ...manifestPayload, digest: evidenceDigest(manifestPayload) };
  const manifestPath = path.join(root, "review-set.json");
  writeEvidenceJson(manifestPath, manifestValue);
  return {
    reviewSet: loadControlReviewSet(manifestPath, { scaffold, topology, inventory }),
    manifestPath,
  };
}

function writeEvidenceJson(target, value) {
  const bytes = Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
  fs.writeFileSync(target, bytes);
  return bytes;
}

export function topologyFixture(inventory, bundleId, id, options = {}) {
  const bundle = options.bundle ?? {
    id: bundleId,
    cohort: "C01",
    authority_components: [`component-${id}`],
    identities: [id],
    atomic_reason: "One runtime and catalog contract",
    composition: {
      kind: "single-component",
      target_packages: [{ domain: "math", family: "basic" }],
      authored_write_set: [],
      shared_authority_sources: [],
      evidence: ["fixture topology review"],
    },
    identity_targets: [{ identity: id, domain: "math", family: "basic", classification: "preserved", evidence: ["fixture topology review"] }],
    review: { status: "reviewed", evidence: ["fixture topology review"] },
  };
  const identityRow = options.identity ?? {
    identity: id,
    bundle_id: bundleId,
    cohort: "C01",
    domain: "math",
    family: "basic",
    disposition: { kind: "canonical", canonical: null, reason: null, source: "reviewed-input" },
    classification: "preserved",
    evidence: ["fixture topology review"],
  };
  const candidatePayload = {
    schema_version: 1,
    kind: "runmat-builtin-migration-topology-candidate",
    authority: "composed-unreviewed-candidate",
    program: "RM-1064/C00-C07",
    baseline: {
      revision: inventory.source.revision,
      inventory_digest: inventory.digest,
      component_graph_digest: `sha256:${"e".repeat(64)}`,
      control_draft_digest: options.controlDraftDigest ?? `sha256:${"f".repeat(64)}`,
    },
    inputs: {
      c01_c03_review: `sha256:${"1".repeat(64)}`,
      c04_c05_review: `sha256:${"2".repeat(64)}`,
      c06_c07_review: `sha256:${"3".repeat(64)}`,
      reconciliation: `sha256:${"4".repeat(64)}`,
      stability_corrections: `sha256:${"5".repeat(64)}`,
    },
    bundles: { [bundleId]: bundle },
    identities: { [id]: identityRow },
    summary: { components: 1, identities: 1, bundles: 1, cohorts: ["C01"], target_packages: 1 },
    review: { status: "unreviewed", evidence: [] },
  };
  const candidate = { ...candidatePayload, digest: evidenceDigest(candidatePayload) };
  const attestation = {
    schema_version: 1,
    kind: "runmat-builtin-topology-attestation",
    authority: "reviewer-authored-development-input",
    program: "RM-1064/C00-C07",
    candidate_digest: candidate.digest,
    input_digests: candidateInputDigests(candidate),
    review: { status: "reviewed", evidence: ["fixture topology review"] },
  };
  const frozen = freezeReviewedTopology(candidate, attestation, candidate);
  return reviewedTopologyView(parseReviewedTopology(frozen, candidate, attestation, candidate));
}

export function compiledInventoryFixture(id = "foo", options = {}) {
  const catalog = catalogEntryFixture(id);
  const manifestEntries = [{ kind: "builtin", declaration: id, variant: "default", builtin_path: `builtins::${id}` }, ...(options.legacyGroup ? [{ kind: "gpu_spec", declaration: "FIXTURE_GPU_SPEC", variant: null, builtin_path: `builtins::${id}` }] : [])].sort((left, right) => `${left.kind}\0${left.declaration}`.localeCompare(`${right.kind}\0${right.declaration}`, "en", { sensitivity: "variant" }));
  const registrationManifest = {
    schema_version: 1,
    digest: contentDigest(Buffer.from(JSON.stringify(manifestEntries))).slice("sha256:".length),
    counts: { builtin: 1, constant: 0, gpu_spec: options.legacyGroup ? 1 : 0, fusion_spec: 0 },
    entries: manifestEntries,
  };
  const snapshot = {
    build: { architecture: "aarch64", operating_system: "macos", family: "unix", pointer_width: 64, endianness: "little", crate_feature_inventory: { crate_name: "runmat-runtime", schema_version: 1, known_features: ["blas-lapack", "blas-only", "gui", "interaction-test-hooks", "occt-native", "occt-wasm-host", "plot-core", "plot-web", "test-classes", "wgpu"], enabled_features: [] } },
    declared: { namespace_scope: "function_callables_and_constants_are_reported_separately", catalog_schema_version: 6, catalog_fingerprint: "a".repeat(64), catalog_entries: [catalog], catalog_aliases: [], catalog_provenance: [{ identity: { builtin: { name: id }, variant: "default" }, provenance: { source_file: `crates/runmat-builtins/src/catalog/entries/math/basic/${id}/mod.rs`, module_path: `catalog::${id}` } }], constants: [], legacy_functions: [], legacy_documentation: [] },
    observed: { registration_manifest: registrationManifest, runtime_constants: [], runtime_bindings: [{ name: id, variant: "default", native_symbol: nativeSymbol(id, "default") }], implementation_provenance: [{ name: id, binding_variant: "default", source_file: `crates/runmat-runtime/src/builtins/math/basic/${id}.rs`, module_path: `builtins::${id}`, function: `${id}_builtin`, builtin_path: `builtins::${id}`, authority: "canonical_binding" }], gpu_specs: options.legacyGroup ? [gpuGroupFixture(id)] : [], fusion_specs: [] },
    validation: { status: "valid", errors: [], migration_readiness: options.finding ? { status: "incomplete", findings: [options.finding] } : { status: "ready", findings: [] } },
  };
  const value = contentDigest(Buffer.from(JSON.stringify(snapshot))).slice("sha256:".length);
  return { schema_version: 3, kind: "runmat-compiled-builtin-migration-inventory", authority: "derived-read-only-evidence", digest: { algorithm: "sha256", value }, snapshot };
}

function catalogEntryFixture(id) {
  return {
    identity: { name: id }, category: "math/basic",
    documentation: { authority: "Catalog", title: "Foo", slug: id, summary: "Foo", description: "Foo.", keywords: [], related: [], sections: [], examples: [], example_exemption: "Fixture", faqs: [], links: [], media: [], evidence: { implementation: [], verification: [], notes: [] }, introduced: null, status: "Stable" },
    descriptor: { signatures: [], output_mode: "Fixed", completion_policy: "Public", errors: [] },
    contract: { maturity: "Complete", inference_rule: { Identity: {} }, compatibility: "Matlab", async_behavior: "NeverSuspends", purity: "Pure", semantic_kind: "General", workspace_effect: null, environment_effect: null, effects: [], capabilities: [] },
    placement: { portability: "NativeAndWasm", accelerator: "Forbidden", residency: "Host", fusion: "Never", distributed: "Unsupported" },
    link: { reachability: "Always", policy: "PortableRuntime", execution_stack: "Any", artifact_dependencies: [] },
    bindings: [{ variant: "default", availability: "Required" }], extensions: [], integer_capabilities: [], integer_audit: null, suppress_auto_output: false,
  };
}

function nativeSymbol(name, variant) { return `runmat_builtin_binding_v1_${Buffer.from(name).toString("hex")}_${Buffer.from(variant).toString("hex")}`; }

function gpuGroupFixture(id) {
  return { key: "data.*", declaration: "FIXTURE_GPU_SPEC", builtin_path: `builtins::${id}`, source_file: `crates/runmat-runtime/src/builtins/math/basic/${id}.rs`, module_path: `builtins::${id}`, owner: { kind: "legacy_group", raw: "data.*", affected_identities: [{ name: id }] }, operation: "custom:data", supported_precisions: [], broadcast: "none", provider_hooks: [], constant_strategy: "inline_literal", residency: "inherit_inputs", nan_mode: "include", two_pass_threshold: null, workgroup_size: null, accepts_nan_mode: false, notes: "fixture" };
}

export function fixtureGatePlans(inventory, additionalTargets = []) {
  const source = (sourcePath) => inventory.source.files.find((entry) => entry.path === sourcePath).content_digest;
  const targets = [inventory.compiled_inventory.build, ...additionalTargets]
    .map(({ operating_system, architecture }) => ({ operating_system, architecture }))
    .sort((left, right) => `${left.operating_system}\0${left.architecture}`.localeCompare(`${right.operating_system}\0${right.architecture}`));
  const approvedToolchains = (roles) => targets.map((target) => ({
    ...target,
    tools: [...roles].sort().map((role) => {
      const executable = role === "node" ? fs.realpathSync(process.execPath)
        : role === "git" ? executableOnPath("git") : rustToolPath(role);
      return { role, content_digest: contentDigest(fs.readFileSync(executable)) };
    }),
  }));
  const script = { kind: "repository_script", path: "scripts/development/check-architecture-boundaries.mjs", content_digest: source("scripts/development/check-architecture-boundaries.mjs"), approved_toolchains: approvedToolchains(["git", "node"]) };
  const documentation = { kind: "repository_script", path: "scripts/development/builtin-migration/documentation-export-cli.mjs", content_digest: source("scripts/development/builtin-migration/documentation-export-cli.mjs"), approved_toolchains: approvedToolchains(["cargo", "git", "node", "rustc"]) };
  const generated = { kind: "repository_script", path: "scripts/development/verify-builtin-generated-products.mjs", content_digest: source("scripts/development/verify-builtin-generated-products.mjs"), approved_toolchains: approvedToolchains(["cargo", "node", "rustc"]) };
  const cargo = { kind: "cargo_binary", package: "runmat-runtime", binary: "export_builtin_migration_inventory", manifest_path: "Cargo.toml", manifest_digest: source("Cargo.toml"), approved_toolchains: approvedToolchains(["cargo", "rustc"]) };
  return [
    { gate: "architecture", program: script, arguments: [], working_directory: "repository", parser: "exit_status", expected_artifact_roles: [] },
    { gate: "catalog-contract", program: cargo, arguments: [], working_directory: "repository", parser: "compiled_inventory", expected_artifact_roles: ["compiled-inventory"] },
    { gate: "deterministic-products", program: generated, arguments: ["--product", "wasm-registry"], working_directory: "repository", parser: "generated_products", expected_artifact_roles: ["generated-products"] },
    { gate: "documentation-cutover", program: documentation, arguments: [], working_directory: "repository", parser: "documentation_cutover", expected_artifact_roles: ["documentation-reconciliation"] },
    { gate: "focused-tests", program: script, arguments: [], working_directory: "repository", parser: "exit_status", expected_artifact_roles: [] },
    { gate: "format-diff", program: script, arguments: [], working_directory: "repository", parser: "exit_status", expected_artifact_roles: [] },
    { gate: "inventory-delta", program: script, arguments: [], working_directory: "repository", parser: "inventory_delta", expected_artifact_roles: ["inventory-delta"] },
    { gate: "native-link", program: cargo, arguments: [], working_directory: "repository", parser: "compiled_inventory", expected_artifact_roles: ["compiled-inventory"] },
    { gate: "runtime-binding", program: cargo, arguments: [], working_directory: "repository", parser: "compiled_inventory", expected_artifact_roles: ["compiled-inventory"] },
    { gate: "strict-clippy", program: script, arguments: [], working_directory: "repository", parser: "exit_status", expected_artifact_roles: [] },
  ];
}

export function gate(fixture, name, artifactId = `gate-${name}`, subject = fixture.inventory) {
  const namedProducer = producer(name);
  const plan = fixture.control.bundles.get(fixture.bundleId).gate_plans.get(name);
  const repository = fs.realpathSync(fixture.repository);
  const reviewed = gatePlanEvidence(plan, subject.compiled_inventory.build, repository);
  const sourceDigest = reviewed.source_digest;
  const environment = {
    CARGO_TARGET_DIR: "/private/tmp/runmat-integration-tmp/cargo-target",
    TMPDIR: "/private/tmp/runmat-integration-tmp/tmp",
    TMP: "/private/tmp/runmat-integration-tmp/tmp",
    TEMP: "/private/tmp/runmat-integration-tmp/tmp",
  };
  const tools = reviewed.tools.map((entry) => ({
    role: entry.role,
    path: entry.role === "node" ? fs.realpathSync(process.execPath)
      : entry.role === "git" ? executableOnPath("git") : rustToolPath(entry.role),
  }));
  const executable = tools.find((entry) => entry.role === reviewed.primary_tool).path;
  const byRole = new Map(tools.map((entry) => [entry.role, entry.path]));
  const toolEnvironment = {
    path_prepend: byRole.has("cargo") ? path.dirname(byRole.get("cargo")) : null,
    rustc: byRole.get("rustc") ?? null,
    rustdoc: byRole.get("rustdoc") ?? null,
    rustfmt: byRole.get("rustfmt") ?? null,
    removed: [
      "CARGO_BUILD_RUSTC", "CARGO_BUILD_RUSTC_WRAPPER", "CARGO_BUILD_TARGET_DIR",
      "CARGO_ENCODED_RUSTFLAGS", "CARGO_ENCODED_RUSTDOCFLAGS", "RUSTC", "RUSTC_WORKSPACE_WRAPPER",
      "RUSTC_WRAPPER", "RUSTDOC", "RUSTDOCFLAGS", "RUSTFLAGS", "RUSTFMT",
    ],
  };
  const invocation = { executable, arguments: reviewed.arguments, cwd: repository, environment, tools, tool_environment: toolEnvironment };
  const processEvidence = { exit_code: 0, signal: null, stdout_digest: `sha256:${"c".repeat(64)}`, stderr_digest: `sha256:${"d".repeat(64)}` };
  let documentationArtifact = null;
  if (name === "documentation-cutover") {
    const catalogExport = { schema_version: 1, inventory: { documents: 0, catalog_identities: 0, legacy_sidecars: 0, missing_catalog_documentation: [] }, builtins: [] };
    documentationArtifact = buildDocumentationCutoverArtifact({
      catalog_export: catalogExport,
      catalog_export_bytes: JSON.stringify(catalogExport),
      source_dispositions: [],
      expected_sources: { [fixture.id]: [] },
      provenance: {
        source_revision: subject.source.revision,
        source_digest: subject.source.digest,
        compiled_inventory_digest: subject.compiled_inventory.digest,
        control_manifest_digest: fixture.control.digest,
        bundle_id: fixture.bundleId,
        identities: [fixture.id],
      },
    });
    processEvidence.stdout_digest = documentationArtifact.catalog_export.evidence_digest;
  }
  const artifacts = plan.expected_artifact_roles.map((role) => {
    const artifactPath = path.join(repository, "..", `${artifactId}-${role}.json`);
    const bytes = Buffer.from(documentationArtifact && role === "documentation-reconciliation"
      ? `${JSON.stringify(documentationArtifact)}\n`
      : `{\"role\":\"${role}\"}\n`);
    fs.writeFileSync(artifactPath, bytes);
    return { role, path: artifactPath, byte_length: bytes.length, content_digest: contentDigest(bytes) };
  });
  return {
    schema_version: 7, kind: "runmat-builtin-migration-gate-result", authority: "machine-verification-only",
    producer: namedProducer, producer_evidence: { schema_version: 3, kind: `${namedProducer}-evidence`, contract: { reviewed_source_revision: fixture.inventory.source.revision, primary_tool: reviewed.primary_tool, tools: reviewed.tools, producer_source_digest: sourceDigest }, invocation, process: processEvidence, captured_process_digest: evidenceDigest(processEvidence) }, artifact_id: artifactId, produced_at: "2026-09-11T00:00:30.000Z",
    execution_target: { operating_system: subject.compiled_inventory.build.operating_system, architecture: subject.compiled_inventory.build.architecture },
    source_revision: subject.source.revision,
    source_digest: subject.source.digest, control_baseline_inventory_digest: fixture.inventory.digest,
    lease_base_inventory_digest: fixture.lease.value.lease_base_inventory.inventory_digest,
    subject_inventory_digest: subject.digest,
    control_manifest_digest: fixture.control.digest, bundle_id: fixture.bundleId,
    lease_id: fixture.lease.value.lease_id, lease_digest: fixture.lease.value.digest,
    identities: [fixture.id],
    gate: name, result: "pass", checks: documentationArtifact ? documentationCutoverChecks(documentationArtifact) : [{ id: `${name}:${fixture.id}`, result: "pass", evidence_digest: `sha256:${"a".repeat(64)}` }], artifacts,
    storage_admission: { profile_id: "fixture-host", execution_host: os.hostname(), observed_at: "2026-09-11T00:00:00.000Z", path_bindings: {
      repository: { role: "repository", path: repository, filesystem_id: "posix-dev:1" },
      cargo_target: { role: "cargo-target", path: environment.CARGO_TARGET_DIR, filesystem_id: "posix-dev:2" },
      temporary: { role: "temporary", path: environment.TMPDIR, filesystem_id: "posix-dev:2" },
      artifacts: artifacts.map((entry) => ({ role: entry.role, path: entry.path, filesystem_id: "posix-dev:2" })),
    }, volumes: [
      { role: "source-worktree", evidence_path: "/System/Volumes/Data", filesystem_id: "posix-dev:1", available_bytes: 10, minimum_free_bytes: 1, pause_below_bytes: 2, status: "admitted" },
      { role: "target-temp", evidence_path: "/private/tmp/runmat-integration-tmp", filesystem_id: "posix-dev:2", available_bytes: 10, minimum_free_bytes: 1, pause_below_bytes: 2, status: "admitted" },
    ] },
  };
}

function rustToolPath(role) {
  const sysroot = execFileSync("rustc", ["--print", "sysroot"], { encoding: "utf8" }).trim();
  const executable = process.platform === "win32" ? `${role}.exe` : role;
  return fs.realpathSync(path.join(sysroot, "bin", executable));
}

function executableOnPath(name) {
  const executable = execFileSync(process.platform === "win32" ? "where" : "which", [name], { encoding: "utf8" }).trim().split(/\r?\n/, 1)[0];
  return fs.realpathSync(executable);
}

export function digestReference(pathname, artifactId, value) { return { path: pathname, artifact_id: artifactId, digest: evidenceDigest(value) }; }

export function fixtureTargetPolicy(migrationExecutionTargets) {
  return {
    schema_version: 1,
    kind: "runmat-builtin-migration-target-policy",
    migration_execution_targets: migrationExecutionTargets,
    terminal_qualification: {
      phase: "R31",
      lanes: R31_REQUIRED_LANES.map((lane) => ({
        product_id: lane.product_id,
        gate_id: lane.gate_id,
        purpose: lane.purpose,
        targets: [
          qualificationTarget("linux", "aarch64", 1),
          qualificationTarget("linux", "x86_64", 1),
          qualificationTarget("macos", "aarch64", 1),
          qualificationTarget("macos", "x86_64", 1),
          qualificationTarget("windows", "x86_64", 2),
        ],
      })),
    },
  };
}

export function fixtureIntegrationProducts(inventory, composition = false) {
  const source = new Map(
    inventory.source.files.map((entry) => [entry.path, entry.content_digest]),
  );
  return {
    ...(composition ? { "runtime-math-parent": {
      path: "crates/runmat-runtime/src/builtins/math/mod.rs",
      producer: "integration",
      generator: {
        path: "scripts/regenerate-wasm-registry.mjs",
        baseline_digest: source.get("scripts/regenerate-wasm-registry.mjs"),
      },
      baseline_digest: source.get("crates/runmat-runtime/src/builtins/math/mod.rs"),
      verification: {
        kind: "rust_module_composition", crate_role: "runtime",
        module_path: "crate::builtins::math",
      },
    } } : {}),
    "wasm-registry": {
      path: "crates/runmat-runtime/src/builtins/generated_wasm_registry.rs",
      producer: "integration",
      generator: {
        path: "scripts/regenerate-wasm-registry.mjs",
        baseline_digest: source.get("scripts/regenerate-wasm-registry.mjs"),
      },
      baseline_digest: source.get(
        "crates/runmat-runtime/src/builtins/generated_wasm_registry.rs",
      ),
      verification: { kind: "native_wasm_registration_manifest" },
    },
  };
}

function qualificationTarget(operatingSystem, architecture, executionOrder) {
  return { operating_system: operatingSystem, architecture, applicability: "required", execution_order: executionOrder, reason: null, evidence: [] };
}

function producer(gate) { return ({ "catalog-contract": "runmat-builtins-catalog-validator", "runtime-binding": "runmat-runtime-binding-validator", "documentation-cutover": "builtin-documentation-cutover-audit", "native-link": "runmat-native-link-validator", architecture: "runmat-architecture-boundary-validator", "focused-tests": "runmat-test-result-adapter", "strict-clippy": "runmat-clippy-result-adapter", "format-diff": "runmat-format-diff-validator", "deterministic-products": "runmat-generated-product-validator", "inventory-delta": "builtin-inventory-delta-validator" })[gate]; }
function write(root, relative, contents) { const target = path.join(root, relative); fs.mkdirSync(path.dirname(target), { recursive: true }); fs.writeFileSync(target, contents); }
