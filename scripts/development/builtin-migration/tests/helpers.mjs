import fs from "node:fs";
import { execFileSync } from "node:child_process";
import os from "node:os";
import path from "node:path";
import { MATURITY_GATES, parseControlManifest } from "../control.mjs";
import { contentDigest, evidenceDigest } from "../evidence.mjs";
import { buildInventory } from "../inventory.mjs";
import { issueLease, parseLease } from "../lease.mjs";

export const REVISION = `git:${"1".repeat(40)}`;
const fixtureRoots = new Set();

export function repositoryFixture({ sidecar = false, identity = "foo" } = {}) {
  const fixtureRoot = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-migration-factory-"));
  fixtureRoots.add(fixtureRoot);
  const root = path.join(fixtureRoot, "repository");
  fs.mkdirSync(root);
  write(root, "Cargo.toml", "[workspace]\nresolver = \"2\"\n");
  write(root, "scripts/development/check-architecture-boundaries.mjs", "process.exit(0);\n");
  write(root, "scripts/regenerate-wasm-registry.mjs", "// fixture generator identity\n");
  write(root, "scripts/development/verify-builtin-generated-products.mjs", `
import fs from "node:fs"; import crypto from "node:crypto";
const digest = (bytes) => "sha256:" + crypto.createHash("sha256").update(bytes).digest("hex");
const observed = fs.readFileSync("crates/runmat-runtime/src/builtins/generated_wasm_registry.rs");
const generator = fs.readFileSync("scripts/regenerate-wasm-registry.mjs");
const value = { schema_version: 1, kind: "runmat-builtin-generated-products-proof", authority: "machine-derived-integration-evidence", products: [{ product_id: "wasm-registry", path: "crates/runmat-runtime/src/builtins/generated_wasm_registry.rs", generator: { path: "scripts/regenerate-wasm-registry.mjs", content_digest: digest(generator) }, checked_in: { byte_length: observed.length, content_digest: digest(observed) }, first: { byte_length: observed.length, content_digest: digest(observed) }, second: { byte_length: observed.length, content_digest: digest(observed) }, deterministic: true, synchronized: true }], result: "pass" };
process.stdout.write(JSON.stringify(value) + "\\n");
`);
  write(root, `crates/runmat-runtime/src/builtins/math/basic/${identity}.rs`, `
#[runtime_builtin(name = "${identity}", category = "math/basic", builtin_path = "crate::builtins::math::basic::${identity}")]
fn ${identity.replaceAll(".", "_")}_builtin() {}
#[cfg(test)] mod tests {}
`);
  write(root, "crates/runmat-runtime/src/builtins/generated_wasm_registry.rs", `crate::builtins::math::basic::${identity}::__runmat_wasm_register_builtin_${identity}_builtin();`);
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
  for (const fixtureRoot of fixtureRoots) fs.rmSync(fixtureRoot, { recursive: true, force: true });
  fixtureRoots.clear();
}

export function controlledFixture(options = {}) {
  const repository = repositoryFixture(options);
  const id = options.identity ?? "foo";
  const compiledInventory = compiledInventoryFixture(id, options);
  const seedInventory = buildInventory(repository, undefined, { revision: REVISION, compiledInventory });
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
  const inventory = buildInventory(repository, dispositions, { revision: REVISION, compiledInventory });
  const bundleId = "math-basic-foo";
  const maturity = Object.fromEntries(MATURITY_GATES.map((gate) => [gate, gate === "identity" || gate === "disposition" || gate === "catalog-contract" || gate === "runtime-binding" || gate === "documentation"
    ? { applicability: "required", reason: null, evidence: [] }
    : { applicability: "not-applicable", reason: "Reviewed as outside this fixture's behavior", evidence: ["fixture review"] }]));
  const controlValue = {
    schema_version: 1, kind: "runmat-builtin-migration-control-manifest", authority: "reviewed-development-control", program: "RM-1064/C00-C07", control_draft_digest: `sha256:${"f".repeat(64)}`,
    baseline: { revision: REVISION, source_digest: inventory.source.digest, inventory_digest: inventory.digest, dispositions_digest: inventory.dispositions_digest, migration_findings_digest: inventory.migration_findings_digest, compiled_target: { operating_system: inventory.compiled_inventory.build.operating_system, architecture: inventory.compiled_inventory.build.architecture } },
    cohorts: ["prerequisite", "A", "B", "C", "D", "E", "F", "G"].map((semantic, order) => ({ id: `C0${order}`, semantic, order })),
    bundles: { [bundleId]: { id: bundleId, identities: [id], domain: "math", family: "basic", atomic_reason: "One runtime and catalog contract", prerequisites: [], authored_write_set: [{ kind: "tree", path: "crates/runmat-builtins/src/catalog/entries/math/basic/foo" }, { kind: "file", path: "crates/runmat-runtime/src/builtins/math/basic/foo.rs" }], integration_outputs: [{ product_id: "wasm-registry", path: "crates/runmat-runtime/src/builtins/generated_wasm_registry.rs", producer: "integration" }], gate_plans: fixtureGatePlans(inventory), owner_role: "builtin-migrator", complexity: { class: "low", weight: 1, basis: ["single identity"] }, review: { status: "reviewed", evidence: ["fixture review"] } } },
    identities: { [id]: { identity: id, public_spelling: id, disposition: { kind: "canonical", target: id }, cohort: "C01", bundle_id: bundleId, domain: "math", family: "basic", runtime_owner: `crates/runmat-runtime/src/builtins/math/basic/${id}.rs`, shared_dependencies: [], complexity: { class: "low", weight: 1, basis: ["single identity"] }, maturity, expected_authorities: { catalog_package: `crates/runmat-builtins/src/catalog/entries/math/basic/${id}/mod.rs`, catalog_entry_count: 1, documentation: "catalog", runtime_bindings: [{ path: `crates/runmat-runtime/src/builtins/math/basic/${id}.rs`, function: `${id}_builtin`, variant: "default" }], native_link: "not-applicable", wasm_registry: "not-applicable" }, expected_removals: [], baseline_evidence: [], owner: "fixture", review: { status: "reviewed", evidence: ["fixture review"] } } },
    migration_findings: { schema_version: 1, kind: "runmat-builtin-migration-finding-dispositions", rows: inventory.migration_findings.map((finding) => ({ finding_digest: evidenceDigest(finding), ...finding, disposition: "bundle-work", bundle_id: bundleId, reason: "Fixture migration work", evidence: ["fixture review"] })), review: { status: "reviewed", evidence: ["fixture review"] } },
    exception_manifest: { entries: [], review: { status: "reviewed", evidence: ["fixture review"] } },
    storage_policy: { volume_roles: {
      source_worktree: { role: "source-worktree", mount_path: "/System/Volumes/Data", filesystem_id: "posix-dev:1", minimum_free_bytes: 1, pause_below_bytes: 2, maximum_observation_age_seconds: 60 },
      target_temp: { role: "target-temp", mount_path: "/private/tmp/runmat-integration-tmp", filesystem_id: "posix-dev:2", minimum_free_bytes: 1, pause_below_bytes: 2, maximum_observation_age_seconds: 60 },
    }, targets_must_be_disjoint: true, occt_default: "disabled-unless-affected" },
    review: { status: "reviewed", evidence: ["fixture review"] },
  };
  const control = parseControlManifest(controlValue, inventory);
  const leaseRequest = { schema_version: 1, kind: "runmat-builtin-migration-lease-request", authority: "reviewed-development-request", control_manifest_digest: control.digest, bundle_id: bundleId, lease_id: "lease-foo", owner: "fixture", issued_at: "2026-01-01T00:00:00.000Z", expires_at: "2027-01-01T00:00:00.000Z", review: { status: "reviewed", evidence: ["fixture review"] } };
  const leaseValue = issueLease(leaseRequest, control);
  const lease = parseLease(leaseValue, control);
  return { repository, inventory, compiledInventory, controlValue, control, leaseValue, lease, id, bundleId };
}

export function compiledInventoryFixture(id = "foo", options = {}) {
  const catalog = catalogEntryFixture(id);
  const snapshot = {
    build: { architecture: "aarch64", operating_system: "macos", family: "unix", pointer_width: 64, endianness: "little", crate_feature_inventory: { crate_name: "runmat-runtime", schema_version: 1, known_features: ["blas-lapack", "blas-only", "gui", "interaction-test-hooks", "occt-native", "occt-wasm-host", "plot-core", "plot-web", "test-classes", "wgpu"], enabled_features: [] } },
    declared: { namespace_scope: "function_callables_and_constants_are_reported_separately", catalog_schema_version: 5, catalog_fingerprint: "a".repeat(64), catalog_entries: [catalog], catalog_provenance: [{ identity: { builtin: { name: id }, variant: "default" }, provenance: { source_file: `crates/runmat-builtins/src/catalog/entries/math/basic/${id}/mod.rs`, module_path: `catalog::${id}` } }], constants: [], legacy_functions: [], legacy_documentation: [] },
    observed: { runtime_constants: [], runtime_bindings: [{ name: id, variant: "default", native_symbol: nativeSymbol(id, "default") }], implementation_provenance: [{ name: id, binding_variant: "default", source_file: `crates/runmat-runtime/src/builtins/math/basic/${id}.rs`, module_path: `builtins::${id}`, function: `${id}_builtin`, builtin_path: `builtins::${id}`, authority: "canonical_binding" }], gpu_specs: options.legacyGroup ? [gpuGroupFixture()] : [], fusion_specs: [] },
    validation: { status: "valid", errors: [], migration_readiness: options.finding ? { status: "incomplete", findings: [options.finding] } : { status: "ready", findings: [] } },
  };
  const value = contentDigest(Buffer.from(JSON.stringify(snapshot))).slice("sha256:".length);
  return { schema_version: 1, kind: "runmat-compiled-builtin-migration-inventory", authority: "derived-read-only-evidence", digest: { algorithm: "sha256", value }, snapshot };
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

function gpuGroupFixture() {
  return { key: "data.*", owner: { kind: "legacy_group", raw: "data.*" }, operation: "custom:data", supported_precisions: [], broadcast: "none", provider_hooks: [], constant_strategy: "inline_literal", residency: "inherit_inputs", nan_mode: "include", two_pass_threshold: null, workgroup_size: null, accepts_nan_mode: false, notes: "fixture" };
}

function fixtureGatePlans(inventory) {
  const source = (sourcePath) => inventory.source.files.find((entry) => entry.path === sourcePath).content_digest;
  const approved = (executable) => [{ operating_system: inventory.compiled_inventory.build.operating_system, architecture: inventory.compiled_inventory.build.architecture, content_digest: contentDigest(fs.readFileSync(fs.realpathSync(executable))) }];
  const script = { kind: "repository_script", path: "scripts/development/check-architecture-boundaries.mjs", content_digest: source("scripts/development/check-architecture-boundaries.mjs"), approved_executables: approved(process.execPath) };
  const generated = { kind: "repository_script", path: "scripts/development/verify-builtin-generated-products.mjs", content_digest: source("scripts/development/verify-builtin-generated-products.mjs"), approved_executables: approved(process.execPath) };
  const cargo = { kind: "cargo_binary", package: "runmat-runtime", binary: "export_builtin_migration_inventory", manifest_path: "Cargo.toml", manifest_digest: source("Cargo.toml"), approved_executables: approved(execFileSync("/usr/bin/which", ["cargo"], { encoding: "utf8" }).trim()) };
  return [
    { gate: "architecture", program: script, arguments: [], working_directory: "repository", parser: "exit_status", expected_artifact_roles: [] },
    { gate: "catalog-contract", program: cargo, arguments: [], working_directory: "repository", parser: "compiled_inventory", expected_artifact_roles: ["compiled-inventory"] },
    { gate: "deterministic-products", program: generated, arguments: [], working_directory: "repository", parser: "generated_products", expected_artifact_roles: ["generated-products"] },
    { gate: "documentation-cutover", program: script, arguments: [], working_directory: "repository", parser: "documentation_cutover", expected_artifact_roles: ["documentation-reconciliation"] },
    { gate: "inventory-delta", program: script, arguments: [], working_directory: "repository", parser: "inventory_delta", expected_artifact_roles: ["inventory-delta"] },
    { gate: "runtime-binding", program: cargo, arguments: [], working_directory: "repository", parser: "compiled_inventory", expected_artifact_roles: ["compiled-inventory"] },
  ];
}

export function gate(fixture, name, artifactId = `gate-${name}`, subject = fixture.inventory) {
  const namedProducer = producer(name);
  const plan = fixture.control.bundles.get(fixture.bundleId).gate_plans.get(name);
  const sourceDigest = plan.program.kind === "repository_script" ? plan.program.content_digest : plan.program.manifest_digest;
  const executableDigest = plan.program.approved_executables[0].content_digest;
  const invocation = plan.program.kind === "repository_script"
    ? { executable: process.execPath, arguments: [path.join(fixture.repository, plan.program.path), ...plan.arguments], cwd: fixture.repository }
    : { executable: execFileSync("/usr/bin/which", ["cargo"], { encoding: "utf8" }).trim(), arguments: ["run", "--quiet", "-p", plan.program.package, "--bin", plan.program.binary, "--", ...plan.arguments], cwd: fixture.repository };
  const processEvidence = { exit_code: 0, signal: null, stdout_digest: `sha256:${"c".repeat(64)}`, stderr_digest: `sha256:${"d".repeat(64)}` };
  const artifacts = plan.expected_artifact_roles.map((role) => {
    const artifactPath = path.join(fixture.repository, "..", `${artifactId}-${role}.json`);
    const bytes = Buffer.from(`{\"role\":\"${role}\"}\n`);
    fs.writeFileSync(artifactPath, bytes);
    return { role, path: artifactPath, byte_length: bytes.length, content_digest: contentDigest(bytes) };
  });
  return {
    schema_version: 2, kind: "runmat-builtin-migration-gate-result", authority: "machine-verification-only",
    producer: namedProducer, producer_evidence: { schema_version: 1, kind: `${namedProducer}-evidence`, contract: { reviewed_source_revision: fixture.inventory.source.revision, executable_digest: executableDigest, producer_source_digest: sourceDigest }, invocation, process: processEvidence, captured_process_digest: evidenceDigest(processEvidence) }, artifact_id: artifactId, produced_at: "2026-09-11T00:00:30.000Z", source_revision: subject.source.revision,
    source_digest: subject.source.digest, baseline_inventory_digest: fixture.inventory.digest, subject_inventory_digest: subject.digest,
    control_manifest_digest: fixture.control.digest, bundle_id: fixture.bundleId, identities: [fixture.id],
    gate: name, result: "pass", checks: [{ id: `${name}:${fixture.id}`, result: "pass", evidence_digest: `sha256:${"a".repeat(64)}` }], artifacts,
    storage_admission: { observed_at: "2026-09-11T00:00:00.000Z", volumes: [
      { role: "source-worktree", evidence_path: "/System/Volumes/Data", filesystem_id: "posix-dev:1", available_bytes: 10, minimum_free_bytes: 1, pause_below_bytes: 2, status: "admitted" },
      { role: "target-temp", evidence_path: "/private/tmp/runmat-integration-tmp", filesystem_id: "posix-dev:2", available_bytes: 10, minimum_free_bytes: 1, pause_below_bytes: 2, status: "admitted" },
    ] },
  };
}

export function digestReference(pathname, artifactId, value) { return { path: pathname, artifact_id: artifactId, digest: evidenceDigest(value) }; }

function producer(gate) { return ({ "catalog-contract": "runmat-builtins-catalog-validator", "runtime-binding": "runmat-runtime-binding-validator", "documentation-cutover": "builtin-documentation-cutover-audit", architecture: "runmat-architecture-boundary-validator", "deterministic-products": "runmat-generated-product-validator", "inventory-delta": "builtin-inventory-delta-validator" })[gate]; }
function write(root, relative, contents) { const target = path.join(root, relative); fs.mkdirSync(path.dirname(target), { recursive: true }); fs.writeFileSync(target, contents); }
