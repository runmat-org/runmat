import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { bundleBaselineEvidence } from "../baseline-evidence.mjs";
import { MATURITY_GATES, validateControlManifestStructure } from "../control.mjs";
import { validateControlReviewChain } from "../control-authoring/authority.mjs";
import { composeControlCandidate, controlCandidateInputDigests } from "../control-authoring/compose.mjs";
import { loadControlReviewSet } from "../control-authoring/review-set.mjs";
import { buildControlOverlayScaffold } from "../control-authoring/scaffold.mjs";
import { identityControlAuthorityTemplate } from "../control-authoring/identity-template.mjs";
import { contentDigest, evidenceDigest } from "../evidence.mjs";
import { composeTopologyCandidate } from "../topology/compose.mjs";
import { candidateInputDigests, freezeReviewedTopology, parseReviewedTopology, reviewedTopologyView } from "../topology/freeze.mjs";
import { fullTopologyChainFixture } from "../topology/tests/full-chain-fixture.mjs";
import { fixtureTargetPolicy } from "./helpers.mjs";

export function writeFullControlWorkflow(directory, revision, sourceContentDigest = undefined, sourceMode = undefined) {
  const input = fullTopologyChainFixture({ revision, sourceContentDigest, sourceMode });
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
  const policies = fullChainControl(input.baselineInventory, topology, scaffold);
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
  const acceptedSealSet = { schema_version: 1, kind: "runmat-builtin-migration-accepted-seal-set", authority: "derived-from-validated-seals", control_manifest_digest: control.digest, seals: [] };
  const barrierSealSet = { schema_version: 1, kind: "runmat-builtin-migration-barrier-seal-set", authority: "derived-from-validated-seals", control_manifest_digest: control.digest, bundle_id: bundleId, seals: [] };
  const queueStatePayload = { schema_version: 3, kind: "runmat-builtin-migration-queue-state", authority: "reviewed-monotonic-scheduling-input", control_manifest_digest: control.digest, predecessor: null, bundles: {}, seals: [] };
  const queueState = { ...queueStatePayload, digest: evidenceDigest(queueStatePayload) };
  const queueCheckpointPayload = { schema_version: 1, kind: "runmat-builtin-migration-queue-checkpoint", authority: "reviewer-authored-current-queue-checkpoint", control_manifest_digest: control.digest, queue_state_digest: queueState.digest, predecessor_checkpoint_digest: null, head_seal: null, source_revision: input.baselineInventory.source.revision, source_digest: input.baselineInventory.source.digest, accepted_seal_set_digest: evidenceDigest(acceptedSealSet), review: reviewed("full-chain current queue checkpoint") };
  const queueCheckpoint = { ...queueCheckpointPayload, digest: evidenceDigest(queueCheckpointPayload) };
  const leaseRequest = {
    schema_version: 4,
    kind: "runmat-builtin-migration-lease-request",
    authority: "reviewed-development-request",
    control_manifest_digest: control.digest,
    bundle_id: bundleId,
    lease_id: "full-chain-cli-lease",
    owner: "fixture-migrator",
    base_revision: revision,
    lease_base_inventory: { source_revision: input.baselineInventory.source.revision, source_digest: input.baselineInventory.source.digest, inventory_digest: input.baselineInventory.digest, compiled_inventory_digest: input.baselineInventory.compiled_inventory.digest },
    queue_checkpoint_digest: queueCheckpoint.digest,
    accepted_seals: [],
    accepted_seal_set_digest: evidenceDigest(acceptedSealSet),
    barrier_seals: [],
    barrier_seal_set_digest: evidenceDigest(barrierSealSet),
    issued_at: "2020-01-01T00:00:00.000Z",
    expires_at: "2099-01-01T00:00:00.000Z",
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
    queueState,
    queueCheckpoint,
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

export function cleanFactoryCliRepository() {
  const repository = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-factory-cli-repository-"));
  fs.cpSync(path.resolve("scripts"), path.join(repository, "scripts"), { recursive: true });
  execFileSync("git", ["init", "--quiet"], { cwd: repository });
  execFileSync("git", ["add", "scripts"], { cwd: repository });
  execFileSync("git", [
    "-c", "user.name=RunMat Test",
    "-c", "user.email=test@runmat.invalid",
    "-c", "commit.gpgsign=false",
    "commit", "--quiet", "-m", "factory CLI fixture",
  ], { cwd: repository });
  return repository;
}

function fullChainControl(inventory, topology, scaffold) {
  const review = reviewed("full-chain control review");
  const sourceDigest = inventory.source.files[0].content_digest;
  const approvedToolchains = [{
    operating_system: inventory.compiled_inventory.build.operating_system,
    architecture: inventory.compiled_inventory.build.architecture,
    tools: [{ role: "node", content_digest: `sha256:${"7".repeat(64)}` }],
  }];
  const integrationToolchains = approvedToolchains.map((toolchain) => ({
    operating_system: toolchain.operating_system,
    architecture: toolchain.architecture,
    tools: ["cargo", "node", "rustc"].map((role) => ({
      role,
      content_digest: `sha256:${"7".repeat(64)}`,
    })),
  }));
  const script = {
    kind: "repository_script",
    path: "scripts/development/check-architecture-boundaries.mjs",
    content_digest: sourceDigest,
    approved_toolchains: approvedToolchains,
  };
  const gatePlans = [
    { gate: "architecture", program: script, arguments: [], working_directory: "repository", parser: "exit_status", expected_artifact_roles: [] },
    fixtureGatePlan("catalog-contract", "compiled_inventory", ["compiled-inventory"], sourceDigest, approvedToolchains),
    { gate: "deterministic-products", program: { ...script, approved_toolchains: integrationToolchains }, arguments: ["--no-products"], working_directory: "repository", parser: "generated_products", expected_artifact_roles: ["generated-products"] },
    { gate: "focused-tests", program: script, arguments: [], working_directory: "repository", parser: "exit_status", expected_artifact_roles: [] },
    { gate: "format-diff", program: script, arguments: [], working_directory: "repository", parser: "exit_status", expected_artifact_roles: [] },
    { gate: "inventory-delta", program: script, arguments: [], working_directory: "repository", parser: "inventory_delta", expected_artifact_roles: ["inventory-delta"] },
    { gate: "strict-clippy", program: script, arguments: [], working_directory: "repository", parser: "exit_status", expected_artifact_roles: [] },
  ];
  const inventoryByIdentity = new Map(inventory.identities.map((entry) => [entry.identity, entry]));
  const bundleControls = Object.fromEntries(Object.keys(topology.bundles).sort().map((bundleId) => [bundleId, {
    prerequisites: [],
    additional_authored_write_set: [...new Set(topology.bundles[bundleId].identities
      .map((identity) => inventoryByIdentity.get(identity).ownership.runtime[0]))]
      .sort()
      .map((sourcePath) => ({ kind: "file", path: sourcePath })),
    integration_product_refs: [],
    expected_removals: [],
    baseline_evidence: bundleBaselineEvidence(inventory, topology.bundles[bundleId].identities),
    gate_plans: gatePlans,
    owner_role: "fixture-migrator",
    complexity: { class: "low", weight: topology.bundles[bundleId].identities.length, basis: ["full-chain fixture review"] },
    review,
  }]));
  const identityControls = Object.fromEntries(Object.keys(topology.identities).sort().map((identity) => {
    const maturity = Object.fromEntries(MATURITY_GATES.map((gate) => [gate, gate === "identity"
      ? { applicability: "required", reason: null, evidence: [] }
      : { applicability: "not-applicable", reason: "Outside this control-workflow fixture", evidence: ["full-chain fixture review"] }]));
    return [identity, {
      ...identityControlAuthorityTemplate(scaffold, identity),
      shared_dependencies: [],
      complexity: { class: "low", weight: 1, basis: ["full-chain fixture review"] },
      maturity,
      expected_authorities: {
        catalog_package: null,
        catalog_alias_package: null,
        catalog_constant_package: null,
        catalog_entry_count: 0,
        catalog_constant_count: 0,
        documentation: "none",
        native_link: "not-applicable",
        wasm_registry: "not-applicable",
      },
      owner: "fixture-migrator",
      review,
    }];
  }));
  const payload = {
    schema_version: 3,
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
    integration_products: {},
    migration_findings: {
      schema_version: 1,
      kind: "runmat-builtin-migration-finding-dispositions",
      rows: [],
      review,
    },
    exception_manifest: { entries: [], review },
    target_policy: fixtureTargetPolicy([{
      operating_system: inventory.compiled_inventory.build.operating_system,
      architecture: inventory.compiled_inventory.build.architecture,
    }]),
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
    schema_version: 2,
    kind: "runmat-builtin-migration-global-control-review",
    authority: "reviewer-authored-development-input",
    program: "RM-1064/C00-C07",
    bindings: {
      scaffold_digest: scaffold.digest,
      topology_digest: topology.digest,
      migration_finding_rows_digest: evidenceDigest(scaffold.migration_finding_rows),
    },
    program_profiles: programProfiles,
    integration_products: policies.integration_products,
    migration_findings: policies.migration_findings,
    exception_manifest: policies.exception_manifest,
    target_policy: policies.target_policy,
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
      authority_proposal_digest: scaffold.authority_proposals.identity_rows
        .find((row) => row.identity === identity).proposal_digest,
    }));
    const bundlePayload = {
      schema_version: 3,
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

export function copyReviewSetAsAuthoringFiles(manifestPath, target) {
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

export function reviewed(evidence) {
  return { status: "reviewed", evidence: [evidence] };
}

export function runtimeConstant(name) {
  return {
    name,
    source_file: "crates/runmat-runtime/src/builtins/constants/mod.rs",
    module_path: "runmat_runtime::builtins::constants",
    builtin_path: "builtins::constants",
  };
}

export function setRegistrationManifest(snapshot, entries) {
  entries.sort((left, right) => {
    const leftKey = `${left.kind}\0${left.declaration}\0${left.variant ?? ""}\0${left.builtin_path}`;
    const rightKey = `${right.kind}\0${right.declaration}\0${right.variant ?? ""}\0${right.builtin_path}`;
    return leftKey < rightKey ? -1 : leftKey > rightKey ? 1 : 0;
  });
  snapshot.observed.registration_manifest = {
    schema_version: 1,
    digest: contentDigest(Buffer.from(JSON.stringify(entries))).slice("sha256:".length),
    counts: Object.fromEntries(["builtin", "constant", "gpu_spec", "fusion_spec"]
      .map((kind) => [kind, entries.filter((entry) => entry.kind === kind).length])),
    entries,
  };
}

export function parseFixtureControl(fixture, value = fixture.controlValue, inventory = fixture.inventory, topology = fixture.topology) {
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

export function resealEvidence(value) {
  delete value.digest;
  value.digest = evidenceDigest(value);
  return value;
}
