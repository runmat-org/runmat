import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { assertControlBaseline, assertControlSubject } from "./control.mjs";
import { publishEvidenceBytes } from "./atomic-evidence-publication.mjs";
import { contentDigest, evidenceDigest } from "./evidence.mjs";
import { GATE_PRODUCERS, parseGateResult } from "./gate-result.mjs";
import { prepareGateProcessInput } from "./gate-input.mjs";
import { gatePlanEvidence } from "./gate-plan.mjs";
import { parseCompiledInventory } from "./compiled-inventory.mjs";
import { parseInventoryEvidence } from "./inventory.mjs";
import { assertActiveLease, assertLeaseBaseInventory } from "./lease.mjs";
import { reviewedIntegrationProducts } from "./integration-products.mjs";
import { deriveEffectiveModuleComposition } from "./module-composition/effective-state.mjs";
import { parseExampleProducer } from "./producer-adapters/example.mjs";
import { parseGeneratedProductsProducer } from "./producer-adapters/generated-products.mjs";
import { parseCompiledInventoryProducer, parseInventoryDeltaProducer } from "./producer-adapters/inventory.mjs";
import { parseDocumentationProducer } from "./producer-adapters/documentation.mjs";
import { absolutePath, exact } from "./schema.mjs";
import { prepareGateStorage } from "./storage-admission.mjs";

const REPOSITORY = fs.realpathSync(
  path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../.."),
);
// The reviewed control owns the complete command. Callers select only a bundle
// and gate; they cannot supply executable, argv, cwd, checks, or process facts.
export function runGateProducer(input, clock = Date.now) {
  exact(input, ["control", "lease", "queue_state", "queue_checkpoint", "control_baseline_inventory", "lease_base_inventory", "subject_inventory", "subject_compiled_inventory", "bundle_id", "gate", "artifact_id", "inputs"], "gate producer request");
  const controlBaseline = parseInventoryEvidence(input.control_baseline_inventory);
  const leaseBase = parseInventoryEvidence(input.lease_base_inventory);
  const subject = parseInventoryEvidence(input.subject_inventory);
  const subjectCompiled = parseCompiledInventory(input.subject_compiled_inventory);
  if (subject.compiled_inventory.digest !== subjectCompiled.digest
    || JSON.stringify(subject.compiled_inventory.build) !== JSON.stringify(subjectCompiled.snapshot.build)) {
    throw new Error("gate subject compiled inventory differs from its inventory evidence");
  }
  if (subject.source.dirty !== false) throw new Error("gate subject must be a clean committed source snapshot");
  const control = input.control;
  assertControlBaseline(control, controlBaseline);
  assertControlSubject(control, subject);
  const lease = assertActiveLease(input.lease, control, clock);
  assertLeaseBaseInventory(lease, control, leaseBase);
  const bundle = control.bundles.get(input.bundle_id);
  if (!bundle) throw new Error(`${input.bundle_id}: unknown gate bundle`);
  if (lease.bundle.id !== bundle.id) throw new Error(`${input.bundle_id}: gate lease belongs to another bundle`);
  const plan = bundle.gate_plans.get(input.gate);
  if (!plan) throw new Error(`${input.bundle_id}/${input.gate}: no reviewed gate plan is registered`);
  if (input.inputs && Object.hasOwn(input.inputs, "module_composition_projection")) {
    throw new Error("module composition projection is derived authority and cannot be supplied by a caller");
  }
  const repository = REPOSITORY;
  const command = commandFor(plan, repository, controlBaseline, subject.compiled_inventory.build);
  const tools = resolveReviewedTools(command.tools, plan.program.kind);
  const executable = tools.find((entry) => entry.role === command.primary_tool)?.path;
  if (!executable) throw new Error(`${input.gate}: reviewed primary tool is absent from the resolved toolchain`);
  const storage = prepareGateStorage(
    control.value.storage_policy,
    subject.compiled_inventory.build,
    repository,
    plannedArtifactOutputs(plan, input.inputs),
    { control_digest: control.digest, bundle_id: bundle.id, artifact_id: input.artifact_id },
  );
  const toolSelection = reviewedToolEnvironment(tools);
  const executionEnvironment = toolEnvironment(toolSelection, storage.environment);
  const integrationProducts = reviewedIntegrationProducts(
    bundle.integration_product_refs, control.integrationProducts, bundle.id,
  );
  const hasCompositionProducts = integrationProducts.some((product) =>
    product.verification.kind === "rust_module_composition");
  const moduleCompositionProjection = hasCompositionProducts
    ? deriveEffectiveModuleComposition({
      control,
      queueState: input.queue_state,
      queueCheckpoint: input.queue_checkpoint,
      lease,
      clock,
    })
    : null;
  const evidenceStorage = storage.admission.volumes.find((entry) => entry.role === "target-temp");
  if (!evidenceStorage) throw new Error(`${input.gate}: target-temp storage admission is absent`);
  const processInput = prepareGateProcessInput(plan, input.inputs, storage.admission.path_bindings.temporary.path, {
    source_revision: subject.source.revision,
    identities: bundle.identities,
    integration_products: integrationProducts,
    native_registration_manifest: subjectCompiled.snapshot.observed.registration_manifest,
    evidence_storage: evidenceStorage,
  }, moduleCompositionProjection);
  const parser = parserFor(plan.parser, input.gate, {
    input, controlBaseline, leaseBase, subject, control, bundle, tools,
    manifestEvidence: processInput?.manifest_evidence ?? null,
    evidenceRoot: processInput?.evidence_root ?? null,
    moduleCompositionProjection: processInput?.module_composition_projection ?? null,
    integrationProducts,
  });
  const processResult = spawnSync(executable, command.arguments, {
    cwd: repository,
    encoding: "utf8",
    env: executionEnvironment,
    maxBuffer: 128 * 1024 * 1024,
    ...(processInput ? { input: processInput.stdin } : {}),
  });
  if (processResult.error) throw new Error(`${input.gate}: could not execute producer: ${processResult.error.message}`);
  const stdout = processResult.stdout ?? "";
  const stderr = processResult.stderr ?? "";
  const status = processResult.status;
  if (status === null) throw new Error(`${input.gate}: producer terminated by signal ${processResult.signal ?? "unknown"}`);
  const identities = [...bundle.identities];
  const parsed = parser(stdout, stderr, status, identities);
  const checks = parsed.checks;
  const artifacts = parsed.artifacts;
  const result = status === 0 && checks.every((entry) => entry.result === "pass") ? "pass" : "fail";
  const processEvidence = {
    exit_code: status, signal: processResult.signal,
    stdout_digest: evidenceDigest(stdout), stderr_digest: evidenceDigest(stderr),
  };
  const expected = {
    source_revision: subject.source.revision, source_digest: subject.source.digest,
    control_baseline_source_revision: controlBaseline.source.revision,
    control_baseline_inventory_digest: controlBaseline.digest,
    lease_base_inventory_digest: leaseBase.digest,
    subject_inventory_digest: subject.digest, control_manifest_digest: control.digest,
    lease_id: lease.value.lease_id, lease_digest: lease.value.digest,
    queue_phase: lease.value.queue_phase,
    bundle_id: bundle.id, storage_policy: control.value.storage_policy,
    gate_plans: bundle.gate_plans, compiled_build: subject.compiled_inventory.build,
    execution_targets: control.executionTargets, repository,
    subject_compiled_inventory_digest: subject.compiled_inventory.digest,
    ...(input.gate === "documentation-cutover"
      ? { documentation_source_dispositions: input.inputs.source_dispositions }
      : {}),
  };
  return parseGateResult({
    schema_version: 8, kind: "runmat-builtin-migration-gate-result", authority: "machine-verification-only",
    producer: GATE_PRODUCERS[input.gate],
    producer_evidence: {
      schema_version: 3, kind: `${GATE_PRODUCERS[input.gate]}-evidence`,
      contract: {
        reviewed_source_revision: controlBaseline.source.revision,
        primary_tool: command.primary_tool,
        tools: command.tools,
        producer_source_digest: command.sourceDigest,
      },
      invocation: { executable, arguments: command.arguments, cwd: repository, environment: storage.environment, tools, tool_environment: toolSelection },
      process: processEvidence, captured_process_digest: evidenceDigest(processEvidence),
    },
    artifact_id: input.artifact_id, produced_at: new Date().toISOString(),
    execution_target: {
      operating_system: subject.compiled_inventory.build.operating_system,
      architecture: subject.compiled_inventory.build.architecture,
    },
    source_revision: subject.source.revision, source_digest: subject.source.digest,
    control_baseline_inventory_digest: controlBaseline.digest,
    lease_base_inventory_digest: leaseBase.digest, subject_inventory_digest: subject.digest,
    control_manifest_digest: control.digest, lease_id: lease.value.lease_id, lease_digest: lease.value.digest,
    queue_phase: lease.value.queue_phase,
    bundle_id: bundle.id, identities, gate: input.gate, result, checks, artifacts,
    storage_admission: storage.admission,
  }, expected);
}

function plannedArtifactOutputs(plan, inputs) {
  if (plan.expected_artifact_roles.length === 0) return [];
  if (plan.expected_artifact_roles.length !== 1 || typeof inputs?.artifact_output !== "string") {
    throw new Error(`${plan.gate}: reviewed artifact output is required before storage admission`);
  }
  return [{ role: plan.expected_artifact_roles[0], output: inputs.artifact_output }];
}

function commandFor(plan, repository, baselineInventory, subjectBuild) {
  const reviewed = gatePlanEvidence(plan, subjectBuild, repository);
  const sourcePath = plan.program.kind === "repository_script"
    ? plan.program.path
    : plan.program.manifest_path;
  const expectedDigest = plan.program.kind === "repository_script"
    ? plan.program.content_digest
    : plan.program.manifest_digest;
  const sourceDigest = pinnedSourceDigest(
    repository, sourcePath, expectedDigest, baselineInventory,
  );
  return { ...reviewed, sourceDigest };
}

function resolveReviewedTools(reviewedTools, programKind) {
  const needsRustToolRoot = programKind !== "repository_script"
    || reviewedTools.some((entry) => !["git", "node"].includes(entry.role));
  const rustToolRoot = needsRustToolRoot ? resolveRustToolRoot() : null;
  return reviewedTools.map((reviewed) => {
    const candidate = reviewed.role === "node"
      ? process.execPath
      : reviewed.role === "git"
        ? resolveExecutable("git")
        : path.join(rustToolRoot, platformExecutableName(reviewed.role));
    const resolved = fs.realpathSync(candidate);
    const observedDigest = contentDigest(fs.readFileSync(resolved));
    if (observedDigest !== reviewed.content_digest) {
      throw new Error(`${reviewed.role}: tool bytes differ from the reviewed platform plan`);
    }
    return { role: reviewed.role, path: resolved };
  });
}

function resolveRustToolRoot() {
  const rustcLocator = resolveExecutable("rustc");
  const sysroot = execFileSync(rustcLocator, ["--print", "sysroot"], { encoding: "utf8" }).trim();
  if (!path.isAbsolute(sysroot)) throw new Error("rustc returned a non-absolute sysroot");
  return fs.realpathSync(path.join(sysroot, "bin"));
}

function platformExecutableName(role) {
  return process.platform === "win32" ? `${role}.exe` : role;
}

const REMOVED_TOOL_ENVIRONMENT = Object.freeze([
  "CARGO_BUILD_RUSTC", "CARGO_BUILD_RUSTC_WRAPPER", "CARGO_BUILD_TARGET_DIR",
  "CARGO_ENCODED_RUSTFLAGS", "CARGO_ENCODED_RUSTDOCFLAGS", "RUSTC", "RUSTC_WORKSPACE_WRAPPER",
  "RUSTC_WRAPPER", "RUSTDOC", "RUSTDOCFLAGS", "RUSTFLAGS", "RUSTFMT",
]);

function reviewedToolEnvironment(tools) {
  const byRole = new Map(tools.map((entry) => [entry.role, entry.path]));
  const cargo = byRole.get("cargo") ?? null;
  return {
    path_prepend: cargo ? path.dirname(cargo) : null,
    rustc: byRole.get("rustc") ?? null,
    rustdoc: byRole.get("rustdoc") ?? null,
    rustfmt: byRole.get("rustfmt") ?? null,
    removed: [...REMOVED_TOOL_ENVIRONMENT],
  };
}

function toolEnvironment(selection, storageEnvironment) {
  const environment = { ...process.env, ...storageEnvironment };
  for (const name of selection.removed) delete environment[name];
  if (selection.rustc) environment.RUSTC = selection.rustc;
  if (selection.rustdoc) environment.RUSTDOC = selection.rustdoc;
  if (selection.rustfmt) environment.RUSTFMT = selection.rustfmt;
  if (selection.path_prepend) environment.PATH = environment.PATH
    ? `${selection.path_prepend}${path.delimiter}${environment.PATH}`
    : selection.path_prepend;
  return environment;
}

function pinnedSourceDigest(repository, sourcePath, expected, inventory) {
  const entry = inventory.source.files.find((candidate) => candidate.path === sourcePath);
  if (!entry || entry.content_digest !== expected) throw new Error(`${sourcePath}: reviewed producer source differs from the frozen inventory`);
  const absolute = path.join(repository, sourcePath);
  const stat = fs.lstatSync(absolute);
  if (!stat.isFile() || contentDigest(fs.readFileSync(absolute)) !== expected) throw new Error(`${sourcePath}: producer bytes differ from the reviewed plan at execution time`);
  return expected;
}

function parserFor(kind, gate, context) {
  if (kind === "exit_status") return (_stdout, _stderr, status, identities) => ({
    checks: primaryChecks(gate, identities, status === 0), artifacts: [],
  });
  if (kind === "compiled_inventory") return (stdout, _stderr, status, identities) => {
    return parseCompiledInventoryProducer(stdout, status, identities, gate, context, producerServices());
  };
  if (kind === "documentation_cutover") return (stdout, _stderr, status, identities) => {
    return parseDocumentationProducer(stdout, status, identities, gate, context, producerServices());
  };
  if (kind === "generated_products") return (stdout, _stderr, status, identities) => {
    return parseGeneratedProductsProducer(stdout, status, identities, gate, context, producerServices());
  };
  if (kind === "example_reconciliation") return (stdout, _stderr, status, identities) => {
    return parseExampleProducer(stdout, status, identities, gate, context, producerServices());
  };
  if (kind === "inventory_delta") return (stdout, _stderr, status, identities) => {
    return parseInventoryDeltaProducer(stdout, status, identities, gate, context, producerServices());
  };
  throw new Error(`${gate}: reviewed parser ${kind} has no authoritative machine contract`);
}

function failedProducer(gate, identities) {
  return { checks: primaryChecks(gate, identities, false), artifacts: [] };
}

function producerServices() {
  return { failedProducer, producerArtifactOutput, writeProducerArtifact, canonicalPotentialPath, isWithin, repository: REPOSITORY };
}

function producerArtifactOutput(context, gate) {
  if (!context.input.inputs) throw new Error(`${gate}: producer requires an external artifact output`);
  const fields = ["native-examples", "browser-examples"].includes(gate)
    ? ["artifact_output", "evidence_root", "example_manifest"]
    : ["artifact_output"];
  exact(context.input.inputs, fields, `${gate} producer inputs`);
  return canonicalPotentialPath(absolutePath(context.input.inputs.artifact_output, `${gate} evidence output`));
}

function writeProducerArtifact(output, role, contents) {
  if (isWithin(REPOSITORY, output)) throw new Error(`${role} evidence output must be outside the canonical repository`);
  const bytes = Buffer.from(contents);
  publishEvidenceBytes(output, bytes);
  return { role, path: output, byte_length: bytes.length, content_digest: contentDigest(bytes) };
}

function primaryChecks(gate, identities, passed) {
  return identities.map((identity) => ({ id: `${gate}:${identity}`, result: passed ? "pass" : "fail", evidence_digest: evidenceDigest({ gate, identity, passed }) }));
}

function resolveExecutable(name) {
  if (path.isAbsolute(name)) return fs.realpathSync(name);
  const extensions = process.platform === "win32"
    ? (process.env.PATHEXT ?? ".COM;.EXE;.BAT;.CMD").split(";")
    : [""];
  for (const directory of (process.env.PATH ?? "").split(path.delimiter)) {
    if (!directory) continue;
    for (const extension of extensions) {
      const candidate = path.join(directory, `${name}${extension}`);
      if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) return fs.realpathSync(candidate);
    }
  }
  throw new Error(`${name}: executable is absent from PATH`);
}

function canonicalPotentialPath(target) {
  const remainder = [];
  let existing = path.resolve(target);
  while (!fs.existsSync(existing)) {
    const parent = path.dirname(existing);
    if (parent === existing) throw new Error(`cannot resolve existing ancestor for ${target}`);
    remainder.unshift(path.basename(existing));
    existing = parent;
  }
  return path.join(fs.realpathSync(existing), ...remainder);
}

function isWithin(parent, child) {
  const relative = path.relative(parent, child);
  return relative === "" || (!relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative));
}
