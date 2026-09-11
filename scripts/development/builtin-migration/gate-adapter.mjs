import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { parseCompiledInventory } from "./compiled-inventory.mjs";
import { parseControlManifest } from "./control.mjs";
import { canonicalJson, contentDigest, evidenceDigest } from "./evidence.mjs";
import { GATE_PRODUCERS, parseGateResult } from "./gate-result.mjs";
import { parseInventoryEvidence } from "./inventory.mjs";
import { buildDocumentationCutoverArtifact, documentationCutoverChecks, parseDocumentationCutoverArtifact } from "./documentation-cutover.mjs";
import { generatedProductChecks, parseGeneratedProductsProof } from "./generated-products.mjs";
import { buildInventoryDeltaProof, inventoryDeltaChecks } from "./inventory-delta.mjs";
import { parseExampleGateProof } from "./example-gate.mjs";
import { absolutePath, exact } from "./schema.mjs";
import { sourceFieldBaselineSource } from "./source-fields.mjs";

const REPOSITORY = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
// The reviewed control owns the complete command. Callers select only a bundle
// and gate; they cannot supply executable, argv, cwd, checks, or process facts.
export function runGateProducer(input) {
  exact(input, ["control", "baseline_inventory", "subject_inventory", "bundle_id", "gate", "artifact_id", "inputs"], "gate producer request");
  const baseline = parseInventoryEvidence(input.baseline_inventory);
  const subject = parseInventoryEvidence(input.subject_inventory);
  if (subject.source.dirty !== false) throw new Error("gate subject must be a clean committed source snapshot");
  const control = parseControlManifest(input.control.value ?? input.control, baseline);
  const bundle = control.bundles.get(input.bundle_id);
  if (!bundle) throw new Error(`${input.bundle_id}: unknown gate bundle`);
  const plan = bundle.gate_plans.get(input.gate);
  if (!plan) throw new Error(`${input.bundle_id}/${input.gate}: no reviewed gate plan is registered`);
  const parser = parserFor(plan.parser, input.gate, { input, baseline, subject, control, bundle });
  const repository = fs.realpathSync(REPOSITORY);
  const command = commandFor(plan, repository, baseline);
  const executable = resolveExecutable(command.executable);
  const executableDigest = contentDigest(fs.readFileSync(executable));
  if (executableDigest !== command.executableDigest) throw new Error(`${input.gate}: executable bytes differ from the reviewed platform plan`);
  const processResult = spawnSync(executable, command.arguments, {
    cwd: repository, encoding: "utf8", env: process.env, maxBuffer: 128 * 1024 * 1024,
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
    baseline_source_revision: baseline.source.revision,
    baseline_inventory_digest: baseline.digest, subject_inventory_digest: subject.digest,
    control_manifest_digest: control.digest,
    bundle_id: bundle.id, storage_policy: control.value.storage_policy,
    gate_plans: bundle.gate_plans, compiled_build: subject.compiled_inventory.build, repository,
  };
  return parseGateResult({
    schema_version: 2, kind: "runmat-builtin-migration-gate-result", authority: "machine-verification-only",
    producer: GATE_PRODUCERS[input.gate],
    producer_evidence: {
      schema_version: 1, kind: `${GATE_PRODUCERS[input.gate]}-evidence`,
      contract: {
        reviewed_source_revision: baseline.source.revision,
        executable_digest: executableDigest,
        producer_source_digest: command.sourceDigest,
      },
      invocation: { executable, arguments: command.arguments, cwd: repository },
      process: processEvidence, captured_process_digest: evidenceDigest(processEvidence),
    },
    artifact_id: input.artifact_id, produced_at: new Date().toISOString(),
    source_revision: subject.source.revision, source_digest: subject.source.digest,
    baseline_inventory_digest: baseline.digest, subject_inventory_digest: subject.digest,
    control_manifest_digest: control.digest,
    bundle_id: bundle.id, identities, gate: input.gate, result, checks, artifacts,
    storage_admission: observeStorage(control.value.storage_policy),
  }, expected);
}

function commandFor(plan, repository, inventory) {
  const build = inventory.compiled_inventory.build;
  const executableDigest = plan.program.approved_executables.find((entry) => entry.operating_system === build.operating_system && entry.architecture === build.architecture).content_digest;
  if (plan.program.kind === "repository_script") {
    const sourceDigest = pinnedSourceDigest(repository, plan.program.path, plan.program.content_digest, inventory);
    return { executable: process.execPath, arguments: [path.join(repository, plan.program.path), ...plan.arguments], sourceDigest, executableDigest };
  }
  const sourceDigest = pinnedSourceDigest(repository, plan.program.manifest_path, plan.program.manifest_digest, inventory);
  return {
    executable: "cargo",
    arguments: ["run", "--quiet", "-p", plan.program.package, "--bin", plan.program.binary, "--", ...plan.arguments],
    sourceDigest, executableDigest,
  };
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
    if (status !== 0) return failedProducer(gate, identities);
    const artifactOutput = producerArtifactOutput(context, gate);
    const compiled = parseCompiledInventory(JSON.parse(stdout));
    if (compiled.digest !== context.subject.compiled_inventory.digest) throw new Error(`${gate}: producer output differs from the subject compiled inventory`);
    return {
      checks: identities.map((identity) => ({
        id: `${gate}:${identity}`,
        result: compiledCheck(gate, authority(compiled, identity)) ? "pass" : "fail",
        evidence_digest: compiled.digest,
      })),
      artifacts: [writeProducerArtifact(artifactOutput, "compiled-inventory", stdout)],
    };
  };
  if (kind === "documentation_cutover") return (stdout, _stderr, status, identities) => {
    if (status !== 0) return failedProducer(gate, identities);
    if (!context.input.inputs) throw new Error(`${gate}: documentation producer requires reviewed source disposition inputs`);
    exact(context.input.inputs, ["source_dispositions", "artifact_output"], "documentation producer inputs");
    const artifact = buildDocumentationCutoverArtifact({
      catalog_export: JSON.parse(stdout), catalog_export_bytes: stdout,
      source_dispositions: context.input.inputs.source_dispositions,
      expected_sources: Object.fromEntries(identities.map((identity) => {
        const row = context.baseline.identities.find((entry) => entry.identity === identity);
        const ownership = row?.lexical_observations?.ownership;
        const paths = [...(ownership?.sidecars ?? []), ...(ownership?.runtime_documentation_shadows ?? [])].sort();
        return [identity, paths.map((sourcePath) => {
          const frozen = context.baseline.source.files.find((entry) => entry.path === sourcePath);
          if (!frozen) throw new Error(`${sourcePath}: documentation source is absent from the frozen inventory`);
          const bytes = execFileSync("git", ["show", `${context.baseline.source.revision.slice("git:".length)}:${sourcePath}`], { cwd: REPOSITORY });
          return sourceFieldBaselineSource(sourcePath, bytes, frozen.content_digest);
        })];
      })),
      provenance: {
        source_revision: context.subject.source.revision, source_digest: context.subject.source.digest,
        compiled_inventory_digest: context.subject.compiled_inventory.digest,
        control_manifest_digest: context.control.digest, bundle_id: context.bundle.id, identities,
      },
    });
    parseDocumentationCutoverArtifact(artifact, {
      source_revision: context.subject.source.revision,
      source_digest: context.subject.source.digest,
      compiled_inventory_digest: context.subject.compiled_inventory.digest,
      control_manifest_digest: context.control.digest,
      bundle_id: context.bundle.id,
      identities,
      catalog_export_digest: artifact.catalog_export.content_digest,
      source_dispositions: context.input.inputs.source_dispositions,
    });
    const artifactOutput = canonicalPotentialPath(absolutePath(context.input.inputs.artifact_output, "documentation evidence output"));
    if (isWithin(REPOSITORY, artifactOutput)) throw new Error("documentation evidence output must be outside the canonical repository");
    const artifactBytes = `${JSON.stringify(artifact, null, 2)}\n`;
    return {
      checks: documentationCutoverChecks(artifact),
      artifacts: [writeProducerArtifact(artifactOutput, "documentation-reconciliation", artifactBytes)],
    };
  };
  if (kind === "generated_products") return (stdout, _stderr, status, identities) => {
    if (status !== 0) return failedProducer(gate, identities);
    const artifactOutput = producerArtifactOutput(context, gate);
    const proof = parseGeneratedProductsProof(JSON.parse(stdout), {
      integration_outputs: context.bundle.integration_outputs,
      source_files: context.subject.source.files,
    });
    return {
      checks: generatedProductChecks(proof, identities),
      artifacts: [writeProducerArtifact(artifactOutput, "generated-products", stdout)],
    };
  };
  if (kind === "example_reconciliation") return (stdout, _stderr, status, identities) => {
    if (status !== 0) return failedProducer(gate, identities);
    const artifactOutput = producerArtifactOutput(context, gate);
    const proof = parseExampleGateProof(JSON.parse(stdout), {
      source_revision: context.subject.source.revision,
      identities,
    });
    const rows = new Map(proof.rows.map((entry) => [entry.identity, entry]));
    const checks = identities.map((identity) => {
      const row = rows.get(identity);
      const required = context.control.identities.get(identity).maturity.examples.applicability === "required";
      const passed = row.status === "passed" || (!required && row.status === "absent");
      return { id: `${gate}:${identity}`, result: passed ? "pass" : "fail", evidence_digest: row.evidence_digest };
    });
    return {
      checks,
      artifacts: [writeProducerArtifact(artifactOutput, "example-reconciliation", stdout)],
    };
  };
  if (kind === "inventory_delta") return (stdout, _stderr, status, identities) => {
    if (status !== 0) return failedProducer(gate, identities);
    const artifactOutput = producerArtifactOutput(context, gate);
    const compiled = parseCompiledInventory(JSON.parse(stdout));
    if (compiled.digest !== context.subject.compiled_inventory.digest) throw new Error(`${gate}: producer output differs from the subject compiled inventory`);
    const proof = buildInventoryDeltaProof(REPOSITORY, context.baseline, context.subject, {
      ...context.control,
      active_bundle_id: context.bundle.id,
    });
    if (JSON.stringify(proof.identities.map((entry) => entry.identity)) !== JSON.stringify(identities)) {
      throw new Error(`${gate}: inventory delta identity coverage differs from the reviewed bundle`);
    }
    return {
      checks: inventoryDeltaChecks(proof),
      artifacts: [writeProducerArtifact(artifactOutput, "inventory-delta", `${canonicalJson(proof)}\n`)],
    };
  };
  throw new Error(`${gate}: reviewed parser ${kind} has no authoritative machine contract`);
}

function failedProducer(gate, identities) {
  return { checks: primaryChecks(gate, identities, false), artifacts: [] };
}

function producerArtifactOutput(context, gate) {
  if (!context.input.inputs) throw new Error(`${gate}: producer requires an external artifact output`);
  exact(context.input.inputs, ["artifact_output"], `${gate} producer inputs`);
  return canonicalPotentialPath(absolutePath(context.input.inputs.artifact_output, `${gate} evidence output`));
}

function writeProducerArtifact(output, role, contents) {
  if (isWithin(REPOSITORY, output)) throw new Error(`${role} evidence output must be outside the canonical repository`);
  const bytes = Buffer.from(contents);
  fs.writeFileSync(output, bytes, { flag: "wx" });
  return { role, path: output, byte_length: bytes.length, content_digest: contentDigest(bytes) };
}

function compiledCheck(gate, authorityValue) {
  if (gate === "catalog-contract") return authorityValue.catalog_entries.length === 1;
  if (gate === "runtime-binding") return authorityValue.runtime_bindings.length > 0 && authorityValue.implementation_provenance.some((entry) => entry.authority === "canonical_binding");
  return false;
}

function authority(compiled, identity) {
  const matches = (rows, selector = (entry) => entry.name) => rows.filter((entry) => selector(entry).toLowerCase() === identity.toLowerCase());
  return {
    catalog_entries: matches(compiled.snapshot.declared.catalog_entries, (entry) => entry.identity.name),
    runtime_bindings: matches(compiled.snapshot.observed.runtime_bindings),
    implementation_provenance: matches(compiled.snapshot.observed.implementation_provenance),
  };
}

function primaryChecks(gate, identities, passed) {
  return identities.map((identity) => ({ id: `${gate}:${identity}`, result: passed ? "pass" : "fail", evidence_digest: evidenceDigest({ gate, identity, passed }) }));
}

function resolveExecutable(name) {
  if (path.isAbsolute(name)) return name;
  return execFileSync("/usr/bin/which", [name], { encoding: "utf8" }).trim();
}

function observeStorage(policy) {
  const observedAt = new Date().toISOString();
  const volumes = [policy.volume_roles.source_worktree, policy.volume_roles.target_temp].map((configured) => {
    const stats = fs.statSync(configured.mount_path, { bigint: true });
    const space = fs.statfsSync(configured.mount_path, { bigint: true });
    const filesystemId = process.platform === "win32" ? `windows-volume:${stats.dev.toString(16).padStart(8, "0")}` : `posix-dev:${stats.dev}`;
    const availableBytes = Number(space.bavail * space.bsize);
    return { role: configured.role, evidence_path: configured.mount_path, filesystem_id: filesystemId, available_bytes: availableBytes, minimum_free_bytes: configured.minimum_free_bytes, pause_below_bytes: configured.pause_below_bytes, status: availableBytes >= configured.pause_below_bytes ? "admitted" : "paused" };
  });
  return { observed_at: observedAt, volumes };
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
