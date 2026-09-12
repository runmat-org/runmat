import fs from "node:fs";
import path from "node:path";

import { compareCodePoint } from "./constants.mjs";
import { contentDigest, evidenceDigest } from "./evidence.mjs";
import { executionTargetKey, parseExecutionTarget, parseExecutionTargets, sameExecutionTarget } from "./execution-target.mjs";
import { gatePlanEvidence } from "./gate-plan.mjs";
import { GATE_PRODUCERS } from "./gate-kinds.mjs";
import { SAFE_IDENTITY, absolutePath, array, digest, enumValue, exact, filesystemIdentity, integer, kind, nonempty, sourceRevision, stableId, timestamp, uniqueStrings } from "./schema.mjs";
import { storageStatus } from "./storage-admission.mjs";

export { GATE_PRODUCERS } from "./gate-kinds.mjs";

export function parseGateResult(value, expected) {
  kind(value, 5, "runmat-builtin-migration-gate-result", "gate result");
  exact(value, ["schema_version", "kind", "authority", "producer", "producer_evidence", "artifact_id", "produced_at", "execution_target", "source_revision", "source_digest", "baseline_inventory_digest", "subject_inventory_digest", "control_manifest_digest", "bundle_id", "identities", "gate", "result", "checks", "artifacts", "storage_admission"], "gate result");
  if (value.authority !== "machine-verification-only") throw new Error("gate result has invalid authority");
  const gate = enumValue(value.gate, Object.keys(GATE_PRODUCERS), "gate result gate");
  if (value.producer !== GATE_PRODUCERS[gate]) throw new Error(`${gate}: unexpected gate producer`);
  const producedAt = timestamp(value.produced_at, "gate produced_at");
  const executionTarget = parseExecutionTarget(value.execution_target, "gate execution target");
  const producerEvidence = parseProducerEvidence(value.producer_evidence, value.producer);
  stableId(value.artifact_id, "gate artifact id");
  sourceRevision(value.source_revision, "gate source revision");
  digest(value.source_digest, "gate source digest");
  digest(value.baseline_inventory_digest, "gate baseline inventory digest");
  digest(value.subject_inventory_digest, "gate subject inventory digest");
  digest(value.control_manifest_digest, "gate control manifest digest");
  stableId(value.bundle_id, "gate bundle id");
  const identities = uniqueStrings(value.identities, "gate identities", { pattern: SAFE_IDENTITY, lower: true });
  enumValue(value.result, ["pass", "fail", "unavailable"], "gate result");
  const checks = array(value.checks, "gate checks").map((entry) => {
    exact(entry, ["id", "result", "evidence_digest"], "gate check");
    nonempty(entry.id, "gate check id");
    enumValue(entry.result, ["pass", "fail", "unavailable"], "gate check result");
    digest(entry.evidence_digest, "gate check evidence digest");
    return entry;
  });
  if (new Set(checks.map((entry) => entry.id)).size !== checks.length) throw new Error("gate check ids must be unique");
  for (const id of identities) if (!checks.some((entry) => entry.id === `${gate}:${id}`)) throw new Error(`${gate}: typed producer evidence is missing the ${id} identity check`);
  const artifacts = array(value.artifacts, "gate artifacts", { empty: true }).map(parseArtifact);
  const artifactRoles = artifacts.map((entry) => entry.role);
  if (new Set(artifactRoles).size !== artifactRoles.length) throw new Error(`${gate}: gate artifact roles must be unique`);
  if (JSON.stringify(artifactRoles) !== JSON.stringify([...artifactRoles].sort(compareCodePoint))) throw new Error(`${gate}: gate artifacts must use canonical role ordering`);
  const derivedResult = value.producer_evidence.process.exit_code === 0 && checks.every((entry) => entry.result === "pass") ? "pass" : "fail";
  if (value.result !== derivedResult) throw new Error("gate result conflicts with captured process status or checks");
  if (expected?.compiled_build && !sameExecutionTarget(executionTarget, expected.compiled_build)) {
    throw new Error(`${gate}: execution target differs from the subject build target`);
  }
  if (expected?.execution_targets) {
    const reviewedTargets = parseExecutionTargets(expected.execution_targets);
    if (!reviewedTargets.some((entry) => sameExecutionTarget(entry, executionTarget))) {
      throw new Error(`${gate}: execution target is absent from the reviewed control target set`);
    }
  }
  const storageAdmission = parseStorageAdmission(
    value.storage_admission, value.result, expected, producedAt, executionTarget, artifacts,
  );
  validateExecutionEnvironment(producerEvidence.invocation.environment, storageAdmission);
  if (expected) {
    if (value.source_revision !== expected.source_revision || value.source_digest !== expected.source_digest || value.baseline_inventory_digest !== expected.baseline_inventory_digest || value.subject_inventory_digest !== expected.subject_inventory_digest || value.control_manifest_digest !== expected.control_manifest_digest || value.bundle_id !== expected.bundle_id) {
      throw new Error(`${gate}: stale or mismatched gate provenance`);
    }
    if (expected.gate_plans) validateReviewedPlan(
      gate, value.result, producerEvidence, artifacts, expected, executionTarget, storageAdmission,
    );
  }
  verifyArtifactFiles(artifacts);
  return { ...value, identities: identities.sort(compareCodePoint), checks, artifacts };
}

function parseArtifact(value) {
  exact(value, ["role", "path", "byte_length", "content_digest"], "gate artifact");
  nonempty(value.role, "gate artifact role");
  const artifactPath = absolutePath(value.path, `${value.role} artifact path`);
  integer(value.byte_length, `${value.role} artifact byte length`, 0);
  digest(value.content_digest, `${value.role} artifact content digest`);
  return { ...value, path: artifactPath };
}

function verifyArtifactFiles(artifacts) {
  for (const artifact of artifacts) {
    const bytes = readCanonicalRegularFile(artifact.path, `${artifact.role} artifact`);
    if (bytes.length !== artifact.byte_length || contentDigest(bytes) !== artifact.content_digest) {
      throw new Error(`${artifact.role}: artifact bytes differ from gate evidence`);
    }
  }
}

function parseProducerEvidence(value, producer) {
  kind(value, 3, `${producer}-evidence`, "typed producer evidence");
  exact(value, ["schema_version", "kind", "contract", "invocation", "process", "captured_process_digest"], "typed producer evidence");
  exact(value.contract, ["reviewed_source_revision", "primary_tool", "tools", "producer_source_digest"], "producer contract evidence");
  sourceRevision(value.contract.reviewed_source_revision, "producer reviewed source revision");
  const contractTools = parseContractTools(value.contract.tools);
  const primaryTool = enumValue(value.contract.primary_tool, contractTools.map((entry) => entry.role), "producer primary tool");
  digest(value.contract.producer_source_digest, "producer source digest");
  exact(value.invocation, ["executable", "arguments", "cwd", "environment", "tools", "tool_environment"], "producer invocation");
  const executable = absolutePath(value.invocation.executable, "producer executable");
  const invocationTools = parseInvocationTools(value.invocation.tools, contractTools);
  parseToolEnvironment(value.invocation.tool_environment, invocationTools);
  const primaryPath = invocationTools.find((entry) => entry.role === primaryTool).path;
  if (executable !== primaryPath) throw new Error("producer executable differs from the primary reviewed tool");
  array(value.invocation.arguments, "producer arguments", { empty: true }).forEach((entry) => nonempty(entry, "producer argument"));
  absolutePath(value.invocation.cwd, "producer working directory");
  exact(value.invocation.environment, ["CARGO_TARGET_DIR", "TMPDIR", "TMP", "TEMP"], "producer environment");
  for (const [name, environmentPath] of Object.entries(value.invocation.environment)) {
    absolutePath(environmentPath, `producer ${name}`);
  }
  exact(value.process, ["exit_code", "signal", "stdout_digest", "stderr_digest"], "producer process result");
  integer(value.process.exit_code, "producer exit code");
  if (value.process.signal !== null) nonempty(value.process.signal, "producer signal");
  digest(value.process.stdout_digest, "producer stdout digest");
  digest(value.process.stderr_digest, "producer stderr digest");
  digest(value.captured_process_digest, "producer captured process digest");
  if (value.captured_process_digest !== evidenceDigest(value.process)) throw new Error("producer captured process digest is inconsistent");
  return value;
}

function parseContractTools(value) {
  const tools = array(value, "producer contract tools").map((entry) => {
    exact(entry, ["role", "content_digest"], "producer contract tool");
    const role = enumValue(entry.role, ["cargo", "cargo-clippy", "cargo-fmt", "clippy-driver", "git", "node", "rustc", "rustdoc", "rustfmt"], "producer contract tool role");
    digest(entry.content_digest, `${role} producer tool digest`);
    return entry;
  });
  requireCanonicalToolRoles(tools, "producer contract tools");
  return tools;
}

function parseToolEnvironment(value, tools) {
  exact(value, ["path_prepend", "rustc", "rustdoc", "rustfmt", "removed"], "producer tool environment");
  const byRole = new Map(tools.map((entry) => [entry.role, entry.path]));
  const expected = {
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
  if (JSON.stringify(value) !== JSON.stringify(expected)) {
    throw new Error("producer tool environment does not enforce the reviewed tool selection");
  }
}

function parseInvocationTools(value, contractTools) {
  const tools = array(value, "producer invocation tools").map((entry) => {
    exact(entry, ["role", "path"], "producer invocation tool");
    return { role: nonempty(entry.role, "producer invocation tool role"), path: absolutePath(entry.path, "producer invocation tool path") };
  });
  requireCanonicalToolRoles(tools, "producer invocation tools");
  if (JSON.stringify(tools.map((entry) => entry.role)) !== JSON.stringify(contractTools.map((entry) => entry.role))) {
    throw new Error("producer invocation tools differ from the contract tool roles");
  }
  for (let index = 0; index < tools.length; index += 1) {
    if (contentDigest(readCanonicalRegularFile(tools[index].path, `${tools[index].role} producer tool`)) !== contractTools[index].content_digest) {
      throw new Error(`${tools[index].role}: producer tool bytes differ from contract evidence`);
    }
  }
  return tools;
}

function readCanonicalRegularFile(filePath, label) {
  const stat = fs.lstatSync(filePath);
  if (!stat.isFile() || fs.realpathSync(filePath) !== filePath) throw new Error(`${label} must be a canonical regular file`);
  return fs.readFileSync(filePath);
}

function requireCanonicalToolRoles(tools, label) {
  const roles = tools.map((entry) => entry.role);
  if (new Set(roles).size !== roles.length) throw new Error(`${label} must be unique by role`);
  if (JSON.stringify(roles) !== JSON.stringify([...roles].sort(compareCodePoint))) {
    throw new Error(`${label} must use canonical role ordering`);
  }
}

function validateReviewedPlan(gate, result, evidence, artifacts, expected, executionTarget, storageAdmission) {
  const plan = expected.gate_plans.get(gate);
  if (!plan) throw new Error(`${gate}: gate evidence has no reviewed bundle plan`);
  const contract = gatePlanEvidence(plan, executionTarget, storageAdmission.path_bindings.repository.path);
  if (evidence.contract.reviewed_source_revision !== expected.baseline_source_revision
    || evidence.contract.primary_tool !== contract.primary_tool
    || JSON.stringify(evidence.contract.tools) !== JSON.stringify(contract.tools)
    || evidence.contract.producer_source_digest !== contract.source_digest) {
    throw new Error(`${gate}: producer contract differs from the reviewed gate plan`);
  }
  if (JSON.stringify(evidence.invocation.arguments) !== JSON.stringify(contract.arguments) || evidence.invocation.cwd !== contract.cwd) throw new Error(`${gate}: producer invocation differs from the reviewed gate plan`);
  const actualRoles = artifacts.map((entry) => entry.role);
  if (actualRoles.some((role) => !plan.expected_artifact_roles.includes(role))) throw new Error(`${gate}: producer emitted an unreviewed artifact role`);
  if (result === "pass" && JSON.stringify(actualRoles) !== JSON.stringify(plan.expected_artifact_roles)) throw new Error(`${gate}: passing evidence does not cover the reviewed artifact roles`);
}

function parseStorageAdmission(value, gateResult, expected, producedAt, executionTarget, artifacts) {
  exact(value, ["profile_id", "execution_host", "observed_at", "path_bindings", "volumes"], "gate storage admission");
  const profileId = stableId(value.profile_id, "gate storage profile id");
  const executionHost = nonempty(value.execution_host, "gate storage execution host");
  const observedAt = timestamp(value.observed_at, "gate storage observed_at");
  const pathBindings = parsePathBindings(value.path_bindings, artifacts);
  const volumes = array(value.volumes, "gate storage volumes");
  const expectedRoles = ["source-worktree", "target-temp"];
  if (JSON.stringify(volumes.map((entry) => entry.role).sort()) !== JSON.stringify([...expectedRoles].sort())) throw new Error("gate storage admission must cover both named volume roles exactly once");
  for (const entry of volumes) {
    exact(entry, ["role", "evidence_path", "filesystem_id", "available_bytes", "minimum_free_bytes", "pause_below_bytes", "status"], "gate storage volume");
    enumValue(entry.role, expectedRoles, "gate storage role");
    absolutePath(entry.evidence_path, `${entry.role} storage evidence path`);
    filesystemIdentity(entry.filesystem_id, `${entry.role} storage filesystem id`);
    integer(entry.available_bytes, `${entry.role} available bytes`);
    integer(entry.minimum_free_bytes, `${entry.role} minimum free bytes`, 1);
    integer(entry.pause_below_bytes, `${entry.role} pause below bytes`, 1);
    if (entry.pause_below_bytes < entry.minimum_free_bytes) throw new Error(`${entry.role}: pause threshold is below minimum`);
    const expectedStatus = storageStatus(entry.available_bytes, entry.minimum_free_bytes, entry.pause_below_bytes);
    if (entry.status !== expectedStatus) throw new Error(`${entry.role}: storage status conflicts with observed bytes`);
    if (gateResult === "pass" && entry.status !== "admitted") throw new Error(`${entry.role}: a product gate cannot pass below its pause threshold`);
    if (expected?.storage_policy) {
      const key = entry.role === "source-worktree" ? "source_worktree" : "target_temp";
      const profile = expected.storage_policy.host_profiles[profileId];
      if (!profile) throw new Error(`gate storage profile ${profileId} is absent from reviewed control policy`);
      if (executionHost !== profile.execution_host) {
        throw new Error(`gate storage execution host differs from reviewed profile ${profileId}`);
      }
      if (executionTargetKey(profile) !== executionTargetKey(executionTarget)) {
        throw new Error(`gate storage profile ${profileId} differs from the subject build target`);
      }
      const configured = profile.volume_roles[key];
      const age = (Date.parse(producedAt) - Date.parse(observedAt)) / 1000;
      if (age < 0 || age > configured.maximum_observation_age_seconds) throw new Error(`${entry.role}: storage observation is outside the reviewed time bound`);
      if (entry.evidence_path !== configured.mount_path || entry.filesystem_id !== configured.filesystem_id || entry.minimum_free_bytes !== configured.minimum_free_bytes || entry.pause_below_bytes !== configured.pause_below_bytes) throw new Error(`${entry.role}: storage evidence differs from reviewed control policy`);
    }
  }
  const filesystems = new Map(volumes.map((entry) => [entry.role, entry.filesystem_id]));
  if (pathBindings.repository.filesystem_id !== filesystems.get("source-worktree")) {
    throw new Error("repository path is not bound to the reviewed source-worktree volume");
  }
  for (const binding of [pathBindings.cargo_target, pathBindings.temporary, ...pathBindings.artifacts]) {
    if (binding.filesystem_id !== filesystems.get("target-temp")) {
      throw new Error(`${binding.role}: path is not bound to the reviewed target-temp volume`);
    }
  }
  if (expected?.repository && pathBindings.repository.path !== fs.realpathSync(expected.repository)) {
    throw new Error("gate repository path differs from the trusted execution repository");
  }
  return { ...value, path_bindings: pathBindings };
}

function parsePathBindings(value, artifacts) {
  exact(value, ["repository", "cargo_target", "temporary", "artifacts"], "gate storage path bindings");
  const repository = parsePathBinding(value.repository, "repository");
  const cargoTarget = parsePathBinding(value.cargo_target, "cargo-target");
  const temporary = parsePathBinding(value.temporary, "temporary");
  const artifactBindings = array(value.artifacts, "gate artifact path bindings", { empty: true })
    .map((entry) => parsePathBinding(entry, entry?.role ?? "artifact"));
  const roles = artifactBindings.map((entry) => entry.role);
  if (new Set(roles).size !== roles.length) throw new Error("gate artifact path binding roles must be unique");
  if (JSON.stringify(roles) !== JSON.stringify([...roles].sort(compareCodePoint))) {
    throw new Error("gate artifact path bindings must use canonical role ordering");
  }
  const expected = artifacts.map((entry) => ({ role: entry.role, path: entry.path }));
  const observed = artifactBindings.map((entry) => ({ role: entry.role, path: entry.path }));
  if (JSON.stringify(observed) !== JSON.stringify(expected)) {
    throw new Error("gate artifact paths differ from their reviewed storage bindings");
  }
  return { repository, cargo_target: cargoTarget, temporary, artifacts: artifactBindings };
}

function parsePathBinding(value, role) {
  exact(value, ["role", "path", "filesystem_id"], `${role} storage path binding`);
  if (value.role !== role) throw new Error(`${role}: storage path binding has the wrong role`);
  return {
    role,
    path: absolutePath(value.path, `${role} storage path`),
    filesystem_id: filesystemIdentity(value.filesystem_id, `${role} storage filesystem id`),
  };
}

function validateExecutionEnvironment(environment, admission) {
  const cargoTarget = admission.path_bindings.cargo_target.path;
  const temporary = admission.path_bindings.temporary.path;
  if (environment.CARGO_TARGET_DIR !== cargoTarget
    || environment.TMPDIR !== temporary
    || environment.TMP !== temporary
    || environment.TEMP !== temporary) {
    throw new Error("producer environment differs from reviewed storage path bindings");
  }
}
