import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

import { compareCodePoint } from "./constants.mjs";
import { pathAllowed } from "./control-graph.mjs";
import { assertValidatedControl } from "./control.mjs";
import { evidenceDigest } from "./evidence.mjs";
import { deepImmutable } from "./immutable.mjs";
import { assertActiveLease } from "./lease.mjs";
import { parseMigrationPhases } from "./migration-phase-schema.mjs";
import { sourceRevision, stableId } from "./schema.mjs";

export { parseMigrationPhases } from "./migration-phase-schema.mjs";

const VALIDATED_PHASES = new WeakSet();

export function captureMigrationPhases(
  repository, lease, control, subject, authoredRevision, clock = Date.now,
) {
  assertActiveLease(lease, control, clock);
  assertValidatedControl(control);
  const root = canonicalRepository(repository);
  if (subject.source.dirty !== false) throw new Error("integrated subject must be a clean committed source snapshot");
  const integratedRevision = repositoryRevision(root, "git:HEAD");
  if (subject.source.revision !== integratedRevision) {
    throw new Error("integrated subject revision differs from the repository HEAD");
  }
  const phaseValue = phaseValueFromRepository(root, {
    leaseBaseRevision: lease.value.base_revision,
    authoredRevision,
    integratedRevision,
    bundle: lease.bundle,
    controlBaselineRevision: control.baseline.revision,
  });
  return validated(phaseValue);
}

export function validateSealedMigrationPhases(repository, control, bundleId, value) {
  assertValidatedControl(control);
  const bundle = control.bundles.get(stableId(bundleId, "phase bundle id"));
  if (!bundle) throw new Error(`migration phases reference unknown bundle ${bundleId}`);
  const parsed = parseMigrationPhases(value);
  const root = canonicalRepository(repository);
  const expected = phaseValueFromRepository(root, {
    leaseBaseRevision: parsed.lease_base_revision,
    authoredRevision: parsed.authored_revision,
    integratedRevision: parsed.integrated_revision,
    bundle,
    controlBaselineRevision: control.baseline.revision,
  });
  if (JSON.stringify(parsed) !== JSON.stringify(expected)) {
    throw new Error("migration phase evidence differs from the exact repository revisions and reviewed bundle policy");
  }
  return validated(expected);
}

export function assertValidatedMigrationPhases(value) {
  if (!VALIDATED_PHASES.has(value)) throw new Error("operation requires exact validated migration phase evidence");
  return value;
}

function phaseValueFromRepository(repository, options) {
  const leaseBaseRevision = repositoryRevision(repository, options.leaseBaseRevision);
  const authoredRevision = repositoryRevision(repository, options.authoredRevision);
  const integratedRevision = repositoryRevision(repository, options.integratedRevision);
  assertAncestor(repository, options.controlBaselineRevision, leaseBaseRevision,
    "lease base does not descend from the frozen control baseline");
  assertAncestor(repository, leaseBaseRevision, authoredRevision,
    "authored revision does not descend from the lease base");
  assertAncestor(repository, authoredRevision, integratedRevision,
    "integrated revision does not descend from the authored revision");

  const authoredChangedPaths = changedPaths(repository, leaseBaseRevision, authoredRevision);
  const integrationChangedPaths = changedPaths(repository, authoredRevision, integratedRevision);
  validateMigrationPhasePaths(options.bundle, authoredChangedPaths, integrationChangedPaths);
  const reviewedAuthoredWriteSet = structuredClone(options.bundle.authored_write_set);
  const reviewedIntegrationOutputs = options.bundle.integration_outputs.map(
    ({ product_id, path: outputPath, producer }) => ({ product_id, path: outputPath, producer }),
  );
  return {
    lease_base_revision: leaseBaseRevision,
    authored_revision: authoredRevision,
    integrated_revision: integratedRevision,
    authored_changed_paths: authoredChangedPaths,
    integration_changed_paths: integrationChangedPaths,
    reviewed_authored_write_set: reviewedAuthoredWriteSet,
    reviewed_integration_outputs: reviewedIntegrationOutputs,
    authored_write_set_digest: evidenceDigest(reviewedAuthoredWriteSet),
    integration_outputs_digest: evidenceDigest(reviewedIntegrationOutputs),
  };
}

export function validateMigrationPhasePaths(bundle, authoredChangedPaths, integrationChangedPaths) {
  const integrationPaths = new Set(bundle.integration_outputs.map((entry) => entry.path));
  const authoredViolations = authoredChangedPaths.filter((sourcePath) =>
    integrationPaths.has(sourcePath) || !pathAllowed(bundle.authored_write_set, sourcePath));
  if (authoredViolations.length) {
    throw new Error(`authored phase changed paths outside its lease: ${authoredViolations.join(", ")}`);
  }
  const integrationViolations = integrationChangedPaths.filter((sourcePath) => !integrationPaths.has(sourcePath));
  if (integrationViolations.length) {
    throw new Error(`integration phase changed non-product paths: ${integrationViolations.join(", ")}`);
  }
}

function changedPaths(repository, fromRevision, toRevision) {
  const output = execFileSync("git", [
    "-c", "core.quotepath=false", "diff", "--name-only", "--no-renames", "-z",
    fromRevision.slice("git:".length), toRevision.slice("git:".length), "--",
  ], { cwd: repository, encoding: "utf8" });
  return [...new Set(output.split("\0").filter(Boolean))].sort(compareCodePoint);
}

function canonicalRepository(repository) {
  if (typeof repository !== "string" || !path.isAbsolute(repository)) {
    throw new Error("migration phase validation requires an absolute repository path");
  }
  const canonical = fs.realpathSync(repository);
  const root = fs.realpathSync(execFileSync("git", ["rev-parse", "--show-toplevel"], {
    cwd: canonical, encoding: "utf8",
  }).trim());
  if (root !== canonical) throw new Error("migration phase repository must be the canonical Git worktree root");
  return canonical;
}

function repositoryRevision(repository, revision) {
  if (typeof revision !== "string" || !revision.startsWith("git:") || revision.length <= "git:".length) {
    throw new Error("migration phase revision must use a nonempty git: reference");
  }
  const raw = revision.slice("git:".length);
  const resolved = execFileSync("git", ["rev-parse", "--verify", `${raw}^{commit}`], {
    cwd: repository, encoding: "utf8",
  }).trim().toLowerCase();
  const result = `git:${resolved}`;
  return sourceRevision(result, "migration phase revision");
}

function assertAncestor(repository, ancestor, descendant, message) {
  const result = spawnSync("git", [
    "merge-base", "--is-ancestor", ancestor.slice("git:".length), descendant.slice("git:".length),
  ], { cwd: repository, encoding: "utf8" });
  if (result.error) throw new Error(`could not verify migration phase ancestry: ${result.error.message}`);
  if (result.status === 1) throw new Error(message);
  if (result.status !== 0) {
    throw new Error(`could not verify migration phase ancestry: ${result.stderr.trim() || `git exited ${result.status}`}`);
  }
}

function validated(value) {
  const result = deepImmutable(value);
  VALIDATED_PHASES.add(result);
  return result;
}
