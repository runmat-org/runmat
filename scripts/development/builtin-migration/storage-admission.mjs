import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { executionTargetKey } from "./execution-target.mjs";

export function prepareGateStorage(policy, build, repository, artifactOutputs, execution) {
  const executionHost = os.hostname();
  const matches = Object.entries(policy.host_profiles).filter(([, profile]) =>
    executionTargetKey(profile) === executionTargetKey(build) && profile.execution_host === executionHost);
  if (matches.length !== 1) {
    throw new Error(`storage policy must select exactly one profile for ${build.operating_system}/${build.architecture}/${executionHost}`);
  }
  const [profileId, profile] = matches[0];
  const sourceVolume = observeVolume(profile.volume_roles.source_worktree);
  const targetVolume = observeVolume(profile.volume_roles.target_temp);
  assertAdmitted(sourceVolume);
  assertAdmitted(targetVolume);

  const repositoryBinding = bindPath("repository", repository, sourceVolume.filesystem_id);
  const artifactBindings = artifactOutputs.map(({ role, output }) =>
    bindPath(role, output, targetVolume.filesystem_id));
  const executionRoot = path.join(
    fs.realpathSync(profile.volume_roles.target_temp.mount_path),
    "runmat-rm1064",
    execution.control_digest.slice("sha256:".length),
    execution.bundle_id,
  );
  const cargoTarget = path.join(executionRoot, "cargo-target");
  const temporary = path.join(executionRoot, "tmp", execution.artifact_id);
  fs.mkdirSync(cargoTarget, { recursive: true });
  fs.mkdirSync(temporary, { recursive: true });
  const cargoBinding = bindPath("cargo-target", cargoTarget, targetVolume.filesystem_id);
  const temporaryBinding = bindPath("temporary", temporary, targetVolume.filesystem_id);
  const environment = {
    CARGO_TARGET_DIR: cargoBinding.path,
    TMPDIR: temporaryBinding.path,
    TMP: temporaryBinding.path,
    TEMP: temporaryBinding.path,
  };
  return {
    environment,
    admission: {
      profile_id: profileId,
      execution_host: executionHost,
      observed_at: new Date().toISOString(),
      path_bindings: {
        repository: repositoryBinding,
        cargo_target: cargoBinding,
        temporary: temporaryBinding,
        artifacts: artifactBindings,
      },
      volumes: [sourceVolume, targetVolume],
    },
  };
}

export function storageStatus(availableBytes, minimumFreeBytes, pauseBelowBytes) {
  if (availableBytes < minimumFreeBytes) return "rejected";
  if (availableBytes < pauseBelowBytes) return "paused";
  return "admitted";
}

function observeVolume(configured) {
  const evidencePath = fs.realpathSync(configured.mount_path);
  const space = fs.statfsSync(evidencePath, { bigint: true });
  const availableBytes = Number(space.bavail * space.bsize);
  const filesystemId = filesystemIdentity(evidencePath);
  if (filesystemId !== configured.filesystem_id) {
    throw new Error(`${configured.role}: observed filesystem differs from the reviewed storage profile`);
  }
  return {
    role: configured.role,
    evidence_path: evidencePath,
    filesystem_id: filesystemId,
    available_bytes: availableBytes,
    minimum_free_bytes: configured.minimum_free_bytes,
    pause_below_bytes: configured.pause_below_bytes,
    status: storageStatus(availableBytes, configured.minimum_free_bytes, configured.pause_below_bytes),
  };
}

function assertAdmitted(volume) {
  if (volume.status !== "admitted") {
    throw new Error(`${volume.role}: storage admission is ${volume.status} at ${volume.available_bytes} available bytes`);
  }
}

function bindPath(role, candidate, expectedFilesystemId) {
  const canonical = canonicalPotentialPath(candidate);
  const filesystemId = filesystemIdentity(existingAncestor(canonical));
  if (filesystemId !== expectedFilesystemId) {
    throw new Error(`${role}: path is not on its reviewed storage volume`);
  }
  return { role, path: canonical, filesystem_id: filesystemId };
}

function filesystemIdentity(candidate) {
  const stats = fs.statSync(candidate, { bigint: true });
  return process.platform === "win32"
    ? `windows-volume:${stats.dev.toString(16).padStart(8, "0")}`
    : `posix-dev:${stats.dev}`;
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

function existingAncestor(target) {
  let candidate = target;
  while (!fs.existsSync(candidate)) {
    const parent = path.dirname(candidate);
    if (parent === candidate) throw new Error(`cannot resolve existing ancestor for ${target}`);
    candidate = parent;
  }
  return candidate;
}
