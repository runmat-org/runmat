import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

import { assertControlSubject } from "./control.mjs";
import { digest, exact, sourceRevision } from "./schema.mjs";
import { sourceSnapshot } from "./snapshot.mjs";

export function leaseBaseInventoryBinding(inventory) {
  return {
    source_revision: inventory.source.revision,
    source_digest: inventory.source.digest,
    inventory_digest: inventory.digest,
    compiled_inventory_digest: inventory.compiled_inventory.digest,
  };
}

export function parseLeaseBaseInventoryBinding(value, label) {
  exact(value, ["source_revision", "source_digest", "inventory_digest", "compiled_inventory_digest"], label);
  return {
    source_revision: sourceRevision(value.source_revision, `${label} source revision`),
    source_digest: digest(value.source_digest, `${label} source digest`),
    inventory_digest: digest(value.inventory_digest, `${label} inventory digest`),
    compiled_inventory_digest: digest(value.compiled_inventory_digest, `${label} compiled inventory digest`),
  };
}

export function validateLeaseBaseInventory(repository, inventory, control, request) {
  assertControlSubject(control, inventory);
  if (inventory.source.dirty !== false) throw new Error("lease base inventory must describe a clean source snapshot");
  if (JSON.stringify(leaseBaseInventoryBinding(inventory)) !== JSON.stringify(request.lease_base_inventory)) {
    throw new Error("lease request base inventory binding differs from the supplied inventory");
  }
  const observedSource = sourceSnapshot(repository, inventory.generated_from, inventory.source.revision);
  if (observedSource.dirty !== false || observedSource.digest !== inventory.source.digest
    || JSON.stringify(observedSource.files) !== JSON.stringify(inventory.source.files)) {
    throw new Error("lease base inventory source differs from the clean integration repository");
  }
  return inventory;
}

export function validateAcceptedIntegrationCheckpoint(repository, baseRevision, base, frozen, sealedBundles) {
  if (sealedBundles.length === 0) {
    if (baseRevision !== frozen.revision || base.digest !== frozen.inventory_digest) {
      throw new Error("first lease base must equal the exact frozen control baseline inventory");
    }
    return;
  }
  let exactCheckpoint = false;
  for (const sealed of sealedBundles) {
    assertAncestor(repository, sealed.integrated_revision, baseRevision,
      `${sealed.reference.bundle_id}: lease base does not descend from its accepted seal`);
    if (sealed.integrated_revision === baseRevision) {
      exactCheckpoint = true;
      if (sealed.source_digest !== base.source.digest
        || sealed.subject_inventory_digest !== base.digest) {
        throw new Error(`${sealed.reference.bundle_id}: lease base source or inventory differs from its accepted seal`);
      }
    }
  }
  if (!exactCheckpoint) throw new Error("lease base must equal an accepted sealed integration checkpoint");
}

export function assertIssuableIntegrationBase(repository, baseRevision, controlBaseline) {
  const repositoryRoot = canonicalRepository(repository);
  const head = repositoryRevision(repositoryRoot, "HEAD");
  assertRecordedIntegrationBase(repositoryRoot, baseRevision, controlBaseline);
  if (head !== baseRevision) throw new Error("lease request base revision is stale; it must equal the current integration HEAD");
  const status = execFileSync("git", ["status", "--porcelain=v1", "--untracked-files=all"], {
    cwd: repositoryRoot,
    encoding: "utf8",
  });
  if (status.length !== 0) throw new Error("lease integration base must have a clean worktree");
}

export function assertRecordedIntegrationBase(repository, baseRevision, controlBaseline) {
  const repositoryRoot = canonicalRepository(repository);
  repositoryRevision(repositoryRoot, baseRevision);
  repositoryRevision(repositoryRoot, controlBaseline);
  assertAncestor(repositoryRoot, controlBaseline, baseRevision, "lease base does not descend from the frozen control baseline");
}

function canonicalRepository(repository) {
  if (typeof repository !== "string" || !path.isAbsolute(repository)) {
    throw new Error("lease issuance requires an absolute integration repository path");
  }
  const canonical = fs.realpathSync(repository);
  const root = fs.realpathSync(execFileSync("git", ["rev-parse", "--show-toplevel"], {
    cwd: canonical,
    encoding: "utf8",
  }).trim());
  if (root !== canonical) throw new Error("lease integration repository must be the canonical Git worktree root");
  return canonical;
}

function repositoryRevision(repository, revision) {
  const raw = revision.startsWith("git:") ? revision.slice("git:".length) : revision;
  const resolved = execFileSync("git", ["rev-parse", "--verify", `${raw}^{commit}`], {
    cwd: repository,
    encoding: "utf8",
  }).trim().toLowerCase();
  const result = `git:${resolved}`;
  sourceRevision(result, "repository revision");
  if (revision !== "HEAD" && result !== revision) throw new Error("lease base revision does not resolve exactly");
  return result;
}

function assertAncestor(repository, ancestor, descendant, message) {
  const result = spawnSync("git", [
    "merge-base", "--is-ancestor",
    ancestor.slice("git:".length), descendant.slice("git:".length),
  ], { cwd: repository, encoding: "utf8" });
  if (result.error) throw new Error(`could not verify lease ancestry: ${result.error.message}`);
  if (result.status === 1) throw new Error(message);
  if (result.status !== 0) throw new Error(`could not verify lease ancestry: ${result.stderr.trim() || `git exited ${result.status}`}`);
}
