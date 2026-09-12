import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

import { contentDigest } from "../evidence.mjs";
import { sourceRevision } from "../schema.mjs";
import { gitTreeOid, signerFingerprint } from "./baseline-schema.mjs";

export function canonicalGitRepository(repository, gitOverride) {
  if (typeof repository !== "string" || !path.isAbsolute(repository)) {
    throw new Error("module composition baseline requires an absolute repository path");
  }
  const root = fs.realpathSync(repository);
  if (!fs.statSync(root).isDirectory()) throw new Error("module composition repository must be a directory");
  const git = gitOverride ?? defaultGit;
  const top = fs.realpathSync(git(root, ["rev-parse", "--show-toplevel"]).trim());
  if (top !== root) throw new Error("module composition repository must be the canonical Git worktree root");
  return root;
}

export function observeSignedCleanHead(root, trustedSignerFingerprint, options = {}) {
  const trusted = signerFingerprint(trustedSignerFingerprint, "trusted source signer fingerprint");
  const git = options.git ?? defaultGit;
  const revision = sourceRevision(`git:${git(root, ["rev-parse", "HEAD"]).trim().toLowerCase()}`, "module composition source revision");
  const tree = gitTreeOid(
    git(root, ["rev-parse", "HEAD^{tree}"]).trim().toLowerCase(),
    "module composition source tree",
  );
  const allowedArtifacts = allowedCompositionArtifacts(root, options.allowedCompositionArtifacts);
  const dirty = git(root, ["status", "--porcelain=v1", "--untracked-files=all"])
    .split("\n").filter(Boolean)
    .filter((line) => !(options.allowCompositionLock
      && line === "?? .runmat-module-composition.lock/owner.json"))
    .filter((line) => !allowedArtifacts.has(line));
  if (dirty.length) throw new Error("module composition baseline requires a clean source HEAD");
  git(root, ["verify-commit", "HEAD"]);
  const signature = git(root, ["log", "-1", "--format=%G?%x00%GF%x00%GP", "HEAD"]).trim().split("\0");
  if (!["G", "U"].includes(signature[0])) throw new Error("module composition source HEAD does not have a valid signature");
  if (!signature.slice(1).filter(Boolean).includes(trusted)) {
    throw new Error("module composition source HEAD was not signed by the trusted signer fingerprint");
  }
  return { revision, tree_oid: tree, signer_fingerprint: trusted };
}

function allowedCompositionArtifacts(root, artifacts) {
  if (artifacts === undefined) return new Set();
  if (!Array.isArray(artifacts)) throw new Error("allowed composition artifacts must be an array");
  const result = new Set();
  for (const artifact of artifacts) {
    if (typeof artifact !== "string" || !path.isAbsolute(artifact)) {
      throw new Error("allowed composition artifact must be an absolute path");
    }
    const relative = path.relative(root, artifact);
    if (!relative || relative === ".." || relative.startsWith(`..${path.sep}`)
      || path.isAbsolute(relative) || relative.includes("\n") || relative.includes("\r")) {
      throw new Error("allowed composition artifact must be beneath the repository");
    }
    result.add(`?? ${relative.split(path.sep).join("/")}`);
  }
  return result;
}

export function verifySignedHeadFiles(root, revision, files, options = {}) {
  const git = options.git ?? defaultGit;
  const platform = options.platform ?? process.platform;
  const commit = revision.slice("git:".length);
  const result = [];
  for (const file of files) {
    const treeEntry = parseTreeEntry(git(root, ["ls-tree", "-z", commit, "--", file.path]), file.path);
    if (treeEntry.type !== "blob") throw new Error(`${file.path}: signed source entry must be a Git blob`);
    if (!["100644", "100755"].includes(treeEntry.mode)) {
      throw new Error(`${file.path}: signed source entry has an unsupported Git mode`);
    }
    const expectedMode = file.working_executable ? "100755" : "100644";
    if (platform !== "win32" && treeEntry.mode !== expectedMode) {
      throw new Error(`${file.path}: working mode does not match the signed source HEAD`);
    }
    const committed = git(root, ["show", `${commit}:${file.path}`]);
    if (contentDigest(committed) !== file.content_digest) {
      throw new Error(`${file.path}: observed bytes do not match the signed source HEAD`);
    }
    result.push({
      path: file.path,
      git_mode: treeEntry.mode,
      content_digest: file.content_digest,
    });
  }
  return result;
}

function parseTreeEntry(value, expectedPath) {
  const rows = value.split("\0").filter(Boolean);
  if (rows.length !== 1) throw new Error(`${expectedPath}: signed source tree entry must resolve exactly once`);
  const match = /^(\d{6}) ([a-z]+) ([a-f0-9]{40,64})\t(.+)$/.exec(rows[0]);
  if (!match || match[4] !== expectedPath) throw new Error(`${expectedPath}: signed source tree entry is invalid`);
  return { mode: match[1], type: match[2] };
}

function defaultGit(repository, arguments_) {
  const result = spawnSync("git", arguments_, { cwd: repository, encoding: "utf8" });
  if (result.error || result.status !== 0) {
    throw new Error(`git ${arguments_.join(" ")} failed: ${result.error?.message ?? result.stderr.trim()}`);
  }
  return result.stdout;
}
