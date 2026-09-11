import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

import { compareCodePoint } from "./constants.mjs";
import { contentDigest, evidenceDigest } from "./evidence.mjs";
import { filesUnder } from "./source-scan.mjs";

export function sourceSnapshot(repository, roots, revision = null) {
  const files = [...new Set(roots.flatMap((root) => {
    const absolute = path.join(repository, root);
    return fs.existsSync(absolute) && fs.lstatSync(absolute).isFile() ? [root] : filesUnder(repository, root);
  }))].sort(compareCodePoint);
  const entries = files.map((sourcePath) => {
    const absolute = path.join(repository, sourcePath);
    const stat = fs.lstatSync(absolute);
    if (!stat.isFile()) throw new Error(`Source snapshot only accepts regular files: ${sourcePath}`);
    return {
      path: sourcePath,
      mode: stat.mode & 0o777,
      content_digest: contentDigest(fs.readFileSync(absolute)),
    };
  });
  const sourceRevision = revision ?? repositoryRevision(repository);
  const canonicalRoots = [...new Set(roots)].sort(compareCodePoint);
  return {
    revision: sourceRevision,
    dirty: repositoryDirty(repository),
    roots: canonicalRoots,
    files: entries,
    digest: evidenceDigest({ roots: canonicalRoots, files: entries }),
  };
}

export function repositoryRevision(repository) {
  try {
    const revision = execFileSync("git", ["rev-parse", "HEAD"], { cwd: repository, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim().toLowerCase();
    if (/^[a-f0-9]{40}$/.test(revision)) return `git:${revision}`;
  } catch {
    // Synthetic test repositories intentionally have no Git metadata.
  }
  return null;
}

function repositoryDirty(repository) {
  try {
    return execFileSync("git", ["status", "--porcelain=v1", "--untracked-files=all"], {
      cwd: repository, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"],
    }).length > 0;
  } catch {
    return null;
  }
}
