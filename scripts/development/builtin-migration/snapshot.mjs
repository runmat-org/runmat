import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

import { compareCodePoint } from "./constants.mjs";
import { contentDigest, evidenceDigest } from "./evidence.mjs";
import { array, digest, exact, repositoryPath, sourceRevision } from "./schema.mjs";
import { filesUnder } from "./source-scan.mjs";

export function sourceSnapshot(repository, roots, revision = null) {
  const canonicalRoots = [...new Set(roots)].sort(compareCodePoint);
  const indexModes = repositoryIndexModes(repository, canonicalRoots);
  const files = [...new Set(canonicalRoots.flatMap((root) => {
    const absolute = path.join(repository, root);
    return fs.existsSync(absolute) && fs.lstatSync(absolute).isFile() ? [root] : filesUnder(repository, root);
  }))].sort(compareCodePoint);
  const entries = files.map((sourcePath) => {
    const absolute = path.join(repository, sourcePath);
    const stat = fs.lstatSync(absolute);
    if (!stat.isFile()) throw new Error(`Source snapshot only accepts regular files: ${sourcePath}`);
    return {
      path: sourcePath,
      mode: canonicalRegularFileMode(stat.mode, indexModes, sourcePath),
      content_digest: contentDigest(fs.readFileSync(absolute)),
    };
  });
  const sourceRevision = revision ?? repositoryRevision(repository);
  return {
    revision: sourceRevision,
    dirty: repositoryDirty(repository, canonicalRoots),
    roots: canonicalRoots,
    files: entries,
    digest: evidenceDigest({ roots: canonicalRoots, files: entries }),
  };
}

export function parseSourceSnapshot(value, label = "source snapshot", { emptyFiles = false } = {}) {
  exact(value, ["revision", "dirty", "roots", "files", "digest"], label);
  sourceRevision(value.revision, `${label} revision`);
  if (![true, false, null].includes(value.dirty)) throw new Error(`${label} dirty must be boolean or null`);
  const roots = array(value.roots, `${label} roots`);
  roots.forEach((entry) => repositoryPath(entry, `${label} root`));
  if (JSON.stringify(roots) !== JSON.stringify([...new Set(roots)].sort(compareCodePoint))) {
    throw new Error(`${label} roots must be unique and canonically ordered`);
  }
  const files = array(value.files, `${label} files`, { empty: emptyFiles });
  files.forEach((entry) => {
    exact(entry, ["path", "mode", "content_digest"], `${label} file`);
    repositoryPath(entry.path, `${label} file path`);
    if (![0o644, 0o755].includes(entry.mode)) throw new Error(`${label} file mode must be canonical 0644 or 0755`);
    digest(entry.content_digest, `${label} file digest`);
  });
  const paths = files.map((entry) => entry.path);
  if (JSON.stringify(paths) !== JSON.stringify([...new Set(paths)].sort(compareCodePoint))) {
    throw new Error(`${label} files must be unique and canonically ordered`);
  }
  for (const sourcePath of paths) {
    if (!roots.some((root) => sourcePath === root || sourcePath.startsWith(`${root}/`))) {
      throw new Error(`${label} file is outside its declared roots: ${sourcePath}`);
    }
  }
  digest(value.digest, `${label} digest`);
  if (evidenceDigest({ roots, files }) !== value.digest) throw new Error(`${label} digest is inconsistent`);
  return value;
}

function canonicalRegularFileMode(mode, indexModes, sourcePath) {
  // Git's portable regular-file modes preserve executable intent only. Host
  // umasks may remove read/write bits from a checkout without changing the
  // committed source identity.
  if (indexModes !== null) {
    const indexMode = indexModes.get(sourcePath);
    if (indexMode === undefined) throw new Error(`Source snapshot file is absent from the Git index: ${sourcePath}`);
    if (indexMode === "100644") return 0o644;
    if (indexMode === "100755") return 0o755;
    throw new Error(`Source snapshot only accepts regular Git blobs, observed mode ${indexMode}`);
  }
  return (mode & 0o111) === 0 ? 0o644 : 0o755;
}

function repositoryIndexModes(repository, roots) {
  if (roots.length === 0) return new Map();
  if (!fs.existsSync(path.join(repository, ".git"))) return null;
  let inside;
  try {
    inside = execFileSync("git", ["rev-parse", "--is-inside-work-tree"], {
      cwd: repository,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
  } catch (error) {
    throw new Error(`Source snapshot could not validate Git repository identity: ${error.message}`);
  }
  if (inside !== "true") throw new Error("Source snapshot repository is not a Git worktree");
  let output;
  try {
    output = execFileSync("git", ["ls-files", "--stage", "-z", "--", ...roots], {
      cwd: repository,
      encoding: "utf8",
      maxBuffer: 64 * 1024 * 1024,
      stdio: ["ignore", "pipe", "pipe"],
    });
  } catch (error) {
    const detail = error?.stderr?.trim() || error.message;
    throw new Error(`Source snapshot could not read the Git index: ${detail}`);
  }
  const result = new Map();
  for (const record of output.split("\0").filter(Boolean)) {
    const match = /^(\d{6}) [a-f0-9]+ (\d)\t([\s\S]+)$/.exec(record);
    if (!match || match[2] !== "0" || result.has(match[3])) {
      throw new Error("Source snapshot encountered an invalid or conflicted Git index entry");
    }
    if (!["100644", "100755"].includes(match[1])) {
      throw new Error(`Source snapshot only accepts regular Git blobs, observed mode ${match[1]} for ${match[3]}`);
    }
    result.set(match[3], match[1]);
  }
  return result;
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

function repositoryDirty(repository, roots) {
  if (roots.length === 0) return false;
  try {
    return execFileSync("git", ["status", "--porcelain=v1", "--untracked-files=all", "--", ...roots], {
      cwd: repository, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"],
    }).length > 0;
  } catch {
    return null;
  }
}
