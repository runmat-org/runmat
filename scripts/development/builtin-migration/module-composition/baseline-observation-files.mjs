import fs from "node:fs";
import path from "node:path";

import { compareCodePoint } from "../constants.mjs";
import { contentDigest } from "../evidence.mjs";
import { repositoryPath } from "../schema.mjs";

export function fileEvidence(root, sourcePath) {
  const target = resolve(root, sourcePath);
  const stat = fs.lstatSync(target);
  if (!stat.isFile() || fs.realpathSync(target) !== target) throw new Error(`${sourcePath}: source must be a canonical regular file`);
  return {
    path: sourcePath,
    working_executable: Boolean(stat.mode & 0o111),
    content_digest: contentDigest(fs.readFileSync(target)),
  };
}

export function resolve(root, sourcePath) {
  const target = path.resolve(root, ...repositoryPath(sourcePath, "module composition source path").split("/"));
  const relative = path.relative(root, target);
  if (relative === ".." || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) throw new Error(`${sourcePath}: source escapes repository`);
  const existing = nearestExisting(target);
  const canonical = fs.realpathSync(existing);
  const canonicalRelative = path.relative(root, canonical);
  if (canonicalRelative === ".." || canonicalRelative.startsWith(`..${path.sep}`) || path.isAbsolute(canonicalRelative)) {
    throw new Error(`${sourcePath}: source escapes repository through a symbolic link`);
  }
  if (canonical !== existing) throw new Error(`${sourcePath}: source has a noncanonical repository ancestor`);
  return target;
}

export function uniqueEvidence(values) {
  const byPath = new Map();
  for (const value of values) {
    const prior = byPath.get(value.path);
    if (prior && JSON.stringify(prior) !== JSON.stringify(value)) throw new Error(`${value.path}: conflicting source evidence`);
    byPath.set(value.path, value);
  }
  return [...byPath.values()].sort((left, right) => compareCodePoint(left.path, right.path));
}

export function lstat(target) {
  try { return fs.lstatSync(target); }
  catch (error) { if (["ENOENT", "ENOTDIR"].includes(error?.code)) return null; throw error; }
}

function nearestExisting(target) {
  let current = target;
  while (!fs.existsSync(current)) {
    const parent = path.dirname(current);
    if (parent === current) throw new Error(`no existing ancestor for ${target}`);
    current = parent;
  }
  return current;
}
