#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";

const prefix = "architecture boundary check failed: ";
const feaRoots = [
  "crates/runmat-analysis/core/",
  "crates/runmat-analysis/language/",
  "crates/runmat-analysis/fea/",
  "crates/runmat-runtime/src/analysis/",
];

// Git's blob identity for the exact UTF-8 ls-tree output. Hashing in-process
// avoids a large synchronous stdin transfer to a second Git process.
export function gitBlobDigest(text) {
  const bytes = Buffer.from(text, "utf8");
  return createHash("sha1").update(`blob ${bytes.length}\0`).update(bytes).digest("hex");
}

// Counts matter: duplicate diagnostics from independent rules remain visible.
export function matchesReviewedDebt(actual, expected) {
  if (!Array.isArray(expected) || expected.length !== 107 || !actual.length) return false;
  const counts = new Map();
  for (const line of expected) {
    if (typeof line !== "string" || !line.startsWith(prefix)) return false;
    const source = line.slice(prefix.length).split(" ", 1)[0];
    if (!feaRoots.some((root) => source.startsWith(root))) return false;
    counts.set(line, (counts.get(line) ?? 0) + 1);
  }
  for (const line of actual) {
    const count = counts.get(line) ?? 0;
    if (!count) return false;
    if (count === 1) counts.delete(line);
    else counts.set(line, count - 1);
  }
  return counts.size === 0;
}

function main() {
  const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
  const run = (command, args, options = {}) => spawnSync(command, args, {
    cwd: repo, encoding: "utf8", maxBuffer: 16 * 1024 * 1024, ...options,
  });
  // Always execute both original checks. Debt never hides the language ledger.
  const raw = run(process.execPath, ["scripts/development/check-architecture-boundaries.mjs"]);
  process.stdout.write(raw.stdout ?? "");
  process.stderr.write(raw.stderr ?? "");
  const language = run(process.execPath, ["scripts/fea/check-language-surface-ledger.mjs"], { stdio: "inherit" });
  if (language.status !== 0 || language.error || language.signal) return 1;
  if (raw.status === 0 && !raw.error && !raw.signal) return 0;
  if (raw.status !== 1 || raw.error || raw.signal || raw.stdout?.trim()) return 1;
  let reviewed;
  try {
    reviewed = JSON.parse(fs.readFileSync(path.join(repo, "qualification/quality/fea-architecture-debt-0.7.json"), "utf8"));
  } catch {
    return 1;
  }
  const inputs = ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo", "crates", "scripts", "docs", "bindings", "verification"];
  const git = (args, options) => run("git", args, options);
  const validCommit = /^[0-9a-f]{40}$/.test(reviewed.reviewed_commit ?? "");
  const ancestor = validCommit ? git(["merge-base", "--is-ancestor", reviewed.reviewed_commit, "HEAD"]) : null;
  const reviewedTree = validCommit ? git(["ls-tree", "-r", reviewed.reviewed_commit, "--", ...inputs]) : null;
  const reviewedDigest = reviewedTree?.status === 0 ? gitBlobDigest(reviewedTree.stdout) : null;
  const clean = git(["diff", "--quiet", "HEAD", "--", ...inputs]);
  const untracked = git(["ls-files", "--others", "--exclude-standard", "--", ...inputs]);
  const rustc = run("rustc", ["--version"]);
  const lines = raw.stderr.trimEnd().split("\n");
  const pinsMatch = reviewed.schema_version === 1 && reviewed.issue === "RM-1287"
    && validCommit && ancestor?.status === 0
    && /^[0-9a-f]{40}$/.test(reviewed.architecture_inputs ?? "")
    && reviewedDigest === reviewed.architecture_inputs
    && clean.status === 0 && untracked.status === 0 && !untracked.stdout.trim()
    && rustc.status === 0 && rustc.stdout.trim() === reviewed.rustc_version;
  if (!pinsMatch || !matchesReviewedDebt(lines, reviewed.diagnostics)) {
    console.error("Reviewed FEA architecture exception refused: review anchor, toolchain or diagnostic multiset differs.");
    return 1;
  }
  console.error("Reviewed RM-1287 FEA architecture debt: 107 findings retained above; all other architecture and language-ledger checks passed.");
  return 0;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.exitCode = main();
}
