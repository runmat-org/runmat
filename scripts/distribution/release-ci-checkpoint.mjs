#!/usr/bin/env node
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import { pathToFileURL } from "node:url";

export const BASELINE = "0c8e1e15ed26b501846021c2b6076bfac5d6253d";
export const CI_RUN = 43367308557;
const REPOSITORY = "runmat-org/runmat";
const COHORTS = ["authored-cli-physics", "driver-recovery-physics", "circuit-large-board-performance", "cad-airliner-drawing-memory"];
const MANIFESTS = new Set(["crates/runmat-cad/wasm/Cargo.toml", "crates/runmat-design-tools/Cargo.toml"]);
const ALLOWED = new Set([
  ...MANIFESTS, ".github/workflows/ci.yml", ".github/workflows/release.yml",
  "scripts/verify-release-version.mjs", "scripts/publish-crates.mjs", "scripts/publish-crates.test.mjs",
  "scripts/distribution/release-ci-checkpoint.mjs", "scripts/distribution/release-ci-checkpoint.test.mjs",
  "qualification/scientific/evidence/publication-metadata-20261113.json",
  ...COHORTS.flatMap(name => [
    `qualification/scientific/receipts/${name}.json`,
    `qualification/scientific/reviews/publication-metadata-20261113-${name}.json`,
  ]),
]);

export function validateDelta(changes, before, after) {
  for (const change of changes) {
    assert(ALLOWED.has(change.path), `Full CI required for unqualified input: ${change.path}`);
    assert(change.beforeMode === null || ["100644", "100755"].includes(change.beforeMode), `Unexpected input mode: ${change.path}`);
    assert.equal(change.afterMode, change.beforeMode ?? "100644", `Unexpected input mode/deletion: ${change.path}`);
    if (MANIFESTS.has(change.path)) {
      assert.equal(after(change.path), before(change.path).replace(/^publish = false\n/m, ""),
        `Full CI required for dependency, feature or other manifest changes: ${change.path}`);
    }
    if (change.path === ".github/workflows/ci.yml") {
      const normalized = after(change.path).replace(/^          scripts\/publish-crates\.test\.mjs\n/m, "")
        .replace(/^          scripts\/distribution\/release-ci-checkpoint\.test\.mjs\n/m, "");
      assert.equal(normalized, before(change.path), "Full CI required for changed ordinary test configuration");
    }
    if (change.path === ".github/workflows/release.yml") {
      const nativeBuild = text => text.match(/^  build-release:\n[\s\S]*?(?=^  create-release:)/m)?.[0];
      const original = nativeBuild(before(change.path));
      assert(original, "Missing native release build definition");
      assert.equal(nativeBuild(after(change.path)), original, "Native release build policy changed");
    }
  }
}

export function requiredJobs(ci) {
  const groups = [...ci.matchAll(/^            name: ([a-z][a-z0-9-]+)\s*$/gm)].map(match => match[1]);
  assert.equal(groups.length, 17, "Unexpected baseline test matrix");
  return [...groups, "changes", "distribution-contracts", "Verify committed scientific qualification", "docs-lint",
    "Cross build macos-aarch64", "Cross build macos-x86_64", "Cross build linux-x86_64", "Cross build windows-x86_64",
    "runtime-artifacts-ready"];
}

export function validateRun(run, jobs, expected) {
  assert.equal(run.id, CI_RUN, "Wrong CI run");
  assert.equal(run.head_sha, BASELINE, "CI source does not match the qualified checkpoint");
  assert.equal(run.head_repository?.full_name, REPOSITORY, "CI ran on a different repository");
  assert.equal(run.event, "push", "CI was not the main push qualification");
  assert.equal(run.head_branch, "main", "CI was not the main qualification");
  assert.equal(run.path, ".github/workflows/ci.yml", "Unexpected CI workflow");
  assert.equal(run.status, "completed", "CI is not finished");
  assert.equal(run.conclusion, "success", "CI did not pass");
  assert.equal(jobs.length, expected.length, "Incomplete or unexpected CI job inventory");
  const names = new Set(jobs.map(job => job.name));
  assert.equal(names.size, jobs.length, "Duplicate CI job identity");
  for (const name of expected) {
    const job = jobs.find(job => job.name === name);
    assert(job, `Missing CI job: ${name}`);
    assert.equal(job.status, "completed", `CI job unfinished: ${name}`);
    assert.equal(job.conclusion, "success", `CI job did not pass: ${name}`);
  }
}

const git = (...args) => execFileSync("git", args, { encoding: "utf8", maxBuffer: 32 * 1024 * 1024 });
const api = route => JSON.parse(execFileSync("gh", ["api", `repos/${REPOSITORY}/${route}`],
  { encoding: "utf8", timeout: 30_000, maxBuffer: 8 * 1024 * 1024 }));
function entries(ref) {
  return Object.fromEntries(git("ls-tree", "-r", "-z", ref).split("\0").filter(Boolean).map(row => {
    const [meta, name] = row.split("\t"); return [name, meta.split(" ")];
  }));
}

export function verifyCheckpoint() {
  const source = git("rev-parse", "HEAD").trim();
  git("merge-base", "--is-ancestor", BASELINE, source);
  const original = entries(BASELINE), current = entries(source);
  const changed = [...new Set([...Object.keys(original), ...Object.keys(current)])]
    .filter(name => JSON.stringify(original[name]) !== JSON.stringify(current[name]))
    .sort().map(name => ({ path: name, beforeMode: original[name]?.[0] ?? null, afterMode: current[name]?.[0] ?? null }));
  validateDelta(changed, name => git("show", `${BASELINE}:${name}`), name => git("show", `${source}:${name}`));
  const ci = git("show", `${BASELINE}:.github/workflows/ci.yml`);
  const run = api(`actions/runs/${CI_RUN}`), jobs = [];
  for (let page = 1; ; page++) {
    const response = api(`actions/runs/${CI_RUN}/jobs?filter=latest&per_page=100&page=${page}`);
    jobs.push(...response.jobs);
    if (response.jobs.length < 100) break;
  }
  validateRun(run, jobs, requiredJobs(ci));
  const proof = { schema_version: 1, source_revision: source, source_tree: git("rev-parse", "HEAD^{tree}").trim(),
    baseline_revision: BASELINE, ci_run: CI_RUN, ci_attempt: run.run_attempt, ci_url: run.html_url,
    changed_paths: changed.map(change => change.path),
    jobs: jobs.map(job => ({ id: job.id, name: job.name, conclusion: job.conclusion, url: job.html_url })),
    scope: "Passing full baseline CI; executable source, dependencies, features and native build policy unchanged. Release publication tooling and publish metadata are checked separately." };
  fs.writeFileSync("release-ci-checkpoint.json", JSON.stringify(proof, null, 2) + "\n");
  console.log(`Verified ${jobs.length} successful CI jobs at ${BASELINE}; ${changed.length} publication-only paths at ${source}.`);
  return proof;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) verifyCheckpoint();
