#!/usr/bin/env node
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { validateArtifactManifest } from "../runtime/builtin-example-verifier/artifacts.mjs";

export const PLATFORMS = ["linux-x86_64", "windows-x86_64", "macos-x86_64", "macos-aarch64"];
const REPOSITORY = "runmat-org/runmat";
const ARTIFACT_STEP = "Publish release artifacts";
const hash = bytes => `sha256:${createHash("sha256").update(bytes).digest("hex")}`;

export function latestJobs(jobs) {
  const groups = new Map();
  for (const job of jobs) {
    assert.ok(Number.isSafeInteger(job.run_attempt) && job.run_attempt > 0, "Missing job attempt");
    const attempts = groups.get(job.name) ?? new Map();
    assert.ok(!attempts.has(job.run_attempt), `Duplicate job in one attempt: ${job.name}`);
    attempts.set(job.run_attempt, job); groups.set(job.name, attempts);
  }
  return [...groups.values()].map(attempts => attempts.get(Math.max(...attempts.keys())));
}

export function validateFrozenProducts(run, jobs, { tag, source, completed = true }) {
  assert.match(tag, /^v\d+\.\d+\.\d+$/);
  assert.match(source, /^[a-f0-9]{40}$/);
  assert.equal(run.head_sha, source);
  assert.equal(run.head_repository?.full_name, REPOSITORY);
  assert.equal(run.head_branch, tag);
  assert.equal(run.path, ".github/workflows/release.yml");
  assert.equal(run.event, "push");
  if (completed) assert.equal(run.status, "completed", "Original release run must be terminal before finalization");
  for (const name of ["validate-release-state", "Verify qualified CI checkpoint and publication contracts",
    "Verify meshing catalogs and scientific release evidence", "Publish wasm bindings to npm / build-and-publish"]) {
    const matches = jobs.filter(job => job.name === name);
    assert.equal(matches.length, 1, `Missing or duplicate source gate: ${name}`);
    assert.equal(matches[0].conclusion, "success", name);
  }
  return PLATFORMS.map(platform => {
    const matches = jobs.filter(job => job.name === `Build ${platform}`);
    assert.equal(matches.length, 1, `Missing or duplicate native job: ${platform}`);
    const job = matches[0];
    assert.ok(Number.isSafeInteger(job.run_attempt) && job.run_attempt > 0, "Missing native job attempt");
    const boundary = job.steps.findIndex(step => step.name === ARTIFACT_STEP);
    assert.ok(boundary >= 0, `Missing product upload: ${platform}`);
    for (const step of job.steps.slice(0, boundary + 1)) {
      assert.equal(step.status, "completed", `${platform}: ${step.name}`);
      assert.ok(["success", "skipped"].includes(step.conclusion), `${platform}: ${step.name} did not pass`);
    }
    const system = platform.startsWith("windows") ? "Windows" : "Unix";
    for (const name of [`Build release CLI with embedded AOT runtime (${system})`,
      `Stage and accept distributed product (${system})`, `Package as archive (${system})`, ARTIFACT_STEP,
      ...(platform.startsWith("macos") ? ["Sign and notarize release artifact (macOS)"] : [])]) {
      assert.equal(job.steps.find(step => step.name === name)?.conclusion, "success", `${platform}: ${name}`);
    }
    // Cancellation is acceptable only after all build, signing, acceptance and
    // packaging work has passed. A cancelled compiler never satisfies this gate.
    for (const step of job.steps.slice(boundary + 1)) {
      assert.ok(step.name === "Save rolling native checkpoint" || step.name === "Upload rolling native checkpoint"
        || step.name.startsWith("Post ") || step.name === "Complete job", `Unexpected post-product step: ${step.name}`);
    }
    return { platform, job: job.id, attempt: job.run_attempt,
      artifact: `runmat-${tag}-${platform}-${job.run_attempt}` };
  });
}

export function validateRetirement(run, jobs, options) {
  const products = validateFrozenProducts(run, jobs, { ...options, completed: false });
  const allowed = new Set([...PLATFORMS.map(platform => `Build ${platform}`), "Create Release"]);
  for (const job of jobs) {
    if (job.status !== "completed") assert.ok(allowed.has(job.name), `Still-active work must finish before retirement: ${job.name}`);
  }
  return products;
}

export function validateAcceptance(manifest, receipt, source) {
  validateArtifactManifest(manifest, { sourceRevision: source });
  assert.equal(manifest.product, "native-cli");
  assert.equal(manifest.artifactProfile, "embedded-aot");
  assert.equal(receipt.status, "passed");
  assert.equal(receipt.artifactManifestDigest, manifest.artifactManifestDigest);
  assert.deepEqual(receipt.processDiagnostics, []);
  assert.equal(receipt.results.length, 5);
  assert.deepEqual(receipt.results.slice(0, 4).map(result => [result.name, result.exitCode, result.signal]), [
    ["first-launch", 0, null], ["scalar-division", 0, null], ["callback", 0, null], ["original-error", 1, null],
  ]);
  const probe = receipt.results[4];
  assert.equal(probe.sourceRevision, source);
  assert.equal(probe.artifactManifestDigest, manifest.artifactManifestDigest);
  assert.equal(probe.probeKind, "native-embedded-aot-compile-and-execute-v1");
  assert.deepEqual(probe.result, { kind: "compile-and-execute", status: "passed", failureKind: "none", prepareExitCode: 0, executeExitCode: 0 });
}

const json = filename => JSON.parse(fs.readFileSync(filename, "utf8"));
const api = route => JSON.parse(execFileSync("gh", ["api", `repos/${REPOSITORY}/${route}`],
  { encoding: "utf8", timeout: 30_000, maxBuffer: 16 * 1024 * 1024 }));
function pages(route, key) {
  const result = [];
  for (let page = 1; ; page++) {
    const response = api(`${route}${route.includes("?") ? "&" : "?"}per_page=100&page=${page}`);
    result.push(...response[key]);
    if (response[key].length < 100) return result;
  }
}

export function stageFrozenRelease({ runId, tag, source, sourceRoot, output, downloadRoot }) {
  assert.match(String(runId), /^[1-9]\d*$/);
  const run = api(`actions/runs/${runId}`);
  assert.equal(String(run.id), String(runId));
  const jobs = latestJobs(pages(`actions/runs/${runId}/jobs?filter=all`, "jobs"));
  const products = validateFrozenProducts(run, jobs, { tag, source });
  const artifacts = pages(`actions/runs/${runId}/artifacts`, "artifacts");
  assert.equal(execFileSync("git", ["rev-parse", "HEAD"], { cwd: sourceRoot, encoding: "utf8" }).trim(), source);
  assert.equal(execFileSync("git", ["rev-parse", `${tag}^{commit}`], { cwd: sourceRoot, encoding: "utf8" }).trim(), source);
  fs.mkdirSync(output, { recursive: false });
  fs.mkdirSync(downloadRoot, { recursive: false });
  const records = [];
  for (const product of products) {
    const matching = artifacts.filter(artifact => artifact.name === product.artifact && !artifact.expired);
    assert.equal(matching.length, 1, `Missing or duplicate exact artifact: ${product.artifact}`);
    assert.equal(matching[0].workflow_run?.head_sha, source);
    assert.equal(String(matching[0].workflow_run?.id), String(runId));
    const destination = path.join(downloadRoot, product.platform);
    execFileSync("gh", ["run", "download", String(runId), "-R", REPOSITORY, "-n", product.artifact, "-D", destination], { stdio: "inherit" });
    const root = path.join(destination, "artifacts/release-product", product.platform);
    const manifestPath = path.join(root, `${product.platform}.artifact-manifest.json`);
    const acceptancePath = path.join(root, `${product.platform}.acceptance.json`);
    const manifest = json(manifestPath);
    validateAcceptance(manifest, json(acceptancePath), source);
    const zip = product.platform.startsWith("windows");
    const filename = `runmat-${tag}-${product.platform}.${zip ? "zip" : "tar.gz"}`;
    const archive = path.join(destination, filename);
    const entries = execFileSync(zip ? "unzip" : "tar", zip ? ["-Z1", archive] : ["-tzf", archive], { encoding: "utf8", maxBuffer: 1024 * 1024 }).trim().split("\n").filter(entry => !entry.endsWith("/"));
    assert.deepEqual(entries.toSorted(), manifest.files.map(file => file.path).toSorted());
    const extraction = path.join(root, "product");
    fs.mkdirSync(extraction);
    execFileSync(zip ? "unzip" : "tar", zip ? ["-q", archive, "-d", extraction] : ["-xzf", archive, "-C", extraction]);
    validateArtifactManifest(manifest, { sourceRevision: source, verifyFiles: true, manifestPath });
    for (const license of ["LICENSE.md", "FREE-USE.md"]) assert.deepEqual(fs.readFileSync(path.join(extraction, license)), fs.readFileSync(path.join(sourceRoot, license)));
    for (const file of [archive, manifestPath, acceptancePath]) {
      fs.copyFileSync(file, path.join(output, path.basename(file)), fs.constants.COPYFILE_EXCL);
      records.push({ name: path.basename(file), size: fs.statSync(file).size, sha256: hash(fs.readFileSync(file)) });
    }
  }
  const evidence = artifacts.filter(artifact => artifact.name === `meshing-release-evidence-${runId}` && !artifact.expired);
  assert.equal(evidence.length, 1, "Missing or duplicate qualification evidence");
  assert.equal(evidence[0].workflow_run?.head_sha, source);
  assert.equal(String(evidence[0].workflow_run?.id), String(runId));
  const archive = execFileSync("gh", ["api", `repos/${REPOSITORY}/actions/artifacts/${evidence[0].id}/zip`], { maxBuffer: 32 * 1024 * 1024 });
  assert.equal(hash(archive), evidence[0].digest, "Qualification ZIP differs from GitHub artifact digest");
  const name = `runmat-${tag}-qualification-evidence.zip`;
  fs.writeFileSync(path.join(output, name), archive, { flag: "wx" });
  records.push({ name, size: archive.length, sha256: hash(archive) });
  const proof = { source, tag, runId, products, files: records };
  fs.writeFileSync(path.join(output, "release-publication.json"), JSON.stringify(proof, null, 2) + "\n", { flag: "wx" });
  return proof;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [runId, tag, source, sourceRoot, output, downloadRoot] = process.argv.slice(2);
  assert.ok(sourceRoot && output && downloadRoot, "Expected run ID, tag, source SHA, frozen checkout, asset directory and download directory");
  console.log(JSON.stringify(stageFrozenRelease({ runId, tag, source, sourceRoot, output, downloadRoot })));
}
