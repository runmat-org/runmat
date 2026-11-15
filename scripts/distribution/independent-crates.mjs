#!/usr/bin/env node
import assert from "node:assert/strict";
import { execFileSync, spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { setTimeout as sleep } from "node:timers/promises";
import { publishWithBackoff } from "../publish-crates.mjs";

export function independentPlan(metadata, published, version, blocked) {
  const members = new Set(metadata.workspace_members);
  const workspace = metadata.packages.filter(pkg => members.has(pkg.id));
  const names = new Set(workspace.map(pkg => pkg.name));
  assert.equal(names.size, workspace.length, "Duplicate workspace package name");
  const packages = workspace.filter(pkg => pkg.publish === null || pkg.publish.length > 0);
  assert.ok(packages.length > 0);
  for (const pkg of packages) assert.equal(pkg.version, version, pkg.name);
  const available = new Set(published);
  let remaining = packages.filter(pkg => !available.has(pkg.name));
  const plan = [];
  for (;;) {
    const candidate = remaining.find(pkg => !blocked.has(pkg.name)
      && pkg.dependencies.every(dep => !names.has(dep.name) || available.has(dep.name)));
    if (!candidate) break;
    plan.push(candidate);
    available.add(candidate.name);
    remaining = remaining.filter(pkg => pkg !== candidate);
  }
  return { packages, plan, deferred: remaining.map(pkg => ({ name: pkg.name,
    blocked: blocked.has(pkg.name), dependencies: [...new Set(pkg.dependencies
      .filter(dep => names.has(dep.name) && !available.has(dep.name)).map(dep => dep.name))] })) };
}

function publishPackage(name, sourceRoot) {
  return new Promise((resolve, reject) => {
    const child = spawn("cargo", ["publish", "--package", name, "--allow-dirty", "--no-verify"],
      { cwd: sourceRoot, stdio: ["ignore", "pipe", "pipe"] });
    let output = "";
    for (const [stream, destination] of [[child.stdout, process.stdout], [child.stderr, process.stderr]]) {
      stream.on("data", chunk => {
        destination.write(chunk);
        output = (output + chunk.toString()).slice(-256 * 1024);
      });
    }
    child.on("error", reject);
    child.on("close", (code, signal) => resolve({ code: code ?? signal ?? "unknown", output }));
  });
}

export async function publishPlan(plan, { exists, publish, wait = sleep, onPublished = () => {} }) {
  for (const pkg of plan) {
    if (await exists(pkg.name)) continue;
    await publishWithBackoff({ run: () => publish(pkg.name), wait });
    let visible = false;
    for (let attempt = 0; attempt < 45 && !visible; attempt++) {
      visible = await exists(pkg.name);
      if (!visible) await wait(2_000);
    }
    assert.ok(visible, `${pkg.name}: Cargo returned success but registry version remains absent`);
    await onPublished(pkg.name);
  }
}

export async function publishIndependent({ sourceRoot, source, tag, output, request = fetch,
  publish = name => publishPackage(name, sourceRoot), wait = sleep }) {
  assert.match(source, /^[a-f0-9]{40}$/); assert.match(tag, /^v\d+\.\d+\.\d+$/);
  const git = (...args) => execFileSync("git", args, { cwd: sourceRoot, encoding: "utf8" }).trim();
  assert.equal(git("rev-parse", "HEAD"), source);
  assert.equal(git("rev-parse", `${tag}^{commit}`), source);
  const metadata = JSON.parse(execFileSync("cargo", ["metadata", "--locked", "--no-deps", "--format-version", "1"],
    { cwd: sourceRoot, encoding: "utf8", maxBuffer: 16 * 1024 * 1024 }));
  const version = tag.slice(1);
  const { packages } = independentPlan(metadata, new Set(), version, new Set());
  async function exists(name) {
    assert.match(name, /^[a-zA-Z0-9_-]+$/);
    const response = await request(`https://crates.io/api/v1/crates/${name}/${version}?verification=${Date.now()}`, {
      headers: { "User-Agent": `RunMat-Release-Publication/${version} (https://github.com/runmat-org/runmat)` },
      cache: "no-store", signal: AbortSignal.timeout(60_000),
    });
    if (response.status === 404) return false;
    assert.ok(response.ok, `${name}: registry HTTP ${response.status}`);
    const record = (await response.json()).version;
    assert.equal(record.num, version); assert.equal(record.yanked, false);
    return true;
  }
  const published = new Set();
  for (const pkg of packages) {
    if (await exists(pkg.name)) published.add(pkg.name);
    await wait(100);
  }
  // This is the separate HTTP 413 blocker. Its dependents remain deferred;
  // optional, target-specific and dev dependencies all participate in the plan.
  const { plan, deferred } = independentPlan(metadata, published, version, new Set(["runmat-runtime"]));
  const proof = { source, tag, status: "in_progress", previouslyPublished: [...published].toSorted(),
    planned: plan.map(pkg => pkg.name), newlyPublished: [], deferred };
  const record = () => fs.writeFileSync(output, JSON.stringify(proof, null, 2) + "\n");
  record();
  for (const pkg of plan) {
    assert.ok(path.resolve(pkg.manifest_path).startsWith(`${path.resolve(sourceRoot)}${path.sep}`));
  }
  await publishPlan(plan, { exists, publish, wait, onPublished: name => {
    proof.newlyPublished.push(name);
    record();
  } });
  proof.status = "independent_subset_published";
  record();
  return proof;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  if (!process.env.CARGO_REGISTRY_TOKEN) throw new Error("CARGO_REGISTRY_TOKEN is required");
  const [sourceRoot, source, tag, output] = process.argv.slice(2);
  assert.ok(sourceRoot && source && tag && output);
  console.log(JSON.stringify(await publishIndependent({ sourceRoot, source, tag, output })));
}
