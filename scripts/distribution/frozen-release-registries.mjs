#!/usr/bin/env node
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

export function expectedPackages(metadata, version) {
  const members = new Set(metadata.workspace_members);
  const packages = metadata.packages.filter(pkg => members.has(pkg.id) && (pkg.publish === null || pkg.publish.length > 0));
  assert.ok(packages.length > 0);
  for (const pkg of packages) { assert.equal(pkg.version, version); assert.match(pkg.name, /^[a-zA-Z0-9_-]+$/); }
  assert.equal(new Set(packages.map(pkg => pkg.name)).size, packages.length);
  return packages;
}
const digest = bytes => createHash("sha256").update(bytes).digest("hex");
export function sdkPublicationProof({ source, tag, npmIntegrity, npm, tags }) {
  assert.match(source, /^[a-f0-9]{40}$/); assert.match(tag, /^v\d+\.\d+\.\d+$/);
  assert.match(npmIntegrity, /^sha512-[A-Za-z0-9+/]+={0,2}$/);
  const version = tag.slice(1);
  assert.equal(npm.name, "runmat"); assert.equal(npm.version, version); assert.equal(npm.dist.integrity, npmIntegrity);
  if (npm.gitHead !== undefined) assert.equal(npm.gitHead, source);
  assert.equal(tags.prod, version); assert.equal(tags.latest, version);
  return { status: "passed", source, tag, npm: { version, integrity: npmIntegrity,
    tags: { prod: tags.prod, latest: tags.latest } } };
}
export async function verifyRegistries({ sourceRoot, source, tag, npmIntegrity, output, cache, npmOnly = false, request = fetch }) {
  assert.match(source, /^[a-f0-9]{40}$/); assert.match(tag, /^v\d+\.\d+\.\d+$/);
  assert.match(npmIntegrity, /^sha512-[A-Za-z0-9+/]+={0,2}$/);
  assert.equal(execFileSync("git", ["rev-parse", "HEAD"], { cwd: sourceRoot, encoding: "utf8" }).trim(), source);
  assert.equal(typeof npmOnly, "boolean");
  const version = tag.slice(1);
  fs.mkdirSync(cache, { recursive: true });
  async function get(url) {
    const response = await request(url, { headers: { "User-Agent": `RunMat-Release-Verification/${version} (https://github.com/runmat-org/runmat)` }, signal: AbortSignal.timeout(60_000) });
    assert.ok(response.ok, `${url}: HTTP ${response.status}`); return response;
  }
  const npm = await (await get(`https://registry.npmjs.org/runmat/${version}`)).json();
  const tags = await (await get("https://registry.npmjs.org/-/package/runmat/dist-tags")).json();
  const sdk = sdkPublicationProof({ source, tag, npmIntegrity, npm, tags });
  if (npmOnly) {
    fs.writeFileSync(output, JSON.stringify(sdk, null, 2) + "\n", { flag: "wx" });
    return sdk;
  }
  const metadata = JSON.parse(execFileSync("cargo", ["metadata", "--locked", "--no-deps", "--format-version", "1"], { cwd: sourceRoot, encoding: "utf8", maxBuffer: 16 * 1024 * 1024 }));
  const packages = expectedPackages(metadata, version);
  const readTar = (archive, member) => execFileSync("tar", ["-xOf", archive, member], { maxBuffer: 1024 * 1024 });
  const results = [], queue = [...packages];
  async function worker() {
    while (queue.length) {
      const pkg = queue.shift();
      const metadata = await (await get(`https://crates.io/api/v1/crates/${pkg.name}`)).json();
      const record = metadata.versions.find(record => record.num === version);
      assert.ok(record, `${pkg.name}@${version} is not published`); assert.equal(record.yanked, false);
      assert.match(record.checksum, /^[a-f0-9]{64}$/);
      const archive = path.join(cache, `${pkg.name}-${version}.crate`);
      if (!fs.existsSync(archive)) fs.writeFileSync(archive, Buffer.from(await (await get(`https://static.crates.io/crates/${pkg.name}/${pkg.name}-${version}.crate`)).arrayBuffer()));
      assert.equal(digest(fs.readFileSync(archive)), record.checksum);
      const prefix = `${pkg.name}-${version}/`, vcs = JSON.parse(readTar(archive, `${prefix}.cargo_vcs_info.json`));
      assert.equal(vcs.git.sha1, source, `${pkg.name}: source identity differs`);
      const license = fs.readFileSync(path.join(sourceRoot, pkg.name === "runmat-clarabel" ? "crates/runmat-clarabel/LICENSE.md" : "LICENSE.md"));
      const files = execFileSync("tar", ["-tzf", archive], { encoding: "utf8", maxBuffer: 8 * 1024 * 1024 }).trim().split("\n");
      assert.ok(files.some(file => /\/LICENSE(?:\.md)?$/.test(file) && readTar(archive, file).equals(license)), `${pkg.name}: license differs`);
      if (pkg.name !== "runmat-clarabel") assert.deepEqual(readTar(archive, `${prefix}FREE-USE.md`), fs.readFileSync(path.join(sourceRoot, "FREE-USE.md")));
      results.push({ name: pkg.name, checksum: record.checksum, source: vcs.git.sha1, dirty: vcs.git.dirty });
      await new Promise(resolve => setTimeout(resolve, 200));
    }
  }
  await Promise.all(Array.from({ length: 4 }, worker));
  const proof = { ...sdk, crates: { expected: packages.length, verified: results.length,
    files: results.toSorted((a, b) => a.name.localeCompare(b.name)) } };
  fs.writeFileSync(output, JSON.stringify(proof, null, 2) + "\n", { flag: "wx" });
  return proof;
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [sourceRoot, source, tag, npmIntegrity, output, cache, mode = "complete"] = process.argv.slice(2);
  assert.ok(["complete", "npm-only"].includes(mode), "Expected complete or npm-only registry verification");
  console.log(JSON.stringify(await verifyRegistries({ sourceRoot, source, tag, npmIntegrity, output, cache, npmOnly: mode === "npm-only" })));
}
