#!/usr/bin/env node
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const REPOSITORY = "runmat-org/runmat";
const CRATES_PENDING_NOTICE = "\n\nCrates.io publication is in progress.";
export function reconcileAssets(existing, expected, complete = false) {
  assert.equal(new Set(expected.map(file => file.name)).size, expected.length, "Duplicate staged asset");
  assert.equal(new Set(existing.map(file => file.name)).size, existing.length, "Duplicate published asset");
  for (const asset of existing) {
    const file = expected.find(file => file.name === asset.name);
    assert.ok(file, `Unexpected existing release asset: ${asset.name}`);
    assert.equal(asset.size, file.size, `Existing release asset size differs: ${asset.name}`);
    assert.equal(asset.digest, file.sha256, `Existing release asset bytes differ: ${asset.name}`);
  }
  const missing = expected.filter(file => !existing.some(asset => asset.name === file.name));
  if (complete) assert.equal(missing.length, 0, "Release upload is incomplete");
  return missing;
}

export async function publishStagedRelease({ directory, tag, source, deferCrates = false, token = process.env.GH_TOKEN,
  request = fetch, upload = files => execFileSync("gh", ["release", "upload", tag, "-R", REPOSITORY, ...files], { stdio: "inherit" }) }) {
  assert.match(tag, /^v\d+\.\d+\.\d+$/);
  assert.match(source, /^[a-f0-9]{40}$/);
  assert.ok(token, "GH_TOKEN is required");
  assert.equal(typeof deferCrates, "boolean");
  const read = name => JSON.parse(fs.readFileSync(path.join(directory, name), "utf8"));
  const proof = read("release-publication.json"), sdk = read("release-sdk.json");
  assert.equal(proof.source, source); assert.equal(proof.tag, tag);
  assert.equal(sdk.source, source); assert.equal(sdk.tag, tag); assert.equal(sdk.status, "passed");
  assert.equal(sdk.npm.version, tag.slice(1));
  assert.equal(sdk.npm.tags.prod, tag.slice(1)); assert.equal(sdk.npm.tags.latest, tag.slice(1));
  assert.match(sdk.npm.integrity, /^sha512-[A-Za-z0-9+/]+={0,2}$/);
  const registryComplete = fs.existsSync(path.join(directory, "release-registries.json"));
  assert.ok(registryComplete || deferCrates, "Complete registry proof is required unless crate deferral is explicit");
  if (registryComplete) {
    const registries = read("release-registries.json");
    assert.equal(registries.source, source); assert.equal(registries.tag, tag);
    assert.equal(registries.status, "passed");
    assert.ok(registries.crates.expected > 0);
    assert.equal(registries.crates.verified, registries.crates.expected);
    assert.deepEqual(registries.npm, sdk.npm);
  }
  const names = [...proof.files.map(file => file.name), "release-publication.json", "release-sdk.json",
    ...(registryComplete ? ["release-registries.json"] : [])].toSorted();
  assert.deepEqual(fs.readdirSync(directory).toSorted(), names, "Unaccounted release staging file");
  const files = names.map(name => {
    assert.equal(path.basename(name), name);
    const filename = path.join(directory, name), stat = fs.lstatSync(filename);
    assert.ok(stat.isFile() && !stat.isSymbolicLink(), name);
    const file = { name, size: stat.size, sha256: `sha256:${createHash("sha256").update(fs.readFileSync(filename)).digest("hex")}` };
    const declared = proof.files.find(candidate => candidate.name === name);
    if (declared) assert.deepEqual(file, declared, `Staged asset changed after verification: ${name}`);
    return file;
  });
  async function api(route, { method = "GET", body, missing = false } = {}) {
    const response = await request(`https://api.github.com/repos/${REPOSITORY}/${route}`, {
      method, headers: { Authorization: `Bearer ${token}`, Accept: "application/vnd.github+json", "Content-Type": "application/json" },
      ...(body ? { body: JSON.stringify(body) } : {}), signal: AbortSignal.timeout(60_000),
    });
    if (missing && response.status === 404) return null;
    assert.ok(response.ok, `${method} ${route}: HTTP ${response.status}`);
    return response.json();
  }
  let object = (await api(`git/ref/tags/${tag}`)).object;
  if (object.type === "tag") object = (await api(`git/tags/${object.sha}`)).object;
  assert.equal(object.type, "commit"); assert.equal(object.sha, source, "Remote release tag moved");
  let release = await api(`releases/tags/${tag}`, { missing: true });
  if (release === null) release = await api("releases", { method: "POST", body: {
    tag_name: tag, target_commitish: source, name: `RunMat ${tag}`, draft: true, prerelease: false,
    body: `[Changelog](https://github.com/${REPOSITORY}/blob/${source}/docs/CHANGELOG.md)\n\nDownload the archive for your platform below.${registryComplete ? "" : CRATES_PENDING_NOTICE}`,
  } });
  assert.equal(release.tag_name, tag); assert.equal(release.prerelease, false);
  const assets = await api(`releases/${release.id}/assets?per_page=100`);
  const missing = reconcileAssets(assets, files);
  if (missing.length) upload(missing.map(file => path.join(directory, file.name)));
  reconcileAssets(await api(`releases/${release.id}/assets?per_page=100`), files, true);
  const bodyUpdate = registryComplete && release.body?.includes(CRATES_PENDING_NOTICE)
    ? { body: release.body.replace(CRATES_PENDING_NOTICE, "") } : {};
  if (release.draft || Object.keys(bodyUpdate).length) release = await api(`releases/${release.id}`, {
    method: "PATCH", body: { ...(release.draft ? { draft: false } : {}), ...bodyUpdate } });
  assert.equal(release.draft, false); assert.equal(release.tag_name, tag);
  return { tag, source, url: release.html_url, crates: registryComplete ? "verified" : "pending", assets: files };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [directory, tag, source, mode = "complete"] = process.argv.slice(2);
  assert.ok(["complete", "defer-crates"].includes(mode), "Expected complete or defer-crates publication");
  console.log(JSON.stringify(await publishStagedRelease({ directory, tag, source, deferCrates: mode === "defer-crates" })));
}
