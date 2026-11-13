#!/usr/bin/env node
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const REPOSITORY = "runmat-org/runmat";
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

export async function publishStagedRelease({ directory, tag, source, token = process.env.GH_TOKEN,
  request = fetch, upload = files => execFileSync("gh", ["release", "upload", tag, "-R", REPOSITORY, ...files], { stdio: "inherit" }) }) {
  assert.match(tag, /^v\d+\.\d+\.\d+$/);
  assert.match(source, /^[a-f0-9]{40}$/);
  assert.ok(token, "GH_TOKEN is required");
  const read = name => JSON.parse(fs.readFileSync(path.join(directory, name), "utf8"));
  const proof = read("release-publication.json"), registries = read("release-registries.json");
  assert.equal(proof.source, source); assert.equal(proof.tag, tag);
  assert.equal(registries.source, source); assert.equal(registries.tag, tag);
  assert.equal(registries.status, "passed");
  assert.ok(registries.crates.expected > 0);
  assert.equal(registries.crates.verified, registries.crates.expected);
  assert.equal(registries.npm.version, tag.slice(1));
  assert.equal(registries.npm.tags.prod, tag.slice(1));
  assert.equal(registries.npm.tags.latest, tag.slice(1));
  const names = [...proof.files.map(file => file.name), "release-publication.json", "release-registries.json"].toSorted();
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
    body: `[Changelog](https://github.com/${REPOSITORY}/blob/${source}/docs/CHANGELOG.md)\n\nDownload the archive for your platform below.`,
  } });
  assert.equal(release.tag_name, tag); assert.equal(release.prerelease, false);
  const assets = await api(`releases/${release.id}/assets?per_page=100`);
  const missing = reconcileAssets(assets, files);
  if (missing.length) upload(missing.map(file => path.join(directory, file.name)));
  reconcileAssets(await api(`releases/${release.id}/assets?per_page=100`), files, true);
  if (release.draft) release = await api(`releases/${release.id}`, { method: "PATCH", body: { draft: false } });
  assert.equal(release.draft, false); assert.equal(release.tag_name, tag);
  return { tag, source, url: release.html_url, assets: files };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [directory, tag, source] = process.argv.slice(2);
  console.log(JSON.stringify(await publishStagedRelease({ directory, tag, source })));
}
