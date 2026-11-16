import assert from "node:assert/strict";
import test from "node:test";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { reconcileAssets, publishStagedRelease } from "./frozen-release-publish.mjs";

const file = { name: "native.tar.gz", size: 3, sha256: `sha256:${"a".repeat(64)}` };
const asset = { name: file.name, size: file.size, digest: file.sha256 };
test("resumes matching assets and refuses overwrite or unaccounted files", () => {
  assert.deepEqual(reconcileAssets([], [file]), [file]);
  assert.deepEqual(reconcileAssets([asset], [file], true), []);
  for (const existing of [[{ ...asset, digest: `sha256:${"b".repeat(64)}` }], [{ ...asset, size: 4 }],
    [{ ...asset, name: "unintended.js" }], [asset, asset]]) assert.throws(() => reconcileAssets(existing, [file]));
  assert.throws(() => reconcileAssets([], [file], true));
  assert.throws(() => reconcileAssets([], [file, file]));
});

function fixture(t) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-publish-test-"));
  t.after(() => fs.rmSync(directory, { recursive: true }));
  const source = "a".repeat(40), tag = "v0.7.1", bytes = Buffer.from("abc");
  const files = [{ name: file.name, size: 3, sha256: `sha256:${createHash("sha256").update(bytes).digest("hex")}` }];
  fs.writeFileSync(path.join(directory, file.name), bytes);
  fs.writeFileSync(path.join(directory, "release-publication.json"), JSON.stringify({ source, tag, files }));
  const sdk = { status: "passed", source, tag, npm: { version: "0.7.1",
    integrity: `sha512-${Buffer.alloc(64).toString("base64")}`, tags: { prod: "0.7.1", latest: "0.7.1" } } };
  fs.writeFileSync(path.join(directory, "release-sdk.json"), JSON.stringify(sdk));
  fs.writeFileSync(path.join(directory, "release-registries.json"), JSON.stringify({ ...sdk,
    crates: { expected: 90, verified: 90 } }));
  return { directory, source, tag };
}
test("publishes a draft only after every uploaded asset is verified", async t => {
  const input = fixture(t), calls = [];
  const release = { id: 1, tag_name: input.tag, draft: true, prerelease: false, html_url: "https://github.com/runmat-org/runmat/releases/tag/v0.7.1" };
  let assets = [];
  const request = async (url, options) => {
    const route = url.split("/runmat-org/runmat/")[1]; calls.push([options.method, route]);
    if (route.startsWith("git/ref/tags/")) return { ok: true, json: async () => ({ object: { type: "commit", sha: input.source } }) };
    if (route.startsWith("releases/tags/")) return { status: 404 };
    if (route === "releases" && options.method === "POST") { assert.equal(JSON.parse(options.body).draft, true); return { ok: true, json: async () => release }; }
    if (route.endsWith("/assets?per_page=100")) return { ok: true, json: async () => assets };
    if (options.method === "PATCH") { assert.equal(assets.length, 4); return { ok: true, json: async () => ({ ...release, draft: false }) }; }
    throw Error(`Unexpected API request: ${route}`);
  };
  const result = await publishStagedRelease({ ...input, token: "test", request, upload: files => {
    assets = files.map(filename => ({ name: path.basename(filename), size: fs.statSync(filename).size,
      digest: `sha256:${createHash("sha256").update(fs.readFileSync(filename)).digest("hex")}` }));
  } });
  assert.equal(result.assets.length, 4);
  assert.equal(result.crates, "verified");
  assert.deepEqual(calls.at(-1), ["PATCH", "releases/1"]);
});

test("requires explicit crate deferral and still requires a valid source-bound SDK proof", async t => {
  const input = fixture(t), registry = path.join(input.directory, "release-registries.json");
  fs.rmSync(registry);
  const request = () => { throw Error("API should not run"); };
  await assert.rejects(publishStagedRelease({ ...input, token: "test", request }), /Complete registry proof/);
  const sdkFile = path.join(input.directory, "release-sdk.json"), original = fs.readFileSync(sdkFile);
  for (const mutate of [p => { p.source = "b".repeat(40); }, p => { p.status = "pending"; },
    p => { p.npm.tags.latest = "0.7.2"; }, p => { p.npm.integrity = "invalid"; }]) {
    const sdk = JSON.parse(original); mutate(sdk); fs.writeFileSync(sdkFile, JSON.stringify(sdk));
    await assert.rejects(publishStagedRelease({ ...input, deferCrates: true, token: "test", request }));
  }
  fs.rmSync(sdkFile);
  await assert.rejects(publishStagedRelease({ ...input, deferCrates: true, token: "test", request }), /ENOENT/);
});

test("deferral never accepts a supplied incomplete registry proof", async t => {
  const input = fixture(t), filename = path.join(input.directory, "release-registries.json");
  const proof = JSON.parse(fs.readFileSync(filename)); proof.crates.verified = 53;
  fs.writeFileSync(filename, JSON.stringify(proof));
  await assert.rejects(publishStagedRelease({ ...input, deferCrates: true, token: "test",
    request: () => { throw Error("API should not run"); } }));
});

test("refuses to downgrade an already complete release to deferred publication", async t => {
  const input = fixture(t);
  const assets = fs.readdirSync(input.directory).map(name => ({ name, size: fs.statSync(path.join(input.directory, name)).size,
    digest: `sha256:${createHash("sha256").update(fs.readFileSync(path.join(input.directory, name))).digest("hex")}` }));
  fs.rmSync(path.join(input.directory, "release-registries.json"));
  await assert.rejects(publishStagedRelease({ ...input, deferCrates: true, token: "test", request: async (url, options) => {
    assert.equal(options.method, "GET");
    const route = url.split("/runmat-org/runmat/")[1];
    const data = route.startsWith("git/ref/tags/") ? { object: { type: "commit", sha: input.source } }
      : route.startsWith("releases/tags/") ? { id: 1, tag_name: input.tag, draft: false, prerelease: false } : assets;
    return { ok: true, json: async () => data };
  }, upload: () => { throw Error("No upload should occur"); } }), /Unexpected existing release asset/);
});

test("publishes native and SDK first, then adds crate proof without replacing existing assets", async t => {
  const input = fixture(t), filename = path.join(input.directory, "release-registries.json");
  const registry = fs.readFileSync(filename); fs.rmSync(filename);
  let release = null;
  const assets = [], uploads = [];
  const request = async (url, options) => {
    const route = url.split("/runmat-org/runmat/")[1];
    if (route.startsWith("git/ref/tags/")) return { ok: true, json: async () => ({ object: { type: "commit", sha: input.source } }) };
    if (route.startsWith("releases/tags/")) return release ? { ok: true, json: async () => ({ ...release }) } : { status: 404 };
    if (route === "releases" && options.method === "POST") {
      release = { ...JSON.parse(options.body), id: 1, html_url: "https://example.test/release" };
      assert.equal(release.draft, true); assert.match(release.body, /Crates.io publication is in progress/);
      return { ok: true, json: async () => ({ ...release }) };
    }
    if (route.endsWith("/assets?per_page=100")) return { ok: true, json: async () => [...assets] };
    if (options.method === "PATCH") {
      Object.assign(release, JSON.parse(options.body));
      return { ok: true, json: async () => ({ ...release }) };
    }
    throw Error(`Unexpected API request: ${route}`);
  };
  const upload = files => {
    uploads.push(files.map(filename => path.basename(filename)));
    for (const filename of files) assets.push({ name: path.basename(filename), size: fs.statSync(filename).size,
      digest: `sha256:${createHash("sha256").update(fs.readFileSync(filename)).digest("hex")}` });
  };
  const first = await publishStagedRelease({ ...input, deferCrates: true, token: "test", request, upload });
  assert.equal(first.crates, "pending"); assert.equal(release.draft, false);
  assert.equal(assets.length, 3);
  const originalAssets = structuredClone(assets);
  release.body += "\n\nAdditional release note.";
  fs.writeFileSync(filename, registry);
  const complete = await publishStagedRelease({ ...input, token: "test", request, upload });
  assert.equal(complete.crates, "verified"); assert.equal(assets.length, 4);
  assert.deepEqual(uploads[1], ["release-registries.json"]);
  assert.deepEqual(assets.slice(0, 3), originalAssets);
  assert.doesNotMatch(release.body, /Crates.io publication is in progress/);
  assert.match(release.body, /Additional release note/);
});
test("rejects incomplete registry proof and modified staged bytes before any API write", async t => {
  const input = fixture(t);
  fs.writeFileSync(path.join(input.directory, file.name), "changed");
  await assert.rejects(publishStagedRelease({ ...input, token: "test", request: () => { throw Error("API should not run"); } }), /Staged asset changed/);
  const filename = path.join(input.directory, "release-registries.json"), proof = JSON.parse(fs.readFileSync(filename));
  proof.crates.verified = 39; fs.writeFileSync(filename, JSON.stringify(proof));
  await assert.rejects(publishStagedRelease({ ...input, token: "test" }));
});
test("refuses a moved immutable tag", async t => {
  const input = fixture(t);
  await assert.rejects(publishStagedRelease({ ...input, token: "test", request: async () => ({ ok: true,
    json: async () => ({ object: { type: "commit", sha: "b".repeat(40) } }) }) }), /Remote release tag moved/);
});
test("verifies an annotated tag and existing matching release without mutation", async t => {
  const input = fixture(t);
  const assets = fs.readdirSync(input.directory).map(name => ({ name, size: fs.statSync(path.join(input.directory, name)).size,
    digest: `sha256:${createHash("sha256").update(fs.readFileSync(path.join(input.directory, name))).digest("hex")}` }));
  const request = async (url, options) => {
    assert.equal(options.method, "GET");
    const route = url.split("/runmat-org/runmat/")[1];
    const data = route.startsWith("git/ref/tags/") ? { object: { type: "tag", sha: "c".repeat(40) } }
      : route.startsWith("git/tags/") ? { object: { type: "commit", sha: input.source } }
        : route.startsWith("releases/tags/") ? { id: 1, tag_name: input.tag, draft: false, prerelease: false }
          : assets;
    return { ok: true, json: async () => data };
  };
  await publishStagedRelease({ ...input, token: "test", request, upload: () => { throw Error("Matching assets must not upload again"); } });
});
