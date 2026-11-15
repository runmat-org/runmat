import assert from "node:assert/strict";
import test from "node:test";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { expectedPackages, sdkPublicationProof, verifyRegistries } from "./frozen-release-registries.mjs";

const metadata = () => ({ workspace_members: ["one", "two"], packages: [
  { id: "one", name: "runmat-one", version: "0.7.1", publish: null },
  { id: "two", name: "runmat-private", version: "0.7.1", publish: [] },
  { id: "external", name: "external", version: "1.0.0", publish: null },
] });
test("derives every expected publishable version from the frozen workspace", () => {
  assert.deepEqual(expectedPackages(metadata(), "0.7.1").map(pkg => pkg.name), ["runmat-one"]);
  const more = metadata(); more.packages[1].publish = ["crates-io"];
  assert.equal(expectedPackages(more, "0.7.1").length, 2);
});

const sdkInput = () => ({ source: "a".repeat(40), tag: "v0.7.1",
  npmIntegrity: `sha512-${Buffer.alloc(64).toString("base64")}`,
  npm: { name: "runmat", version: "0.7.1", dist: { integrity: `sha512-${Buffer.alloc(64).toString("base64")}` } },
  tags: { prod: "0.7.1", latest: "0.7.1", dev: "0.7.1-dev.6" } });

test("SDK proof is stable across unrelated development-tag changes", () => {
  const input = sdkInput(), proof = sdkPublicationProof(input);
  input.tags.dev = "0.7.2-dev.1";
  assert.deepEqual(sdkPublicationProof(input), proof);
  assert.deepEqual(proof.npm.tags, { prod: "0.7.1", latest: "0.7.1" });
  assert.equal("crates" in proof, false);
});

test("SDK verification requires exact identity, integrity and production tags", () => {
  for (const mutate of [p => { p.npm.name = "other"; }, p => { p.npm.version = "0.7.2"; },
    p => { p.npm.dist.integrity = "sha512-AAAA"; }, p => { p.npm.gitHead = "b".repeat(40); },
    p => { p.tags.prod = "0.7.2"; }, p => { p.tags.latest = "0.7.0"; }]) {
    const input = sdkInput(); mutate(input); assert.throws(() => sdkPublicationProof(input));
  }
});

test("npm-only verification makes no Cargo or crates.io request", async t => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-sdk-proof-test-"));
  t.after(() => fs.rmSync(directory, { recursive: true }));
  execFileSync("git", ["init", "--quiet", directory]);
  const hooks = path.join(directory, "test-hooks"); fs.mkdirSync(hooks);
  execFileSync("git", ["-c", "user.name=Release test", "-c", "user.email=release-test@example.invalid",
    "-c", "commit.gpgsign=false", "-c", `core.hooksPath=${hooks}`,
    "commit", "--quiet", "--allow-empty", "-m", "fixture"], { cwd: directory });
  const source = execFileSync("git", ["rev-parse", "HEAD"], { cwd: directory, encoding: "utf8" }).trim();
  const input = sdkInput(), urls = [];
  const output = path.join(directory, "sdk.json");
  const proof = await verifyRegistries({ sourceRoot: directory, source, tag: input.tag,
    npmIntegrity: input.npmIntegrity, output, cache: path.join(directory, "cache"), npmOnly: true,
    request: async url => {
      urls.push(url); assert.match(url, /^https:\/\/registry\.npmjs\.org\//);
      return { ok: true, json: async () => url.endsWith("dist-tags") ? input.tags : input.npm };
    } });
  assert.equal(urls.length, 2); assert.equal(proof.source, source);
  assert.equal("crates" in proof, false);
  assert.deepEqual(JSON.parse(fs.readFileSync(output)), proof);
});

test("workflow makes crate deferral explicit and keeps SDK and native gates unconditional", () => {
  const workflow = fs.readFileSync(new URL("../../.github/workflows/finalize-release.yml", import.meta.url), "utf8");
  assert.match(workflow, /defer_crates:\n        description:.*\n        type: boolean\n        default: false/);
  const sdk = workflow.slice(workflow.indexOf("      - name: Verify published SDK"), workflow.indexOf("      - name: Verify complete registry publication"));
  assert.match(sdk, /npm-only/); assert.doesNotMatch(sdk, /\n        if:/);
  const native = workflow.slice(workflow.indexOf("      - name: Stage exact accepted native assets"), workflow.indexOf("      - name: Publish and verify"));
  assert.match(native, /scripts\/distribution\/frozen-release\.mjs/); assert.doesNotMatch(native, /\n        if:/);
  assert.match(workflow, /Verify complete registry publication\n        if: inputs\.defer_crates == false/);
});
test("rejects wrong versions, duplicate identities, unsafe names and empty inventories", () => {
  for (const edit of [m => { m.packages[0].version = "0.7.0"; }, m => { m.packages[0].name = "../other"; },
    m => { m.packages[0].publish = []; }, m => { m.workspace_members.push("extra"); m.packages.push({ ...m.packages[0], id: "extra" }); }]) {
    const input = metadata(); edit(input); assert.throws(() => expectedPackages(input, "0.7.1"));
  }
});
