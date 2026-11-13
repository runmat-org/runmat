#!/usr/bin/env node
import assert from "node:assert/strict";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { runHeadlessChrome } from "./builtin-example-verifier/browser-runner.mjs";
import { createRunnerHtml } from "./builtin-example-verifier/browser-template.mjs";
import { resolveArtifactPath, validateArtifactManifest } from "./builtin-example-verifier/artifacts.mjs";

const manifestPath = resolve(process.argv[2]);
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
validateArtifactManifest(manifest, { verifyFiles: true, manifestPath });
assert.equal(manifest.product, "browser-wasm");
const sources = [
  ["scalar-division", "x=[1 2 4]; assert(isequal(x/2,[0.5 1 2])); assert(isequal(x.^(-2),[1 0.25 0.0625]));"],
  ["typed-integers", "x=uint64([0x0020000000000001u64 0xFFFFFFFFFFFFFFFFu64]); assert(isequal(x,x));"],
  ["strings", 's=["a" "b"]; assert(isequal(s+"!",["a!" "b!"]));'],
  ["callback", "f=@(x) x.*x; assert(isequal(arrayfun(f,[1 2 3]),[1 4 9]));"],
];
const cases = sources.map(([name, input], id) => ({ id, builtin: name,
  description: name, input: `${input}\ndisp('acceptance-passed');`,
  compatibility: "RunMat", harness: "BrowserCpu", fixture: "None" }));
const results = await runHeadlessChrome({
  repoRoot: resolve("."), chromeWrapper: resolve("scripts/runtime/chrome-headless.sh"),
  wasmModule: resolveArtifactPath(manifest, manifestPath, "wasm-js"),
  wasmBinary: resolveArtifactPath(manifest, manifestPath, "wasm-binary"),
  runnerHtml: createRunnerHtml(90000, 2, 10000, false),
  casesJson: JSON.stringify(cases), totalCases: cases.length, overallTimeoutMs: 240000,
});
validateArtifactManifest(manifest, { verifyFiles: true, manifestPath });
const receipt = { artifactManifestDigest: manifest.artifactManifestDigest, results };
writeFileSync(resolve(process.argv[3] ?? "bindings/ts/browser-product.acceptance.json"), JSON.stringify(receipt, null, 2) + "\n");
assert.equal(results.length, cases.length);
assert.deepEqual(new Set(results.map(result => result.id)), new Set(cases.map(test => test.id)));
for (const result of results) {
  assert.equal(result.errorText, "", JSON.stringify(result));
  assert.match(result.stdoutText, /acceptance-passed/, JSON.stringify(result));
}
console.log(`${results.length} bounded Chrome worker cases passed against the manifested browser bytes.`);
