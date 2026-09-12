import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const repository = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../..");
const verifier = path.join(repository, "scripts/development/verify-builtin-generated-products.mjs");

function run(argumentsList) {
  return spawnSync(process.execPath, [verifier, ...argumentsList], { cwd: repository, encoding: "utf8" });
}

test("generated product verification requires an explicit reviewed product selection", () => {
  const none = run(["--no-products"]);
  assert.equal(none.status, 0, none.stderr);
  const proof = JSON.parse(none.stdout);
  assert.deepEqual(proof.products, []);
  assert.equal(proof.result, "pass");

  for (const argumentsList of [[], ["--product"], ["--product", "unknown"], ["--no-products", "--product", "wasm-registry"], ["--product", "wasm-registry", "--product", "wasm-registry"]]) {
    assert.notEqual(run(argumentsList).status, 0, `unexpectedly accepted ${JSON.stringify(argumentsList)}`);
  }
});
