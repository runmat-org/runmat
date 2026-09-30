import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = fs.readFileSync(new URL("../../check-architecture-boundaries.mjs", import.meta.url), "utf8");

// Exercise the actual checker sections without loading a complete workspace.
function section(start, end) {
  const from = source.indexOf(start);
  const to = source.indexOf(end, from);
  assert.ok(from >= 0 && to > from, `missing checker section: ${start}`);
  return source.slice(from, to);
}

const helpers = section("function productionRust(text)", "function enforceTransientSequenceQuarantine()");
const uploadGuard = section("const gpuHelpers =", "for (const { path: sourcePath, text } of rustSources(binaryArithmeticRuntimeRoot))");
const arrayfunGuard = section(
  'for (const { path: sourcePath, text } of rustSources(\n  "crates/runmat-runtime/src/builtins/acceleration/gpu/arrayfun"',
  "const arrayfunRuntimeMacro =",
);

function failures(guard, context) {
  const messages = [];
  vm.runInNewContext(helpers + guard, {
    ...context,
    fail: (message) => messages.push(message),
  }, { timeout: 1000 });
  return messages;
}

test("exact integer upload guard accepts typed errors and detects discarded failures", () => {
  const valid = `pub fn upload_exact_integer_scalar_like() -> NumericUploadResult<Option<GpuTensorHandle>> {
    provider.upload().map(Some)
  }`;
  const check = (text) => failures(uploadGuard, { read: () => text });
  assert.deepEqual(check(valid), []);
  assert.equal(check(valid.replace("NumericUploadResult", "Option")).length, 1);
  assert.equal(check(valid.replace("provider.upload().map(Some)", "provider.upload().ok()")).length, 1);
  assert.equal(check("pub fn unrelated() {}").length, 1);
  // A neighbouring function is not part of the owning operation's guard.
  assert.deepEqual(check(valid + "\npub fn unrelated() { optional.ok(); }"), []);
});

test("arrayfun rejects production identity switches but permits test fixtures", () => {
  const check = (text, path = "arrayfun/gpu/admission.rs") => failures(arrayfunGuard, {
    rustSources: () => [{ path, text }],
  });
  const typed = "fn admit(operation: ArrayfunGpuCallback) { match operation { _ => () } }";
  assert.deepEqual(check(typed), []);
  assert.deepEqual(check('fixture("plus");', "arrayfun/gpu/admission/tests/cases.rs"), []);
  assert.equal(check('fn admit(name: &str) { if name == "plus" {} }').length, 1);
});
