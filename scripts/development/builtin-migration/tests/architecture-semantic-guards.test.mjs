import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import vm from "node:vm";
import {
  productionRust,
  rustFunctionRegions,
} from "../rust-production-source.mjs";

const source = fs.readFileSync(new URL("../../check-architecture-boundaries.mjs", import.meta.url), "utf8");

// Exercise the actual checker sections without loading a complete workspace.
function section(start, end) {
  const from = source.indexOf(start);
  const to = source.indexOf(end, from);
  assert.ok(from >= 0 && to > from, `missing checker section: ${start}`);
  return source.slice(from, to);
}

const uploadGuard = section("const gpuHelpers =", "for (const { path: sourcePath, text } of rustSources(binaryArithmeticRuntimeRoot))");
const arrayfunGuard = section(
  'for (const { path: sourcePath, text } of rustSources(\n  "crates/runmat-runtime/src/builtins/acceleration/gpu/arrayfun"',
  "const arrayfunRuntimeMacro =",
);

function failures(guard, context) {
  const messages = [];
  vm.runInNewContext(guard, {
    ...context,
    fail: (message) => messages.push(message),
    productionRust,
    rustFunctionRegions,
  }, { timeout: 1000 });
  return messages;
}

function production(text) {
  return productionRust(text);
}

test("production Rust masking removes only cfg-test items", () => {
  const input = `
pub fn before() { canonical_before(); }
#[cfg(test)]
mod tests {
  const TEXT: &str = r###"} #[cfg(test)] {"###;
  const BRACES: [char; 2] = ['{', '\\u{7d}'];
  /* nested { /* block } */ comment */
  fn fixture() { project_legacy_builtin_value_abi(test_value()); }
}
pub fn after() { project_legacy_builtin_value_abi(production_value()); }
`;
  const output = production(input);
  assert.match(output, /canonical_before\(\)/);
  assert.doesNotMatch(output, /fixture|test_value/);
  assert.match(output, /fn after\(\)/);
  assert.match(output, /project_legacy_builtin_value_abi\(production_value\(\)\)/);
  assert.equal(output.split("\n").length, input.split("\n").length);
});

test("cfg-test text in trivia and literals cannot hide later production", () => {
  const input = `
const RAW: &str = r#"#[cfg(test)] mod hidden { }"#;
// #[cfg(test)] fn hidden_by_comment() {}
/* #[cfg(test)] mod hidden_by_block { { } } */
pub fn retained() { Value::OutputList(vec![]); }
pub fn lifetime<'a>(value: &'a str) -> &'a str { value }
`;
  const output = production(input);
  assert.match(output, /fn retained\(\)/);
  assert.match(output, /Value::OutputList/);
  assert.match(output, /fn lifetime<'a>/);
});

test("cfg-test imports and constrained modules do not truncate following items", () => {
  const input = `
#[cfg(test)] use crate::test_support::{self, Fixture};
#[cfg(all(unix, test))]
mod unix_tests { fn only_test() {} }
#[cfg(any(test, unix))]
mod also_production_on_unix { fn retained_branch() {} }
pub fn final_item() {}
`;
  const output = production(input);
  assert.doesNotMatch(output, /test_support|only_test/);
  assert.match(output, /also_production_on_unix/);
  assert.match(output, /final_item/);
});

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
