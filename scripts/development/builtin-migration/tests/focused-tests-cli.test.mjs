import assert from "node:assert/strict";
import test from "node:test";

import { parseFocusedTestFilters, runFocusedTests } from "../focused-tests-cli.mjs";

test("focused filters require explicit, unique Rust test paths", () => {
  assert.deepEqual(parseFocusedTestFilters([
    "--filter", "builtins::constants::tests",
    "--filter", "builtins::array::creation::true_false::tests",
  ]), ["builtins::constants::tests", "builtins::array::creation::true_false::tests"]);
  assert.throws(() => parseFocusedTestFilters([]), /one or more/);
  assert.throws(() => parseFocusedTestFilters(["--filter", "builtins::tests", "--filter", "builtins::tests"]), /duplicate/);
  assert.throws(() => parseFocusedTestFilters(["--filter", "tests;exit 0"]), /invalid/);
});

test("focused groups run in separate serial Rust harnesses", () => {
  const calls = [];
  const result = runFocusedTests(["builtins::constants::tests", "builtins::logical::rel::isequal"], (program, args, options) => {
    calls.push({ program, args, options });
    return { status: 0, stdout: `test ${args[4]}::one ... ok\ntest ${args[4]}::two ... ok\ntest result: ok. 2 passed; 0 failed; 0 ignored; 123 filtered out\n` };
  });
  assert.equal(result.passed, 4);
  assert.equal(result.tests.length, 4);
  assert.equal(calls.length, 2);
  assert.deepEqual(calls[0].args, ["test", "-p", "runmat-runtime", "--lib", "builtins::constants::tests", "--", "--test-threads=1"]);
  assert.ok(calls.every((call) => call.options.timeout > 0));
});

test("a zero-match filter cannot produce passing gate evidence", () => {
  assert.throws(() => runFocusedTests(["builtins::missing"], () => ({
    status: 0,
    stdout: "test result: ok. 0 passed; 0 failed; 0 ignored; 123 filtered out\n",
  })), /no tests matched/);
});

test("failed, signaled, or truncated executions cannot pass", () => {
  const filter = ["builtins::constants::tests"];
  assert.throws(() => runFocusedTests(filter, () => ({ status: 101, stdout: "test result: FAILED" })), /exited 101/);
  assert.throws(() => runFocusedTests(filter, () => ({ signal: "SIGTERM" })), /SIGTERM/);
  assert.throws(() => runFocusedTests(filter, () => ({ error: new Error("maxBuffer exceeded") })), /maxBuffer exceeded/);
});

test("ignored, overlapping, and unaccounted tests cannot pass", () => {
  assert.throws(() => runFocusedTests(["first"], () => ({
    status: 0,
    stdout: "test first::one ... ok\ntest result: ok. 1 passed; 0 failed; 1 ignored; 0 filtered out\n",
  })), /ignored/);
  assert.throws(() => runFocusedTests(["first"], () => ({
    status: 0,
    stdout: "test result: ok. 1 passed; 0 failed; 0 ignored; 0 filtered out\n",
  })), /inventory differs/);
  assert.throws(() => runFocusedTests(["first", "second"], () => ({
    status: 0,
    stdout: "test shared::one ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 filtered out\n",
  })), /multiple filters/);
});
