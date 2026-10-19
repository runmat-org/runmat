import assert from "node:assert/strict";
import test from "node:test";

import { DOCUMENTATION_EXPORT_ARGUMENTS, runDocumentationExport } from "../documentation-export.mjs";

test("documentation producer runs the aggregate transition exporter through the reviewed Cargo path", () => {
  const calls = [];
  const result = runDocumentationExport((executable, argumentsList, options) => {
    calls.push({ executable, argumentsList, options });
    return { status: 0, signal: null, stdout: "{\"schema_version\":2}\n", stderr: "" };
  });
  assert.deepEqual(calls, [{
    executable: "cargo",
    argumentsList: DOCUMENTATION_EXPORT_ARGUMENTS,
    options: { encoding: "utf8", maxBuffer: 128 * 1024 * 1024 },
  }]);
  assert.equal(result.stdout, "{\"schema_version\":2}\n");
  assert.equal(result.status, 0);
});

test("documentation producer preserves failures and rejects missing or signalled executions", () => {
  assert.deepEqual(
    runDocumentationExport(() => ({ status: 7, signal: null, stdout: "", stderr: "failed" })),
    { status: 7, signal: null, stdout: "", stderr: "failed" },
  );
  assert.throws(
    () => runDocumentationExport(() => ({ status: null, signal: "SIGTERM", stdout: "", stderr: "" })),
    /terminated by signal SIGTERM/,
  );
  assert.throws(
    () => runDocumentationExport(() => ({ error: new Error("missing") })),
    /could not execute.*missing/,
  );
});
