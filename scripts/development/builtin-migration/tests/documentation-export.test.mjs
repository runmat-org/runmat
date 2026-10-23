import assert from "node:assert/strict";
import test from "node:test";

import { DOCUMENTATION_EXPORT_ARGUMENTS, readDocumentationExport, runDocumentationExport } from "../documentation-export.mjs";

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

function payload() {
  return { schema_version: 2, inventory: { documents: 1, catalog_identities: 1, legacy_sidecars: 0, missing_catalog_documentation: [] }, builtins: [{ key: "abs", module_stem: "abs", authority: "catalog" }] };
}
function consume(value, options = {}) {
  return readDocumentationExport({ repository: "/source", ...options }, () => ({ status: 0, stdout: JSON.stringify(value) }));
}

test("canonical consumers select exact source cwd and strict arguments, including prebuilt executables", () => {
  const calls = [];
  const spawn = (...argumentsList) => { calls.push(argumentsList); return { status: 0, stdout: JSON.stringify(payload()) }; };
  assert.deepEqual(readDocumentationExport({ repository: "/source" }, spawn), payload());
  assert.deepEqual(calls[0], ["cargo", DOCUMENTATION_EXPORT_ARGUMENTS.slice(0, -1), { encoding: "utf8", maxBuffer: 128 * 1024 * 1024, cwd: "/source" }]);
  readDocumentationExport({ repository: "/source", binary: "target/export-docs", mode: "transition" }, spawn);
  assert.deepEqual(calls[1], ["/source/target/export-docs", ["--transition"], { encoding: "utf8", maxBuffer: 128 * 1024 * 1024, cwd: "/source" }]);
});

test("consumers reject previews, duplicate identities, incomplete inventories and accidental legacy authority", () => {
  const preview = payload(); preview.preview = { final_acceptance: false };
  assert.throws(() => consume(preview), /aggregate builtin documentation export/);
  const duplicate = payload(); duplicate.builtins.push({ ...duplicate.builtins[0], key: "ABS" }); duplicate.inventory.documents += 1;
  assert.throws(() => consume(duplicate), /Duplicate builtin documentation identity/);
  const missing = payload(); missing.inventory.documents += 1;
  assert.throws(() => consume(missing), /aggregate builtin documentation export/);
  const legacy = payload(); legacy.builtins[0].authority = "legacy_sidecar"; legacy.inventory.legacy_sidecars = 1;
  assert.throws(() => consume(legacy), /Strict builtin documentation consumption/);
  assert.deepEqual(consume(legacy, { mode: "transition" }), legacy);
  const unfinished = payload(); unfinished.inventory.missing_catalog_documentation.push("unfinished");
  assert.throws(() => consume(unfinished), /Strict builtin documentation consumption/);
  assert.deepEqual(consume(unfinished, { mode: "transition" }), unfinished);
  assert.throws(() => consume(payload(), { mode: "preview" }), /mode must be strict or transition/);
  assert.throws(() => consume(payload(), { exportPath: "export.json", binary: "binary" }), /not both/);
  assert.throws(() => readDocumentationExport({ repository: "/source" }, () => ({ status: 9, stderr: "duplicate authority" })), /exited 9: duplicate authority/);
});
