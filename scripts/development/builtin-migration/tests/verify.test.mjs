import assert from "node:assert/strict";
import test from "node:test";

import { buildMachineReport, combineMachineReports } from "../../../runtime/builtin-example-verifier/reporting.mjs";
import { createInventory, shardInventory } from "../../../runtime/builtin-example-verifier/sharding.mjs";
import { canonicalJson } from "../evidence.mjs";
import { parseVerificationManifest } from "../verify-schema.mjs";
import { verifyBatch } from "../verify.mjs";

const SOURCE = "git:abc";
const FACTORY_DIGEST = `sha256:${"a".repeat(64)}`;

function exampleReport(harness = "Portable", matches = true, shard = { index: 0, count: 1 }, artifact = "examples-0") {
  const testCase = {
    id: 1, exampleKey: "foo#scalar", builtin: "foo", authority: "catalog", compatibility: "Matlab",
    harness, input: "foo(1)", expectedOutput: "1", hasExpectedOutput: true, exampleIndex: 0,
  };
  const inventory = createInventory([testCase]);
  const selected = shardInventory(inventory, shard);
  const rows = selected.cases.map((entry) => ({
    testCase: entry, normalizedExpected: "1", normalizedActual: matches ? "1" : "2",
    imageRelPath: "", imageError: "", matches,
  }));
  return buildMachineReport({ rows, inventory, shard, range: selected.range, source: SOURCE, artifact });
}

function factoryAudit() {
  return {
    schema_version: 2,
    kind: "runmat-builtin-migration-audit",
    authority: "development-verification-only",
    metadata: {
      source: SOURCE,
      artifact: "factory-audit-1",
      inventory: {
        schema_version: 1, kind: "runmat-builtin-migration-inventory", digest: FACTORY_DIGEST, identities: ["foo"],
      },
    },
    requested_identities: ["foo"],
    global_diagnostics: [],
    summary: { identities: 1, passed: 1, failed: 0, global_errors: 0 },
    result: "pass",
    identities: [{ identity: "foo", result: "pass", failures: [], evidence: null }],
  };
}

function manifest(report, requiredLanes = ["browser", "native"]) {
  return {
    schema_version: 1,
    kind: "runmat-builtin-migration-verification-manifest",
    batch: {
      artifact: "batch-1", source: SOURCE, identities: ["foo"],
      factory_inventory_digest: FACTORY_DIGEST,
      example_inventory_digest: report.metadata.inventory.digest,
      combined_example_artifact: report.metadata.index === undefined ? report.metadata.artifact : "batch-1-examples",
    },
    factory_audit: { path: "audit.json", artifact: "factory-audit-1" },
    example_reports: [{ path: "examples.json", artifact: report.metadata.artifact }],
    expectations: [{ identity: "foo", example_keys: ["foo#scalar"], required_lanes: requiredLanes }],
  };
}

function verify(report, audit = factoryAudit(), requiredLanes) {
  const input = manifest(report, requiredLanes);
  return verifyBatch(
    input,
    { reference: input.factory_audit, value: audit },
    [{ reference: input.example_reports[0], value: report }],
  );
}

test("verification passes exact factory, inventory, example, and lane evidence", () => {
  const result = verify(exampleReport());
  assert.equal(result.result, "pass");
  assert.deepEqual(result.identities[0].observed_lanes, ["browser", "native"]);
  assert.equal(result.metadata.artifacts.example_combined, "batch-1-examples");
  assert.equal(result.authority, "development-verification-evidence-only");
});

test("stale factory and example identities fail as explicit evidence diagnostics", () => {
  const report = exampleReport();
  const staleFactory = factoryAudit();
  staleFactory.metadata.source = "git:old";
  assert.deepEqual(verify(report, staleFactory).global_failures.map((entry) => entry.code), ["factory-evidence-invalid"]);
  const input = manifest(report);
  input.batch.example_inventory_digest = `sha256:${"b".repeat(64)}`;
  const result = verifyBatch(input, { reference: input.factory_audit, value: factoryAudit() }, [{ reference: input.example_reports[0], value: report }]);
  assert.match(result.global_failures[0].detail, /Stale example inventory/);
});

test("failed examples and partial lane coverage fail the identity", () => {
  assert.deepEqual(verify(exampleReport("Portable", false)).identities[0].failures.map((entry) => entry.code), ["example-failed"]);
  const partial = verify(exampleReport("Browser"), factoryAudit(), ["browser", "native"]);
  assert.deepEqual(partial.identities[0].failures, [{ code: "required-lane-missing", lane: "native" }]);
});

test("verification accepts one exact combined report or one complete shard set", () => {
  const unsharded = exampleReport();
  const combined = combineMachineReports([unsharded], SOURCE, "batch-1-examples");
  const combinedResult = verify(combined);
  assert.equal(combinedResult.result, "pass");
  assert.deepEqual(combinedResult.metadata.artifacts.example_constituents, ["examples-0"]);

  const first = exampleReport("Portable", true, { index: 0, count: 2 }, "examples-0");
  const second = exampleReport("Portable", true, { index: 1, count: 2 }, "examples-1");
  const input = manifest(first);
  input.example_reports.push({ path: "examples-1.json", artifact: "examples-1" });
  const result = verifyBatch(
    input,
    { reference: input.factory_audit, value: factoryAudit() },
    [
      { reference: input.example_reports[0], value: first },
      { reference: input.example_reports[1], value: second },
    ],
  );
  assert.equal(result.result, "pass");
  assert.deepEqual(result.metadata.artifacts.example_constituents, ["examples-0", "examples-1"]);
});

test("manifest and shard reconciliation reject guessed or incomplete evidence", () => {
  const report = exampleReport();
  const badManifest = manifest(report);
  badManifest.expectations[0].identity = "bar";
  assert.throws(() => verifyBatch(badManifest, null, []), /cover each batch identity/);

  const first = exampleReport("Portable", true, { index: 0, count: 2 }, "examples-0");
  const input = manifest(first);
  input.example_reports.push({ path: "examples-1.json", artifact: "examples-1" });
  const incomplete = verifyBatch(input, { reference: input.factory_audit, value: factoryAudit() }, [{ reference: input.example_reports[0], value: first }]);
  assert.match(incomplete.global_failures[0].detail, /count does not match/);
});

test("schemas reject extra fields and exact combined artifact mismatches", () => {
  const report = exampleReport();
  const extra = manifest(report);
  extra.batch.sorce = SOURCE;
  assert.throws(() => verifyBatch(extra, null, []), /verification batch fields/);
  const future = manifest(report);
  future.schema_version = 2;
  assert.throws(() => verifyBatch(future, null, []), /schema_version 1/);

  const malformedAudit = factoryAudit();
  malformedAudit.metadata.inventory.extra = true;
  assert.match(verify(report, malformedAudit).global_failures[0].detail, /factory audit inventory fields/);
  const oldAudit = factoryAudit();
  oldAudit.schema_version = 1;
  assert.match(verify(report, oldAudit).global_failures[0].detail, /schema_version 2/);

  const combined = combineMachineReports([report], SOURCE, "actual-combined");
  const wrongIdentity = manifest(combined);
  wrongIdentity.batch.combined_example_artifact = "expected-combined";
  const result = verifyBatch(wrongIdentity, { reference: wrongIdentity.factory_audit, value: factoryAudit() }, [{ reference: wrongIdentity.example_reports[0], value: combined }]);
  assert.match(result.global_failures[0].detail, /Combined example artifact mismatch/);
});

test("canonical evidence uses code-point order and rejects ambiguous values", () => {
  const encoded = canonicalJson({ "𐀀": 2, "": 1 });
  assert.ok(encoded.indexOf("") < encoded.indexOf("𐀀"));
  for (const value of [undefined, NaN, Infinity, -Infinity, -0, 1n, () => {}, new Date(0)]) {
    assert.throws(() => canonicalJson(value), /Unsupported/);
  }
  const sparse = [];
  sparse.length = 1;
  assert.throws(() => canonicalJson(sparse), /Sparse/);
  const cyclic = {};
  cyclic.self = cyclic;
  assert.throws(() => canonicalJson(cyclic), /Cyclic/);
});

test("manifest artifacts and expectations use explicit code-point ordering", () => {
  const input = manifest(exampleReport());
  input.example_reports = [
    { path: "astral.json", artifact: "𐀀" },
    { path: "private.json", artifact: "" },
  ];
  input.expectations[0].example_keys = ["foo#𐀀", "foo#"];
  const parsed = parseVerificationManifest(input);
  assert.deepEqual(parsed.example_reports.map((entry) => entry.artifact), ["", "𐀀"]);
  assert.deepEqual(parsed.expectations[0].example_keys, ["foo#", "foo#𐀀"]);
});
