import assert from "node:assert/strict";
import test from "node:test";

import {
    createInventory,
    exampleKey,
    resolveShardConfig,
    shardFileSuffix,
    shardInventory
} from "./sharding.mjs";
import { buildMachineReport, combineMachineReports } from "./reporting.mjs";

const cases = ["zeta#one", "alpha#two", "alpha#one", "beta#one"].map((key, id) => ({
    id, exampleIndex: id, exampleKey: key, builtin: key.split("#")[0], authority: "catalog", compatibility: "Matlab",
    harness: "Portable", input: `x=${id}`, expectedOutput: "", hasExpectedOutput: false
}));

test("example keys prefer explicit stable ids and shard the sorted inventory into exact ranges", () => {
    assert.equal(exampleKey("FFT", { id: "Matrix" }, 9), "fft#matrix");
    assert.equal(exampleKey("FFT", { input: "x = 1" }, 1), exampleKey("FFT", { input: "x = 1" }, 99));
    assert.match(exampleKey("FFT", { input: "x = 1" }, 1), /^fft#legacy-[a-f0-9]{64}$/);
    assert.throws(() => createInventory([cases[0], { ...cases[0] }]), /Duplicate builtin example key/);
    const inventory = createInventory(cases);
    assert.deepEqual(inventory.keys, ["alpha#one", "alpha#two", "beta#one", "zeta#one"]);
    assert.deepEqual(shardInventory(inventory, { index: 1, count: 3 }).cases.map((entry) => entry.exampleKey), ["alpha#two"]);
    assert.deepEqual(shardInventory(inventory, { index: 2, count: 3 }).range, { startInclusive: 2, endExclusive: 4 });
});

test("shard configuration is strict and zero based", () => {
    assert.deepEqual(resolveShardConfig({}), { index: 0, count: 1 });
    assert.deepEqual(resolveShardConfig({ RUNMAT_EXAMPLE_SHARD_INDEX: "2", RUNMAT_EXAMPLE_SHARD_COUNT: "3" }), { index: 2, count: 3 });
    assert.throws(() => resolveShardConfig({ RUNMAT_EXAMPLE_SHARD_INDEX: "1" }), /set together/);
    assert.throws(() => resolveShardConfig({ RUNMAT_EXAMPLE_SHARD_INDEX: "3", RUNMAT_EXAMPLE_SHARD_COUNT: "3" }), /zero-based/);
});

test("shard filenames sort lexically and excess shards are valid empty ranges", () => {
    assert.equal(shardFileSuffix({ index: 2, count: 12 }), ".shard-0002-of-0012");
    const inventory = createInventory(cases.slice(0, 1));
    const empty = shardInventory(inventory, { index: 3, count: 4 });
    assert.deepEqual(empty.range, { startInclusive: 0, endExclusive: 1 });
    assert.equal(empty.cases.length, 1);
    const leadingEmpty = shardInventory(inventory, { index: 0, count: 4 });
    assert.deepEqual(leadingEmpty.range, { startInclusive: 0, endExclusive: 0 });
    assert.equal(leadingEmpty.cases.length, 0);
});

function reports(count = 2) {
    const inventory = createInventory(cases);
    return Array.from({ length: count }, (_, index) => {
        const selected = shardInventory(inventory, { index, count });
        const rows = selected.cases.map((testCase) => ({
            testCase, normalizedExpected: "", normalizedActual: "", imageRelPath: "", imageError: "", matches: true
        }));
        return buildMachineReport({ rows, inventory, shard: { index, count }, range: selected.range, source: "commit:abc", artifact: `shard-${index}` });
    });
}

test("combined reports are deterministic regardless of input order", () => {
    const input = reports();
    assert.deepEqual(combineMachineReports([...input].reverse(), "commit:abc", "combined-a"), combineMachineReports(input, "commit:abc", "combined-a"));
});

test("combined report validation rejects missing, duplicate, stale, and inconsistent shards", () => {
    const input = reports();
    assert.throws(() => combineMachineReports(input, "commit:abc"), /combined artifact identity/);
    const extra = structuredClone(input);
    extra[0].metadata.inventory.total = extra[0].metadata.inventory.count;
    assert.throws(() => combineMachineReports(extra, "commit:abc", "combined-a"), /report inventory fields/);
    const future = structuredClone(input);
    future[0].schemaVersion = "runmat.builtin-example-report.v3";
    assert.throws(() => combineMachineReports(future, "commit:abc", "combined-a"), /Unsupported shard report schema/);
    const sealedV1 = structuredClone(input);
    sealedV1[0].schemaVersion = "runmat.builtin-example-report.v1";
    assert.throws(() => combineMachineReports(sealedV1, "commit:abc", "combined-a"), /Unsupported shard report schema/);
    assert.throws(() => combineMachineReports([input[0]], "commit:abc", "combined-a"), /Missing shard reports/);
    assert.throws(() => combineMachineReports([input[0], input[0]], "commit:abc", "combined-a"), /Duplicate shard index/);
    assert.throws(() => combineMachineReports(input, "commit:new", "combined-a"), /Stale shard source/);
    const inconsistent = structuredClone(input);
    inconsistent[1].metadata.inventory.digest = "sha256:stale";
    assert.throws(() => combineMachineReports(inconsistent, "commit:abc", "combined-a"), /stale shard inventories|report inventory/);
    const missingResult = structuredClone(input);
    missingResult[1].results.pop();
    missingResult[1].summary = { total: 1, passed: 1, failed: 0 };
    assert.throws(() => combineMachineReports(missingResult, "commit:abc", "combined-a"), /stale or inconsistent inventory range|summary/);
});

test("combined reports preserve failing results for process-level failure", () => {
    const input = reports();
    input[1].results[0].matches = false;
    input[1].summary = { total: 2, passed: 1, failed: 1 };
    const combined = combineMachineReports(input, "commit:abc", "combined-failures");
    assert.deepEqual(combined.summary, { total: 4, passed: 3, failed: 1 });
    assert.equal(combined.metadata.artifact, "combined-failures");
    assert.deepEqual(combined.metadata.constituentArtifacts, ["shard-0", "shard-1"]);
});
