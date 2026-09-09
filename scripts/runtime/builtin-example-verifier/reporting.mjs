// @ts-check

import { REPORT_SCHEMA, shardInventory } from "./sharding.mjs";

export function buildMachineReport({ rows, inventory, shard, range, source, artifact }) {
    const results = rows.map((row) => ({
        key: row.testCase.exampleKey,
        builtin: row.testCase.builtin,
        exampleIndex: row.testCase.exampleIndex,
        harness: row.testCase.harness,
        matches: row.matches,
        expected: row.normalizedExpected,
        actual: row.normalizedWasm,
        image: row.imageRelPath || null,
        imageError: row.imageError || null
    }));
    const failures = results.filter((result) => !result.matches).length;
    return {
        schemaVersion: REPORT_SCHEMA,
        metadata: {
            index: shard.index,
            count: shard.count,
            source,
            artifact,
            lane: "harness-defined-browser-and-native",
            inventory: {
                digest: inventory.digest,
                count: inventory.keys.length,
                keys: inventory.keys,
                range
            }
        },
        summary: { total: results.length, passed: results.length - failures, failed: failures },
        results
    };
}

export function combineMachineReports(reports, expectedSource) {
    if (!Array.isArray(reports) || reports.length === 0) throw new Error("No shard reports were provided");
    const parsed = reports.map(validateReportShape);
    const first = parsed[0];
    const count = first.metadata.count;
    if (parsed.length !== count) throw new Error(`Missing shard reports: expected ${count}, received ${parsed.length}`);
    if (expectedSource !== undefined && first.metadata.source !== expectedSource) {
        throw new Error(`Stale shard source: expected ${expectedSource}, received ${first.metadata.source}`);
    }
    const byIndex = new Map();
    const artifacts = new Set();
    for (const report of parsed) {
        const metadata = report.metadata;
        if (metadata.count !== count) throw new Error("Inconsistent shard counts");
        if (metadata.source !== first.metadata.source) throw new Error("Inconsistent or stale shard sources");
        if (metadata.inventory.digest !== first.metadata.inventory.digest
            || metadata.inventory.count !== first.metadata.inventory.count
            || JSON.stringify(metadata.inventory.keys) !== JSON.stringify(first.metadata.inventory.keys)) {
            throw new Error("Inconsistent or stale shard inventories");
        }
        if (metadata.lane !== first.metadata.lane) throw new Error("Inconsistent shard lanes");
        if (byIndex.has(metadata.index)) throw new Error(`Duplicate shard index: ${metadata.index}`);
        if (artifacts.has(metadata.artifact)) throw new Error(`Duplicate shard artifact: ${metadata.artifact}`);
        byIndex.set(metadata.index, report);
        artifacts.add(metadata.artifact);
    }
    for (let index = 0; index < count; index += 1) {
        if (!byIndex.has(index)) throw new Error(`Missing shard index: ${index}`);
    }

    const inventory = { cases: first.metadata.inventory.keys.map((exampleKey) => ({ exampleKey })) };
    const resultsByKey = new Map();
    for (let index = 0; index < count; index += 1) {
        const report = byIndex.get(index);
        const expected = shardInventory(inventory, { index, count });
        const expectedKeys = expected.cases.map((entry) => entry.exampleKey);
        const actualKeys = report.results.map((result) => result.key);
        if (JSON.stringify(report.metadata.inventory.range) !== JSON.stringify(expected.range)
            || JSON.stringify(actualKeys) !== JSON.stringify(expectedKeys)) {
            throw new Error(`Shard ${index} has a stale or inconsistent inventory range`);
        }
        for (const result of report.results) {
            if (resultsByKey.has(result.key)) throw new Error(`Duplicate example result: ${result.key}`);
            resultsByKey.set(result.key, result);
        }
    }
    if (resultsByKey.size !== first.metadata.inventory.count) {
        throw new Error(`Missing example results: expected ${first.metadata.inventory.count}, received ${resultsByKey.size}`);
    }
    const results = first.metadata.inventory.keys.map((key) => resultsByKey.get(key));
    const failed = results.filter((result) => !result.matches).length;
    const combinedInventory = {
        ...first.metadata.inventory,
        range: { startInclusive: 0, endExclusive: first.metadata.inventory.count }
    };
    return {
        schemaVersion: REPORT_SCHEMA,
        metadata: {
            source: first.metadata.source,
            artifact: "combined",
            lane: first.metadata.lane,
            inventory: combinedInventory,
            shards: [...byIndex.keys()].sort((left, right) => left - right)
        },
        summary: { total: results.length, passed: results.length - failed, failed },
        results
    };
}

function validateReportShape(report) {
    if (!report || typeof report !== "object" || report.schemaVersion !== REPORT_SCHEMA) {
        throw new Error("Unsupported shard report schema");
    }
    const metadata = report.metadata;
    if (!metadata || !Number.isInteger(metadata.index) || !Number.isInteger(metadata.count)
        || metadata.index < 0 || metadata.count < 1 || metadata.index >= metadata.count) {
        throw new Error("Invalid shard index/count metadata");
    }
    if (typeof metadata.source !== "string" || !metadata.source
        || typeof metadata.artifact !== "string" || !metadata.artifact
        || typeof metadata.lane !== "string" || !metadata.lane) {
        throw new Error("Missing shard source/artifact/lane metadata");
    }
    if (!metadata.inventory || typeof metadata.inventory.digest !== "string"
        || !Number.isInteger(metadata.inventory.count) || !Array.isArray(metadata.inventory.keys)
        || !metadata.inventory.range || !Array.isArray(report.results)) {
        throw new Error("Invalid shard inventory metadata");
    }
    const { startInclusive, endExclusive } = metadata.inventory.range;
    if (metadata.inventory.keys.length !== metadata.inventory.count
        || !metadata.inventory.keys.every((key) => typeof key === "string" && key.length > 0)
        || !Number.isInteger(startInclusive) || !Number.isInteger(endExclusive)
        || startInclusive < 0 || endExclusive < startInclusive || endExclusive > metadata.inventory.count) {
        throw new Error("Invalid shard inventory keys/range metadata");
    }
    if (!report.results.every((result) => result && typeof result === "object"
        && typeof result.key === "string" && typeof result.matches === "boolean")) {
        throw new Error("Invalid shard result records");
    }
    const failed = report.results.filter((result) => !result.matches).length;
    if (!report.summary || report.summary.total !== report.results.length
        || report.summary.passed !== report.results.length - failed || report.summary.failed !== failed) {
        throw new Error("Inconsistent shard summary");
    }
    return report;
}
