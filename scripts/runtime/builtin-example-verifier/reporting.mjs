// @ts-check

import { validateCombinedMachineReport, validateMachineReport } from "./report-schema.mjs";
import { REPORT_SCHEMA, shardInventory } from "./sharding.mjs";

export { validateCombinedMachineReport, validateMachineReport } from "./report-schema.mjs";

export function buildMachineReport({ rows, inventory, shard, range, source, artifact }) {
    const results = rows.map((row) => ({
        key: row.testCase.exampleKey,
        builtin: row.testCase.builtin,
        exampleIndex: row.testCase.exampleIndex,
        harness: row.testCase.harness,
        matches: row.matches,
        expected: row.normalizedExpected,
        actual: row.normalizedActual,
        image: row.imageRelPath || null,
        imageError: row.imageError || null
    }));
    const failures = results.filter((result) => !result.matches).length;
    return validateMachineReport({
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
    });
}

export function combineMachineReports(reports, expectedSource, combinedArtifact) {
    if (!Array.isArray(reports) || reports.length === 0) throw new Error("No shard reports were provided");
    if (typeof combinedArtifact !== "string" || !combinedArtifact.trim()) throw new Error("A combined artifact identity is required");
    const parsed = reports.map(validateMachineReport);
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
    return validateCombinedMachineReport({
        schemaVersion: REPORT_SCHEMA,
        metadata: {
            source: first.metadata.source,
            artifact: combinedArtifact.trim(),
            lane: first.metadata.lane,
            inventory: combinedInventory,
            shards: [...byIndex.keys()].sort((left, right) => left - right),
            constituentArtifacts: Array.from({ length: count }, (_, index) => byIndex.get(index).metadata.artifact)
        },
        summary: { total: results.length, passed: results.length - failed, failed },
        results
    });
}
