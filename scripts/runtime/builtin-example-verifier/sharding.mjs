// @ts-check

import { createHash } from "node:crypto";

export const REPORT_SCHEMA = "runmat.builtin-example-report.v2";

export function exampleKey(identity, example, exampleIndex) {
    const owner = normalizeKey(identity, "builtin identity");
    const explicit = typeof example?.id === "string" ? example.id.trim().toLowerCase() : "";
    const legacy = createHash("sha256")
        .update(typeof example?.input === "string" ? example.input.trim() : `missing-${exampleIndex + 1}`)
        .digest("hex");
    return `${owner}#${explicit || `legacy-${legacy}`}`;
}

export function resolveShardConfig(env = process.env) {
    const rawIndex = env.RUNMAT_EXAMPLE_SHARD_INDEX;
    const rawCount = env.RUNMAT_EXAMPLE_SHARD_COUNT;
    if (rawIndex === undefined && rawCount === undefined) return { index: 0, count: 1 };
    if (rawIndex === undefined || rawCount === undefined) {
        throw new Error("RUNMAT_EXAMPLE_SHARD_INDEX and RUNMAT_EXAMPLE_SHARD_COUNT must be set together");
    }
    const index = parseInteger(rawIndex, "RUNMAT_EXAMPLE_SHARD_INDEX");
    const count = parseInteger(rawCount, "RUNMAT_EXAMPLE_SHARD_COUNT");
    if (count < 1) throw new Error("RUNMAT_EXAMPLE_SHARD_COUNT must be at least 1");
    if (index < 0 || index >= count) {
        throw new Error(`RUNMAT_EXAMPLE_SHARD_INDEX must be zero-based and less than ${count}`);
    }
    return { index, count };
}

export function createInventory(cases) {
    const ordered = [...cases].sort((left, right) => left.exampleKey < right.exampleKey ? -1 : left.exampleKey > right.exampleKey ? 1 : 0);
    const keys = ordered.map((testCase) => testCase.exampleKey);
    const duplicate = keys.find((key, index) => index > 0 && key === keys[index - 1]);
    if (duplicate) throw new Error(`Duplicate builtin example key: ${duplicate}`);
    const digestRecords = ordered.map((testCase) => ({
        key: testCase.exampleKey,
        builtin: testCase.builtin,
        authority: testCase.authority,
        compatibility: testCase.compatibility,
        harness: testCase.harness,
        input: testCase.input,
        expectedOutput: testCase.expectedOutput,
        hasExpectedOutput: testCase.hasExpectedOutput,
        verification: testCase.verification ?? null
    }));
    return {
        cases: ordered,
        keys,
        digest: `sha256:${createHash("sha256").update(JSON.stringify(digestRecords)).digest("hex")}`
    };
}

export function shardInventory(inventory, shard) {
    const start = Math.floor((inventory.cases.length * shard.index) / shard.count);
    const end = Math.floor((inventory.cases.length * (shard.index + 1)) / shard.count);
    return {
        cases: inventory.cases.slice(start, end),
        range: { startInclusive: start, endExclusive: end }
    };
}

export function shardFileSuffix(shard) {
    return shard.count === 1
        ? ""
        : `.shard-${String(shard.index).padStart(4, "0")}-of-${String(shard.count).padStart(4, "0")}`;
}

function normalizeKey(value, label) {
    const normalized = String(value ?? "").trim().toLowerCase();
    if (!normalized) throw new Error(`Missing ${label}`);
    return normalized;
}

function parseInteger(value, name) {
    if (!/^(0|[1-9]\d*)$/.test(value)) throw new Error(`${name} must be a non-negative integer`);
    return Number(value);
}
