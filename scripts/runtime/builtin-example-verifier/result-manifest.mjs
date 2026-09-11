// @ts-check

import { compareUtf8, digestObject } from "./identity.mjs";
import { isExecutionLane, laneProduct } from "./lanes.mjs";
import { requiredArtifactRoles } from "./plan.mjs";
import { digest, enumValue, exactKeys, gitRevision, integer } from "./schema.mjs";

export const SHARD_RESULT_SCHEMA = "runmat.builtin-examples.shard-result.v1";
export const RESULT_STATUSES = Object.freeze(["passed", "failed", "unavailable", "infra-error", "timed-out"]);

export function buildShardResult(fields) {
    const results = [...fields.results].sort((left, right) => compareUtf8(left.executionIdentity, right.executionIdentity));
    const summary = summarize(results);
    const manifest = {
        schema: SHARD_RESULT_SCHEMA,
        sourceRevision: fields.sourceRevision,
        sourceState: fields.sourceState,
        inventoryDigest: fields.inventoryDigest,
        planDigest: fields.planDigest,
        runnerDigest: fields.runnerDigest,
        lane: fields.lane,
        shardIndex: fields.shardIndex,
        shardCount: fields.shardCount,
        assignmentDigest: fields.assignmentDigest,
        product: fields.product,
        artifactProfile: fields.artifactProfile,
        artifactManifestDigest: fields.artifactManifestDigest ?? null,
        environment: fields.environment,
        limits: fields.limits,
        adapter: fields.adapter,
        status: shardStatus(results, fields.adapter),
        results,
        summary,
        shardResultDigest: ""
    };
    manifest.shardResultDigest = digestObject(manifest, ["shardResultDigest"]);
    return manifest;
}

export function validateShardResult(manifest) {
    if (!manifest || manifest.schema !== SHARD_RESULT_SCHEMA) throw new Error("Unsupported builtin example shard result schema");
    exactKeys(manifest, ["schema", "sourceRevision", "sourceState", "inventoryDigest", "planDigest", "runnerDigest", "lane", "shardIndex", "shardCount", "assignmentDigest", "product", "artifactProfile", "artifactManifestDigest", "environment", "limits", "adapter", "status", "results", "summary", "shardResultDigest"], "shard result");
    gitRevision(manifest.sourceRevision);
    enumValue(manifest.sourceState, ["clean", "dirty"], "shard source state");
    for (const [value, label] of [[manifest.inventoryDigest, "shard inventory digest"], [manifest.planDigest, "shard plan digest"], [manifest.runnerDigest, "shard runner digest"], [manifest.assignmentDigest, "shard assignment digest"], [manifest.shardResultDigest, "shard result digest"]]) digest(value, label);
    enumValue(manifest.product, ["native-cli", "browser-wasm"], "shard product");
    if (!isExecutionLane(manifest.lane) || laneProduct(manifest.lane) !== manifest.product) throw new Error("Shard lane and product are inconsistent");
    requiredArtifactRoles(manifest.product, manifest.artifactProfile);
    if (manifest.artifactManifestDigest !== null) digest(manifest.artifactManifestDigest, "shard artifact manifest digest");
    integer(manifest.shardIndex, "shard index");
    integer(manifest.shardCount, "shard count", 1);
    if (manifest.shardIndex >= manifest.shardCount) throw new Error("Shard index is outside the shard count");
    exactKeys(manifest.environment, ["platform", "architecture", "node"], "shard environment");
    for (const [name, value] of Object.entries(manifest.environment)) if (typeof value !== "string" || !value) throw new Error(`Invalid shard environment ${name}`);
    exactKeys(manifest.limits, ["perCaseTimeoutMs", "shardWallTimeoutMs", "concurrency", "maxOutputBytes", "maxResultBytes", "maxFigureBytes"], "shard limits");
    for (const [name, value] of Object.entries(manifest.limits)) integer(value, `shard limit ${name}`, 1);
    exactKeys(manifest.adapter, ["available", "reason"], "shard adapter");
    if (typeof manifest.adapter.available !== "boolean" || typeof manifest.adapter.reason !== "string") throw new Error("Invalid shard adapter declaration");
    if (manifest.adapter.available && manifest.adapter.reason !== "") throw new Error("Available shard adapter cannot have an unavailable reason");
    if (!manifest.adapter.available && !manifest.adapter.reason.trim()) throw new Error("Unavailable shard adapter requires a reason");
    if (manifest.adapter.available && manifest.artifactManifestDigest === null) throw new Error("Available shard requires an artifact manifest digest");
    exactKeys(manifest.summary, ["total", "byStatus"], "shard summary");
    exactKeys(manifest.summary.byStatus, RESULT_STATUSES, "shard status counts");
    if (digestObject(manifest, ["shardResultDigest"]) !== manifest.shardResultDigest) throw new Error("Shard result digest mismatch");
    if (!Array.isArray(manifest.results)) throw new Error("Shard result records must be an array");
    const seen = new Set();
    for (const result of manifest.results) {
        exactKeys(result, ["executionIdentity", "exampleIdentity", "definitionDigest", "builtinKey", "exampleId", "lane", "status", "errorIdentifier", "errorText", "normalizedExpected", "normalizedActual", "imageRelativePath"], `execution result ${result?.executionIdentity ?? "unknown"}`);
        if (!result || typeof result.executionIdentity !== "string" || seen.has(result.executionIdentity)) throw new Error(`Invalid or duplicate shard result identity: ${result?.executionIdentity}`);
        seen.add(result.executionIdentity);
        if (!RESULT_STATUSES.includes(result.status)) throw new Error(`Invalid execution result status: ${result.status}`);
        for (const [value, label] of [[result.executionIdentity, "execution identity"], [result.exampleIdentity, "result example identity"], [result.definitionDigest, "result definition digest"]]) digest(value, label);
        for (const field of ["builtinKey", "exampleId", "lane", "errorIdentifier", "errorText", "normalizedExpected", "normalizedActual"]) {
            if (typeof result[field] !== "string") throw new Error(`Invalid ${field} for ${result.executionIdentity}`);
        }
        if (result.lane !== manifest.lane) throw new Error(`Execution result lane differs from its shard: ${result.executionIdentity}`);
        if (!manifest.adapter.available && result.status !== "unavailable") throw new Error(`Unavailable shard contains an executable result: ${result.executionIdentity}`);
        if (result.imageRelativePath !== null && (typeof result.imageRelativePath !== "string" || !result.imageRelativePath || result.imageRelativePath.startsWith("/") || result.imageRelativePath.split(/[\\/]/).includes(".."))) throw new Error(`Invalid image path for ${result.executionIdentity}`);
    }
    const ordered = [...manifest.results].sort((left, right) => compareUtf8(left.executionIdentity, right.executionIdentity));
    if (JSON.stringify(ordered) !== JSON.stringify(manifest.results)) throw new Error("Shard results are not in canonical identity order");
    if (JSON.stringify(summarize(manifest.results)) !== JSON.stringify(manifest.summary)) throw new Error("Shard result summary mismatch");
    if (shardStatus(manifest.results, manifest.adapter) !== manifest.status) throw new Error("Shard result status mismatch");
    return manifest;
}

export function summarize(results) {
    const byStatus = Object.fromEntries(RESULT_STATUSES.map((status) => [status, 0]));
    for (const result of results) byStatus[result.status] += 1;
    return { total: results.length, byStatus };
}

function shardStatus(results, adapter = { available: true }) {
    if (!adapter.available) return "unavailable";
    if (results.some((result) => result.status === "infra-error")) return "infra-error";
    if (results.some((result) => result.status === "timed-out")) return "timed-out";
    if (results.some((result) => result.status === "unavailable")) return "unavailable";
    if (results.some((result) => result.status === "failed")) return "failed";
    return "passed";
}
