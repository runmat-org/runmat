// @ts-check

import { mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { basename, dirname, join, resolve, sep } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { laneAdapterAvailability } from "./lanes.mjs";
import { buildShardResult } from "./result-manifest.mjs";
import { resolveArtifactPath } from "./artifacts.mjs";

const moduleDirectory = dirname(fileURLToPath(import.meta.url));

export function executeShard({ inventory, plan, lanePlan, shard, artifactManifest, artifactManifestPath, resultPath, workDirectory, resolveAdapterAvailability = laneAdapterAvailability }) {
    const units = new Map(inventory.executionUnits.map((unit) => [unit.executionIdentity, unit]));
    const expected = shard.executionIdentities.map((id) => {
        const unit = units.get(id);
        if (!unit) throw new Error(`Planned execution identity is absent from inventory: ${id}`);
        return unit;
    });
    const availability = resolveAdapterAvailability(lanePlan.lane);
    if (!availability.available) {
        return writeResult(resultPath, shardResult(plan, lanePlan, shard, artifactManifest, {
            available: false,
            reason: availability.reason
        }, expected.map((unit) => baseResult(unit, "unavailable", { errorText: availability.reason }))));
    }
    if (!artifactManifest) throw new Error(`Lane ${lanePlan.lane} requires a ${lanePlan.product} artifact manifest`);
    const absoluteResult = resolve(resultPath);
    const shardWork = resolve(workDirectory ?? join(dirname(absoluteResult), `.${basename(absoluteResult)}.work`));
    mkdirSync(shardWork, { recursive: true });
    const keysPath = join(shardWork, "case-keys.json");
    const rawPath = join(shardWork, "raw-results.json");
    const legacyReportPath = join(shardWork, "legacy-report.json");
    if ([keysPath, rawPath, legacyReportPath].some(safeExists)) throw new Error(`Shard work directory contains prior verifier evidence: ${shardWork}`);
    writeFileSync(keysPath, `${JSON.stringify(expected.map((unit) => unit.runnerKey), null, 2)}\n`, "utf8");
    const env = {
        ...process.env,
        RUNMAT_EXAMPLE_CASE_KEYS_FILE: keysPath,
        RUNMAT_EXAMPLE_EXECUTION_LANE: lanePlan.lane,
        RUNMAT_EXAMPLE_OUTPUT_DIR: shardWork,
        RUNMAT_EXAMPLE_RAW_RESULT_JSON: rawPath,
        RUNMAT_EXAMPLE_REPORT_JSON: legacyReportPath,
        RUNMAT_EXAMPLE_SOURCE: plan.sourceRevision,
        RUNMAT_EXAMPLE_ARTIFACT: artifactManifest.artifactManifestDigest,
        RUNMAT_EXAMPLE_TIMEOUT_MS: String(lanePlan.limits.perCaseTimeoutMs),
        RUNMAT_EXAMPLE_NATIVE_TIMEOUT_MS: String(lanePlan.limits.perCaseTimeoutMs),
        RUNMAT_EXAMPLE_TOTAL_TIMEOUT_MS: String(lanePlan.limits.shardWallTimeoutMs),
        RUNMAT_EXAMPLE_CONCURRENCY: String(lanePlan.limits.concurrency),
        RUNMAT_EXAMPLE_SHARD_INDEX: "0",
        RUNMAT_EXAMPLE_SHARD_COUNT: "1"
    };
    if (lanePlan.product === "native-cli") env.RUNMAT_EXAMPLE_NATIVE_BINARY = resolveArtifactPath(artifactManifest, artifactManifestPath, "runmat-binary");
    if (lanePlan.product === "browser-wasm") {
        env.RUNMAT_EXAMPLE_WASM_MODULE = resolveArtifactPath(artifactManifest, artifactManifestPath, "wasm-js");
        env.RUNMAT_EXAMPLE_WASM_BINARY = resolveArtifactPath(artifactManifest, artifactManifestPath, "wasm-binary");
    }
    if (lanePlan.product === "desktop-native") env.RUNMAT_EXAMPLE_DESKTOP_BINARY = resolveArtifactPath(artifactManifest, artifactManifestPath, "runmat-desktop-binary");
    const completed = spawnSync(process.execPath, [join(moduleDirectory, "legacy-runner.mjs"), "--all"], {
        cwd: repositoryRoot(moduleDirectory),
        env,
        encoding: "utf8",
        timeout: lanePlan.limits.shardWallTimeoutMs,
        maxBuffer: lanePlan.limits.maxOutputBytes
    });
    let results;
    if (completed.error || !safeExists(rawPath)) {
        const timedOut = completed.error?.code === "ETIMEDOUT";
        const detail = completed.error?.message ?? completed.stderr?.trim() ?? `runner exited with status ${completed.status}`;
        results = expected.map((unit) => baseResult(unit, timedOut ? "timed-out" : "infra-error", { errorText: detail }));
    } else {
        if (statSync(rawPath).size > lanePlan.limits.maxResultBytes) {
            results = expected.map((unit) => baseResult(unit, "infra-error", { errorText: "runner result exceeded the planned byte limit" }));
        } else {
            try {
                results = normalizeRunnerRecords(expected, JSON.parse(readFileSync(rawPath, "utf8")), shardWork, lanePlan.limits);
            } catch (error) {
                const detail = error instanceof Error ? error.message : String(error);
                results = expected.map((unit) => baseResult(unit, "infra-error", { errorText: `runner result was not valid JSON: ${detail}` }));
            }
        }
    }
    return writeResult(resultPath, shardResult(plan, lanePlan, shard, artifactManifest, { available: true, reason: "" }, results));
}

export function normalizeRunnerRecords(expected, raw, shardWork, limits) {
    if (!Array.isArray(raw)) return expected.map((unit) => baseResult(unit, "infra-error", { errorText: "runner result was not an array" }));
    const expectedKeys = new Set(expected.map((unit) => unit.runnerKey));
    const byKey = new Map();
    for (const record of raw) {
        if (!record || typeof record.exampleKey !== "string" || byKey.has(record.exampleKey) || !expectedKeys.has(record.exampleKey)) {
            return expected.map((unit) => baseResult(unit, "infra-error", { errorText: "runner returned a duplicate, malformed, or unplanned record" }));
        }
        byKey.set(record.exampleKey, record);
    }
    return expected.map((unit) => {
        const record = byKey.get(unit.runnerKey);
        if (!record) return baseResult(unit, "infra-error", { errorText: "runner omitted the planned execution unit" });
        const normalizedExpected = typeof record.normalizedExpected === "string" ? record.normalizedExpected : "";
        const normalizedActual = typeof record.normalizedActual === "string" ? record.normalizedActual : "";
        const errorText = typeof record.result?.errorText === "string"
            ? record.result.errorText
            : typeof record.imageError === "string" ? record.imageError : "";
        if (Buffer.byteLength(normalizedExpected) + Buffer.byteLength(normalizedActual) + Buffer.byteLength(errorText) > limits.maxOutputBytes) {
            return baseResult(unit, "infra-error", { errorText: "runner output exceeded the planned byte limit" });
        }
        let imageRelativePath = typeof record.imageRelPath === "string" ? record.imageRelPath : null;
        if (imageRelativePath) {
            const imagePath = resolve(shardWork, imageRelativePath);
            const root = `${resolve(shardWork)}${sep}`;
            if (!imagePath.startsWith(root) || !safeExists(imagePath) || statSync(imagePath).size > limits.maxFigureBytes) {
                return baseResult(unit, "infra-error", { errorText: "runner figure was absent, escaped the shard directory, or exceeded the planned byte limit" });
            }
        }
        const timedOut = /timeout/i.test(errorText);
        const unavailable = record.result?.availability === "unavailable";
        return baseResult(unit, unavailable ? "unavailable" : record.matches === true ? "passed" : timedOut ? "timed-out" : "failed", {
            errorText,
            errorIdentifier: typeof record.result?.errorIdentifier === "string" ? record.result.errorIdentifier : "",
            normalizedExpected,
            normalizedActual,
            imageRelativePath
        });
    });
}

function shardResult(plan, lanePlan, shard, artifactManifest, adapter, results) {
    const productPlan = plan.products.find((product) => product.kind === lanePlan.product);
    if (!productPlan) throw new Error(`Plan contains no artifact contract for ${lanePlan.product}`);
    return buildShardResult({
        sourceRevision: plan.sourceRevision,
        sourceState: plan.sourceState,
        inventoryDigest: plan.inventoryDigest,
        planDigest: plan.planDigest,
        runnerDigest: plan.runnerDigest,
        lane: lanePlan.lane,
        shardIndex: shard.index,
        shardCount: lanePlan.shardCount,
        assignmentDigest: shard.assignmentDigest,
        product: lanePlan.product,
        artifactProfile: productPlan.artifactProfile,
        artifactManifestDigest: artifactManifest?.artifactManifestDigest ?? null,
        environment: { platform: process.platform, architecture: process.arch, node: process.version },
        limits: lanePlan.limits,
        adapter,
        results
    });
}

function baseResult(unit, status, details) {
    return {
        executionIdentity: unit.executionIdentity,
        exampleIdentity: unit.exampleIdentity,
        definitionDigest: unit.definitionDigest,
        builtinKey: unit.builtinKey,
        exampleId: unit.exampleId,
        lane: unit.lane,
        status,
        errorIdentifier: details.errorIdentifier ?? "",
        errorText: details.errorText ?? "",
        normalizedExpected: details.normalizedExpected ?? "",
        normalizedActual: details.normalizedActual ?? "",
        imageRelativePath: details.imageRelativePath ?? null
    };
}

function writeResult(path, manifest) {
    const target = resolve(path);
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
    return manifest;
}

function safeExists(path) {
    try { readFileSync(path); return true; } catch { return false; }
}

function repositoryRoot(start) {
    return resolve(start, "../../..");
}
