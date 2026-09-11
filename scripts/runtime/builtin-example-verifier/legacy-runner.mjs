#!/usr/bin/env node
// @ts-check

import {
    existsSync,
    mkdirSync,
    readFileSync,
    unlinkSync,
    writeFileSync
} from "fs";
import { basename, dirname, join, resolve } from "path";
import { fileURLToPath } from "url";
import {
    mergeLaneResults,
    usesBrowserLane,
    usesNativeLane
} from "./lanes.mjs";
import { runNativeCases } from "./native.mjs";
import {
    createInventory,
    exampleKey,
    resolveShardConfig,
    shardFileSuffix,
    shardInventory
} from "./sharding.mjs";
import { buildMachineReport } from "./reporting.mjs";
import { createRunnerHtml } from "./browser-template.mjs";
import { runHeadlessChrome } from "./browser-runner.mjs";
import { collectCases, matchesVerification, printInventory, readBuiltinDocuments, resolveReportSource } from "./documents.mjs";
import { resolveConcurrency, resolveLogIntervalMs, resolveNativeTimeoutMs, resolveOverallTimeoutMs, resolveTimeoutMs } from "./configuration.mjs";
import { buildReportHtml, buildReportMarkdown, filterReportRows, formatExecutionOutput, normalizeOutput, resolveReportMode, writePlotImageArtifact } from "./presentation.mjs";

/**
 * @typedef {import("../metadata/BuiltinMetadataSpecification").BuiltinMetadata} BuiltinMetadata
 * @typedef {import("../metadata/BuiltinMetadataSpecification").Example} BuiltinExample
 */

/**
 * @typedef {object} ExampleCase
 * @property {number} id
 * @property {string} exampleKey
 * @property {string} builtin
 * @property {string} file
 * @property {string} description
 * @property {string} input
 * @property {string} expectedOutput
 * @property {boolean} hasExpectedOutput
 * @property {number} exampleIndex
 * @property {string} category
 * @property {boolean} isPlotExample
 * @property {string} authority
 * @property {string} compatibility
 * @property {string} harness
 * @property {unknown} fixture
 * @property {unknown} requirements
 * @property {unknown} verification
 */

/**
 * @typedef {object} RunnerResult
 * @property {number} id
 * @property {string} stdoutText
 * @property {string} valueText
 * @property {string} errorText
 * @property {string} errorIdentifier
 * @property {"unavailable"} [availability]
 * @property {string} [figurePngBase64]
 * @property {string} [figureImageError]
 */

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = findRepoRoot(scriptDir);
const outputDir = process.env.RUNMAT_EXAMPLE_OUTPUT_DIR
    ? resolve(process.env.RUNMAT_EXAMPLE_OUTPUT_DIR)
    : join(repoRoot, "scripts", "example-output-reports");
const imageOutputDir = join(outputDir, "plot-example-images");
const chromeWrapper = join(repoRoot, "scripts", "runtime", "chrome-headless.sh");
const wasmModule = process.env.RUNMAT_EXAMPLE_WASM_MODULE
    ? resolve(process.env.RUNMAT_EXAMPLE_WASM_MODULE)
    : join(repoRoot, "bindings", "ts", "dist", "pkg-web", "runmat_wasm_web.js");
const wasmBinary = process.env.RUNMAT_EXAMPLE_WASM_BINARY
    ? resolve(process.env.RUNMAT_EXAMPLE_WASM_BINARY)
    : join(repoRoot, "bindings", "ts", "dist", "pkg-web", "runmat_wasm_web_bg.wasm");
const documents = readBuiltinDocuments(repoRoot);

if (process.argv.includes("--check-inventory")) {
    printInventory(documents);
    process.exit(process.exitCode ?? 0);
}

const completeInventory = createInventory(collectCases(documents));
const requestedKeys = readRequestedCaseKeys();
const inventory = requestedKeys
    ? createInventory(completeInventory.cases.filter((testCase) => requestedKeys.has(testCase.exampleKey)))
    : completeInventory;
if (requestedKeys && inventory.cases.length !== requestedKeys.size) {
    const found = new Set(inventory.cases.map((testCase) => testCase.exampleKey));
    const missing = [...requestedKeys].filter((key) => !found.has(key));
    throw new Error(`Requested builtin example keys are absent from the current inventory: ${missing.join(", ")}`);
}
if (inventory.cases.length === 0) {
    if (process.env.RUNMAT_EXAMPLE_BUILTIN || process.env.RUNMAT_EXAMPLE_FILTER) {
        console.error("No examples matched the requested builtin or text filter.");
        process.exit(1);
    }
    console.log("No examples found to run.");
    process.exit(0);
}
const shard = resolveShardConfig();
const selected = shardInventory(inventory, shard);
const cases = selected.cases;
const reportSuffix = shardFileSuffix(shard);
const reportPath = join(outputDir, `example-output-report${reportSuffix}.html`);
const markdownReportPath = join(outputDir, `example-output-report${reportSuffix}.md`);
const machineReportPath = process.env.RUNMAT_EXAMPLE_REPORT_JSON
    ? resolve(repoRoot, process.env.RUNMAT_EXAMPLE_REPORT_JSON)
    : join(outputDir, `example-output-report${reportSuffix}.json`);

const timeoutMs = resolveTimeoutMs();
const nativeTimeoutMs = resolveNativeTimeoutMs(timeoutMs);
const concurrency = resolveConcurrency();
const requestedLane = process.env.RUNMAT_EXAMPLE_EXECUTION_LANE ?? "";
const browserCases = cases.filter((testCase) => requestedLane
    ? requestedLane.startsWith("browser-")
    : usesBrowserLane(testCase.harness));
const nativeCases = cases.filter((testCase) => requestedLane
    ? requestedLane.startsWith("native-")
    : usesNativeLane(testCase.harness));
const unsupportedCases = requestedLane
    ? []
    : cases.filter((testCase) => !usesBrowserLane(testCase.harness) && !usesNativeLane(testCase.harness));
if (browserCases.length > 0 && (!existsSync(wasmModule) || !existsSync(wasmBinary))) {
    console.error("Missing wasm artifacts required by the selected browser examples. Build bindings/ts dist before running this script.");
    console.error(`Expected: ${wasmModule}`);
    console.error(`Expected: ${wasmBinary}`);
    process.exit(1);
}
const overallTimeoutMs = resolveOverallTimeoutMs(timeoutMs, concurrency, browserCases.length);
const logIntervalMs = resolveLogIntervalMs();
const reportMode = resolveReportMode();
const runnerHtml = createRunnerHtml(
    timeoutMs,
    concurrency,
    logIntervalMs,
    requestedLane ? requestedLane === "browser-wgpu" : true
);
const casesJson = JSON.stringify(browserCases);

const browserResults = browserCases.length > 0
    ? await runHeadlessChrome({ repoRoot, chromeWrapper, runnerHtml, casesJson, overallTimeoutMs, totalCases: browserCases.length, wasmModule, wasmBinary })
    : [];
const nativeResults = nativeCases.length > 0 ? await runNativeCases(repoRoot, nativeCases, nativeTimeoutMs) : [];
const results = mergeLaneResults(cases, browserResults, nativeResults, unsupportedCases);

const resultsById = new Map(results.map((result) => [result.id, result]));

mkdirSync(outputDir, { recursive: true });
mkdirSync(imageOutputDir, { recursive: true });

const rows = cases.map((testCase) => {
    const result = resultsById.get(testCase.id);
    const executionOutput = formatExecutionOutput(result);
    const normalizedExpected = normalizeOutput(testCase.expectedOutput);
    const normalizedActual = normalizeOutput(executionOutput);
    const imageRelPath = writePlotImageArtifact(imageOutputDir, testCase, result);
    const captureError = result && typeof result.figureImageError === "string" ? result.figureImageError : "";
    const executionError = result && typeof result.errorText === "string" ? result.errorText : "";
    const imageError = captureError || (executionError ? "image capture skipped because example execution returned an error" : "");
    const hasExecutionError = Boolean(result && typeof result.errorText === "string" && result.errorText.trim().length > 0);
    return {
        testCase,
        normalizedExpected,
        normalizedActual,
        imageRelPath,
        imageError,
        matches: matchesVerification(testCase, result, normalizedExpected, normalizedActual, hasExecutionError, imageRelPath)
    };
});

const plotImageErrors = rows
    .filter((row) => row.testCase.isPlotExample && !row.imageRelPath)
    .map((row) => {
        const why = row.imageError && row.imageError.trim().length > 0 ? row.imageError.trim() : "(no error message)";
        return `#${row.testCase.id} ${row.testCase.file} example ${row.testCase.exampleIndex + 1}\n${why}`;
    });
if (plotImageErrors.length > 0) {
    writeFileSync(join(outputDir, `plot-image-capture-errors${reportSuffix}.txt`), `${plotImageErrors.join("\n\n")}\n`, "utf8");
} else {
    const captureErrorsPath = join(outputDir, `plot-image-capture-errors${reportSuffix}.txt`);
    if (existsSync(captureErrorsPath)) {
        unlinkSync(captureErrorsPath);
    }
}

const reportRows = filterReportRows(rows, reportMode);
const reportHtml = buildReportHtml(rows, reportRows, reportMode);
writeFileSync(reportPath, reportHtml, "utf8");

const reportMarkdown = buildReportMarkdown(rows, reportRows, reportMode);
writeFileSync(markdownReportPath, reportMarkdown, "utf8");

const machineReport = buildMachineReport({
    rows,
    inventory,
    shard,
    range: selected.range,
    source: resolveReportSource(repoRoot),
    artifact: process.env.RUNMAT_EXAMPLE_ARTIFACT ?? basename(machineReportPath)
});
mkdirSync(dirname(machineReportPath), { recursive: true });
writeFileSync(machineReportPath, `${JSON.stringify(machineReport, null, 2)}\n`, "utf8");
writeRawResult(rows, resultsById);

console.log(`Wrote ${shard.count === 1 ? "consolidated" : `shard ${shard.index}/${shard.count}`} reports to:
  HTML: ${reportPath}
  Markdown: ${markdownReportPath}
  JSON: ${machineReportPath}`);
if (rows.some((row) => !row.matches)) {
    process.exitCode = 1;
}

function findRepoRoot(startDir) {
    let current = startDir;
    while (true) {
      if (existsSync(join(current, "rust-toolchain.toml"))) {
            return current;
        }
        const parent = dirname(current);
        if (parent === current) {
            return startDir;
        }
        current = parent;
    }
}

function readRequestedCaseKeys() {
    const path = process.env.RUNMAT_EXAMPLE_CASE_KEYS_FILE;
    if (!path) return null;
    const parsed = JSON.parse(readFileSync(resolve(path), "utf8"));
    if (!Array.isArray(parsed) || parsed.some((key) => typeof key !== "string" || !key)) {
        throw new Error("RUNMAT_EXAMPLE_CASE_KEYS_FILE must contain an array of non-empty strings");
    }
    if (new Set(parsed).size !== parsed.length) {
        throw new Error("RUNMAT_EXAMPLE_CASE_KEYS_FILE contains duplicate keys");
    }
    return new Set(parsed);
}

function writeRawResult(rows, resultsById) {
    const path = process.env.RUNMAT_EXAMPLE_RAW_RESULT_JSON;
    if (!path) return;
    const raw = rows.map((row) => ({
        exampleKey: row.testCase.exampleKey,
        matches: row.matches,
        normalizedExpected: row.normalizedExpected,
        normalizedActual: row.normalizedActual,
        imageRelPath: row.imageRelPath || null,
        imageError: row.imageError || null,
        result: resultsById.get(row.testCase.id) ?? null
    }));
    const target = resolve(path);
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, `${JSON.stringify(raw, null, 2)}\n`, "utf8");
}

/**
 * @returns {BuiltinMetadata[]}
 */
