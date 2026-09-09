#!/usr/bin/env node
// @ts-check

import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { combineMachineReports } from "./builtin-example-verifier/reporting.mjs";

const args = process.argv.slice(2);
let output = "";
let source = "";
const inputs = [];
for (let index = 0; index < args.length; index += 1) {
    if (args[index] === "--output") output = requireValue(args, ++index, "--output");
    else if (args[index] === "--source") source = requireValue(args, ++index, "--source");
    else inputs.push(args[index]);
}
if (!output || inputs.length === 0) {
    throw new Error("Usage: combine-builtin-example-reports.mjs --output <report.json> [--source <source>] <shard.json>...");
}
const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, "../..");
const expectedSource = source || `git:${execFileSync("git", ["rev-parse", "HEAD"], { cwd: repoRoot, encoding: "utf8" }).trim()}`;
const reports = inputs.map((path) => JSON.parse(readFileSync(resolve(path), "utf8")));
const combined = combineMachineReports(reports, expectedSource);
const outputPath = resolve(output);
mkdirSync(dirname(outputPath), { recursive: true });
writeFileSync(outputPath, `${JSON.stringify(combined, null, 2)}\n`, "utf8");
console.log(`Validated ${reports.length} shards and wrote ${outputPath}`);
if (combined.summary.failed > 0) process.exitCode = 1;

function requireValue(values, index, flag) {
    const value = values[index];
    if (!value || value.startsWith("--")) throw new Error(`${flag} requires a value`);
    return value;
}
