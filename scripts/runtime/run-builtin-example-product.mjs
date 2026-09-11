#!/usr/bin/env node

import { readJson } from "./builtin-example-verifier/io.mjs";
import { executeProductPlan } from "./builtin-example-verifier/product-runner.mjs";

const options = parseOptions(process.argv.slice(2));
const artifactManifestPath = required(options["artifact-manifest"], "--artifact-manifest");
const results = executeProductPlan({
    inventory: readJson(required(options.inventory, "--inventory")),
    plan: readJson(required(options.plan, "--plan")),
    artifactManifest: readJson(artifactManifestPath),
    artifactManifestPath,
    outputDirectory: required(options["output-dir"], "--output-dir")
});
const failed = results.filter((result) => result.status !== "passed");
console.log(`Executed ${results.length} product shards; ${failed.length} did not pass`);
if (failed.length) process.exitCode = 1;

function parseOptions(args) {
    const result = {};
    for (let index = 0; index < args.length; index += 2) {
        const name = args[index];
        const value = args[index + 1];
        if (!name?.startsWith("--") || value === undefined) throw new Error("Options must use --name value pairs");
        result[name.slice(2)] = value;
    }
    return result;
}

function required(value, name) {
    if (typeof value !== "string" || !value) throw new Error(`${name} is required`);
    return value;
}
