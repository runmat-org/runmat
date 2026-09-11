#!/usr/bin/env node
// @ts-check

import { stageNativeProduct } from "./builtin-example-verifier/product-staging.mjs";

const options = parse(process.argv.slice(2));
const result = stageNativeProduct({
    binary: required(options.binary, "--binary"),
    destination: required(options.destination, "--destination"),
    executableName: options["executable-name"],
    dependencyDirectories: array(options["dependency-dir"])
});
console.log(`Staged ${result.fileCount} native product files in ${result.destination}`);

function parse(args) {
    const result = {};
    for (let index = 0; index < args.length; index += 1) {
        const name = args[index];
        if (!name.startsWith("--")) throw new Error(`Unexpected positional argument: ${name}`);
        const value = args[++index];
        if (!value || value.startsWith("--")) throw new Error(`${name} requires a value`);
        const key = name.slice(2);
        result[key] = key === "dependency-dir" ? [...array(result[key]), value] : value;
    }
    return result;
}

function required(value, flag) {
    if (typeof value !== "string" || !value) throw new Error(`${flag} is required`);
    return value;
}

function array(value) {
    if (value === undefined) return [];
    return Array.isArray(value) ? value : [value];
}
