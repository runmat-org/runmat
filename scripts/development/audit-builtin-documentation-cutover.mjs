#!/usr/bin/env node
// @ts-check

// Interactive pre-review helper retained for individual migrations. Its v1
// report is not C00 completion evidence. The migration factory's reviewed
// documentation_cutover gate owns the closed, content-addressed leaf and
// destination reconciliation used by audit and verification.

import { execFileSync, spawnSync } from "child_process";
import { existsSync, writeFileSync } from "fs";
import { dirname, join, resolve } from "path";
import { fileURLToPath } from "url";

const FIELD_RULES = Object.freeze({
    // Legacy sidecars used "aliases" for both callable names and search terms.
    // Catalog entries model each callable identity independently, so preserve
    // this metadata in the searchable documentation vocabulary during cutover.
    aliases: subset("keywords"),
    arguments: semantic("catalog.descriptor"),
    backend: semantic("catalog.placement"),
    behaviors: semantic("sections"),
    category: exact("category"),
    compatibility_notes: semantic("sections"),
    coverage: semantic("evidence"),
    defaults: semantic("sections"),
    description: semantic("description"),
    errors: semantic("catalog.descriptor.errors"),
    examples: semantic("examples"),
    extended_capabilities: semantic("catalog.contract.capabilities"),
    faqs: semantic("faqs"),
    fusion: semantic("catalog.placement.fusion"),
    gpu: semantic("catalog.placement"),
    gpu_behavior: semantic("sections"),
    gpu_residency: semantic("sections"),
    gpu_support: semantic("catalog.placement"),
    hero_image: semantic("media"),
    implemented_in: semantic("evidence.implementation"),
    implementation_notes: semantic("sections"),
    integer_forms: semantic("catalog.integer_capabilities"),
    integer_semantics: semantic("catalog.integer_capabilities"),
    introduced: exact("introduced"),
    jsonencode_options: semantic("sections"),
    keywords: subset("keywords"),
    limitations: semantic("sections"),
    links: semantic("links"),
    name: exact("key"),
    notes: semantic("sections"),
    options: semantic("sections"),
    outputs: semantic("catalog.descriptor"),
    related: semantic("related"),
    requires_feature: semantic("catalog.link.artifact_dependencies"),
    returns: semantic("catalog.descriptor"),
    signatures: semantic("catalog.descriptor.signatures"),
    slug: exact("slug"),
    source: semantic("evidence.implementation"),
    status: exact("status"),
    summary: semantic("summary"),
    syntax: semantic("catalog.descriptor"),
    tested: semantic("evidence.verification"),
    tests: semantic("evidence.verification"),
    title: exact("title"),
    validation: semantic("evidence.verification"),
    version_added: exact("introduced")
});

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const repository = findRepositoryRoot(scriptDirectory);
const options = parseArguments(process.argv.slice(2));
const key = options.name.toLowerCase();
const referencePath = `docs/builtins/reference/${options.name}.json`;
const runtimeShadowPath = `crates/runmat-runtime/src/builtins/builtins-json/${options.name}.json`;

const sources = [referencePath, runtimeShadowPath]
    .map((path) => readBaselineDocument(options.baseline, path))
    .filter((source) => source !== null);

if (sources.length === 0 && !options.isNew) {
    fail(
        `No documentation input exists for ${options.name} at ${options.baseline}. ` +
        "Use --new only after confirming that the public identity never had a sidecar or runtime shadow."
    );
}
if (sources.length > 0 && options.isNew) {
    fail(`--new is invalid because ${sources.map((source) => source.path).join(", ")} exists at ${options.baseline}`);
}

for (const path of [referencePath, runtimeShadowPath]) {
    if (existsSync(join(repository, path))) {
        fail(`${path} remains in the working tree; delete the old editable authority before auditing the cutover`);
    }
}

const current = readCatalogDocument(key);
validateCanonicalDocument(current, options.name, sources);

const fieldResults = [];
for (const source of sources) {
    for (const [field, value] of Object.entries(source.document).sort(([left], [right]) => left.localeCompare(right))) {
        if (!isPopulated(value)) {
            fieldResults.push(result(source.path, field, "empty", "not populated", null, value));
            continue;
        }
        const rule = FIELD_RULES[field];
        if (!rule) {
            fieldResults.push(result(source.path, field, "unmapped", "no typed destination is declared", null, value));
            continue;
        }
        const targetValue = getPath(current, rule.target);
        if (targetValue === undefined) {
            fieldResults.push(result(source.path, field, "missing-target", rule.target, targetValue, value));
            continue;
        }
        const automatic = rule.kind === "exact"
            ? equivalentDirectValue(field, value, targetValue, key)
            : rule.kind === "subset"
                ? stringSubset(value, targetValue)
                : false;
        if (automatic) {
            fieldResults.push(result(source.path, field, "verified", rule.target, targetValue, value));
        } else if (options.reviewed.has(field)) {
            fieldResults.push(result(source.path, field, "reviewed", rule.target, targetValue, value));
        } else {
            fieldResults.push(result(source.path, field, "review-required", rule.target, targetValue, value));
        }
    }
}

const usedReviews = new Set(fieldResults.filter((entry) => entry.status === "reviewed").map((entry) => entry.field));
const unusedReviews = [...options.reviewed].filter((field) => !usedReviews.has(field)).sort();
const failures = fieldResults.filter((entry) => entry.status === "unmapped" || entry.status === "missing-target" || entry.status === "review-required");
const report = {
    schema_version: 1,
    builtin: key,
    baseline: options.baseline,
    new_documentation: options.isNew,
    source_paths: sources.map((source) => source.path),
    catalog_authority: current.authority,
    catalog_document: current,
    field_results: fieldResults,
    reviewed_fields: [...options.reviewed].sort(),
    unused_reviewed_fields: unusedReviews,
    result: failures.length === 0 && unusedReviews.length === 0 ? "pass" : "fail"
};

if (options.output) {
    writeFileSync(resolve(repository, options.output), `${JSON.stringify(report, null, 2)}\n`, "utf8");
}

printSummary(report, failures);
if (unusedReviews.length > 0) {
    console.error(`Unused --reviewed fields: ${unusedReviews.join(", ")}`);
}
if (failures.length > 0 || unusedReviews.length > 0) {
    process.exit(1);
}

function exact(target) {
    return { kind: "exact", target };
}

function subset(target) {
    return { kind: "subset", target };
}

function semantic(target) {
    return { kind: "semantic", target };
}

function parseArguments(arguments_) {
    let name = null;
    let baseline = "HEAD";
    let output = null;
    let isNew = false;
    const reviewed = new Set();
    for (let index = 0; index < arguments_.length; index += 1) {
        const argument = arguments_[index];
        if (argument === "--baseline") {
            baseline = requireValue(arguments_, ++index, argument);
        } else if (argument === "--output") {
            output = requireValue(arguments_, ++index, argument);
        } else if (argument === "--reviewed") {
            for (const field of requireValue(arguments_, ++index, argument).split(",")) {
                if (field.trim()) reviewed.add(field.trim());
            }
        } else if (argument === "--new") {
            isNew = true;
        } else if (argument.startsWith("-")) {
            fail(`Unknown option: ${argument}`);
        } else if (name === null) {
            name = argument;
        } else {
            fail(`Unexpected argument: ${argument}`);
        }
    }
    if (name === null) {
        fail("Usage: audit-builtin-documentation-cutover.mjs NAME [--baseline REF] [--reviewed field,...] [--output PATH] [--new]");
    }
    if (!/^[A-Za-z][A-Za-z0-9_.]*$/.test(name) || name.includes("..")) {
        fail(`Unsafe builtin identity: ${name}`);
    }
    return { name, baseline, output, isNew, reviewed };
}

function requireValue(arguments_, index, option) {
    const value = arguments_[index];
    if (!value || value.startsWith("--")) fail(`${option} requires a value`);
    return value;
}

function findRepositoryRoot(start) {
    let current = resolve(start);
    while (true) {
        if (existsSync(join(current, "rust-toolchain.toml"))) return current;
        const parent = dirname(current);
        if (parent === current) fail("Could not locate the RunMat repository root");
        current = parent;
    }
}

function readBaselineDocument(baseline, path) {
    const shown = spawnSync("git", ["show", `${baseline}:${path}`], {
        cwd: repository,
        encoding: "utf8",
        maxBuffer: 16 * 1024 * 1024
    });
    if (shown.status !== 0) return null;
    try {
        return { path, document: JSON.parse(shown.stdout) };
    } catch (error) {
        fail(`${baseline}:${path} is not valid JSON: ${error instanceof Error ? error.message : String(error)}`);
    }
}

function readCatalogDocument(name) {
    const encoded = execFileSync(
        "cargo",
        ["run", "--quiet", "-p", "runmat-builtins", "--bin", "export_builtin_documentation", "--", "--transition"],
        { cwd: repository, encoding: "utf8", maxBuffer: 128 * 1024 * 1024 }
    );
    const payload = JSON.parse(encoded);
    const matches = payload.builtins.filter((document) => String(document.key).toLowerCase() === name);
    if (matches.length !== 1) fail(`Expected one exported document for ${name}, found ${matches.length}`);
    return matches[0];
}

function validateCanonicalDocument(document, name, sources) {
    if (document.authority !== "catalog") fail(`${name} is still exported from ${document.authority ?? "an unknown authority"}`);
    if (!String(document.summary ?? "").trim()) fail(`${name} has no catalog summary`);
    if (!String(document.description ?? "").trim()) fail(`${name} has no catalog description`);
    const examples = Array.isArray(document.examples) ? document.examples : [];
    const exemption = String(document.example_exemption ?? "").trim();
    if (examples.length === 0 && !exemption) fail(`${name} has neither typed examples nor a reviewed exemption`);
    const legacyExampleCount = Math.max(0, ...sources.map((source) => Array.isArray(source.document.examples) ? source.document.examples.length : 0));
    if (examples.length < legacyExampleCount) {
        fail(`${name} has ${examples.length} catalog examples but its source inventory contained ${legacyExampleCount}`);
    }
    for (const [index, example] of examples.entries()) {
        if (!example || typeof example !== "object") fail(`${name} example ${index + 1} is not typed`);
        for (const field of ["id", "input", "harness", "verification"]) {
            if (example[field] === null || example[field] === undefined || example[field] === "") {
                fail(`${name} example ${index + 1} has no ${field}`);
            }
        }
    }
}

function isPopulated(value) {
    if (value === null || value === undefined || value === "") return false;
    if (Array.isArray(value)) return value.length > 0;
    if (typeof value === "object") return Object.keys(value).length > 0;
    return true;
}

function getPath(object, path) {
    return path.split(".").reduce((value, part) => value && typeof value === "object" ? value[part] : undefined, object);
}

function equivalentDirectValue(field, source, target, name) {
    const normalize = (value) => String(value ?? "").trim().toLowerCase();
    if (field === "name") return normalize(source) === name;
    if (field === "version_added") return normalize(source) === normalize(target);
    return normalize(source) === normalize(target);
}

function stringSubset(source, target) {
    if (!Array.isArray(source) || !Array.isArray(target)) return false;
    const targetSet = new Set(target.map((value) => String(value).trim().toLowerCase()));
    return source.every((value) => targetSet.has(String(value).trim().toLowerCase()));
}

function result(source, field, status, destination, target, value) {
    return { source, field, status, destination, source_value: value, target_value: target };
}

function printSummary(report, failures) {
    const counts = new Map();
    for (const entry of report.field_results) counts.set(entry.status, (counts.get(entry.status) ?? 0) + 1);
    const rendered = [...counts].sort(([left], [right]) => left.localeCompare(right)).map(([status, count]) => `${status}=${count}`).join(", ");
    console.log(`Builtin documentation cutover: ${report.builtin}`);
    console.log(`Baseline: ${report.baseline}`);
    console.log(`Sources: ${report.source_paths.length > 0 ? report.source_paths.join(", ") : "none (new documentation)"}`);
    console.log(`Fields: ${rendered || "none"}`);
    if (failures.length > 0) {
        console.error("Fields requiring action:");
        for (const entry of failures) console.error(`  ${entry.field}: ${entry.status} -> ${entry.destination}`);
        console.error("Review each semantic change, then pass its field name through --reviewed and retain the JSON report as slice evidence.");
    } else {
        console.log("Result: PASS");
    }
}

function fail(message) {
    console.error(message);
    process.exit(1);
}
