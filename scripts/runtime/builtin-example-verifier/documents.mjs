// @ts-check

import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import { usesBrowserLane, usesNativeLane } from "./lanes.mjs";
import { exampleKey } from "./sharding.mjs";
import {
    NO_EXAMPLE_FIXTURE,
    NO_EXAMPLE_REQUIREMENTS,
    validateBuiltinExampleFixture,
    validateBuiltinExampleRequirements
} from "../../metadata/BuiltinExampleFixtureSchema.mjs";

/** @typedef {import("../../metadata/BuiltinMetadataSpecification").BuiltinMetadata} BuiltinMetadata */
/** @typedef {import("../../metadata/BuiltinMetadataSpecification").Example} BuiltinExample */

export function readBuiltinDocuments(repoRoot) {
    const exporter = process.env.RUNMAT_EXAMPLE_DOCUMENTATION_BINARY;
    const command = exporter ? resolve(repoRoot, exporter) : "cargo";
    const args = exporter
        ? ["--transition"]
        : ["run", "--quiet", "-p", "runmat-builtins", "--bin", "export_builtin_documentation", "--", "--transition"];
    const encoded = execFileSync(
        command,
        args,
        { cwd: repoRoot, encoding: "utf8", maxBuffer: 128 * 1024 * 1024 }
    );
    const payload = JSON.parse(encoded);
    if (![1, 2].includes(payload.schema_version) || !Array.isArray(payload.builtins)) {
        throw new Error("Unsupported builtin documentation export schema");
    }
    return payload.builtins;
}

/**
 * @param {BuiltinMetadata[]} documents
 */
export function printInventory(documents) {
    const authorities = new Map();
    const harnesses = new Map();
    const unsupported = [];
    let examples = 0;
    for (const document of documents) {
        const authority = typeof document.authority === "string" ? document.authority : "unknown";
        authorities.set(authority, (authorities.get(authority) ?? 0) + 1);
        for (const example of Array.isArray(document.examples) ? document.examples : []) {
            examples += 1;
            const harness = typeof example.harness === "string" ? example.harness : "LegacyBrowser";
            harnesses.set(harness, (harnesses.get(harness) ?? 0) + 1);
            if (authority === "catalog" && !usesBrowserLane(harness) && !usesNativeLane(harness)) {
                unsupported.push(`${document.key ?? document.title}:${example.id ?? "unknown"}:${harness}`);
            }
        }
    }
    const render = (values) => [...values.entries()]
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, count]) => `${key}=${count}`)
        .join(", ");
    console.log(`Builtin documentation inventory: documents=${documents.length}, examples=${examples}`);
    console.log(`Authorities: ${render(authorities)}`);
    console.log(`Harnesses: ${render(harnesses)}`);
    if (unsupported.length > 0) {
        console.error(`Catalog examples without an executable adapter: ${unsupported.join(", ")}`);
        process.exitCode = 1;
    }
}

/**
 * @param {BuiltinMetadata[]} documents
 * @returns {ExampleCase[]}
 */
export function collectCases(documents) {
    /** @type {ExampleCase[]} */
    const cases = [];
    let id = 1;

    for (const parsed of documents) {
        const builtinKey = String(parsed.key ?? parsed.title ?? "").toLowerCase();
        const authority = typeof parsed.authority === "string" ? parsed.authority : "unknown";
        const file = authority === "catalog" ? `${builtinKey} (catalog)` : `${builtinKey}.json`;
        const category = typeof parsed.category === "string" ? parsed.category : "";
        const examples = Array.isArray(parsed.examples) ? parsed.examples : [];
        for (let i = 0; i < examples.length; i += 1) {
            const example = examples[i];
            if (!example || typeof example.input !== "string" || example.input.trim().length === 0) {
                continue;
            }
            const harness = typeof example.harness === "string" ? example.harness : "LegacyBrowser";
            const compatibility = typeof example.compatibility === "string" ? example.compatibility : "RunMat";
            const verification = example.verification;
            const fixture = example.fixture ?? NO_EXAMPLE_FIXTURE;
            const requirements = example.requirements ?? {
                ...NO_EXAMPLE_REQUIREMENTS,
                compiler: [],
                runtime: [],
                toolchain: []
            };
            validateBuiltinExampleFixture(fixture, `${builtinKey} example fixture`);
            validateBuiltinExampleRequirements(requirements, `${builtinKey} example requirements`);
            const isPlotExample = is_plot_example(category);
            const hasExpectedOutput = typeof example.output === "string";
            if (!hasExpectedOutput && !isPlotExample && verification === undefined) {
                continue;
            }
            if (hasExpectedOutput && is_comment_only_output(example.output) && verification === undefined) {
                continue;
            }
            const description = typeof example.description === "string" && example.description.trim().length > 0
                ? example.description.trim()
                : `${parsed.title ?? builtinKey} example ${i + 1}`;
            const input = appendVerificationSource(example.input, verification);
            cases.push({
                id: id++,
                exampleKey: exampleKey(builtinKey, example, i),
                builtin: parsed.title ?? builtinKey,
                file,
                description,
                input,
                expectedOutput: hasExpectedOutput ? example.output : "",
                hasExpectedOutput,
                exampleIndex: i,
                category,
                isPlotExample,
                authority,
                compatibility,
                harness,
                fixture,
                requirements,
                verification
            });
        }
    }

    const filtered = applyCaseFilter(cases);
    return applyCaseLimit(filtered);
}

export function resolveReportSource(repoRoot) {
    const override = process.env.RUNMAT_EXAMPLE_SOURCE;
    if (override && override.trim()) return override.trim();
    try {
        return `git:${execFileSync("git", ["rev-parse", "HEAD"], { cwd: repoRoot, encoding: "utf8" }).trim()}`;
    } catch (_error) {
        return `inventory:${inventory.digest}`;
    }
}

/**
 * @param {string} input
 * @param {unknown} verification
 */
function appendVerificationSource(input, verification) {
    if (!verification || typeof verification !== "object" || !("Assertions" in verification)) {
        return input;
    }
    const assertions = verification.Assertions;
    if (!assertions || typeof assertions !== "object" || typeof assertions.source !== "string") {
        return input;
    }
    return `${input.trimEnd()}\n${assertions.source}`;
}

/**
 * @param {ExampleCase} testCase
 * @param {RunnerResult | undefined} result
 * @param {string} normalizedExpected
 * @param {string} normalizedActual
 * @param {boolean} hasExecutionError
 * @param {string} imagePath
 */
export function matchesVerification(testCase, result, normalizedExpected, normalizedActual, hasExecutionError, imagePath) {
    const verification = testCase.verification;
    if (verification === "Succeeds") {
        return !hasExecutionError;
    }
    if (verification && typeof verification === "object") {
        if ("Assertions" in verification) {
            return !hasExecutionError;
        }
        if ("ExpectedError" in verification) {
            const expected = verification.ExpectedError;
            const identifier = expected && typeof expected === "object" ? expected.identifier : "";
            return typeof identifier === "string"
                && identifier.length > 0
                && result?.errorIdentifier === identifier;
        }
        if ("Figure" in verification) {
            return !hasExecutionError && Boolean(imagePath);
        }
    }
    return testCase.hasExpectedOutput ? normalizedExpected === normalizedActual : !hasExecutionError;
}

/**
 * @param {ExampleCase[]} cases
 */
function applyCaseFilter(cases) {
    const builtin = process.env.RUNMAT_EXAMPLE_BUILTIN;
    const raw = process.env.RUNMAT_EXAMPLE_FILTER;
    if (builtin && raw) {
        throw new Error("RUNMAT_EXAMPLE_BUILTIN and RUNMAT_EXAMPLE_FILTER cannot be combined");
    }
    if (builtin) {
        const identity = builtin.trim().toLowerCase();
        if (identity.length === 0) {
            throw new Error("RUNMAT_EXAMPLE_BUILTIN must name one builtin identity");
        }
        return cases.filter((testCase) => testCase.builtin.toLowerCase() === identity);
    }
    if (!raw || raw.trim().length === 0) {
        return cases;
    }
    const needle = raw.toLowerCase();
    return cases.filter((testCase) => {
        const hay = `${testCase.builtin}\n${testCase.file}\n${testCase.description}\n${testCase.input}\n${testCase.category}`.toLowerCase();
        return hay.includes(needle);
    });
}

/**
 * @param {ExampleCase[]} cases
 */
function applyCaseLimit(cases) {
    const raw = process.env.RUNMAT_EXAMPLE_LIMIT;
    if (!raw) {
        return cases;
    }
    const n = Number.parseInt(raw, 10);
    if (!Number.isFinite(n) || n <= 0) {
        return cases;
    }
    return cases.slice(0, n);
}

/**
 * @param {string} category
 */
function is_plot_example(category) {
    return category === "plotting" || category.startsWith("plotting/");
}

/**
 * Treat examples with only comment lines as documentation-only.
 * @param {string} output
 */
function is_comment_only_output(output) {
    const lines = output.split(/\r?\n/);
    let saw_comment = false;
    for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) {
            continue;
        }
        if (trimmed.startsWith("%")) {
            saw_comment = true;
            continue;
        }
        return false;
    }
    return saw_comment;
}
