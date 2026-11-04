// Registry coverage (METHOD §4.3, E-DOC-6): every name in the compiler's closed registries must
// appear, in backticks, on the reference page that owns it.
//
//   node scripts/docs/registry-coverage.mjs [--json]
//
// The names are read from the registries' own literal tables in the source (the match arms or
// constant arrays the compiler uses), so there is no second hand-kept list. If a registry moves or
// changes shape, extraction finds no names and this check fails, naming the registry to re-anchor.
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ARM = /"([A-Za-z_][A-Za-z0-9_^]*)"\s*(?=\||=>)/g;
const OPERATION = /(?:projection|spatial_reduction|extremum|series_operation|case_operation|operation)\(\s*"([a-z_]+)"/g;
const STRING = /"([a-z_0-9]+)"/g;

/** Each registry: where its literal table lives, how to read names from it, and which pages own it. */
export const REGISTRIES = [
    {
        id: "units",
        source: "crates/runmat-types/src/physical/parse.rs",
        from: /^fn unit\(symbol: &str\)/m, to: /^#\[cfg\(test\)\]/m,
        pattern: ARM,
        docs: ["language/types-units-and-expressions.md"],
    },
    {
        id: "types",
        source: "crates/runmat-analysis/language/src/semantic/type_syntax.rs",
        from: /^pub fn parse_type_syntax/m, to: /^fn known_quantity/m,
        pattern: ARM,
        docs: ["language/types-units-and-expressions.md"],
    },
    {
        id: "field locations",
        source: "crates/runmat-analysis/language/src/semantic/procedure/output/fields.rs",
        from: /fn field_location\(/, to: /fn field_reference_type\(/,
        pattern: ARM,
        docs: ["language/outputs-fields-and-observations.md"],
    },
    {
        id: "field quantities",
        source: "crates/runmat-analysis/core/src/study/output/quantity.rs",
        from: /fn from_source_name\(/, to: /^\s*_ => return None|^\s*_ => None/m,
        pattern: ARM,
        docs: ["language/outputs-fields-and-observations.md"],
    },
    {
        id: "output operations",
        source: "crates/runmat-analysis/language/src/operations/output.rs",
        from: /pub fn output_operation_extension\(/, to: /^\s*\],\s*$/m,
        pattern: OPERATION,
        // `magnitude()`, `recovery()`, …: a helper that registers one operation under its own literal.
        helpers: true,
        docs: ["language/outputs-fields-and-observations.md"],
    },
    {
        id: "material members",
        source: "crates/runmat-analysis/language/src/semantic/material/mod.rs",
        from: /const ASSIGNMENTS: &\[&str\]/, to: /let mut seen = BTreeSet::new\(\);/,
        pattern: STRING,
        docs: ["language/reference/declarations.md", "language/reference/fluid-materials.md"],
    },
];

function slice(text, from, to) {
    const start = text.search(from);
    if (start < 0) return "";
    const rest = text.slice(start);
    const end = rest.slice(1).search(to);
    return end < 0 ? rest : rest.slice(0, end + 1);
}

export function registryNames(repo, registry) {
    const text = readFileSync(join(repo, registry.source), "utf8");
    const body = slice(text, registry.from, registry.to);
    const names = [...body.matchAll(registry.pattern)].map(match => match[1]);
    if (registry.helpers) {
        for (const [, helper] of body.matchAll(/\b([a-z_]+)\(\)/g)) {
            const definition = slice(text, new RegExp(`^fn ${helper}\\(\\)`, "m"), /^\}/m);
            const named = [...definition.matchAll(registry.pattern)][0];
            if (named) names.push(named[1]);
        }
    }
    return [...new Set(names)].sort();
}

export function registryCoverage(repo) {
    const root = join(repo, "docs/simulation");
    return REGISTRIES.map(registry => {
        const names = registryNames(repo, registry);
        const text = registry.docs.map(doc => readFileSync(join(root, doc), "utf8")).join("\n");
        const backticked = new Set();
        for (const match of text.matchAll(/`([^`\n]+)`/g)) for (const token of match[1].split(/[^A-Za-z0-9_^]+/)) backticked.add(token);
        const missing = names.filter(name => !backticked.has(name));
        return { id: registry.id, source: registry.source, docs: registry.docs, names: names.length, missing };
    });
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
    const repo = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
    const report = registryCoverage(repo);
    if (process.argv.includes("--json")) console.log(JSON.stringify(report, null, 2));
    let errors = 0;
    for (const entry of report) {
        if (entry.names === 0) { errors++; console.log(`${entry.source}: [registry-coverage] no names extracted for ${entry.id}; re-anchor REGISTRIES in scripts/docs/registry-coverage.mjs`); }
        for (const name of entry.missing) { errors++; console.log(`docs/simulation/${entry.docs[0]}: [registry-coverage] ${entry.id}: \`${name}\` (${entry.source}) is not documented`); }
    }
    console.log(`registry-coverage: ${report.map(entry => `${entry.id} ${entry.names}`).join(", ")}; ${errors} errors`);
    process.exit(errors ? 1 : 0);
}
