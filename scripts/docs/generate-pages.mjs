// Generates the docs pages derived from the capability ledger (docs/simulation/ledger.json), so no
// one edits status by hand:
//   docs/simulation/validation/status.md     what the release runs, the family × procedure matrix, planned features
//   docs/simulation/validation/<area>.md     per family: capabilities, reference cases, test evidence by requirement
//
//   node scripts/docs/generate-pages.mjs           # write the pages
//   node scripts/docs/generate-pages.mjs --check   # exit 1 if a page is stale (CI)
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { markdownFiles } from "./markdown-scan.mjs";
import { pendingReferences } from "./docs-lint.mjs";
import { GENERATED_PAGES, VALIDATION_AREAS, loadLedger } from "./ledger.mjs";

const COMMAND = "node scripts/docs/generate-pages.mjs";
const AREAS = {
    ...VALIDATION_AREAS,
    language: "The `.fea` language", geometry: "Geometry", meshing: "Meshing", execution: "Execution and scale", results: "Results", tooling: "Tooling",
};
const FAMILY_AREA = { conjugate_heat_transfer: "conjugate-heat-transfer", fluid_structure_interaction: "fluid-structure-interaction" };
const PROCEDURE = {
    static: "static", modal: "modal", prestressed_modal: "prestressed modal", buckling: "buckling", transient: "transient", harmonic: "harmonic",
    prestressed_harmonic: "prestressed harmonic", response_spectrum: "response spectrum", random_response: "random response", nonlinear: "nonlinear", adaptive: "adaptive",
};
const FLAG = id => `[not yet in the current release](pending:${id})`;
const anchor = text => text.toLowerCase().replace(/[^a-z0-9\s-]/g, "-").trim().replace(/\s+/g, "-").replace(/-+/g, "-").replace(/^-|-$/g, "");
const HEADER = (description, keywords, docType, title) => [
    "---", `description: ${description}`, `keywords: [${keywords.join(", ")}]`, `doc_type: ${docType}`, `generated: ${COMMAND}`, "---",
    `# ${title}`, "", "<!-- Generated from docs/simulation/ledger.json by scripts/docs/generate-pages.mjs; do not edit by hand. -->", "",
];

function pageTitle(root, file) {
    if (GENERATED_PAGES.includes(file)) {
        const area = file.replace(/^validation\//, "").replace(/\.md$/, "");
        return area === "status" ? "Status" : `${VALIDATION_AREAS[area]} validation`;
    }
    const text = readFileSync(join(root, file), "utf8");
    return text.match(/^#\s+(.+)$/m)?.[1].trim() ?? file.replace(/^.*\//, "").replace(/\.md$/, "").replace(/-/g, " ").replace(/^./, c => c.toUpperCase());
}

const procedureLabel = capability => PROCEDURE[capability.identity.procedure] + (capability.identity.solution_space === "modal_reduced" ? " (modal superposition)" : "");

export function statusPage(source) {
    const root = join(source, "docs/simulation");
    const { ledger } = loadLedger(source);
    const references = pendingReferences(root, markdownFiles(root)).filter(reference => !GENERATED_PAGES.includes(reference.file));
    const capabilities = ledger.capabilities;
    const procedures = capabilities.filter(capability => capability.identity);
    const pending = capabilities.filter(capability => capability.status === "pending");
    const lines = HEADER(
        "What the current RunMat release runs for each physics family and procedure, which loads, constraints and outputs it accepts, and which documented features are planned, with acceptance criteria.",
        ["release status", "supported analyses", "family and procedure matrix", "planned features", "not yet in the current release", "FEA"],
        "release-status", "Status");
    lines.push(
        `This page describes RunMat ${ledger.release}. It is generated from the capability ledger, \`docs/simulation/ledger.json\`, the one record of what the simulation system can do. The same ledger generates the compiler's table of family and procedure pairs, so \`runmat check\` and \`runmat run\` accept exactly what this page lists.`,
        "",
        "A documented feature that the current release does not include yet carries the marker [not yet in the current release](#planned-features) where the docs describe it, and appears below under its id. `runmat check` reports the same id when a study uses it, so the docs, the diagnostics and the RunMat agent share one key.",
        "",
        "## Families and procedures",
        "",
        "| Family | Runs in this release | Planned |",
        "| --- | --- | --- |",
    );
    for (const family of [...new Set(procedures.map(capability => capability.identity.family))]) {
        const rows = procedures.filter(capability => capability.identity.family === family);
        const runs = rows.filter(capability => capability.status === "release").map(procedureLabel).join(", ") || "—";
        const planned = rows.filter(capability => capability.status === "pending").map(capability => `[${procedureLabel(capability)}](#${anchor(capability.id)})`).join(", ") || "—";
        const area = FAMILY_AREA[family] ?? family;
        lines.push(`| [${VALIDATION_AREAS[area]}](./${area}.md) | ${runs} | ${planned} |`);
    }
    lines.push("", "Modal superposition reuses a retained modal basis; every other entry solves the full system directly. The [procedure reference](../language/reference/procedures.md) gives the syntax of each.", "");
    lines.push("## Loads, constraints and outputs", "");
    for (const [kind, label, doc] of [["load", "Loads", "../language/reference/constraints-loads-and-interfaces.md"], ["constraint", "Constraints", "../language/reference/constraints-loads-and-interfaces.md"], ["output", "Output operations", "../language/reference/operations-and-output-queries.md"]]) {
        const names = capabilities.filter(capability => capability.kind === kind && capability.status === "release" && capability.id.startsWith(`${kind}.`)).map(capability => `\`${capability.id.slice(kind.length + 1)}\``);
        lines.push(`- **${label}** (${names.length}): ${names.join(", ")}. See the [reference](${doc}).`);
    }
    lines.push("", "## Planned features", "");
    if (!pending.length) lines.push("Every documented feature is in the current release.", "");
    for (const [area, label] of Object.entries(AREAS)) {
        const features = pending.filter(feature => feature.area === area).sort((a, b) => a.id.localeCompare(b.id));
        if (!features.length) continue;
        lines.push(`### ${label}`, "");
        for (const feature of features) {
            lines.push(`#### ${feature.id}`, "", `**${feature.title}.**`, "", `- Kind: ${feature.kind}.`, `- Ships when: ${feature.acceptance}`);
            if (feature.decision === "owner") lines.push("- Scope: awaiting a product decision on whether this belongs in the final system.");
            if (feature.enforcement) lines.push(`- In this release: ${feature.enforcement}.`);
            const docs = [...new Set(references.filter(reference => reference.id === feature.id).map(reference => reference.file))].sort();
            if (docs.length) lines.push(`- Described in: ${docs.map(file => `[${pageTitle(root, file)}](../${file})`).join(", ")}.`);
            lines.push("");
        }
    }
    lines.push("## Validation by family", "", ...Object.entries(VALIDATION_AREAS).map(([area, label]) => `- [${label}](./${area}.md)`), "", "The [validation methodology](./methodology.md) explains how cases, references and tests are read.", "");
    return lines.join("\n");
}

function evidenceIndex(root) {
    const path = join(root, "qualification/generated/evidence-catalog.json");
    if (!existsSync(path)) return new Map();
    return new Map(JSON.parse(readFileSync(path, "utf8")).requirements.map(requirement => [requirement.id, requirement]));
}

export function familyPage(source, area) {
    const root = join(source, "docs/simulation");
    const { ledger } = loadLedger(source);
    const label = VALIDATION_AREAS[area];
    const capabilities = ledger.capabilities.filter(capability => capability.area === area);
    const cases = ledger.cases.filter(entry => entry.area === area);
    const evidence = evidenceIndex(root);
    const lines = HEADER(
        `How RunMat checks its ${label.toLowerCase()} results: reference cases with their errors and receipts, and the tests behind each requirement, generated from the capability ledger.`,
        [`${label.toLowerCase()} validation`, "verification", "reference cases", "convergence", "FEA accuracy"],
        "validation", `${label} validation`);
    lines.push(
        "Each case below compares a RunMat result with an independent reference: a closed-form solution, a manufactured solution or a published benchmark. Values link to their run receipts. The [validation methodology](./methodology.md) explains how to read them, and [Status](./status.md) lists what the release runs.",
        "",
        "## Capabilities",
        "",
        "| Capability | In this release | Cases |",
        "| --- | --- | --- |",
    );
    for (const capability of capabilities.filter(capability => capability.kind !== "load" && capability.kind !== "constraint")) {
        const status = capability.status === "release" ? "yes" : FLAG(capability.id);
        const caseLinks = capability.verification.map(id => {
            const entry = ledger.cases.find(item => item.id === id);
            return entry.area === area ? `[${entry.title}](#${anchor(entry.title)})` : `[${entry.title}](./${entry.area}.md#${anchor(entry.title)})`;
        }).join(", ") || "—";
        lines.push(`| ${capability.title} | ${status} | ${caseLinks} |`);
    }
    lines.push("", "## Reference cases", "");
    if (!cases.length) lines.push(`No ${label.toLowerCase()} reference case is published on this page yet; the requirement tests below cover the formulation.`, "");
    for (const entry of cases) {
        lines.push(`### ${entry.title}`, "", readFileSync(join(root, entry.narrative), "utf8").trim(), "");
        lines.push(`- References: ${entry.references.join("; ")}.`);
        if (entry.tests.length) lines.push(`- Held in CI by: ${entry.tests.map(test => `\`${test}\``).join(", ")}.`);
        if (entry.benchmarks?.length) lines.push(`- Benchmarks: ${entry.benchmarks.join("; ")}.`);
        lines.push("");
    }
    const requirementIds = [...new Set(capabilities.flatMap(capability => capability.requirements))];
    lines.push("## Test evidence by requirement", "");
    if (!requirementIds.length) lines.push("No requirement is mapped to these capabilities yet.", "");
    else {
        lines.push(
            "Each requirement lists acceptance obligations, and each obligation must hold in stated envelopes: formulation, element order and execution provider. Tests declare the obligation and envelope they support at the test; the counts are envelope cells from the latest evidence run. PASS covers the obligation fully, PARTIAL passes with a narrower boundary than the criterion, and OPEN has no supporting test yet.",
            "",
            "| Requirement | Title | Obligations | PASS | PARTIAL | FAIL | OPEN |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        );
        for (const id of requirementIds) {
            const requirement = ledger.requirements.find(item => item.id === id);
            const run = evidence.get(id);
            const count = key => run ? run[key] ?? 0 : "—";
            lines.push(`| ${id} | ${requirement.title} | ${requirement.obligations.length} | ${count("pass")} | ${count("partial")} | ${count("fail")} | ${count("open")} |`);
        }
        lines.push("");
    }
    return lines.join("\n");
}

export const generatedPages = source => Object.fromEntries([
    ["docs/simulation/validation/status.md", statusPage(source)],
    ...Object.keys(VALIDATION_AREAS).map(area => [`docs/simulation/validation/${area}.md`, familyPage(source, area)]),
]);

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
    const source = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
    let stale = 0;
    for (const [path, text] of Object.entries(generatedPages(source))) {
        let current = "";
        try { current = readFileSync(join(source, path), "utf8"); } catch { /* new page */ }
        if (current === text) continue;
        if (process.argv.includes("--check")) { console.log(`${path} is stale; run ${COMMAND}`); stale += 1; }
        else { writeFileSync(join(source, path), text); console.log(`wrote ${path}`); }
    }
    process.exit(stale ? 1 : 0);
}
