// Generates the docs pages derived from the capability ledger (docs/simulation/ledger.json), so no
// one edits status or evidence by hand:
//   docs/simulation/validation/<area>.md     per family: the reference checks, grouped by analysis type
//   docs/simulation/validation/status.md     one row per family: analysis types checked, check count, last run
//   docs/simulation/language/reference/procedures.md   the family × procedure table between its markers
// Test results and run dates come from the evidence catalog (`evidence_report`), which lists the
// tests whose `// witness:` tags name each check; receipts come from examples/catalog.json.
//
//   node scripts/docs/generate-pages.mjs           # write the pages
//   node scripts/docs/generate-pages.mjs --check   # exit 1 if a page is stale (CI)
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { VALIDATION_AREAS, loadLedger } from "./ledger.mjs";

const COMMAND = "node scripts/docs/generate-pages.mjs";
const FAMILY_AREA = { conjugate_heat_transfer: "conjugate-heat-transfer", fluid_structure_interaction: "fluid-structure-interaction" };
const PROCEDURE = {
    static: "static", modal: "modal", prestressed_modal: "prestressed modal", buckling: "buckling", transient: "transient", harmonic: "harmonic",
    prestressed_harmonic: "prestressed harmonic", response_spectrum: "response spectrum", random_response: "random response", nonlinear: "nonlinear", adaptive: "adaptive",
};
const HEADER = (description, keywords, docType, title) => [
    "---", `description: ${description}`, `keywords: [${keywords.join(", ")}]`, `doc_type: ${docType}`, `generated: ${COMMAND}`, "---",
    `# ${title}`, "", "<!-- Generated from docs/simulation/ledger.json by scripts/docs/generate-pages.mjs; do not edit by hand. -->", "",
];

const procedureLabel = capability => PROCEDURE[capability.identity.procedure] + (capability.identity.solution_space === "modal_reduced" ? " (modal superposition)" : "");

/** The evidence catalog's checks section: check id → tests with results and run dates. */
function checkTests(root) {
    const path = join(root, "qualification/generated/evidence-catalog.json");
    if (!existsSync(path)) return new Map();
    return new Map((JSON.parse(readFileSync(path, "utf8")).checks ?? []).map(entry => [entry.id, entry.tests]));
}

function receiptIndex(root) {
    const catalog = JSON.parse(readFileSync(join(root, "examples/catalog.json"), "utf8")).examples;
    return new Map(catalog.flatMap(example => (example.receipts ?? []).map(receipt => [`${example.id}#${receipt.id}`, receipt])));
}

/** Everything a page needs about one check: its tests, receipts, verdict and last run date. */
function checkEvidence(check, tests, receipts) {
    const ran = tests.get(check.id) ?? [];
    const quoted = (check.receipts ?? []).map(id => ({ ...receipts.get(id), key: id }));
    const dates = [...ran.map(test => test.run_date), ...quoted.map(receipt => receipt.run?.date)].filter(Boolean).sort();
    const failed = ran.filter(test => test.result !== "passed");
    return { tests: ran, receipts: quoted, failed, last: dates.at(-1) ?? null };
}

const meshLabel = mesh => `${mesh.order === "quadratic" ? "Tet10" : mesh.order === "linear" ? "Tet4" : mesh.order}, ${mesh.target_size}, ${mesh.nodes.toLocaleString("en-US")} nodes`;
const signedPercent = ratio => `${ratio < 0 ? "−" : "+"}${Math.abs(ratio * 100).toFixed(2)} %`;
const testName = test => test.name.split("::").at(-1);

function receiptTable(evidence) {
    const lines = ["| Quantity | Mesh | RunMat | Reference | Difference |", "| --- | --- | ---: | ---: | ---: |"];
    const order = [...new Set(evidence.receipts.map(receipt => receipt.quantity))];
    for (const quantity of order) {
        evidence.receipts.filter(receipt => receipt.quantity === quantity).forEach((receipt, index) => {
            const reference = receipt.reference ? `${receipt.reference.value} ${receipt.reference.unit}` : "none";
            const difference = receipt.reference ? signedPercent((receipt.value - receipt.reference.value) / receipt.reference.value) : "";
            lines.push(`| ${index === 0 ? quantity : ""} | ${meshLabel(receipt.mesh)} | [${receipt.value} ${receipt.unit}](receipt:${receipt.key}) | ${reference} | ${difference} |`);
        });
    }
    return lines;
}

function resultLine(evidence) {
    const parts = [];
    if (evidence.tests.length) {
        const count = `${evidence.tests.length} test${evidence.tests.length === 1 ? "" : "s"}`;
        if (evidence.tests.some(test => test.result === "not_run")) parts.push(`Not yet run by the evidence report (${count}).`);
        else if (evidence.failed.length) parts.push(`Failed in the evidence run of ${evidence.last}: ${evidence.failed.map(test => `\`${testName(test)}\``).join(", ")}.`);
        else parts.push(`Passed (${count}) in the evidence run of ${evidence.last}.`);
    }
    if (evidence.receipts.length) parts.push(`Run receipts dated ${[...new Set(evidence.receipts.map(receipt => receipt.run.date))].sort().join(", ")}.`);
    return parts.join(" ");
}

export function familyPage(source, area) {
    const root = join(source, "docs/simulation");
    const { ledger } = loadLedger(source);
    const label = VALIDATION_AREAS[area];
    const tests = checkTests(root);
    const receipts = receiptIndex(root);
    const checks = ledger.checks.filter(check => check.area === area && check.status !== "planned");
    const planned = ledger.checks.filter(check => check.area === area && check.status === "planned");
    const lines = HEADER(
        `Reference checks for RunMat ${label.toLowerCase()} analyses: each problem, its closed-form, manufactured or published reference, the compared quantity, the tolerance and the latest result.`,
        [`${label.toLowerCase()} validation`, "verification", "reference solution", "mesh convergence", "FEA accuracy"],
        "validation", `${label} validation`);
    if (!checks.length) {
        lines.push(`No reference check yet. Planned: ${planned.map(check => check.title).join("; ") || "none named"}.`, "");
        return lines.join("\n");
    }
    lines.push(
        "Each check compares a RunMat result with an independent reference: a closed-form solution, a manufactured solution or a published benchmark. The tests listed under a check run on every change and in the evidence report; receipts record a public study run with its build, commit and mesh. The [validation methodology](./methodology.md) explains how checks are built and how tolerances are set, and [Status](./status.md) summarizes every family.",
        "",
    );
    for (const analysis of [...new Set(checks.map(check => check.analysis))]) {
        lines.push(`## ${analysis}`, "");
        for (const check of checks.filter(entry => entry.analysis === analysis)) {
            const evidence = checkEvidence(check, tests, receipts);
            lines.push(`### ${check.title}`, "");
            lines.push(`**Problem.** ${check.problem}`, "");
            lines.push(`**Reference.** ${check.reference}${check.citation ? ` ${check.citation}` : ""}`, "");
            lines.push(`**Compared.** ${check.quantity}`, "");
            lines.push(`**Tolerance.** ${check.tolerance}`, "");
            lines.push(`**Result.** ${[resultLine(evidence), check.result].filter(Boolean).join(" ")}`, "");
            if (evidence.receipts.length) lines.push(...receiptTable(evidence), "");
            if (evidence.tests.length) lines.push(`Tests: ${evidence.tests.map(test => `\`${testName(test)}\``).join(", ")}.`, "");
        }
    }
    if (planned.length) lines.push("## Planned", "", ...planned.map(check => `- ${check.title}: ${check.reference}`), "");
    return lines.join("\n");
}

export function statusPage(source) {
    const root = join(source, "docs/simulation");
    const { ledger } = loadLedger(source);
    const tests = checkTests(root);
    const receipts = receiptIndex(root);
    const lines = HEADER(
        "Which analysis types of each RunMat physics family have a reference check, how many checks each family has, and when they last ran.",
        ["validation status", "reference checks", "verification", "FEA accuracy"],
        "release-status", "Status");
    lines.push(
        `Reference checks for RunMat ${ledger.release}, by physics family. Each family page lists its checks with the problem, reference, tolerance and result. The last run is the latest evidence-report run or receipt date among the family's checks.`,
        "",
        "| Family | Analysis types with a reference check | Checks | Last run |",
        "| --- | --- | ---: | --- |",
    );
    for (const [area, label] of Object.entries(VALIDATION_AREAS)) {
        const checks = ledger.checks.filter(check => check.area === area && check.status !== "planned");
        const analyses = [...new Set(checks.map(check => check.analysis))].join(", ");
        const last = checks.map(check => checkEvidence(check, tests, receipts).last).filter(Boolean).sort().at(-1);
        const planned = ledger.checks.filter(check => check.area === area && check.status === "planned").map(check => check.title);
        lines.push(`| [${label}](./${area}.md) | ${analyses || `None yet; planned: ${planned.join("; ")}`} | ${checks.length} | ${last ?? "none"} |`);
    }
    lines.push("", "Which procedures each family runs in this release is in the [procedure reference](../language/reference/procedures.md#families-and-procedures).", "");
    return lines.join("\n");
}

/** The family × procedure table generated into the procedure reference between its markers. */
export const PROCEDURE_TABLE = { file: "docs/simulation/language/reference/procedures.md", start: "<!-- ledger:families-and-procedures -->", end: "<!-- /ledger:families-and-procedures -->" };

export function procedureTable(source) {
    const { ledger } = loadLedger(source);
    const procedures = ledger.capabilities.filter(capability => capability.identity);
    const lines = [PROCEDURE_TABLE.start, "", "| Family | Runs in this release | Planned |", "| --- | --- | --- |"];
    for (const family of [...new Set(procedures.map(capability => capability.identity.family))]) {
        const rows = procedures.filter(capability => capability.identity.family === family);
        const runs = rows.filter(capability => capability.status === "release").map(procedureLabel).join(", ") || "none";
        const planned = rows.filter(capability => capability.status === "pending").map(procedureLabel).join(", ") || "none";
        lines.push(`| ${VALIDATION_AREAS[FAMILY_AREA[family] ?? family]} | ${runs} | ${planned} |`);
    }
    lines.push("", "Modal superposition reuses a retained modal basis; every other entry solves the full system directly.", "", PROCEDURE_TABLE.end);
    return lines.join("\n");
}

function withProcedureTable(source) {
    const text = readFileSync(join(source, PROCEDURE_TABLE.file), "utf8");
    const start = text.indexOf(PROCEDURE_TABLE.start), end = text.indexOf(PROCEDURE_TABLE.end);
    if (start < 0 || end < start) throw new Error(`${PROCEDURE_TABLE.file}: missing ${PROCEDURE_TABLE.start} … ${PROCEDURE_TABLE.end}`);
    return text.slice(0, start) + procedureTable(source) + text.slice(end + PROCEDURE_TABLE.end.length);
}

export const generatedPages = source => Object.fromEntries([
    ["docs/simulation/validation/status.md", statusPage(source)],
    [PROCEDURE_TABLE.file, withProcedureTable(source)],
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
