// Generates the docs pages that are derived from registries, so no one edits them by hand:
//   docs/simulation/release-status.md      from docs/simulation/pending-features.json
//   docs/simulation/validation/records.md  from docs/simulation/qualification/*.md
//
//   node scripts/docs/generate-pages.mjs           # write the pages
//   node scripts/docs/generate-pages.mjs --check   # exit 1 if a page is stale (CI)
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { markdownFiles } from "./markdown-scan.mjs";
import { loadRegistry, pendingReferences } from "./docs-lint.mjs";

const COMMAND = "node scripts/docs/generate-pages.mjs";
const AREAS = {
    structural: "Structural mechanics", thermal: "Heat transfer", electromagnetic: "Electromagnetics", acoustic: "Acoustics",
    fluid: "Fluid flow", thermomechanical: "Thermomechanical", electrothermal: "Electrothermal",
    "conjugate-heat-transfer": "Conjugate heat transfer", "fluid-structure-interaction": "Fluid–structure interaction",
    language: "The `.fea` language", geometry: "Geometry", meshing: "Meshing", execution: "Execution and scale", results: "Results", tooling: "Tooling",
};

function pageTitle(root, file) {
    const text = readFileSync(join(root, file), "utf8");
    const fallback = { "generated/family-procedure-matrix.md": "Family and procedure matrix" };
    return text.match(/^#\s+(.+)$/m)?.[1].trim() ?? fallback[file] ?? file;
}

export function releaseStatusPage(source) {
    const root = join(source, "docs/simulation");
    const { byId } = loadRegistry(source);
    const references = pendingReferences(root, markdownFiles(root)).filter(reference => reference.file !== "release-status.md");
    const open = [...byId.values()].filter(feature => !feature.landed_in);
    const lines = [
        "---",
        "description: Planned RunMat simulation features that the docs describe but the current release does not include yet, grouped by area, with acceptance criteria.",
        "keywords: [release status, roadmap, planned features, not yet in the current release, FEA]",
        "doc_type: release-status",
        `generated: ${COMMAND}`,
        "---",
        "# Release status",
        "",
        "The RunMat simulation docs describe the complete system. A feature that the current release does not include yet carries the marker [not yet in the current release](./release-status.md) where the docs describe it, and appears on this page under its id. `runmat check` reports the same id when a study uses the feature, so the docs, the diagnostics and the RunMat agent share one key.",
        "",
        "This page is generated from `docs/simulation/pending-features.json`. When a feature ships, its entry records the release, and its markers are removed from the docs.",
        "",
        "Which procedures each physics family runs is listed in the [family and procedure matrix](./generated/family-procedure-matrix.md).",
        "",
    ];
    if (!open.length) lines.push("Every documented feature is in the current release.", "");
    for (const [area, label] of Object.entries(AREAS)) {
        const features = open.filter(feature => feature.area === area).sort((a, b) => a.id.localeCompare(b.id));
        if (!features.length) continue;
        lines.push(`## ${label}`, "");
        for (const feature of features) {
            lines.push(`### ${feature.id}`, "", `**${feature.title}.**`, "", `- Kind: ${feature.kind}.`, `- Ships when: ${feature.acceptance}`);
            if (feature.decision === "owner") lines.push("- Scope: awaiting a product decision on whether this procedure belongs in the final system.");
            const docs = [...new Set(references.filter(reference => reference.id === feature.id).map(reference => reference.file))].sort();
            if (docs.length) lines.push(`- Described in: ${docs.map(file => `[${pageTitle(root, file)}](./${file})`).join(", ")}.`);
            lines.push("");
        }
    }
    return lines.join("\n");
}

export function recordsPage(source) {
    const root = join(source, "docs/simulation");
    const records = readdirSync(join(root, "qualification")).filter(name => name.endsWith(".md")).sort();
    const lines = [
        "---",
        "description: Index of the RunMat FEA qualification records, one per capability, each stating its acceptance obligations and the tests that hold it to account.",
        "keywords: [qualification records, validation, acceptance criteria, evidence catalog, FEA verification]",
        "doc_type: records",
        `generated: ${COMMAND}`,
        "---",
        "# Validation records",
        "",
        "Each qualification record states what one capability must do, the acceptance obligations that pin it down, and the execution envelopes (formulation, element order, execution provider) it must hold in. The tests that support an obligation declare it at the test, and the evidence catalog reports PASS, FAIL or OPEN for every obligation and envelope from an actual test run. The [validation methodology](./methodology.md) explains how to read them.",
        "",
        "The records live in the source repository under `docs/simulation/qualification/`; the generated evidence catalog is `docs/simulation/qualification/generated/evidence-catalog.md`.",
        "",
        "| Record | File |",
        "| --- | --- |",
        ...records.map(name => `| ${pageTitle(root, `qualification/${name}`).replaceAll("|", "\\|")} | \`qualification/${name}\` |`),
        "",
    ];
    return lines.join("\n");
}

export const generatedPages = source => ({
    "docs/simulation/release-status.md": releaseStatusPage(source),
    "docs/simulation/validation/records.md": recordsPage(source),
});

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

