// The design docs (docs/design): the Design tab's manual, its pages and the feature ledger the
// pages' "not yet in the current release" flags read.
//
// - docs/design/manual.json lists every page: id, route, title, group, order, summary,
//   keywords, narrative. The site builds the Design tab from it (sections in order of first
//   appearance) and keys each page `manual:design:<id>`.
// - docs/design/ledger.json holds one row per documented design feature that is not built yet
//   (status pending) or that has landed since (status release, landed_in). A page flags a pending
//   feature with [not yet in the current release](pending:<id>) or a ```part pending=<id> fence.
// - The examples are checked by crates/runmat-design-tools/tests/docs_snippets.rs, through the
//   same compilers the editor uses.
//
// `designDocsProblems` is the lint; `collectDesign` writes the export's design/ files.
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { readJson, copyInput } from "./documents.mjs";
import { fencedBlocks, headings, links, markdownFiles, parseFrontMatter, proseLines, readText } from "./markdown-scan.mjs";

export const DESIGN_ROOT = "docs/design";
export const DESIGN_PREFIXES = ["/docs/design-language", "/docs/mcad", "/docs/ecad"];
/** Fence languages the docs test compiles (the file extensions). */
export const DESIGN_LANGUAGES = ["part", "assembly", "drawing", "circuit", "circuit.layout"];
/** Other fences a design page may hold: MATLAB, `.fea`, shell commands, JSON and plain text. */
const OTHER_LANGUAGES = ["matlab", "fea", "sh", "json", "text", "toml", "mermaid"];
const FENCE_ATTRIBUTES = ["file", "fragment", "pending", "fails", "m", "occt"];

export const hasDesignDocs = source => existsSync(join(source, DESIGN_ROOT, "manual.json"));

export function loadDesign(source) {
    return { manual: readJson(join(source, DESIGN_ROOT, "manual.json")), ledger: readJson(join(source, DESIGN_ROOT, "ledger.json")) };
}

export const STATUS_PAGE = "docs/design/language/status.md";
const AREA_TITLES = { language: "Design language", mcad: "MCAD", ecad: "ECAD", tooling: "Tools" };

/** The Feature status page, generated from the ledger: one heading per pending feature, its id. */
export function designStatusPage(ledger) {
    const lines = [
        "---",
        "description: Design features that the RunMat design docs describe and the current release does not include yet, with what each needs to ship. Generated from the design feature ledger.",
        "keywords: [feature status, roadmap, not yet released, RunMat MCAD, RunMat ECAD]",
        "---",
        "# Feature status",
        "",
        "The design docs describe the design language as it is meant to work. Where the docs describe a feature the current release does not include yet, a “Not yet in” badge marks it and links here. Each entry below says what the feature needs before the mark comes off.",
        "",
        "This page is generated from `docs/design/ledger.json`; edit the ledger, not this page.",
    ];
    for (const area of Object.keys(AREA_TITLES)) {
        const rows = ledger.capabilities.filter(row => row.area === area && row.status === "pending");
        if (!rows.length) continue;
        lines.push("", `## ${AREA_TITLES[area]}`);
        for (const row of rows) lines.push("", `### ${row.id}`, "", `**${row.title}.** ${row.acceptance}`);
    }
    return lines.join("\n") + "\n";
}

const isDesignRoute = route => DESIGN_PREFIXES.some(prefix => route === prefix || route.startsWith(prefix + "/"));

/** Every problem with the design docs, as { file, line, rule, message }. Empty when the tree is clean. */
export function designDocsProblems(source) {
    const problems = [];
    const problem = (file, line, rule, message) => problems.push({ file, line, rule, message });
    const { manual, ledger } = loadDesign(source);
    const root = join(source, DESIGN_ROOT);
    const routes = new Map();
    const ids = new Set();
    for (const topic of manual.topics) {
        if (ids.has(topic.id)) problem("docs/design/manual.json", 0, "duplicate-id", `topic id \`${topic.id}\` appears twice`);
        ids.add(topic.id);
        if (routes.has(topic.route)) problem("docs/design/manual.json", 0, "duplicate-route", `route \`${topic.route}\` appears twice`);
        routes.set(topic.route, topic);
        if (!isDesignRoute(topic.route)) problem("docs/design/manual.json", 0, "route-prefix", `\`${topic.route}\` is outside ${DESIGN_PREFIXES.join(", ")}`);
        if (!existsSync(join(source, topic.narrative))) problem("docs/design/manual.json", 0, "narrative-missing", `\`${topic.id}\`: ${topic.narrative} does not exist`);
    }
    // Groups are contiguous, so the tab's sections read in the manual's order.
    const groups = manual.topics.map(topic => topic.group);
    for (const [index, group] of groups.entries())
        if (index > 0 && group !== groups[index - 1] && groups.slice(0, index).includes(group))
            problem("docs/design/manual.json", 0, "group-order", `group \`${group}\` is split; keep a group's topics together`);
    const narratives = new Set(manual.topics.map(topic => topic.narrative));
    const pending = new Map(ledger.capabilities.map(row => [row.id, row]));
    const flagged = new Set();
    for (const file of markdownFiles(root, { skip: [] })) {
        const path = `${DESIGN_ROOT}/${file}`;
        if (!narratives.has(path)) problem(path, 1, "page-unlisted", "the page is in no manual topic");
        const text = readText(join(root, file));
        const { data, bodyLine } = parseFrontMatter(text);
        if (!data?.description) problem(path, 1, "front-matter", "front matter needs a `description`");
        if (!headings(text).some(heading => heading.depth === 1)) problem(path, bodyLine, "title", "the page needs a `# Title` heading");
        for (const block of fencedBlocks(text)) {
            if (block.language === "rust" || block.language === "rs") problem(path, block.line, "fence-rust", "design docs show the language, never Rust");
            else if (!DESIGN_LANGUAGES.includes(block.language) && !OTHER_LANGUAGES.includes(block.language))
                problem(path, block.line, "fence-language", `unknown fence language \`${block.language || "(none)"}\`; use ${[...DESIGN_LANGUAGES, ...OTHER_LANGUAGES].join(", ")}`);
            for (const key of Object.keys(block.attributes))
                if (!FENCE_ATTRIBUTES.includes(key)) problem(path, block.line, "fence-attribute", `unknown fence attribute \`${key}\``);
            if (typeof block.attributes.pending === "string") flagged.add(block.attributes.pending);
            if (typeof block.attributes.pending === "string" && pending.get(block.attributes.pending)?.status !== "pending")
                problem(path, block.line, "pending-unknown", `\`${block.attributes.pending}\` is not a pending row of docs/design/ledger.json`);
        }
        for (const link of links(text)) {
            const flag = link.target.match(/^pending:(.+)$/);
            if (flag) {
                flagged.add(flag[1]);
                if (link.text !== "not yet in the current release") problem(path, link.line, "pending-text", "a pending flag reads exactly `not yet in the current release`");
                if (pending.get(flag[1])?.status !== "pending") problem(path, link.line, "pending-unknown", `\`${flag[1]}\` is not a pending row of docs/design/ledger.json`);
                continue;
            }
            if (/^(https?:|mailto:|#)/.test(link.target)) continue;
            if (!link.target.startsWith("/docs/")) { problem(path, link.line, "link-form", `\`${link.target}\`: link docs pages by route (/docs/...)`); continue; }
            const [route, anchor] = link.target.split("#");
            if (!isDesignRoute(route)) continue;
            const topic = routes.get(route);
            if (!topic) { problem(path, link.line, "link-missing", `\`${route}\` is no design docs route`); continue; }
            if (anchor && existsSync(join(source, topic.narrative)) && !headings(readText(join(source, topic.narrative))).some(heading => heading.anchor === anchor))
                problem(path, link.line, "link-anchor", `\`${link.target}\`: ${topic.narrative} has no heading #${anchor}`);
        }
        for (const { line, text: prose } of proseLines(text))
            if (/\bnot yet in the current release\b/.test(prose) && !/\]\(pending:/.test(prose))
                problem(path, line, "pending-form", "write the flag as a link: [not yet in the current release](pending:<id>)");
    }
    if (!existsSync(join(source, STATUS_PAGE)) || readText(join(source, STATUS_PAGE)) !== designStatusPage(ledger))
        problem(STATUS_PAGE, 1, "status-stale", "regenerate it from the ledger: node scripts/docs/design.mjs --write-status");
    for (const row of ledger.capabilities) {
        if (row.status === "pending" && !flagged.has(row.id)) problem("docs/design/ledger.json", 0, "pending-unreferenced", `\`${row.id}\` is flagged by no page`);
        if (row.status === "release" && !row.landed_in) problem("docs/design/ledger.json", 0, "landed-in", `\`${row.id}\` is released; set landed_in`);
        if (row.status === "release" && flagged.has(row.id)) problem("docs/design/ledger.json", 0, "pending-landed", `\`${row.id}\` has landed; remove its flags`);
    }
    return problems;
}

/** Writes design/manual.json and design/ledger.json and copies each page to documents/. */
export function collectDesign(source, output, validate) {
    if (!hasDesignDocs(source)) return;
    const { manual, ledger } = loadDesign(source);
    validate("design-manual", manual);
    validate("design-ledger", ledger);
    const problems = designDocsProblems(source);
    if (problems.length)
        throw new Error(`Design docs: ${problems.map(p => `${p.file}:${p.line} ${p.rule}: ${p.message}`).join("; ")}`);
    for (const topic of manual.topics) copyInput(source, output, topic.narrative, "documents/" + topic.narrative);
    mkdirSync(join(output, "design"), { recursive: true });
    writeFileSync(join(output, "design/manual.json"), JSON.stringify(manual, null, 2) + "\n");
    writeFileSync(join(output, "design/ledger.json"), JSON.stringify(ledger, null, 2) + "\n");
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
    const source = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
    if (process.argv.includes("--write-status")) writeFileSync(join(source, STATUS_PAGE), designStatusPage(loadDesign(source).ledger));
    const problems = designDocsProblems(source);
    for (const p of problems) console.log(`${p.file}:${p.line}: ${p.rule}: ${p.message}`);
    console.log(`design docs: ${loadDesign(source).manual.topics.length} pages, ${problems.length} problems`);
    process.exit(problems.length ? 1 : 0);
}
