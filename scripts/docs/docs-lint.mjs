// Lint for the FEA docs tree: the pending-feature registry and its flags, front matter, and links.
//
//   node scripts/docs/docs-lint.mjs            # lint docs/simulation; exit 1 on any error
//   node scripts/docs/docs-lint.mjs --json     # machine-readable report
//
// Rules (METHOD §3.1 item 8, §4.3 E-DOC-3/E-DOC-5, §5):
// - every `pending:<id>` link and ```fea pending=<id> fence names a registry entry;
// - an entry with `landed_in` set fails while any doc still flags it (the error names file:line to delete);
// - every open entry is flagged by at least one doc, or carries `doc_less` with a reason;
// - a ```fea source=<file> block is a verbatim excerpt of docs/simulation/examples/<file>;
// - front matter, where present, matches docs/schema/doc-front-matter.schema.json;
// - every name in the compiler's closed registries (units, types, field locations and quantities, output
//   operations, material members) appears on its owner reference page (E-DOC-6, registry-coverage.mjs);
// - relative links resolve to a file in docs/simulation and, with `#anchor`, to a heading in it;
//   links may not leave docs/simulation (the site and the agent corpus only contain that tree).
import Ajv from "ajv";
import { existsSync } from "node:fs";
import { dirname, join, posix, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { fencedBlocks, headings, links, markdownFiles, parseFrontMatter, readText } from "./markdown-scan.mjs";
import { REGISTRIES, registryCoverage } from "./registry-coverage.mjs";

const readJson = path => JSON.parse(readText(path));

export function loadRegistry(source) {
    const registry = readJson(join(source, "docs/simulation/pending-features.json"));
    const ajv = new Ajv({ allErrors: true });
    const check = ajv.compile(readJson(join(source, "docs/schema/pending-features.schema.json")));
    if (!check(registry)) throw new Error(`pending-features.json: ${ajv.errorsText(check.errors)}`);
    const byId = new Map();
    for (const feature of registry.features) {
        if (byId.has(feature.id)) throw new Error(`pending-features.json: duplicate id ${feature.id}`);
        byId.set(feature.id, feature);
    }
    return { registry, byId };
}

/** Every pending reference in the tree: { file, line, id, form: "link" | "fence" }. */
export function pendingReferences(root, files) {
    const found = [];
    for (const file of files) {
        const text = readText(join(root, file));
        for (const link of links(text)) {
            if (link.target.startsWith("pending:")) found.push({ file, line: link.line, id: link.target.slice("pending:".length), form: "link" });
        }
        for (const block of fencedBlocks(text)) {
            if (typeof block.attributes.pending === "string") found.push({ file, line: block.line, id: block.attributes.pending, form: "fence" });
        }
    }
    return found;
}

function routeIndex(source) {
    const manual = readJson(join(source, "docs/simulation/manual.json"));
    return new Map(manual.topics.map(topic => [topic.route, topic.narrative.replace(/^docs\/simulation\//, "")]));
}

function resolveTarget(root, routes, fromFile, target) {
    const cut = target.search(/#/);
    const path = cut === -1 ? target : target.slice(0, cut);
    const anchor = cut === -1 ? "" : target.slice(cut + 1);
    if (path === "") return { file: fromFile, anchor };
    let base;
    if (path.startsWith("/docs/simulation")) {
        const route = path.replace(/\/$/, "");
        if (routes.has(route)) return { file: routes.get(route), anchor };
        base = route.slice("/docs/simulation".length).replace(/^\//, "") || "index";
    }
    else if (path.startsWith("/")) return { external: true };
    else base = posix.normalize(posix.join(posix.dirname(fromFile), path)).replace(/\/$/, "");
    if (base.startsWith("../") || base === "..") return { outside: base };
    for (const candidate of [base, `${base}.md`, `${base}/index.md`]) {
        if (candidate.endsWith(".md") && existsSync(join(root, candidate))) return { file: candidate, anchor };
    }
    if (existsSync(join(root, base))) return { asset: base };
    return { missing: base };
}

/**
 * E-DOC-4: a receipt link's text is the catalog value at the precision written, in the catalog unit:
 * `[0.2000 mm](receipt:cantilever-static#tip-deflection)` needs value 0.19995 ≤ v < 0.20005 and unit mm.
 */
export function receiptMismatch(receipt, text) {
    if (!receipt) return "no such receipt in examples/catalog.json";
    const match = /^\s*([-−]?[0-9][0-9,]*)(?:\.([0-9]+))?(?:\s+(.+?))?\s*$/.exec(text.replace(/\u00a0|\u202f/g, " "));
    if (!match) return `link text \`${text}\` is not a value and unit`;
    const decimals = match[2]?.length ?? 0;
    const shown = Number(`${match[1].replace(/,/g, "").replace("−", "-")}.${match[2] ?? "0"}`);
    const unit = match[3] ?? "";
    if (unit !== receipt.unit) return `unit \`${unit}\` differs from the receipt's \`${receipt.unit}\``;
    const rounded = Number(receipt.value.toFixed(decimals));
    if (Math.abs(rounded - shown) > 1e-12 * Math.max(1, Math.abs(shown))) return `text ${text} differs from the receipt value ${receipt.value} ${receipt.unit} at ${decimals} decimals (${rounded})`;
    return null;
}

export function lintDocs(source) {
    const root = join(source, "docs/simulation");
    const files = markdownFiles(root);
    const errors = [];
    const error = (file, line, rule, message) => errors.push({ file: `docs/simulation/${file}`, line, rule, message });

    // Registry and flags.
    const { byId } = loadRegistry(source);
    const references = pendingReferences(root, files);
    for (const reference of references) {
        const feature = byId.get(reference.id);
        if (!feature) error(reference.file, reference.line, "pending-unknown", `unknown pending id \`${reference.id}\`; add it to docs/simulation/pending-features.json or fix the id`);
        else if (feature.landed_in) error(reference.file, reference.line, "pending-landed", `\`${reference.id}\` landed in ${feature.landed_in}; delete this flag`);
    }
    for (const feature of byId.values()) {
        if (feature.landed_in || feature.doc_less) continue;
        if (!references.some(reference => reference.id === feature.id)) errors.push({ file: "docs/simulation/pending-features.json", line: 0, rule: "pending-unreferenced", message: `open entry \`${feature.id}\` is flagged by no doc; flag it where the docs describe it, or set doc_less with a reason` });
    }

    // Front matter.
    const ajv = new Ajv({ allErrors: true });
    const checkFrontMatter = ajv.compile(readJson(join(source, "docs/schema/doc-front-matter.schema.json")));
    const catalog = readJson(join(root, "examples/catalog.json")).examples;
    const exampleIds = new Set(catalog.map(example => example.id));
    const receipts = new Map(catalog.flatMap(example => (example.receipts ?? []).map(receipt => [`${example.id}#${receipt.id}`, receipt])));
    const parsed = new Map();
    for (const file of files) {
        const text = readText(join(root, file));
        let front;
        try { front = parseFrontMatter(text); }
        catch (cause) { error(file, 1, "front-matter", cause.message); continue; }
        parsed.set(file, { text, headings: headings(text) });
        if (!front.data) continue;
        if (!checkFrontMatter(front.data)) error(file, 1, "front-matter", ajv.errorsText(checkFrontMatter.errors, { dataVar: "front matter" }));
        for (const id of front.data.examples ?? []) if (!exampleIds.has(id)) error(file, 1, "front-matter", `examples: unknown catalog id \`${id}\``);
    }

    // Excerpts of canonical example files: ```fea source=<examples path> must be a verbatim slice.
    for (const [file, { text }] of parsed) {
        for (const block of fencedBlocks(text)) {
            if (typeof block.attributes.source !== "string") continue;
            const path = join(root, "examples", block.attributes.source);
            if (!existsSync(path)) error(file, block.line, "source-slice", `source=${block.attributes.source} is not a file in docs/simulation/examples`);
            else if (!readText(path).includes(block.code.trim())) error(file, block.line, "source-slice", `this block is no longer an exact excerpt of examples/${block.attributes.source}; copy it from the file again`);
        }
    }

    // Links.
    const routes = routeIndex(source);
    for (const [file, { text }] of parsed) {
        for (const link of links(text)) {
            const target = link.target;
            if (/^(https?:|mailto:|pending:)/.test(target)) continue;
            if (target.startsWith("receipt:")) {
                if (!/^receipt:[a-z0-9-]+#[a-z0-9-]+$/.test(target)) { error(file, link.line, "link", `malformed receipt link \`${target}\``); continue; }
                const problem = receiptMismatch(receipts.get(target.slice("receipt:".length)), link.text);
                if (problem) error(file, link.line, "receipt", `\`${target}\`: ${problem}`);
                continue;
            }
            const resolved = resolveTarget(root, routes, file, target);
            if (resolved.outside) error(file, link.line, "link-outside", `\`${target}\` leaves docs/simulation; the site and the agent corpus only contain that tree`);
            else if (resolved.missing) error(file, link.line, "link-missing", `\`${target}\` resolves to docs/simulation/${resolved.missing}, which does not exist`);
            else if (resolved.file && resolved.anchor) {
                const page = parsed.get(resolved.file);
                if (page && !page.headings.some(heading => heading.anchor === resolved.anchor)) error(file, link.line, "link-anchor", `\`${target}\`: docs/simulation/${resolved.file} has no heading with anchor #${resolved.anchor}`);
            }
        }
    }
    // E-DOC-6: every closed-registry name is documented on its owner page (skipped for fixture trees without sources).
    if (REGISTRIES.every(registry => existsSync(join(source, registry.source)))) {
        for (const entry of registryCoverage(source)) {
            if (entry.names === 0) errors.push({ file: entry.source, line: 0, rule: "registry-coverage", message: `no names extracted for ${entry.id}; re-anchor REGISTRIES in scripts/docs/registry-coverage.mjs` });
            for (const name of entry.missing) error(entry.docs[0], 0, "registry-coverage", `${entry.id}: \`${name}\` (${entry.source}) is not documented`);
        }
    }
    return { files: files.length, references: references.length, errors };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
    const source = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
    const report = lintDocs(source);
    if (process.argv.includes("--json")) console.log(JSON.stringify(report, null, 2));
    else {
        for (const item of report.errors) console.log(`${item.file}:${item.line}: [${item.rule}] ${item.message}`);
        console.log(`docs-lint: ${report.files} files, ${report.references} pending flags, ${report.errors.length} errors`);
    }
    process.exit(report.errors.length ? 1 : 0);
}
