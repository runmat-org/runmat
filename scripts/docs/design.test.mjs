import test from "node:test";
import assert from "node:assert/strict";
import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import Ajv from "ajv";
import { collectDesign, designDocsProblems, designStatusPage, loadDesign } from "./design.mjs";

const source = resolve(import.meta.dirname, "../..");

function validator() {
    const ajv = new Ajv({ allErrors: true });
    const schemas = new Map(["design-manual", "design-ledger"].map(name => [name, ajv.compile(JSON.parse(readFileSync(join(source, `docs/schema/${name}.schema.json`))))]));
    return (name, value) => { const check = schemas.get(name); if (!check(value)) throw new Error(`${name}: ${ajv.errorsText(check.errors)}`); };
}

/** A copy of the design docs tree to break on purpose. */
function fixture() {
    const root = mkdtempSync(join(tmpdir(), "design-docs-"));
    cpSync(join(source, "docs/design"), join(root, "docs/design"), { recursive: true });
    return root;
}

test("the design docs lint clean and the manual and ledger match their schemas", () => {
    assert.deepEqual(designDocsProblems(source), []);
    const { manual, ledger } = loadDesign(source);
    const validate = validator();
    validate("design-manual", manual);
    validate("design-ledger", ledger);
    for (const topic of manual.topics) assert.match(topic.route, /^\/docs\/(design-language|mcad|ecad)(\/|$)/);
});

test("the export writes design/manual.json, design/ledger.json and every page", () => {
    const output = mkdtempSync(join(tmpdir(), "design-export-"));
    try {
        collectDesign(source, output, validator());
        const manual = JSON.parse(readFileSync(join(output, "design/manual.json")));
        assert.deepEqual(manual, loadDesign(source).manual);
        for (const topic of manual.topics) assert.equal(readFileSync(join(output, "documents", topic.narrative), "utf8"), readFileSync(join(source, topic.narrative), "utf8"));
        assert.deepEqual(JSON.parse(readFileSync(join(output, "design/ledger.json"))), loadDesign(source).ledger);
        assert.deepEqual(JSON.parse(readFileSync(join(output, "design/clips.json"))), loadDesign(source).clips);
    }
    finally { rmSync(output, { recursive: true, force: true }); }
});

test("unknown flags, unlisted pages, broken links, Rust fences and a stale status page are problems", () => {
    const root = fixture();
    try {
        const page = join(root, "docs/design/language/overview.md");
        writeFileSync(page, readFileSync(page, "utf8") + [
            "", "A thing [not yet in the current release](pending:design.nothing.such).",
            "See [nowhere](/docs/mcad/no-such-page).", "```rust", "fn main() {}", "```", "",
        ].join("\n"));
        mkdirSync(join(root, "docs/design/extra"), { recursive: true });
        writeFileSync(join(root, "docs/design/extra/orphan.md"), "---\ndescription: x\n---\n# Orphan\n");
        const ledgerPath = join(root, "docs/design/ledger.json");
        const ledger = JSON.parse(readFileSync(ledgerPath));
        ledger.capabilities[0].title += " (renamed)";
        writeFileSync(ledgerPath, JSON.stringify(ledger));
        const rules = designDocsProblems(root).map(problem => problem.rule);
        for (const rule of ["pending-unknown", "link-missing", "fence-rust", "page-unlisted", "status-stale"]) assert.ok(rules.includes(rule), `${rule} in ${rules}`);
        assert.throws(() => collectDesign(root, mkdtempSync(join(tmpdir(), "design-export-")), validator()), /Design docs/);
    }
    finally { rmSync(root, { recursive: true, force: true }); }
});

test("the status page lists every pending feature under its id", () => {
    const { ledger } = loadDesign(source);
    const page = designStatusPage(ledger);
    for (const row of ledger.capabilities.filter(row => row.status === "pending")) assert.ok(page.includes(`### ${row.id}\n`), row.id);
});

test("ledger prose renders as MDX-safe text on the status page", async () => {
    const { mdxText } = await import("./design.mjs");
    assert.equal(mdxText("part = <name> with { inputs }"), "part = &lt;name&gt; with &#123; inputs &#125;");
    assert.equal(mdxText("keep `quantity.<Kind>` and {x}"), "keep `quantity.<Kind>` and &#123;x&#125;");
    const page = designStatusPage(loadDesign(resolve(import.meta.dirname, "../..")).ledger).split("\n").filter(line => !line.startsWith("---") && !/^(description|keywords):/.test(line)).join("\n");
    const outsideCode = page.replace(/`[^`]*`/g, "");
    assert.doesNotMatch(outsideCode, /[<{}]/);
});

test("a clip marker names a clip in clips.json, and each clip has its alt text, size, length and files", () => {
    const root = fixture();
    try {
        const page = join(root, "docs/design/language/overview.md");
        writeFileSync(page, readFileSync(page, "utf8") + "\n<!-- design:clip:mcad/no-such-clip -->\n");
        const clipsPath = join(root, "docs/design/clips.json");
        const clips = JSON.parse(readFileSync(clipsPath));
        clips.clips.push({ id: "mcad/half", alt: "" });
        writeFileSync(clipsPath, JSON.stringify(clips));
        const rules = designDocsProblems(root).map(problem => problem.rule);
        assert.ok(rules.includes("clip-unknown"), rules.join(" "));
        assert.ok(rules.includes("clip-field"), rules.join(" "));
    }
    finally { rmSync(root, { recursive: true, force: true }); }
});
