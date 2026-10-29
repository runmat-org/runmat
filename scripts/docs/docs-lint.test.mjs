import test from "node:test";
import assert from "node:assert/strict";
import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { lintDocs } from "./docs-lint.mjs";
import { fencedBlocks, headingAnchor, parseFrontMatter } from "./markdown-scan.mjs";

const repository = resolve(dirname(fileURLToPath(import.meta.url)), "../..");

function fixture(pages, features) {
    const root = mkdtempSync(join(tmpdir(), "docs-lint-"));
    for (const file of ["docs/schema/pending-features.schema.json", "docs/schema/doc-front-matter.schema.json"]) {
        mkdirSync(dirname(join(root, file)), { recursive: true });
        cpSync(join(repository, file), join(root, file));
    }
    const docs = join(root, "docs/simulation");
    mkdirSync(join(docs, "examples"), { recursive: true });
    writeFileSync(join(docs, "examples/catalog.json"), JSON.stringify({ examples: [{ id: "cantilever-static" }] }));
    writeFileSync(join(docs, "manual.json"), JSON.stringify({ schema_version: 1, topics: [] }));
    writeFileSync(join(docs, "pending-features.json"), JSON.stringify({ schema_version: 1, features }));
    for (const [path, text] of Object.entries(pages)) {
        mkdirSync(dirname(join(docs, path)), { recursive: true });
        writeFileSync(join(docs, path), text);
    }
    return root;
}

const feature = (id, extra = {}) => ({ id, title: "A feature", area: "acoustic", kind: "procedure", acceptance: "It runs.", witness: [], gaps: [], owner: "q-fluid", landed_in: null, ...extra });
const rules = report => report.errors.map(error => error.rule).sort();

test("the docs tree passes the lint", () => {
    const report = lintDocs(repository);
    assert.deepEqual(report.errors, []);
});

test("flags must name registry entries, and landed entries must lose their flags", () => {
    const root = fixture({
        "a.md": "# A\n\nStatic [not yet in the current release](pending:acoustic.static).\n\n```fea pending=fluid.nope\nstudy s {}\n```\n",
        "b.md": "# B\n\nHarmonic [not yet in the current release](pending:fluid.harmonic).\n",
    }, [feature("acoustic.static"), feature("fluid.harmonic", { landed_in: "0.7.0" }), feature("acoustic.nonlinear"), feature("thermal.quiet", { doc_less: "Tracked by a tooling item." })]);
    try {
        const report = lintDocs(root);
        assert.deepEqual(rules(report), ["pending-landed", "pending-unknown", "pending-unreferenced"]);
        assert.equal(report.errors.find(error => error.rule === "pending-landed").file, "docs/simulation/b.md");
        assert.equal(report.errors.find(error => error.rule === "pending-unknown").line, 5);
    }
    finally { rmSync(root, { recursive: true, force: true }); }
});

test("front matter is checked against its schema and the example catalog", () => {
    const good = "---\ndescription: A page that explains how studies are written, checked and run from the command line.\nkeywords: [study, check, run]\ndoc_type: concept\nexamples:\n  - cantilever-static\n---\n# Good\n";
    const bad = "---\ndescription: Too short.\nkeywords: [study]\ndoc_type: essay\nexamples: [missing-example]\n---\n# Bad\n";
    const root = fixture({ "good.md": good, "bad.md": bad }, []);
    try {
        const report = lintDocs(root);
        assert.ok(report.errors.every(error => error.file === "docs/simulation/bad.md" && error.rule === "front-matter"));
        assert.equal(report.errors.length, 2);
    }
    finally { rmSync(root, { recursive: true, force: true }); }
});

test("links resolve to files and headings inside docs/simulation", () => {
    const root = fixture({
        "index.md": "# Index\n\n[ok](./guide.md#apply-a-load) [route](./guide) [missing](./nope.md) [anchor](./guide.md#nowhere) [out](../../fea/x.md) [flagged](./guide.md#harmonic-sweeps) `[code](./ignored.md)`\n",
        "guide.md": "# Guide\n\n## Apply a load\n\n## Harmonic sweeps [not yet in the current release](pending:acoustic.static)\n",
    }, [feature("acoustic.static")]);
    try {
        assert.deepEqual(rules(lintDocs(root)), ["link-anchor", "link-missing", "link-outside"]);
    }
    finally { rmSync(root, { recursive: true, force: true }); }
});

test("scanner reads fence attributes, front matter and anchors like the website", () => {
    const [block] = fencedBlocks("text\n```fea pending=acoustic.static\nstudy s {}\n```\n");
    assert.equal(block.language, "fea");
    assert.equal(block.attributes.pending, "acoustic.static");
    assert.equal(block.line, 2);
    assert.deepEqual(parseFrontMatter("---\nkeywords:\n  - a\n  - b\ndoc_type: glossary\n---\nbody\n").data, { keywords: ["a", "b"], doc_type: "glossary" });
    assert.equal(headingAnchor("Apply a `structural.force` load [not yet in the current release](pending:x.y)"), "apply-a-structural-force-load");
    assert.equal(readFileSync(join(repository, "docs/simulation/pending-features.json"), "utf8").endsWith("\n"), true);
});
