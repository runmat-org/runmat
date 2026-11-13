import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { collectDocuments } from "./documents.mjs";

function fixture(run) {
    const root = mkdtempSync(join(tmpdir(), "runmat-canonical-documents-"));
    try {
        mkdirSync(join(root, "docs"));
        mkdirSync(join(root, "output"));
        writeFileSync(join(root, "docs/catalog.json"), JSON.stringify({ schema_version: 1, documents: [] }));
        const documents = [
            { key: "abs", module_stem: "abs", authority: "catalog", title: "Absolute value", examples: [{ input: "abs(-1)", verification: { Assertions: { source: "assert(abs(-1) == 1)" } } }] },
            { key: "arraydatastore", module_stem: "arrayDatastore", authority: "catalog", evidence: { implementation: [] } },
            { key: "plot", module_stem: "plot", authority: "catalog", sections: [{ heading: "Behavior", paragraphs: ["Graphics content"] }] },
        ];
        const payload = { schema_version: 2, inventory: { documents: documents.length, catalog_identities: documents.length, legacy_sidecars: 0, missing_catalog_documentation: [] }, builtins: documents };
        const path = join(root, "export.json");
        const write = () => writeFileSync(path, JSON.stringify(payload));
        write();
        run({ root, payload, path, write, collect: () => collectDocuments(root, join(root, "output"), () => {}, path) });
    } finally { rmSync(root, { recursive: true, force: true }); }
}

test("packing uses complete canonical documents without any physical reference directory", () => fixture(({ root, payload, collect }) => {
    collect();
    const output = join(root, "output/builtins/reference");
    assert.deepEqual(readdirSync(output).sort(), ["abs.json", "arrayDatastore.json", "plot.json"]);
    for (const document of payload.builtins) {
        assert.deepEqual(JSON.parse(readFileSync(join(output, document.module_stem + ".json"))), document);
    }
    mkdirSync(join(root, "docs/builtins/reference"), { recursive: true });
    writeFileSync(join(root, "docs/builtins/reference/abs.json"), "not valid JSON and not an authority");
    writeFileSync(join(root, "docs/builtins/reference/stale.json"), "{}");
    collect();
    assert.deepEqual(readdirSync(output).sort(), ["abs.json", "arrayDatastore.json", "plot.json"]);
}));

test("unsafe or colliding exported filenames and duplicate identities reject before builtin writes", () => fixture(({ root, payload, write, collect }) => {
    for (const name of ["../escape", "a/b", "a\\b", "", ".", "..", "bad\u0000name"]) {
        payload.builtins[1].module_stem = name;
        write();
        assert.throws(collect, /Unsafe docs path|Invalid exported builtin document filename/);
    }
    payload.builtins[1].module_stem = "ABS";
    write();
    assert.throws(collect, /Duplicate builtin document filename/);
    payload.builtins[1].key = "ABS";
    write();
    assert.throws(collect, /Duplicate builtin documentation identity/);
    assert.deepEqual(readdirSync(join(root, "output")), ["documents.json"]);
}));
