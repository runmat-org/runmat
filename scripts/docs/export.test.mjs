import test from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync, readFileSync, cpSync, rmSync } from "node:fs";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { buildSourceDocs } from "./export.mjs";
import { hash, verifyArtifact, inventory, writeManifest } from "./manifest.mjs";
function fixture() {
    const root = mkdtempSync(join(tmpdir(), "runmat-docs-export-test-"));
    const source = resolve(import.meta.dirname, "../..");
    for (const file of ["docs/catalog.json", "docs/simulation/manual.json", "docs/simulation/ledger.json", "docs/simulation/examples/catalog.json", "docs/simulation/guides/linear-static.md", "docs/simulation/validation/methodology.md", "docs/simulation/validation/structural.md", "docs/simulation/examples/structural-static.fea", "docs/simulation/examples/geometry/two-box-assembly.step"]) {
        mkdirSync(join(root, file, ".."), { recursive: true });
        cpSync(join(source, file), join(root, file));
    }
    cpSync(join(source, "docs/schema"), join(root, "docs/schema"), { recursive: true });
    mkdirSync(join(root, "docs/builtins/reference"), { recursive: true });
    writeFileSync(join(root, "docs/builtins/reference/disp.json"), '{"title":"disp"}');
    writeFileSync(join(root, "docs/catalog.json"), JSON.stringify({ schema_version: 1, documents: [{ id: "manual", path: "docs/simulation/guides/linear-static.md" }] }));
    const manual = JSON.parse(readFileSync(join(root, "docs/simulation/manual.json")));
    manual.topics = manual.topics.filter(topic => ["linear-static", "validation-methodology"].includes(topic.id));
    writeFileSync(join(root, "builtins.json"), '{"builtins":[]}');
    writeFileSync(join(root, "builtin-documentation.json"), JSON.stringify({ schema_version: 2, inventory: { documents: 2, catalog_identities: 2, legacy_sidecars: 0, missing_catalog_documentation: [] }, builtins: [
        { key: "abs", module_stem: "abs", authority: "catalog", title: "abs", examples: [{ input: "assert(abs(-1) == 1)", verification: "Succeeds" }] },
        { key: "plot", module_stem: "plot", authority: "catalog", title: "plot", sections: [{ heading: "Behavior", paragraphs: ["Canonical graphics documentation"] }] }
    ] }));
    const catalogPath = join(root, "docs/simulation/examples/catalog.json");
    const catalog = JSON.parse(readFileSync(catalogPath)); catalog.examples = [catalog.examples.find(example => example.id === "two-body-static")]; catalog.examples[0].snapshot = false;
    manual.topics[0].examples = ["two-body-static"];
    writeFileSync(join(root, "docs/simulation/manual.json"), JSON.stringify(manual));
    writeFileSync(catalogPath, JSON.stringify(catalog));
    const options = { source: root, output: join(root, "target/docs/source"), builtinDescriptorsPath: join(root, "builtins.json"), builtinDocumentationPath: join(root, "builtin-documentation.json") };
    return { root, options };
}
test("repeat exports preserve exact manifests and complete deterministic example archives", () => {
    const { root, options } = fixture();
    try {
        const first = buildSourceDocs(options);
        const second = buildSourceDocs(options);
        assert.equal(first.content_hash, second.content_hash);
        verifyArtifact(options.output);
        const references = first.files.filter(file => file.path.startsWith("builtins/reference/"));
        assert.deepEqual(references.map(file => file.path), ["builtins/reference/abs.json", "builtins/reference/plot.json"]);
        const canonical = JSON.parse(readFileSync(options.builtinDocumentationPath)).builtins;
        assert.deepEqual(references.map(file => JSON.parse(readFileSync(join(options.output, file.path)))), canonical);
        const catalog = JSON.parse(readFileSync(join(options.output, "examples/catalog.json")));
        assert.deepEqual(catalog.examples[0].files.map(file => file.path), ["structural-static.fea", "geometry/two-box-assembly.step"]);
        const example = catalog.examples[0];
        assert.equal(hash(JSON.stringify(example.package_identity)), example.digest);
        assert.equal(example.package_identity.entry, example.entry);
        assert.deepEqual(example.package_identity.files, example.files.map(({ content, ...metadata }) => metadata));
        assert.equal(JSON.parse(readFileSync(join(options.output, "fea/evidence-index.json"))).records.length, 0);
    }
    finally {
        rmSync(root, { recursive: true, force: true });
    }
});
test("unknown contract references reject without replacing a valid export", () => {
    const { root, options } = fixture();
    try {
        const first = buildSourceDocs(options);
        // Requirement ids come from the capability ledger; an example naming another one rejects.
        const path = join(root, "docs/simulation/examples/catalog.json");
        const catalog = JSON.parse(readFileSync(path));
        catalog.examples[0].requirements.push("FEA-UNKNOWN-001");
        writeFileSync(path, JSON.stringify(catalog));
        assert.throws(() => buildSourceDocs(options), /FEA-UNKNOWN-001|Unknown/);
        assert.equal(verifyArtifact(options.output).content_hash, first.content_hash);
    }
    finally {
        rmSync(root, { recursive: true, force: true });
    }
});
test("invalid canonical documentation never replaces a published complete artifact", () => {
    const { root, options } = fixture();
    try {
        const first = buildSourceDocs(options);
        const payload = JSON.parse(readFileSync(options.builtinDocumentationPath));
        payload.preview = { kind: "catalog", final_acceptance: false };
        writeFileSync(options.builtinDocumentationPath, JSON.stringify(payload));
        assert.throws(() => buildSourceDocs(options), /aggregate builtin documentation export/);
        assert.equal(verifyArtifact(options.output).content_hash, first.content_hash);
        assert.deepEqual(first.files.filter(file => file.path.startsWith("builtins/reference/")).map(file => file.path), ["builtins/reference/abs.json", "builtins/reference/plot.json"]);
    } finally { rmSync(root, { recursive: true, force: true }); }
});
test("omitting the referenced STEP geometry rejects the package", () => {
    const { root, options } = fixture();
    try {
        const path = join(root, "docs/simulation/examples/catalog.json");
        const catalog = JSON.parse(readFileSync(path));
        catalog.examples[0].files = catalog.examples[0].files.filter(file => !file.endsWith(".step"));
        writeFileSync(path, JSON.stringify(catalog));
        assert.throws(() => buildSourceDocs(options), /Undeclared example dependency/);
    }
    finally {
        rmSync(root, { recursive: true, force: true });
    }
});
test("path escapes reject", () => {
    const { root, options } = fixture();
    try {
        const path = join(root, "docs/simulation/examples/catalog.json");
        const catalog = JSON.parse(readFileSync(path));
        catalog.examples[0].files.push("../secret");
        writeFileSync(path, JSON.stringify(catalog));
        assert.throws(() => buildSourceDocs(options), /Unsafe docs path/);
    }
    finally {
        rmSync(root, { recursive: true, force: true });
    }
});
test("editorial example metadata changes the immutable package identity", () => {
    const { root, options } = fixture();
    try {
        buildSourceDocs(options);
        const exported = () => JSON.parse(readFileSync(join(options.output, "examples/catalog.json"))).examples[0];
        const first = exported().digest;
        const path = join(root, "docs/simulation/examples/catalog.json");
        const catalog = JSON.parse(readFileSync(path));
        catalog.examples[0].title = "A revised example title";
        writeFileSync(path, JSON.stringify(catalog));
        buildSourceDocs(options);
        assert.notEqual(exported().digest, first);
    } finally { rmSync(root, { recursive: true, force: true }); }
});
test("evidence outside a requirement's declared scope rejects", () => {
    const { root, options } = fixture();
    try {
        const evidencePath = join(root, "evidence.json");
        const register = JSON.parse(readFileSync(join(root, "docs/simulation/ledger.json")));
        writeFileSync(evidencePath, JSON.stringify({ schema_version: 1, records: [{
                    requirement: register.requirements[0].id,
                    obligation: "undeclared",
                    scope: { formulation: "elasticity", mesh: "tet4", execution: "native" },
                    decision: "passed",
                    summary: "Fixture assertion",
                    reference: "Fixture reference",
                    tested_identity: "fixture",
                    record_url: "https://example.com/verification",
                    limits: "Fixture only"
                }] }));
        assert.throws(() => buildSourceDocs({ ...options, evidencePath }), /undeclared scope/);
    }
    finally {
        rmSync(root, { recursive: true, force: true });
    }
});
test("saved projects bind original inputs and result identity into immutable asset paths", () => {
    const { root, options } = fixture();
    try {
        const catalogPath = join(root, "docs/simulation/examples/catalog.json");
        const catalog = JSON.parse(readFileSync(catalogPath)); const example = catalog.examples[0]; example.snapshot = true;
        writeFileSync(catalogPath, JSON.stringify(catalog));
        const snapshotsRoot = join(root, "snapshots"); const project = join(snapshotsRoot, example.id, "project");
        for (const path of example.files) { mkdirSync(join(project, path, ".."), { recursive: true }); cpSync(join(root, "docs/simulation/examples", path), join(project, path)); }
        mkdirSync(join(project, ".snapshot"), { recursive: true }); mkdirSync(join(project, ".artifacts/runs"), { recursive: true });
        writeFileSync(join(project, ".artifacts/runs/docs-example.json"), "{}");
        writeFileSync(join(project, ".snapshot/view.scene.json"), "saved geometry"); writeFileSync(join(project, ".snapshot/view.png"), "poster fixture");
        const initial_camera = { activeAxes: 0, axes: [{ position: [1, -1, 1], target: [0, 0, 0], up: [0, 0, 1], zoom: 1, aspectRatio: 1.6, projection: { kind: "perspective", fov: 0.8, near: 0.01, far: 100 } }] };
        writeFileSync(join(project, ".snapshot/snapshot.json"), JSON.stringify({ schema_version: 1, run_id: "fixture", publication: { numerically_acceptable: true }, views: [{ initial_camera, id: "geometry", label: "Solver mesh", unit: "m", minimum: null, maximum: null, deformation_scale: 1, scene: "view.scene.json", poster: "view.png" }] }));
        const graphicsRoot = join(root, "graphics"); mkdirSync(graphicsRoot); writeFileSync(join(graphicsRoot, "runmat_wasm_web.js"), "export function importGeometryScene(scene) { return scene; }"); writeFileSync(join(graphicsRoot, "runmat_wasm_web_bg.wasm"), "wasm fixture");
        const exportOptions = { ...options, snapshotsRoot, graphicsRoot };
        buildSourceDocs(exportOptions);
        const first = JSON.parse(readFileSync(join(options.output, "examples/catalog.json"))).examples[0];
        assert.ok(first.playback.project.sha256); assert.ok(first.playback.views[0].scene.sha256);
        assert.deepEqual(first.playback.views[0].initial_camera, initial_camera);
        const captured = JSON.parse(readFileSync(join(project, ".snapshot/snapshot.json")));
        captured.views[0].initial_camera.axes[0].projection.near = -1;
        writeFileSync(join(project, ".snapshot/snapshot.json"), JSON.stringify(captured));
        assert.throws(() => buildSourceDocs(exportOptions), /Invalid saved camera/);
        verifyArtifact(options.output);
        captured.views[0].initial_camera = initial_camera;
        captured.views[0].scalar_scale = 0;
        writeFileSync(join(project, ".snapshot/snapshot.json"), JSON.stringify(captured));
        assert.throws(() => buildSourceDocs(exportOptions), /Invalid saved-view metadata/);
        verifyArtifact(options.output);
        captured.views[0].scalar_scale = 2;
        writeFileSync(join(project, ".snapshot/snapshot.json"), JSON.stringify(captured));
        writeFileSync(join(project, ".snapshot/view.scene.json"), "new saved geometry"); buildSourceDocs(exportOptions);
        const second = JSON.parse(readFileSync(join(options.output, "examples/catalog.json"))).examples[0];
        assert.notEqual(first.package_path, second.package_path);
        writeFileSync(join(project, example.entry), "changed source");
        assert.throws(() => buildSourceDocs(exportOptions), /Snapshot source differs/);
        verifyArtifact(options.output);
    } finally { rmSync(root, { recursive: true, force: true }); }
});


test("file inventory preserves byte-exact hashes across chunks and detects changed content", () => {
    const root = mkdtempSync(join(tmpdir(), "runmat-docs-file-identity-"));
    try {
        mkdirSync(join(root, "nested"));
        const contents = Buffer.from(Array.from({ length: 131079 }, (_, index) => index % 251));
        writeFileSync(join(root, "nested/data.bin"), contents);
        writeFileSync(join(root, "empty.bin"), Buffer.alloc(0));
        assert.deepEqual(inventory(root), [
            { path: "empty.bin", sha256: hash(Buffer.alloc(0)), bytes: 0 },
            { path: "nested/data.bin", sha256: hash(contents), bytes: contents.length },
        ]);
        writeManifest(root, { revision: "file-identity-fixture" });
        verifyArtifact(root);
        contents[65536] ^= 0xff;
        writeFileSync(join(root, "nested/data.bin"), contents);
        assert.throws(() => verifyArtifact(root), /inventory\/hash mismatch/);
    } finally {
        rmSync(root, { recursive: true, force: true });
    }
});
test("every committed grammar ships with an extension index the site selects by", () => {
    const { root, options } = fixture();
    try {
        const source = resolve(import.meta.dirname, "../..");
        cpSync(join(source, "crates/runmat-design-syntax/grammar"), join(root, "crates/runmat-design-syntax/grammar"), { recursive: true });
        buildSourceDocs(options);
        const index = JSON.parse(readFileSync(join(options.output, "design/grammars.json")));
        assert.equal(index.schema_version, 1);
        assert.deepEqual(index.grammars.map(grammar => grammar.extension), ["circuit.layout", "assembly", "circuit", "drawing", "part", "fea"]);
        for (const grammar of index.grammars) {
            const shipped = JSON.parse(readFileSync(join(options.output, grammar.path)));
            assert.equal(shipped.scopeName, grammar.scopeName);
        }
        assert.equal(index.grammars.find(grammar => grammar.extension === "circuit").scopeName, "source.circuit");
    }
    finally {
        rmSync(root, { recursive: true, force: true });
    }
});
