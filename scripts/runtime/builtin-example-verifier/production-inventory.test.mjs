import assert from "node:assert/strict";
import test from "node:test";
import { buildInventory } from "./inventory.mjs";
import { sha256 } from "./identity.mjs";
import { productionExampleDocumentation, validateCurrentExampleInventory } from "./production-inventory.mjs";

const source = {
    sourceRevision: "0123456789abcdef0123456789abcdef01234567",
    sourceState: "clean",
    runnerDigest: sha256("current runner"),
};
const full = {
    schema_version: 1,
    inventory: { documents: 2, catalog_identities: 2, legacy_sidecars: 0, missing_catalog_documentation: [] },
    builtins: ["abs", "plot"].map((key) => ({
        key, authority: "catalog", category: "test",
        examples: [{ id: "example", description: "Original example", input: "disp(1)", harness: "Native", verification: "Succeeds" }],
    })),
};

test("production example inventory requests strict full current documentation and retains compiler failures", () => {
    const calls = [];
    const spawn = (command, args) => {
        calls.push({ command, args });
        return { status: 0, stdout: JSON.stringify(full), stderr: "" };
    };
    assert.deepEqual(productionExampleDocumentation("/fixture", spawn), full);
    assert.deepEqual(calls, [{ command: "cargo", args: ["run", "--quiet", "-p", "runmat-builtins", "--bin", "export_builtin_documentation", "--"] }]);
    assert.throws(() => productionExampleDocumentation("/fixture", () => ({
        status: 1, stdout: "", stderr: "fea.Direction.vector requires an example or exemption",
    })), /fea.Direction.vector requires an example or exemption/);
});

test("closure compares actual full definitions and source inputs, not a self-declared complete subset", () => {
    const inventory = buildInventory(full, source);
    assert.equal(validateCurrentExampleInventory(inventory, full, source), inventory);
    const scopedExport = { ...full, builtins: full.builtins.slice(0, 1) };
    const subset = buildInventory(scopedExport, source);
    assert.equal(subset.scope.kind, "complete");
    assert.throws(() => validateCurrentExampleInventory(subset, full, source), /current full catalog/);
    for (const change of [
        { sourceRevision: "a".repeat(40) },
        { runnerDigest: sha256("changed shared schema or runner") },
    ]) assert.throws(() => validateCurrentExampleInventory(inventory, full, { ...source, ...change }), /source or runner/);
    for (const field of ["input", "description", "verification"]) {
        const changed = structuredClone(full);
        changed.builtins[0].examples[0][field] = field === "verification" ? { Assertions: { source: "assert(true)" } } : "changed";
        assert.throws(() => validateCurrentExampleInventory(inventory, changed, source), /current full catalog/);
    }
    assert.throws(() => validateCurrentExampleInventory(inventory, full, { ...source, sourceState: "dirty" }), /current clean source/);
    assert.throws(() => validateCurrentExampleInventory(buildInventory(full, { ...source, scope: { limit: 1 } }), full, source), /complete inventory/);
});
