import test from "node:test";
import assert from "node:assert/strict";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { REGISTRIES, registryNames } from "./registry-coverage.mjs";
import { readFileSync } from "node:fs";
import { COMMITTED_GRAMMAR, DESIGN_DIALECTS, committedDesignGrammar, designGrammar, designGrammarText, designVocabulary, feaGrammar, grammarText, tokenizeLine, vocabulary } from "./fea-grammar.mjs";

const repository = resolve(dirname(fileURLToPath(import.meta.url)), "../..");

test("every compiler vocabulary group is found in the compiler sources", () => {
    for (const group of vocabulary(repository)) assert.ok(group.names.length > 0, `no names extracted for ${group.id}; re-anchor it in fea-grammar.mjs`);
});

test("the grammar colors every word the compiler knows with its group's scope", () => {
    const grammar = feaGrammar(repository);
    const disagreements = [];
    for (const group of vocabulary(repository))
        for (const name of group.names) {
            const pieces = tokenizeLine(grammar, `${name} {`).filter(([text]) => text.trim());
            if (pieces[0]?.[0] !== name || !pieces[0][1]) disagreements.push(`${group.id}: ${name} -> ${JSON.stringify(pieces[0])}`);
        }
    assert.deepEqual(disagreements, []);
});

test("every unit colors as a unit after a number, alone and in compound units", () => {
    const grammar = feaGrammar(repository);
    const units = registryNames(repository, REGISTRIES.find(registry => registry.id === "units"));
    for (const unit of [...units, "kg/m^3", "W/m/K", "m^2"]) {
        const pieces = tokenizeLine(grammar, `x = 2.5 ${unit}`);
        assert.deepEqual(pieces.find(([text]) => text === unit)?.[1], "support.type.unit.fea", unit);
    }
});

test("reference paths color their compiler segments and leave declared names plain", () => {
    const pieces = tokenizeLine(feaGrammar(repository), "output d = max(magnitude(analysis.static.field.node.displacement), over = geometry.beam.region.tip)");
    const scope = word => pieces.find(([text]) => text === word)?.[1] ?? null;
    assert.equal(scope("static"), "storage.type.fea");
    assert.equal(scope("node"), "constant.language.fea");
    assert.equal(scope("displacement"), "constant.language.fea");
    assert.equal(scope("max"), "support.function.fea");
    assert.equal(scope("beam"), null);
});

test("the committed grammar the Rust hosts embed matches the compiler tables", () => {
    assert.equal(readFileSync(join(repository, COMMITTED_GRAMMAR), "utf8"), grammarText(repository), `regenerate with: node scripts/docs/fea-grammar.mjs --write`);
});

test("every design dialect reads its grammar and statement words from the registries", () => {
    for (const entry of DESIGN_DIALECTS) {
        const groups = Object.fromEntries(designVocabulary(repository, entry).map(group => [group.id, group.names]));
        for (const id of ["declarations", "generic forms", "clauses", "literals"]) assert.ok(groups[id].length > 0, `${entry.name}: no ${id}; re-anchor it in fea-grammar.mjs`);
        if (entry.statements) assert.ok(groups.statements.length > 0, `${entry.name}: no statement words; re-anchor it in fea-grammar.mjs`);
        if (entry.operations) assert.ok(groups.operations.length > 0, `${entry.name}: no operations; re-anchor it in fea-grammar.mjs`);
    }
});

test("a design dialect colors its words, values and units", () => {
    const part = designGrammar(repository, DESIGN_DIALECTS.find(entry => entry.name === "part"));
    const scope = (line, word) => tokenizeLine(part, line).find(([text]) => text === word)?.[1] ?? null;
    assert.equal(scope("part bracket {", "part"), "keyword.control.part");
    assert.equal(scope("    feature body = extrude { profile = p }", "feature"), "storage.type.part");
    assert.equal(scope("    feature body = extrude { profile = p }", "extrude"), "support.function.part");
    assert.equal(scope("    let r = 10 kohm ± 1 %", "%"), "support.type.unit.part");
    assert.equal(scope("    let r = 10 kohm ± 1 %", "±"), "keyword.operator.part");
    assert.equal(scope("    publish top = region { region = top }", "publish"), "keyword.control.part");
    const circuit = designGrammar(repository, DESIGN_DIALECTS.find(entry => entry.name === "circuit"));
    assert.equal(tokenizeLine(circuit, "    component R1 = { part = \"R0402\" }").find(([text]) => text === "component")?.[1], "storage.type.circuit");
});

test("the committed design grammars match the registries", () => {
    for (const entry of DESIGN_DIALECTS)
        assert.equal(readFileSync(join(repository, committedDesignGrammar(entry)), "utf8"), designGrammarText(repository, entry), `regenerate with: node scripts/docs/fea-grammar.mjs --write`);
});
