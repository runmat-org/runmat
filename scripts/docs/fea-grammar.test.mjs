import test from "node:test";
import assert from "node:assert/strict";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { REGISTRIES, registryNames } from "./registry-coverage.mjs";
import { feaGrammar, tokenizeLine, vocabulary } from "./fea-grammar.mjs";

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
