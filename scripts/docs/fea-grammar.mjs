// The `.fea` TextMate grammar, generated from the compiler's own closed tables so that docs, site
// code cards and any TextMate host color exactly the words the compiler knows:
//   keywords            runmat-fea-syntax lexer (FeaKeyword::from_source)
//   statement words     `assessment` (procedure/assessment.rs)
//   families, procedures analysis-core family_label / procedure_label, and the procedure words
//                       language lowering accepts (`preload`, `steady`, `modal_harmonic`, ...)
//   mesh words          element blocks and orders (language semantic/mesh.rs)
//   loads, constraints, output operations, field locations and quantities, material members,
//   types, units        the registries in registry-coverage.mjs and ledger.mjs (SURFACES)
//
//   node scripts/docs/fea-grammar.mjs [--out <file>]   # write the grammar JSON (default: stdout)
//   node scripts/docs/fea-grammar.mjs --write          # refresh the committed copy the Rust hosts embed
//
// The docs export writes it to fea/fea.tmLanguage.json in the docs artifact; the website loads it
// from there. `lintFeaHighlighting` checks every ```fea block in the docs tokenizes with no known
// word left uncolored.
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { REGISTRIES, registryNames } from "./registry-coverage.mjs";
import { SURFACES } from "./ledger.mjs";

/** Word groups, each with the scope it is colored with. Order is match priority. */
export const VOCABULARY = [
    { id: "keywords", scope: "keyword.control.fea", source: "crates/runmat-fea-syntax/src/lexical/token.rs", from: /pub fn from_source\(/, to: /_ => return None/, pattern: /"([a-z_]+)" => Self::/g },
    { id: "statements", scope: "keyword.control.fea", source: "crates/runmat-analysis/language/src/semantic/procedure/assessment.rs", from: /./, to: /$^/, pattern: /== Some\("([a-z_]+)"\)/g },
    { id: "families", scope: "storage.type.fea", source: "crates/runmat-analysis/core/src/study/compatibility.rs", from: /pub const fn family_label\(/, to: /^}/m, pattern: /=> \{?\s*"([a-z_]+)"/g },
    { id: "procedures", scope: "storage.type.fea", source: "crates/runmat-analysis/core/src/study/compatibility.rs", from: /pub const fn procedure_label\(/, to: /^}/m, pattern: /=> "([a-z_]+)"/g },
    { id: "authored procedures", scope: "storage.type.fea", source: "crates/runmat-analysis/language/src/semantic/procedure/lower.rs", from: /fn lower_procedure\(/, to: /^}/m, pattern: /\("[a-z_]+", "([a-z_]+)"\) =>/g },
    { id: "mesh blocks", scope: "storage.type.fea", source: "crates/runmat-analysis/language/src/semantic/mesh.rs", from: /const ELEMENT_BLOCKS/, to: /\];/, pattern: /\("([a-z_]+)", MeshDimension::/g },
    { id: "element orders", scope: "constant.language.fea", source: "crates/runmat-analysis/language/src/semantic/mesh.rs", from: /required_enum\(declaration, settings_block, "order"/, to: /_ =>/, pattern: /"([a-z_]+)" => Some\(ElementOrder/g },
    ...SURFACES.map(({ kind, registry }) => ({ id: `${kind}s`, scope: "support.function.fea", ...registry })),
    { id: "field locations", scope: "constant.language.fea", ...REGISTRIES.find(registry => registry.id === "field locations") },
    { id: "field quantities", scope: "constant.language.fea", ...REGISTRIES.find(registry => registry.id === "field quantities") },
    { id: "material members", scope: "support.type.property-name.fea", ...REGISTRIES.find(registry => registry.id === "material members") },
    { id: "types", scope: "support.type.fea", ...REGISTRIES.find(registry => registry.id === "types") },
];
const UNITS = REGISTRIES.find(registry => registry.id === "units");

/** { id, scope, names } for each vocabulary group, read from the compiler sources. */
export function vocabulary(source) {
    return VOCABULARY.map(group => ({ id: group.id, scope: group.scope, names: registryNames(source, group) }));
}

const escape = word => word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
// A name is a word not glued to a longer identifier. Reference paths are colored segment by
// segment (`analysis.static.field.node.displacement`); dotted names (`structural.fixed`) are whole.
const words = names => `(?<![\\w])(?:${[...names].sort((a, b) => b.length - a.length).map(escape).join("|")})(?![\\w])`;
// Match priority at one position: dotted support names before the family word they start with.
const SCOPE_ORDER = ["support.function.fea", "keyword.control.fea", "storage.type.fea", "constant.language.fea", "support.type.property-name.fea", "support.type.fea"];

export function feaGrammar(source) {
    const groups = vocabulary(source);
    const units = registryNames(source, UNITS);
    const byScope = new Map();
    for (const group of groups) {
        const names = byScope.get(group.scope) ?? new Set();
        for (const name of group.names) names.add(name);
        byScope.set(group.scope, names);
    }
    const number = "-?\\b\\d+(?:\\.\\d+)?(?:[eE][+\\-]?\\d+)?\\b";
    const unit = `(?:${[...units].sort((a, b) => b.length - a.length).map(escape).join("|")})(?:\\s*[/*]\\s*(?:${[...units].map(escape).join("|")})|\\^-?\\d+)*`;
    return {
        name: "fea",
        scopeName: "source.fea",
        fileTypes: ["fea"],
        generated: "scripts/docs/fea-grammar.mjs from the RunMat compiler's closed tables; do not edit by hand",
        patterns: [
            { name: "comment.line.double-slash.fea", match: "//.*$" },
            { name: "string.quoted.double.fea", begin: "\"", end: "\"" },
            // A quantity: number (or a closing vector bracket) followed by a unit expression.
            { match: `(${number}|\\])(\\s*)(${unit})(?![\\w])`, captures: { 1: { patterns: [{ name: "constant.numeric.fea", match: "-?\\d.*" }] }, 3: { name: "support.type.unit.fea" } } },
            // `units = mm` and other bare unit values after `=`.
            { match: `\\b(units)(\\s*=\\s*)(${unit})(?![\\w])`, captures: { 1: { name: "variable.other.assignment.fea" }, 2: { patterns: [{ name: "keyword.operator.fea", match: "\\s*=\\s*" }] }, 3: { name: "support.type.unit.fea" } } },
            ...SCOPE_ORDER.filter(scope => byScope.has(scope)).map(scope => ({ name: scope, match: words(byScope.get(scope)) })),
            { name: "variable.other.assignment.fea", match: "\\b[a-zA-Z_]\\w*(?=\\s*=(?!=))" },
            { name: "constant.numeric.fea", match: number },
            { name: "keyword.operator.fea", match: "[=:;]" },
        ],
        repository: {},
        // The same word classes as plain lists, for hosts that classify lexer tokens instead of
        // running TextMate regexes (runmat-lsp semantic tokens, Desktop). Ignored by TextMate.
        vocabulary: {
            scopes: SCOPE_ORDER.filter(scope => byScope.has(scope)).map(scope => ({ scope, words: [...byScope.get(scope)].sort() })),
            units: { scope: "support.type.unit.fea", words: [...units].sort() },
        },
    };
}

/**
 * A small TextMate-style tokenizer for this grammar's shape (single-line `match` patterns with
 * captures, and a `begin`/`end` string): at each position the earliest match wins, ties go to
 * the first pattern. It returns [text, scope | null] pieces per line.
 */
export function tokenizeLine(grammar, line) {
    const pieces = [];
    let position = 0;
    const compiled = grammar.patterns.map(pattern => ({ pattern, regex: new RegExp(pattern.match ?? pattern.begin, "g") }));
    while (position < line.length) {
        let best = null;
        for (const entry of compiled) {
            entry.regex.lastIndex = position;
            const match = entry.regex.exec(line);
            if (match && match[0].length && (!best || match.index < best.match.index)) best = { ...entry, match };
        }
        if (!best) { pieces.push([line.slice(position), null]); break; }
        if (best.match.index > position) pieces.push([line.slice(position, best.match.index), null]);
        const { pattern, match } = best;
        if (pattern.begin) {
            const close = line.indexOf(pattern.end, match.index + match[0].length);
            const end = close < 0 ? line.length : close + pattern.end.length;
            pieces.push([line.slice(match.index, end), pattern.name]);
            position = end;
            continue;
        }
        if (pattern.captures) {
            let offset = match.index;
            for (let group = 1; group < match.length; group += 1) {
                if (match[group] === undefined) continue;
                const capture = pattern.captures[group];
                const nested = capture?.patterns?.find(inner => new RegExp(`^(?:${inner.match})$`).test(match[group]));
                pieces.push([match[group], capture?.name ?? nested?.name ?? null]);
                offset += match[group].length;
            }
        } else pieces.push([match[0], pattern.name]);
        position = match.index + match[0].length;
    }
    return pieces;
}

/** Every known compiler word that a ```fea block leaves uncolored: { line, word }. */
export function uncoloredKnownWords(grammar, known, code) {
    const problems = [];
    code.split("\n").forEach((line, index) => {
        for (const [text, scope] of tokenizeLine(grammar, line)) {
            if (scope) continue;
            for (const match of text.matchAll(/(?<![\w.])[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*/g)) {
                // A whole dotted name (`structural.fixed`) or each segment of a reference path.
                const words = known.has(match[0]) ? [match[0]] : match[0].split(".");
                for (const word of words) if (known.has(word)) problems.push({ line: index + 1, word });
            }
        }
    });
    return problems;
}

/** The set of compiler words that must always be colored. */
export function knownWords(source) {
    return new Set(vocabulary(source).flatMap(group => group.names));
}

/** The committed copy the Rust hosts embed (runmat-fea-syntax `FEA_TEXTMATE_GRAMMAR`). */
export const COMMITTED_GRAMMAR = "crates/runmat-fea-syntax/grammar/fea.tmLanguage.json";

export function grammarText(source) {
    return JSON.stringify(feaGrammar(source), null, 2) + "\n";
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
    const source = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
    const text = grammarText(source);
    const out = process.argv.indexOf("--out");
    if (process.argv.includes("--write")) writeFileSync(join(source, COMMITTED_GRAMMAR), text);
    else if (out > 0) writeFileSync(process.argv[out + 1], text);
    else process.stdout.write(text);
}
