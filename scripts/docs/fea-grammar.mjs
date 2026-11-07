// The `.fea` TextMate grammar, generated from the compiler's own closed tables so that docs, site
// code cards and any TextMate host color exactly the words the compiler knows:
//   keywords            runmat-design-syntax lexer (the fea_keywords! table)
//   statement words     `assessment` (procedure/assessment.rs)
//   families, procedures analysis-core family_label / procedure_label, and the procedure words
//                       language lowering accepts (`preload`, `steady`, `modal_harmonic`, ...)
//   mesh words          element blocks and orders (language semantic/mesh.rs)
//   loads, constraints, output operations, field locations and quantities, material members,
//   types, units        the registries in registry-coverage.mjs and ledger.mjs (SURFACES)
//
//   node scripts/docs/fea-grammar.mjs [--out <file>]   # write the grammar JSON (default: stdout)
//   node scripts/docs/fea-grammar.mjs --write          # refresh the committed copies the Rust hosts embed
//                                                       (`.fea` and every design dialect)
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
    { id: "keywords", scope: "keyword.control.fea", source: "crates/runmat-design-syntax/src/lexical/token.rs", from: /^fea_keywords! \{/m, to: /^\}/m, pattern: /\w+ => "([a-z_]+)"/g },
    { id: "statements", scope: "keyword.control.fea", source: "crates/runmat-analysis/language/src/semantic/procedure/assessment.rs", from: /./, to: /$^/, pattern: /== Some\("([a-z_]+)"\)/g },
    { id: "families", scope: "storage.type.fea", source: "crates/runmat-analysis/core/src/study/compatibility.rs", from: /pub const fn family_label\(/, to: /^}/m, pattern: /=> \{?\s*"([a-z_]+)"/g },
    { id: "procedures", scope: "storage.type.fea", source: "crates/runmat-analysis/core/src/study/compatibility.rs", from: /pub const fn procedure_label\(/, to: /^}/m, pattern: /=> "([a-z_]+)"/g },
    { id: "authored procedures", scope: "storage.type.fea", source: "crates/runmat-analysis/language/src/semantic/procedure/lower.rs", from: /fn lower_procedure\(/, to: /^}/m, pattern: /\("[a-z_]+", "([a-z_]+)"\) =>/g },
    { id: "mesh blocks", scope: "storage.type.fea", source: "crates/runmat-analysis/language/src/semantic/mesh.rs", from: /const ELEMENT_BLOCKS/, to: /\];/, pattern: /\("([a-z_]+)", MeshDimension::/g },
    { id: "element orders", scope: "constant.language.fea", source: "crates/runmat-analysis/language/src/semantic/mesh.rs", from: /required_enum\(declaration, settings_block, "order"/, to: /_ =>/, pattern: /"([a-z_]+)" => Some\(ElementOrder/g },
    { id: "region selectors", scope: "constant.language.fea", source: "crates/runmat-analysis/language/src/semantic/geometry/command.rs", from: /pub fn region_selector\(/, to: /pub fn region_cardinality\(/, pattern: /"([a-z_]+)"(?= =>| \|)/g },
    { id: "region expectations", scope: "constant.language.fea", source: "crates/runmat-analysis/language/src/semantic/geometry/command.rs", from: /pub fn region_cardinality\(/, to: /pub fn parameter_selector\(/, pattern: /value == "([a-z_]+)"/g },
    { id: "parameter access", scope: "constant.language.fea", source: "crates/runmat-analysis/language/src/semantic/geometry/command.rs", from: /pub fn parameter_access\(/, to: /^}/m, pattern: /access == "([a-z_]+)"/g },
    { id: "reference segments", scope: "keyword.control.fea", source: "crates/runmat-analysis/language/src/semantic/expression/reference.rs", from: /./, to: /$^/, pattern: /(?:state|field) == "([a-z_]+)"/g },
    { id: "step order", scope: "keyword.control.fea", source: "crates/runmat-analysis/language/src/semantic/procedure/header.rs", from: /./, to: /$^/, pattern: /\[_, _, "([a-z_]+)", reference\]/g },
    { id: "retained states", scope: "constant.language.fea", source: "crates/runmat-analysis/language/src/semantic/procedure/lower.rs", from: /./, to: /$^/, pattern: /statements, "([a-z_]+)", diagnostics/g },
    { id: "assessment evidence", scope: "constant.language.fea", source: "crates/runmat-analysis/language/src/semantic/procedure/assessment.rs", from: /./, to: /$^/, pattern: /Enum\(name\) if name == "([a-z_]+)"/g },
    { id: "modal static corrections", scope: "constant.language.fea", source: "crates/runmat-analysis/language/src/semantic/procedure/modal_reduction.rs", from: /fn static_correction_kind\(/, to: /_ => None/, pattern: /"([a-z_]+)" => Some\(ModalStaticCorrection/g },
    { id: "output policies", scope: "constant.language.fea", source: "crates/runmat-analysis/language/src/semantic/procedure/output/policy.rs", from: /fn parse_precision\(/, to: /fn enum_name\(/, pattern: /^\s*"([a-z_0-9]+)"(?: if [^\n]*?)? =>/gm },
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

/** The committed copy the Rust hosts embed (runmat-design-syntax `FEA_TEXTMATE_GRAMMAR`). */
export const COMMITTED_GRAMMAR = "crates/runmat-design-syntax/grammar/fea.tmLanguage.json";

export function grammarText(source) {
    return JSON.stringify(feaGrammar(source), null, 2) + "\n";
}

// ---------------------------------------------------------------------------- design dialects
//
// `.part`, `.assembly`, `.drawing`, `.circuit` and `.circuit.layout` share the `.fea` lexer and
// generic forms; each grammar is read from the same two kinds of table:
//   grammar words     runmat-design-syntax dialect registry (src/dialect.rs): the dialect's
//                     declaration words, the generic form words, value, clause and literal words
//   statement words   the dialect's front-end registry (runmat-cad src/syntax/mod.rs grammars,
//                     runmat-circuit src/syntax/mod.rs and src/syntax/layout.rs `words`)
//   operations        registry words a dialect applies (`.part` feature kinds: FeatureKind)
//   units             the `.fea` unit registry, plus `%`

const DIALECT_SOURCE = "crates/runmat-design-syntax/src/dialect.rs";
const CAD_SYNTAX = "crates/runmat-cad/src/syntax/mod.rs";
// The CAD surface forms' own words (runmat-cad src/syntax/forms.rs `words`).
const CAD_FORMS = "crates/runmat-cad/src/syntax/forms.rs";
const CAD_FORM_STATEMENTS = { source: CAD_FORMS, from: /pub const STATEMENTS/, to: /;/, pattern: /"([a-z_]+)"/g };
const CAD_FORM_OPERATIONS = { source: CAD_FORMS, from: /pub const OPERATIONS/, to: /;/, pattern: /"([a-z_]+)"/g };

// Circuit simulation sections: analysis kinds and measure kinds read as operations, a measure's
// own fields (`spec`, `of`, `at`, …) as clauses (runmat-circuit src/syntax/simulation.rs and
// src/simulation.rs `MeasureKind::ALL`).
const CIRCUIT_SIMULATION = "crates/runmat-circuit/src/syntax/simulation.rs";
const CIRCUIT_SIMULATION_OPERATIONS = [
    { source: CIRCUIT_SIMULATION, from: /^pub const ANALYSES/m, to: /;/, pattern: /"([a-z_]+)"/g },
    { source: "crates/runmat-circuit/src/simulation.rs", from: /pub const ALL: &'static \[\(&'static str, MeasureKind\)\]/, to: /\];/, pattern: /\("([a-z_]+)", MeasureKind/g },
];
const CIRCUIT_MEASURE_FIELDS = { source: CIRCUIT_SIMULATION, from: /^pub const MEASURE_FIELDS/m, to: /;/, pattern: /"([a-z_]+)"/g };
// Circuit quantities read through runmat-rules' `BuiltinUnits`: every SI base symbol under
// every prefix, plus the named units its resolver takes (`mil`, `degC`, `ppm`, …).
const RULES_VALUE = "crates/runmat-rules/src/value.rs";
const CIRCUIT_UNITS = source => {
    const text = readFileSync(join(source, RULES_VALUE), "utf8");
    const block = (from, to) => { const start = text.search(from); const rest = text.slice(start); return rest.slice(0, rest.search(to)); };
    const quoted = body => [...body.matchAll(/\("([^"]+)", /g)].map(match => match[1]);
    const prefixes = quoted(block(/^const PREFIXES/m, /^\];/m));
    const base = quoted(block(/^const BASE/m, /^\];/m));
    // Quotes pair in order (the resolver's first arm is `""`), and the empty symbol is no unit.
    const named = [...block(/match symbol \{/, /_ => \{\}/).matchAll(/"([^"]*)"/g)].map(match => match[1]).filter(Boolean);
    return [...new Set([...base, ...prefixes.flatMap(p => base.map(b => p + b)), ...named, "dB"])];
};
// Quoted endpoints in a circuit: a pin (`"C1.1"`, `"U1.V-"`) and a net written as a string (`"net.+5V"`).
const CIRCUIT_QUOTED = [
    { scope: "variable.other.net", match: "\"net\\.[^\"]+\"" },
    { scope: "variable.other.pin", match: "\"[A-Z][A-Za-z0-9_]*(?:\\[\\d+\\])?\\.[A-Za-z0-9_+\\-~#]+\"" },
];

/** One entry per design dialect: its registry name, file extension and front-end word tables. */
export const DESIGN_DIALECTS = [
    { dialect: "Part", name: "part", extension: "part", statements: [{ source: CAD_SYNTAX, from: /^pub const PART: DocumentGrammar/m, to: /^};/m, pattern: /named\("([a-z_]+)"/g }, CAD_FORM_STATEMENTS], operations: [{ source: "crates/runmat-cad/src/model/feature.rs", from: /^pub enum FeatureKind \{/m, to: /^\}/m, pattern: /^\s+([A-Z][A-Za-z]*)\(/gm, snake: true }, CAD_FORM_OPERATIONS] },
    { dialect: "Assembly", name: "assembly", extension: "assembly", statements: [{ source: CAD_SYNTAX, from: /^pub const ASSEMBLY: DocumentGrammar/m, to: /^};/m, pattern: /named\("([a-z_]+)"/g }, CAD_FORM_STATEMENTS] },
    { dialect: "Drawing", name: "drawing", extension: "drawing", statements: { source: CAD_SYNTAX, from: /^pub const DRAWING: DocumentGrammar/m, to: /^};/m, pattern: /named\("([a-z_]+)"/g } },
    { dialect: "Circuit", name: "circuit", extension: "circuit", statements: { source: "crates/runmat-circuit/src/syntax/mod.rs", from: /^pub mod words \{/m, to: /pub const ALL/, pattern: /: &str = "([a-z_]+)"/g }, operations: CIRCUIT_SIMULATION_OPERATIONS, clauses: CIRCUIT_MEASURE_FIELDS, units: CIRCUIT_UNITS, quoted: CIRCUIT_QUOTED },
    { dialect: "CircuitLayout", name: "circuit-layout", extension: "circuit.layout", statements: { source: "crates/runmat-circuit/src/syntax/layout.rs", from: /^pub mod words \{/m, to: /pub const ALL/, pattern: /: &str = "([a-z_]+)"/g }, units: CIRCUIT_UNITS },
];

const snakeCase = name => name.replace(/[A-Z]/g, (letter, index) => (index ? "_" : "") + letter.toLowerCase());

function wordsIn(source, table) {
    if (!table) return [];
    if (Array.isArray(table)) return table.flatMap(entry => wordsIn(source, entry));
    const names = registryNames(source, table);
    return table.snake ? names.map(snakeCase).sort() : names;
}

/** The grammar words of one dialect, from the dialect registry. */
function dialectWords(source, dialect) {
    const text = readFileSync(join(source, DIALECT_SOURCE), "utf8");
    const between = (from, to) => {
        const start = text.search(from);
        if (start < 0) return "";
        const rest = text.slice(start);
        const end = rest.search(to);
        return end < 0 ? rest : rest.slice(0, end);
    };
    const quoted = body => [...body.matchAll(/"([a-z_]+)"/g)].map(match => match[1]);
    const spec = between(new RegExp(`dialect: Dialect::${dialect},`), /\n    \},/);
    return {
        declarations: spec.includes("declarations") ? quoted(spec.slice(spec.indexOf("declarations")).split("]")[0]) : [],
        forms: [...between(/pub const fn word\(self\)/, /^    \}/m).matchAll(/=> "([a-z_]+)"/g)].map(match => match[1]),
        header: quoted(between(/pub const STD_HEADER_WORD/, /;/)),
        values: quoted(between(/pub const VALUE_WORDS/, /;/)),
        clauses: quoted(between(/pub const CLAUSE_WORDS/, /;/)),
        literals: quoted(between(/pub const VALUE_LITERALS/, /;/)),
    };
}

/** { id, scope, names } groups of one design dialect. */
export function designVocabulary(source, entry) {
    const words = dialectWords(source, entry.dialect);
    const suffix = entry.name;
    const groups = [
        { id: "declarations", scope: `keyword.control.${suffix}`, names: words.declarations },
        { id: "generic forms", scope: `keyword.control.${suffix}`, names: [...words.forms, ...words.header, ...words.values] },
        { id: "clauses", scope: `keyword.other.${suffix}`, names: [...words.clauses, ...wordsIn(source, entry.clauses)] },
        { id: "literals", scope: `constant.language.${suffix}`, names: words.literals },
        { id: "statements", scope: `storage.type.${suffix}`, names: wordsIn(source, entry.statements) },
        { id: "operations", scope: `support.function.${suffix}`, names: wordsIn(source, entry.operations) },
    ];
    return groups.map(group => ({ ...group, names: [...new Set(group.names)].sort() }));
}

export function designGrammar(source, entry) {
    const suffix = entry.name;
    const groups = designVocabulary(source, entry).filter(group => group.names.length);
    const units = [...new Set([...registryNames(source, UNITS), ...(entry.units ? entry.units(source) : []), "%"])];
    const order = ["support.function", "keyword.control", "storage.type", "keyword.other", "constant.language"].map(scope => `${scope}.${suffix}`);
    const byScope = new Map();
    for (const group of groups) {
        const names = byScope.get(group.scope) ?? new Set();
        for (const name of group.names) names.add(name);
        byScope.set(group.scope, names);
    }
    const number = "-?\\b\\d+(?:\\.\\d+)?(?:[eE][+\\-]?\\d+)?\\b";
    const unitWord = units => `(?:${[...units].sort((a, b) => b.length - a.length).map(escape).join("|")})`;
    const unit = `${unitWord(units)}(?:\\s*[/*]\\s*${unitWord(units)}|\\^-?\\d+)*`;
    return {
        name: suffix,
        scopeName: `source.${suffix}`,
        fileTypes: [entry.extension],
        generated: "scripts/docs/fea-grammar.mjs from the RunMat dialect registry and front-end registries; do not edit by hand",
        patterns: [
            { name: `comment.line.double-slash.${suffix}`, match: "//.*$" },
            ...(entry.quoted ?? []).map(q => ({ name: `${q.scope}.${suffix}`, match: q.match })),
            { name: `string.quoted.double.${suffix}`, begin: "\"", end: "\"" },
            { match: `(${number}|\\])(\\s*)(${unit})(?![\\w%])`, captures: { 1: { patterns: [{ name: `constant.numeric.${suffix}`, match: "-?\\d.*" }] }, 3: { name: `support.type.unit.${suffix}` } } },
            ...order.filter(scope => byScope.has(scope)).map(scope => ({ name: scope, match: words(byScope.get(scope)) })),
            { name: `variable.other.assignment.${suffix}`, match: "\\b[a-zA-Z_]\\w*(?=\\s*=(?!=))" },
            { name: `constant.numeric.${suffix}`, match: number },
            { name: `keyword.operator.${suffix}`, match: "±|\\+/-|@|[=:;]" },
        ],
        repository: {},
        vocabulary: {
            scopes: order.filter(scope => byScope.has(scope)).map(scope => ({ scope, words: [...byScope.get(scope)].sort() })),
            units: { scope: `support.type.unit.${suffix}`, words: [...units].sort() },
        },
    };
}

/** The committed copy of a design dialect's grammar (runmat-design-syntax `DESIGN_TEXTMATE_GRAMMARS`). */
export const committedDesignGrammar = entry => `crates/runmat-design-syntax/grammar/${entry.extension}.tmLanguage.json`;

export function designGrammarText(source, entry) {
    return JSON.stringify(designGrammar(source, entry), null, 2) + "\n";
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
    const source = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
    const text = grammarText(source);
    const out = process.argv.indexOf("--out");
    if (process.argv.includes("--write")) {
        writeFileSync(join(source, COMMITTED_GRAMMAR), text);
        for (const entry of DESIGN_DIALECTS) writeFileSync(join(source, committedDesignGrammar(entry)), designGrammarText(source, entry));
    }
    else if (out > 0) writeFileSync(process.argv[out + 1], text);
    else process.stdout.write(text);
}
