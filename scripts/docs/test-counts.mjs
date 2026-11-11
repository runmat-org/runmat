// Counts the automated tests behind each Validation family for the Status page.
//
// A test is every `#[test]` or `#[tokio::test]` function in the simulation crates below. It counts
// toward a family when its function name, or failing that its source path, names the family;
// coupled families are matched first so a conduction-through-fluid test counts as conjugate heat
// transfer, not fluid flow. Tests that name no family (meshing, sparse solvers, study planning,
// receipts) are counted once as shared, because every family runs on them.
//
// The counts change whenever a test is added, so the docs export regenerates Status at publish
// time and `generate-pages.mjs --check` compares Status with these cells masked
// (see `maskTestCounts`).
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

export const TEST_ROOTS = [
    "crates/runmat-analysis", "crates/runmat-meshing", "crates/runmat-sparse-solvers",
    "crates/runmat-fea-source", "crates/runmat-fea-syntax", "crates/runmat-fea-admission-native",
];

// [family area, whole-word tokens, phrases]. Order matters: coupled families first.
const FAMILY_TERMS = [
    ["fluid-structure-interaction", ["fsi"], ["fluid_structure"]],
    ["conjugate-heat-transfer", ["cht"], ["conjugate_heat"]],
    ["thermomechanical", ["thermomechanical"], ["thermal_stress", "thermal_expansion", "thermal_prestress"]],
    ["electrothermal", ["electrothermal", "joule"], []],
    ["acoustic", ["acoustic", "acoustics", "helmholtz", "muffler", "pml", "sound"], ["transmission_loss"]],
    ["fluid", ["cfd", "fluid", "navier", "stokes", "reynolds", "inflow", "outflow", "pipe", "channel", "coolant"],
        ["flow_rate", "pressure_drop", "fluid_flow", "laminar_flow", "flow_field"]],
    ["electromagnetic", ["electrical", "electric", "magnetic", "maxwell", "electrostatic", "electrostatics", "magnetostatic",
        "magnetostatics", "eddy", "conductor", "coil", "electromagnetic", "permeability", "permittivity", "voltage",
        "capacitance", "inductance"], ["line_parameters", "skin_depth"]],
    ["thermal", ["thermal", "heat", "conduction", "convection", "temperature", "radiation"], []],
    ["structural", ["structural", "buckling", "modal", "elastic", "plasticity", "plastic", "stress", "beam", "shell",
        "spectrum", "prestressed", "connector", "contact", "displacement", "cantilever", "solid", "strain", "eigen",
        "frequency"], ["von_mises", "random_response", "response_spectrum"]],
];
const COUPLED = 4;

const names = text => text.split(/[^a-z0-9]+/);
const matches = (text, [, tokens, phrases]) => { const words = names(text); return tokens.some(token => words.includes(token)) || phrases.some(phrase => text.includes(phrase)); };

/** The family area a test belongs to, or "shared". */
export function testFamily(name, path) {
    const both = `${name} ${path}`;
    return FAMILY_TERMS.slice(0, COUPLED).find(terms => matches(both, terms))?.[0]
        ?? FAMILY_TERMS.find(terms => matches(name, terms))?.[0]
        ?? FAMILY_TERMS.find(terms => matches(path, terms))?.[0]
        ?? "shared";
}

const sources = directory => readdirSync(directory).flatMap(entry => {
    if (entry === "target" || entry.startsWith(".")) return [];
    const path = join(directory, entry);
    return statSync(path).isDirectory() ? sources(path) : path.endsWith(".rs") ? [path] : [];
});

/** Test counts by family area, plus `shared` and `total`. */
export function countTests(source) {
    const counts = Object.fromEntries([...FAMILY_TERMS.map(([area]) => area), "shared"].map(area => [area, 0]));
    for (const root of TEST_ROOTS) {
        let files;
        try { files = sources(join(source, root)); } catch { continue; }
        for (const file of files) {
            const path = file.slice(source.length).toLowerCase().replace(/\.rs$/, "");
            const lines = readFileSync(file, "utf8").split("\n");
            lines.forEach((line, index) => {
                if (!/^\s*#\[(tokio::)?test\b/.test(line)) return;
                const name = lines.slice(index + 1, index + 8).map(next => next.match(/\bfn\s+([A-Za-z0-9_]+)/)?.[1]).find(Boolean) ?? "";
                counts[testFamily(name.toLowerCase(), path)] += 1;
            });
        }
    }
    counts.total = Object.values(counts).reduce((sum, count) => sum + count, 0);
    return counts;
}

export const formatCount = count => count.toLocaleString("en-US");

/** Status with the automated-test counts replaced, so `--check` doesn't go stale when a test is added. */
export const maskTestCounts = text => text
    .replace(/^(\| \[[^\]]+\]\([^)]+\) \|[^|]*\| \d+ \| )[\d,]+( \|)/gm, "$1#$2")
    .replace(/(<!-- tests -->)[^<]*(<!-- \/tests -->)/g, "$1#$2");
