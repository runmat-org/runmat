// The capability ledger (docs/simulation/ledger.json): loading, schema validation and the
// ledger ↔ code ↔ docs consistency checks that CI runs through docs-lint.
//
// What the ledger owns (docs/schema/ledger.schema.json):
// - capabilities: status (release | pending), requirement ids, verification cases, docs pages;
//   rows with `identity` generate ANALYSIS_IDENTITIES (crates/runmat-analysis/core/build.rs);
// - cases: reference problems shown on the per-family Validation pages;
// - claims: scope claims mapped to requirement obligations (qualification scope);
// - requirements: the requirement register (crates/runmat-analysis/qualification/build.rs).
import Ajv from "ajv";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { REGISTRIES, registryNames } from "./registry-coverage.mjs";

const readJson = path => JSON.parse(readFileSync(path, "utf8"));

export const LEDGER = "docs/simulation/ledger.json";

export function loadLedger(source) {
    const ledger = readJson(join(source, LEDGER));
    const ajv = new Ajv({ allErrors: true });
    const check = ajv.compile(readJson(join(source, "docs/schema/ledger.schema.json")));
    if (!check(ledger)) throw new Error(`ledger.json: ${ajv.errorsText(check.errors)}`);
    const byId = new Map();
    for (const capability of ledger.capabilities) {
        if (byId.has(capability.id)) throw new Error(`ledger.json: duplicate capability ${capability.id}`);
        byId.set(capability.id, capability);
    }
    return { ledger, byId };
}

/** Expands an obligation's ordered cross-product blocks into envelopes. */
export const expandEnvelopes = blocks => blocks.flatMap(block =>
    block.formulations.flatMap(formulation => block.meshes.flatMap(mesh => block.executions.map(execution => ({ formulation, mesh, execution })))));

/** The register in the shape `export_docs` wrote: obligations carry `required_envelopes`. */
export function registerExport(ledger) {
    return {
        schema_version: 1,
        posture: "discovery_only",
        requirements: ledger.requirements.map(requirement => ({
            ...requirement,
            obligations: requirement.obligations.map(({ envelopes, ...obligation }) => ({ ...obligation, required_envelopes: expandEnvelopes(envelopes) })),
        })),
    };
}

/** The pending-feature view the website's flags read (fea/pending-features.json). */
export function pendingView(ledger) {
    return {
        schema_version: 1,
        features: ledger.capabilities.filter(capability => capability.status === "pending" || capability.landed_in).map(capability => ({
            id: capability.id, title: capability.title, area: capability.area, kind: capability.kind,
            acceptance: capability.acceptance ?? "", landed_in: capability.status === "release" ? (capability.landed_in ?? ledger.release) : null,
            ...(capability.decision ? { decision: capability.decision } : {}),
        })),
    };
}

// Pending ids the code reports (`runmat check` diagnostics and admission rules). The ledger owns
// their status; the code only maps a construct to its id.
const CODE_ROOTS = ["crates/runmat-analysis", "crates/runmat-fea-syntax"];
const CODE_PATTERNS = [
    /\[pending:([a-z0-9._-]+)\]/g, // a literal diagnostic suffix
    /pending:\s*Some\("([a-z0-9._-]+)"\)/g, // StaticRule { pending: Some("id") }
    /Pending\("([a-z0-9._-]+)"\)/g, // a literal ANALYSIS_IDENTITIES row
];

function rustFiles(dir, out = []) {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
        if (entry.name === "target" || entry.name.startsWith(".")) continue;
        const path = join(dir, entry.name);
        if (entry.isDirectory()) rustFiles(path, out);
        else if (entry.name.endsWith(".rs")) out.push(path);
    }
    return out;
}

/** { id, file, line } for every pending id the Rust sources emit. */
export function codePendingIds(source) {
    const found = [];
    for (const root of CODE_ROOTS) {
        if (!existsSync(join(source, root))) continue;
        for (const path of rustFiles(join(source, root))) {
            const text = readFileSync(path, "utf8");
            const file = path.slice(source.length + 1);
            const lineOf = index => text.slice(0, index).split("\n").length;
            for (const pattern of CODE_PATTERNS)
                for (const match of text.matchAll(pattern)) found.push({ id: match[1], file, line: lineOf(match.index) });
            // language/src/semantic/pending.rs: `pub const NAME: &str = "id";`
            if (file.endsWith("semantic/pending.rs"))
                for (const match of text.matchAll(/pub const [A-Z_]+: &str = "([a-z0-9._-]+)";/g)) found.push({ id: match[1], file, line: lineOf(match.index) });
        }
    }
    // Tests may name ids for negative cases; only sources outside tests count.
    return found.filter(entry => !/\/tests?\//.test(entry.file) && !entry.file.endsWith("/tests.rs"));
}

/**
 * The compiler's closed tables of authorable loads, constraints and output operations. Each name
 * is a release ledger row `<kind>.<name>`; a planned one is a pending row the code rejects.
 */
export const SURFACES = [
    { kind: "constraint", registry: { source: "crates/runmat-analysis/language/src/semantic/model/constraints.rs", from: /let kind = match kind \{/, to: /^\s*_ =>|^\s*other =>/m, pattern: /^\s*"([a-z_.]+)"(?:\s*\|\s*"[a-z_.]+")*\s*=>/gm } },
    { kind: "load", registry: { source: "crates/runmat-analysis/language/src/semantic/model/excitations.rs", from: /fn kind_policy\(/, to: /^\s*_ =>|^\s*other =>/m, pattern: /^\s*"([a-z_.]+)"\s*=>/gm } },
    { kind: "output", registry: REGISTRIES.find(registry => registry.id === "output operations") },
];

export function surfaceNames(source) {
    return SURFACES.flatMap(({ kind, registry }) => existsSync(join(source, registry.source)) ? registryNames(source, registry).map(name => ({ kind, name, id: `${kind}.${name}`, file: registry.source })) : []);
}

/** Literal `row(...)` lines of ANALYSIS_IDENTITIES, when compatibility.rs still carries them. */
function literalIdentityRows(source) {
    const path = join(source, "crates/runmat-analysis/core/src/study/compatibility.rs");
    if (!existsSync(path)) return null;
    const text = readFileSync(path, "utf8");
    const rows = [...text.matchAll(/^\s*row\((\w+), P::(\w+), (Direct|ModalReduced), (Runs|Pending\("([^"]+)"\))\),$/gm)];
    return rows.length ? rows.map(match => ({ family: match[1], procedure: match[2], space: match[3], pending: match[5] ?? null })) : null;
}

const FAMILY_CONST = { structural: "STRUCTURAL", thermal: "THERMAL", electromagnetic: "EM", acoustic: "ACOUSTIC", fluid: "FLUID", thermomechanical: "TM", electrothermal: "ET", conjugate_heat_transfer: "CHT", fluid_structure_interaction: "FSI" };
const PROCEDURE_VARIANT = { static: "Static", modal: "ModalExtraction", prestressed_modal: "PrestressedModalExtraction", buckling: "EigenvalueBuckling", transient: "TransientResponse", harmonic: "HarmonicResponse", prestressed_harmonic: "PrestressedHarmonicResponse", response_spectrum: "ModalResponseSpectrum", random_response: "RandomResponse", nonlinear: "Nonlinear", adaptive: "Adaptive" };

/** Ledger integrity and ledger ↔ code ↔ docs consistency. `references` are the docs' pending flags. */
export function lintLedger(source, { ledger, byId }, references, error) {
    const at = (rule, message) => error({ file: LEDGER, line: 0, rule, message });
    const requirementIds = new Set(ledger.requirements.map(requirement => requirement.id));
    const caseIds = new Set(ledger.cases.map(entry => entry.id));
    const root = join(source, "docs/simulation");
    for (const capability of ledger.capabilities) {
        for (const id of capability.requirements) if (!requirementIds.has(id)) at("ledger-requirement", `${capability.id}: unknown requirement ${id}`);
        for (const id of capability.verification) if (!caseIds.has(id)) at("ledger-case", `${capability.id}: unknown case ${id}`);
        for (const doc of capability.docs) if (!existsSync(join(root, doc))) at("ledger-docs", `${capability.id}: docs page ${doc} does not exist`);
        if (capability.status === "pending" && !capability.acceptance) at("ledger-acceptance", `${capability.id}: a pending row states its acceptance`);
        if (capability.status === "pending") {
            const flagged = [...new Set(references.filter(reference => reference.id === capability.id && !reference.generated).map(reference => reference.file))].sort();
            const listed = [...capability.docs].sort();
            if (flagged.join() !== listed.join()) at("ledger-docs", `${capability.id}: docs lists [${listed.join(", ")}] but these pages flag it: [${flagged.join(", ")}]`);
        }
    }
    for (const entry of ledger.cases) {
        if (!existsSync(join(root, entry.narrative))) at("ledger-case", `${entry.id}: narrative ${entry.narrative} does not exist`);
        if (!ledger.capabilities.some(capability => capability.verification.includes(entry.id))) at("ledger-case", `case ${entry.id} verifies no capability`);
    }
    // Claims map to existing requirements and obligations.
    for (const claim of ledger.claims)
        for (const map of claim.maps.split("|")) {
            if (/^(excluded|external):/.test(map)) continue;
            const [requirement, obligation] = map.split("/");
            const found = ledger.requirements.find(entry => entry.id === requirement);
            if (!found) at("ledger-claim", `${claim.id}: unknown requirement ${requirement}`);
            else if (obligation && obligation !== "*" && !found.obligations.some(entry => entry.id === obligation)) at("ledger-claim", `${claim.id}: ${requirement} has no obligation ${obligation}`);
        }
    // Code → ledger: every id the code reports is a pending ledger row; a landed one must leave the code.
    for (const entry of codePendingIds(source)) {
        const capability = byId.get(entry.id);
        if (!capability) error({ file: entry.file, line: entry.line, rule: "ledger-code", message: `\`${entry.id}\` is reported here but is not a ledger capability; add it to ${LEDGER}` });
        else if (capability.status !== "pending") error({ file: entry.file, line: entry.line, rule: "ledger-code", message: `\`${entry.id}\` is in the release per the ledger, but the code still rejects it as pending` });
    }
    // Code tables → ledger rows, and release rows of those kinds → code tables.
    const surface = surfaceNames(source);
    if (surface.length) {
        const named = new Set(surface.map(entry => entry.id));
        for (const entry of surface) {
            const capability = byId.get(entry.id);
            if (!capability) error({ file: entry.file, line: 0, rule: "ledger-code", message: `${entry.kind} \`${entry.name}\` is authorable but has no ledger row \`${entry.id}\`` });
            else if (capability.status !== "release") error({ file: entry.file, line: 0, rule: "ledger-code", message: `${entry.kind} \`${entry.name}\` is authorable, but the ledger lists \`${entry.id}\` as pending` });
        }
        for (const capability of ledger.capabilities)
            if (capability.status === "release" && capability.id.startsWith(`${capability.kind}.`) && SURFACES.some(entry => entry.kind === capability.kind) && !named.has(capability.id))
                at("ledger-code", `${capability.id} is in the release per the ledger, but the compiler does not accept it`);
    }
    // A hand-written ANALYSIS_IDENTITIES must equal the ledger's identity rows, in order.
    const literal = literalIdentityRows(source);
    if (literal) {
        const expected = ledger.capabilities.filter(capability => capability.identity).map(capability => ({
            family: FAMILY_CONST[capability.identity.family], procedure: PROCEDURE_VARIANT[capability.identity.procedure],
            space: capability.identity.solution_space === "direct" ? "Direct" : "ModalReduced", pending: capability.status === "pending" ? capability.id : null,
        }));
        if (JSON.stringify(literal) !== JSON.stringify(expected)) at("ledger-code", "ANALYSIS_IDENTITIES in compatibility.rs differs from the ledger's identity rows");
    }
}

/** Areas with a Validation page, in the order the Validation section lists them. */
export const VALIDATION_AREAS = {
    structural: "Structural mechanics", thermal: "Heat transfer", electromagnetic: "Electromagnetics", acoustic: "Acoustics",
    fluid: "Fluid flow", thermomechanical: "Thermomechanical", electrothermal: "Electrothermal",
    "conjugate-heat-transfer": "Conjugate heat transfer", "fluid-structure-interaction": "Fluid–structure interaction",
};

/** Docs pages generate-pages.mjs writes from the ledger (paths under docs/simulation). */
export const GENERATED_PAGES = ["validation/status.md", ...Object.keys(VALIDATION_AREAS).map(area => `validation/${area}.md`)];
