// @ts-check

import { compareUtf8, digestObject, exampleIdentity, executionIdentity, legacyExampleId, sha256 } from "./identity.mjs";
import { exampleKey as legacyRunnerKey } from "./sharding.mjs";
import { isExecutionLane, requiredExecutionLanes } from "./lanes.mjs";
import { digest, enumValue, exactKeys, gitRevision, integer } from "./schema.mjs";
import {
    NO_EXAMPLE_FIXTURE,
    NO_EXAMPLE_REQUIREMENTS,
    validateBuiltinExampleFixture,
    validateBuiltinExampleRequirements
} from "../../metadata/BuiltinExampleFixtureSchema.mjs";

export const INVENTORY_SCHEMA = "runmat.builtin-examples.inventory.v2";

export function buildInventory(documentExport, options) {
    if (!documentExport || ![1, 2].includes(documentExport.schema_version) || !Array.isArray(documentExport.builtins)) {
        throw new Error("Unsupported builtin documentation export schema");
    }
    const scope = normalizeScope(options.scope);
    gitRevision(options.sourceRevision);
    enumValue(options.sourceState, ["clean", "dirty"], "inventory source state");
    digest(options.runnerDigest, "inventory runner digest");
    const records = [];
    const seen = new Set();
    let sourceExamples = 0;
    for (const document of documentExport.builtins) {
        const builtinKey = normalize(document?.key ?? document?.title, "builtin identity");
        const authority = typeof document?.authority === "string" ? document.authority : "unknown";
        const category = typeof document?.category === "string" ? document.category : "";
        const examples = Array.isArray(document?.examples) ? document.examples : [];
        for (let index = 0; index < examples.length; index += 1) {
            sourceExamples += 1;
            const raw = examples[index];
            const normalized = normalizeExample(raw, { builtinKey, authority, category, index, exportSchemaVersion: documentExport.schema_version });
            const exampleId = normalized.exampleId;
            const key = `${builtinKey}#${exampleId}`;
            if (seen.has(key)) throw new Error(`Duplicate builtin example identity: ${key}`);
            seen.add(key);
            if (!matchesScope(normalized, scope)) continue;
            records.push(normalized);
        }
    }
    records.sort((left, right) => compareUtf8(left.identity, right.identity));
    if (scope.limit !== null) records.splice(scope.limit);
    const documentationOnly = records.filter((example) => example.admission.kind === "documentation-only").length;
    const executionUnits = records
        .filter((example) => example.admission.kind === "executable")
        .flatMap((example) => example.requiredLanes.map((lane) => ({
            executionIdentity: executionIdentity(example.builtinKey, example.exampleId, lane),
            exampleIdentity: example.identity,
            definitionDigest: example.definitionDigest,
            builtinKey: example.builtinKey,
            exampleId: example.exampleId,
            runnerKey: example.runnerKey,
            lane
        })))
        .sort((left, right) => compareUtf8(left.executionIdentity, right.executionIdentity));
    const countsByLane = {};
    for (const unit of executionUnits) countsByLane[unit.lane] = (countsByLane[unit.lane] ?? 0) + 1;
    const inventory = {
        schema: INVENTORY_SCHEMA,
        sourceRevision: requiredString(options.sourceRevision, "source revision"),
        sourceState: options.sourceState,
        exportSchemaVersion: documentExport.schema_version,
        exportDigest: sha256(records),
        runnerDigest: requiredString(options.runnerDigest, "runner digest"),
        scope,
        counts: {
            documents: documentExport.builtins.length,
            sourceExamples,
            selectedExamples: records.length,
            executableExamples: records.length - documentationOnly,
            documentationOnly,
            executionUnitsByLane: Object.fromEntries(Object.entries(countsByLane).sort(([a], [b]) => compareUtf8(a, b)))
        },
        examples: records,
        executionUnits,
        inventoryDigest: ""
    };
    inventory.inventoryDigest = digestObject(inventory, ["inventoryDigest"]);
    return inventory;
}

export function validateInventory(inventory) {
    if (!inventory || inventory.schema !== INVENTORY_SCHEMA) throw new Error("Unsupported builtin example inventory schema");
    exactKeys(inventory, ["schema", "sourceRevision", "sourceState", "exportSchemaVersion", "exportDigest", "runnerDigest", "scope", "counts", "examples", "executionUnits", "inventoryDigest"], "inventory");
    gitRevision(inventory.sourceRevision);
    enumValue(inventory.sourceState, ["clean", "dirty"], "inventory source state");
    digest(inventory.exportDigest, "inventory export digest");
    digest(inventory.runnerDigest, "inventory runner digest");
    digest(inventory.inventoryDigest, "inventory digest");
    if (![1, 2].includes(inventory.exportSchemaVersion)) throw new Error("Inventory export schema version must be 1 or 2");
    exactKeys(inventory.scope, ["kind", "builtin", "filter", "limit"], "inventory scope");
    enumValue(inventory.scope.kind, ["complete", "development"], "inventory scope kind");
    for (const field of ["builtin", "filter"]) {
        const value = inventory.scope[field];
        if (value !== null && (typeof value !== "string" || !value || value !== value.trim().toLowerCase())) throw new Error(`Inventory scope ${field} must be null or a normalized nonempty string`);
    }
    if (inventory.scope.limit !== null) integer(inventory.scope.limit, "inventory scope limit", 1);
    const developmentScope = inventory.scope.builtin !== null || inventory.scope.filter !== null || inventory.scope.limit !== null;
    if ((inventory.scope.kind === "development") !== developmentScope) throw new Error("Inventory scope kind does not match its selectors");
    exactKeys(inventory.counts, ["documents", "sourceExamples", "selectedExamples", "executableExamples", "documentationOnly", "executionUnitsByLane"], "inventory counts");
    for (const [key, value] of Object.entries(inventory.counts).filter(([key]) => key !== "executionUnitsByLane")) integer(value, `inventory count ${key}`);
    if (!inventory.counts.executionUnitsByLane || typeof inventory.counts.executionUnitsByLane !== "object" || Array.isArray(inventory.counts.executionUnitsByLane)) throw new Error("Inventory lane counts must be an object");
    for (const [lane, count] of Object.entries(inventory.counts.executionUnitsByLane)) {
        if (!isExecutionLane(lane)) throw new Error(`Unknown inventory lane count: ${lane}`);
        integer(count, `inventory lane count ${lane}`, 1);
    }
    const orderedCountLanes = Object.keys(inventory.counts.executionUnitsByLane).sort(compareUtf8);
    if (JSON.stringify(orderedCountLanes) !== JSON.stringify(Object.keys(inventory.counts.executionUnitsByLane))) throw new Error("Inventory lane counts are not in canonical order");
    if (digestObject(inventory, ["inventoryDigest"]) !== inventory.inventoryDigest) throw new Error("Builtin example inventory digest mismatch");
    if (!Array.isArray(inventory.examples) || !Array.isArray(inventory.executionUnits)) throw new Error("Invalid builtin example inventory records");
    const examples = new Map(inventory.examples.map((example) => [example.identity, example]));
    if (examples.size !== inventory.examples.length) throw new Error("Duplicate example identity in inventory");
    const units = new Set();
    for (const example of inventory.examples) {
        exactKeys(example, ["identity", "runnerKey", "definitionDigest", "builtinKey", "exampleId", "authority", "category", "description", "program", "displayOutput", "compatibility", "harness", "fixture", "requirements", "verification", "isPlotExample", "admission", "requiredLanes"], `inventory example ${example.identity}`);
        digest(example.identity, "example identity");
        digest(example.definitionDigest, "example definition digest");
        enumValue(example.authority, ["catalog", "legacy_sidecar"], "example documentation authority");
        for (const field of ["runnerKey", "builtinKey", "exampleId", "category", "description", "program", "compatibility", "harness"]) {
            if (typeof example[field] !== "string") throw new Error(`Invalid example ${field}: ${example.identity}`);
        }
        if (!example.runnerKey || !example.builtinKey || !example.exampleId || !example.description || !example.program) throw new Error(`Inventory example has an empty required field: ${example.identity}`);
        if (example.builtinKey !== example.builtinKey.trim().toLowerCase() || example.exampleId !== example.exampleId.trim().toLowerCase()) throw new Error(`Inventory example identity fields are not normalized: ${example.identity}`);
        if (example.displayOutput !== null && typeof example.displayOutput !== "string") throw new Error(`Invalid example display output: ${example.identity}`);
        if (typeof example.isPlotExample !== "boolean") throw new Error(`Invalid plot marker: ${example.identity}`);
        enumValue(example.compatibility, ["RunMat", "Matlab", "Strict"], `example compatibility ${example.identity}`);
        validateBuiltinExampleRequirements(example.requirements, `example requirements ${example.identity}`);
        validateBuiltinExampleFixture(example.fixture, `example fixture ${example.identity}`, { program: example.program, harness: example.harness, requirements: example.requirements });
        validateVerification(example.verification, example.identity);
        if (exampleIdentity(example.builtinKey, example.exampleId) !== example.identity) throw new Error(`Invalid example identity: ${example.identity}`);
        const expectedDefinition = { program: example.program, displayOutput: example.displayOutput, compatibility: example.compatibility, harness: example.harness, fixture: example.fixture, requirements: example.requirements, verification: example.verification, category: example.category };
        if (sha256(["runmat.builtin-example.definition.v2", expectedDefinition]) !== example.definitionDigest) throw new Error(`Invalid example definition digest: ${example.identity}`);
        exactKeys(example.admission, example.admission.kind === "executable" ? ["kind"] : ["kind", "reason"], `example admission ${example.identity}`);
        enumValue(example.admission.kind, ["executable", "documentation-only"], "example admission kind");
        if (example.admission.kind === "documentation-only" && (typeof example.admission.reason !== "string" || !example.admission.reason.trim())) throw new Error(`Documentation-only example lacks a reason: ${example.identity}`);
        if (example.authority === "catalog" && example.admission.kind !== "executable") throw new Error(`Catalog example is not executable: ${example.builtinKey}#${example.exampleId}`);
        const expectedLanes = example.admission.kind === "executable" ? requiredExecutionLanes(example.harness) : [];
        if (JSON.stringify(example.requiredLanes) !== JSON.stringify(expectedLanes)) throw new Error(`Example lanes do not match its harness: ${example.identity}`);
    }
    for (const unit of inventory.executionUnits) {
        exactKeys(unit, ["executionIdentity", "exampleIdentity", "definitionDigest", "builtinKey", "exampleId", "runnerKey", "lane"], `inventory execution unit ${unit.executionIdentity}`);
        if (units.has(unit.executionIdentity)) throw new Error("Duplicate execution identity in inventory");
        units.add(unit.executionIdentity);
        const example = examples.get(unit.exampleIdentity);
        if (!example || example.definitionDigest !== unit.definitionDigest || !example.requiredLanes.includes(unit.lane)
            || example.builtinKey !== unit.builtinKey || example.exampleId !== unit.exampleId || example.runnerKey !== unit.runnerKey) {
            throw new Error(`Inconsistent execution unit: ${unit.executionIdentity}`);
        }
        if (executionIdentity(unit.builtinKey, unit.exampleId, unit.lane) !== unit.executionIdentity) {
            throw new Error(`Invalid execution identity: ${unit.executionIdentity}`);
        }
    }
    const expectedDocumentationOnly = inventory.examples.filter((example) => example.admission.kind === "documentation-only").length;
    const expectedExecutable = inventory.examples.length - expectedDocumentationOnly;
    if (inventory.counts.selectedExamples !== inventory.examples.length
        || inventory.counts.documentationOnly !== expectedDocumentationOnly
        || inventory.counts.executableExamples !== expectedExecutable) {
        throw new Error("Inventory example counts do not match records");
    }
    const expectedByLane = {};
    for (const unit of inventory.executionUnits) expectedByLane[unit.lane] = (expectedByLane[unit.lane] ?? 0) + 1;
    if (JSON.stringify(inventory.counts.executionUnitsByLane) !== JSON.stringify(Object.fromEntries(Object.entries(expectedByLane).sort(([a], [b]) => compareUtf8(a, b))))) {
        throw new Error("Inventory lane counts do not match execution units");
    }
    const expectedUnitIds = inventory.examples.flatMap((example) => example.requiredLanes.map((lane) => executionIdentity(example.builtinKey, example.exampleId, lane))).sort(compareUtf8);
    const actualUnitIds = inventory.executionUnits.map((unit) => unit.executionIdentity);
    if (JSON.stringify(expectedUnitIds) !== JSON.stringify(actualUnitIds)) throw new Error("Inventory does not contain every required execution lane exactly once");
    const orderedExamples = [...inventory.examples].sort((left, right) => compareUtf8(left.identity, right.identity));
    if (JSON.stringify(orderedExamples) !== JSON.stringify(inventory.examples)) throw new Error("Inventory examples are not in canonical order");
    if (inventory.counts.sourceExamples < inventory.counts.selectedExamples) throw new Error("Inventory source example count is smaller than its selected count");
    if (sha256(inventory.examples) !== inventory.exportDigest) throw new Error("Inventory export digest does not match normalized examples");
    return inventory;
}

function validateVerification(value, exampleIdentityValue) {
    if (value === null) return;
    if (value === "Succeeds") return;
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`Invalid example verification: ${exampleIdentityValue}`);
    const variants = Object.keys(value);
    if (variants.length !== 1 || !["Assertions", "ExpectedError", "Figure"].includes(variants[0])) throw new Error(`Unknown example verification: ${exampleIdentityValue}`);
    const variant = variants[0];
    if (variant === "Assertions") {
        exactKeys(value.Assertions, ["source"], `assertion verification ${exampleIdentityValue}`);
        if (typeof value.Assertions.source !== "string" || !value.Assertions.source.trim()) throw new Error(`Empty assertion verification: ${exampleIdentityValue}`);
    } else if (variant === "ExpectedError") {
        exactKeys(value.ExpectedError, ["identifier"], `expected-error verification ${exampleIdentityValue}`);
        if (typeof value.ExpectedError.identifier !== "string" || !value.ExpectedError.identifier.trim()) throw new Error(`Empty expected-error identifier: ${exampleIdentityValue}`);
    } else {
        exactKeys(value.Figure, ["minimum_figures", "assertions"], `figure verification ${exampleIdentityValue}`);
        integer(value.Figure.minimum_figures, `figure minimum for ${exampleIdentityValue}`, 1);
        if (typeof value.Figure.assertions !== "string") throw new Error(`Invalid figure assertions: ${exampleIdentityValue}`);
    }
}

function normalizeExample(raw, context) {
    const object = raw && typeof raw === "object" && !Array.isArray(raw) ? raw : {};
    const program = typeof object.input === "string" ? object.input.trim() : "";
    const compatibility = typeof object.compatibility === "string" ? object.compatibility : "RunMat";
    const harness = typeof object.harness === "string" ? object.harness : "LegacyBrowser";
    const verification = object.verification ?? null;
    const displayOutput = typeof object.output === "string" ? object.output : null;
    if (context.exportSchemaVersion === 2 && (!("fixture" in object) || !("requirements" in object))) {
        throw new Error(`${context.builtinKey} schema-v2 example ${context.index + 1} lacks fixture requirements`);
    }
    const fixture = object.fixture ?? NO_EXAMPLE_FIXTURE;
    const requirements = object.requirements ?? {
        ...NO_EXAMPLE_REQUIREMENTS,
        compiler: [],
        runtime: [],
        toolchain: []
    };
    validateBuiltinExampleRequirements(requirements, `${context.builtinKey} example requirements`);
    validateBuiltinExampleFixture(fixture, `${context.builtinKey} example fixture`, { program, harness, requirements });
    const definition = {
        program,
        displayOutput,
        compatibility,
        harness,
        fixture,
        requirements,
        verification,
        category: context.category
    };
    let exampleId = typeof object.id === "string" ? object.id.trim().toLowerCase() : "";
    if (context.authority === "catalog" && !exampleId) {
        throw new Error(`${context.builtinKey} catalog example ${context.index + 1} has no stable id`);
    }
    if (context.authority === "catalog" && verification === null) {
        throw new Error(`${context.builtinKey} catalog example ${exampleId || context.index + 1} has no verification policy`);
    }
    if (!exampleId) exampleId = legacyExampleId(definition);
    const description = typeof object.description === "string" && object.description.trim()
        ? object.description.trim()
        : `${context.builtinKey} example ${context.index + 1}`;
    const plot = context.category === "plotting" || context.category.startsWith("plotting/");
    const reason = admissionReason(program, displayOutput, verification, plot);
    if (context.authority === "catalog" && reason) {
        throw new Error(`${context.builtinKey} catalog example ${exampleId} is not executable: ${reason}`);
    }
    const requiredLanes = reason ? [] : requiredExecutionLanes(harness);
    const executableProgram = appendAssertions(program, verification);
    const definitionRecord = { ...definition, program: executableProgram };
    return {
        identity: exampleIdentity(context.builtinKey, exampleId),
        runnerKey: legacyRunnerKey(context.builtinKey, object, context.index),
        definitionDigest: sha256(["runmat.builtin-example.definition.v2", definitionRecord]),
        builtinKey: context.builtinKey,
        exampleId,
        authority: context.authority,
        category: context.category,
        description,
        program: executableProgram,
        displayOutput,
        compatibility,
        harness,
        fixture,
        requirements,
        verification,
        isPlotExample: plot,
        admission: reason ? { kind: "documentation-only", reason } : { kind: "executable" },
        requiredLanes
    };
}

function admissionReason(program, output, verification, plot) {
    if (!program) return "missing executable source";
    if (verification !== null || plot) return "";
    if (typeof output !== "string") return "missing verification oracle";
    const meaningful = output.split(/\r?\n/).some((line) => line.trim() && !line.trim().startsWith("%"));
    return meaningful ? "" : "comment-only presentation output has no verification oracle";
}

function appendAssertions(program, verification) {
    const assertions = verification && typeof verification === "object" && verification.Assertions;
    return assertions && typeof assertions.source === "string"
        ? `${program.trimEnd()}\n${assertions.source}`
        : program;
}

function normalizeScope(scope = {}) {
    const builtin = typeof scope.builtin === "string" && scope.builtin.trim() ? scope.builtin.trim().toLowerCase() : null;
    const filter = typeof scope.filter === "string" && scope.filter.trim() ? scope.filter.trim().toLowerCase() : null;
    if (builtin && filter) throw new Error("Builtin and text filters cannot be combined");
    const limit = scope.limit === undefined || scope.limit === null ? null : Number(scope.limit);
    if (limit !== null && (!Number.isSafeInteger(limit) || limit < 1)) throw new Error("Inventory limit must be a positive integer");
    return { kind: builtin || filter || limit !== null ? "development" : "complete", builtin, filter, limit };
}

function matchesScope(example, scope) {
    if (scope.builtin && example.builtinKey !== scope.builtin) return false;
    if (!scope.filter) return true;
    return [example.builtinKey, example.description, example.program, example.category]
        .join("\n").toLowerCase().includes(scope.filter);
}

function normalize(value, label) {
    const normalized = String(value ?? "").trim().toLowerCase();
    if (!normalized) throw new Error(`Missing ${label}`);
    return normalized;
}

function requiredString(value, label) {
    if (typeof value !== "string" || !value.trim()) throw new Error(`Missing ${label}`);
    return value.trim();
}
