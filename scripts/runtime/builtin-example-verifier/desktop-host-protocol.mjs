// @ts-check

import { validateBuiltinExampleFixture } from "../../metadata/BuiltinExampleFixtureSchema.mjs";
import { exactKeys, integer } from "./schema.mjs";
import { isAbsolute } from "node:path";

export const DESKTOP_HOST_MODE_ARGUMENT = "--__runmat-builtin-example-host";
export const DESKTOP_HOST_REQUEST_ENV = "RUNMAT_BUILTIN_EXAMPLE_HOST_REQUEST";
export const DESKTOP_HOST_RESULT_ENV = "RUNMAT_BUILTIN_EXAMPLE_HOST_RESULT";
export const DESKTOP_HOST_REQUEST_SCHEMA = "runmat.builtin-examples.desktop-host-request.v1";
export const DESKTOP_HOST_RESULT_SCHEMA = "runmat.builtin-examples.desktop-host-result.v1";
const MAX_TEXT_BYTES = 64 * 1024;

export function buildDesktopHostRequest(fields) {
    const request = {
        schema: DESKTOP_HOST_REQUEST_SCHEMA,
        exampleKey: requiredText(fields.exampleKey, "desktop request example key"),
        program: requiredText(fields.program, "desktop request program"),
        compatibility: fields.compatibility,
        workspaceRoot: requiredText(fields.workspaceRoot, "desktop request workspace root"),
        fixture: structuredClone(fields.fixture)
    };
    validateDesktopHostRequest(request);
    return request;
}

export function validateDesktopHostRequest(request) {
    if (!request || request.schema !== DESKTOP_HOST_REQUEST_SCHEMA) throw new Error("Unsupported Desktop host request schema");
    exactKeys(request, ["schema", "exampleKey", "program", "compatibility", "workspaceRoot", "fixture"], "Desktop host request");
    requiredText(request.exampleKey, "desktop request example key");
    requiredText(request.program, "desktop request program");
    requiredText(request.workspaceRoot, "desktop request workspace root");
    for (const [value, label] of [[request.exampleKey, "example key"], [request.program, "program"], [request.workspaceRoot, "workspace root"]]) {
        if (new TextEncoder().encode(value).length > MAX_TEXT_BYTES) throw new Error(`Desktop request ${label} exceeds its byte limit`);
    }
    if (!isAbsolute(request.workspaceRoot)) throw new Error("Desktop request workspace root must be absolute");
    if (!["RunMat", "Matlab", "Strict"].includes(request.compatibility)) throw new Error("Desktop request compatibility is invalid");
    if (!request.fixture || typeof request.fixture !== "object" || !("DesktopHostOnly" in request.fixture)) {
        throw new Error("Desktop request requires a DesktopHostOnly fixture");
    }
    validateBuiltinExampleFixture(request.fixture, "Desktop request fixture", {
        program: request.program,
        harness: "InteractiveHost",
        requirements: {
            host: "DesktopHostOnly",
            engine: "Default",
            compiler: [],
            runtime: [],
            toolchain: []
        }
    });
    return request;
}

export function validateDesktopHostResult(result, request) {
    if (!result || result.schema !== DESKTOP_HOST_RESULT_SCHEMA) throw new Error("Unsupported Desktop host result schema");
    exactKeys(result, ["schema", "exampleKey", "stdoutText", "valueText", "errorText", "errorIdentifier", "interactions"], "Desktop host result");
    requiredText(result.exampleKey, "desktop result example key");
    if (result.exampleKey !== request.exampleKey) throw new Error("Desktop host result belongs to a different example");
    for (const field of ["stdoutText", "valueText", "errorText", "errorIdentifier"]) {
        if (typeof result[field] !== "string" || result[field].includes("\0")) throw new Error(`Desktop host result ${field} is invalid`);
    }
    exactKeys(result.interactions, ["expectedCount", "observedCount", "expectedFigureCount", "observedFigureCount", "matched", "figurePresentationsMatched"], "Desktop host interaction result");
    integer(result.interactions.expectedCount, "Desktop host expected interaction count");
    integer(result.interactions.observedCount, "Desktop host observed interaction count");
    integer(result.interactions.expectedFigureCount, "Desktop host expected figure count");
    integer(result.interactions.observedFigureCount, "Desktop host observed figure count");
    if (typeof result.interactions.matched !== "boolean" || typeof result.interactions.figurePresentationsMatched !== "boolean") {
        throw new Error("Desktop host interaction result flags must be booleans");
    }
    const expectedInteractions = request.fixture.DesktopHostOnly.interactions;
    const expectedFigureCount = expectedInteractions.filter((interaction) => "FigurePresentation" in interaction).length;
    if (result.interactions.expectedCount !== expectedInteractions.length) {
        throw new Error("Desktop host result expected count does not match its request");
    }
    if (result.interactions.expectedFigureCount !== expectedFigureCount) {
        throw new Error("Desktop host result expected figure count does not match its request");
    }
    if (result.interactions.matched && result.interactions.expectedCount !== result.interactions.observedCount) {
        throw new Error("Desktop host interaction counts disagree with a matched result");
    }
    if (result.interactions.figurePresentationsMatched && result.interactions.expectedFigureCount !== result.interactions.observedFigureCount) {
        throw new Error("Desktop host figure counts disagree with a matched result");
    }
    return result;
}

function requiredText(value, label) {
    if (typeof value !== "string" || !value.trim() || value.includes("\0")) throw new Error(`${label} must be nonempty text`);
    return value;
}
