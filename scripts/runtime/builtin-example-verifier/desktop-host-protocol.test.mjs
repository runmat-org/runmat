import assert from "node:assert/strict";
import test from "node:test";

import {
    buildDesktopHostRequest,
    DESKTOP_HOST_RESULT_SCHEMA,
    validateDesktopHostRequest,
    validateDesktopHostResult
} from "./desktop-host-protocol.mjs";

function fixture() {
    return {
        DesktopHostOnly: {
            id: { local_name: "dialog-example" },
            entries: [],
            interactions: [{
                LineInput: {
                    prompt: "Name: ",
                    echo: true,
                    outcome: { Line: "Ada" }
                }
            }]
        }
    };
}

test("Desktop host request and result protocols are closed and case-bound", () => {
    const request = buildDesktopHostRequest({
        exampleKey: "input#one",
        program: "name = input('Name: ', 's');",
        compatibility: "RunMat",
        workspaceRoot: "/isolated/workspace",
        fixture: fixture()
    });
    assert.equal(validateDesktopHostRequest(request), request);
    assert.throws(
        () => validateDesktopHostRequest({ ...request, surprise: true }),
        /invalid fields|unknown/u
    );

    const result = {
        schema: DESKTOP_HOST_RESULT_SCHEMA,
        exampleKey: "input#one",
        stdoutText: "",
        valueText: "",
        errorText: "",
        errorIdentifier: "",
        interactions: {
            expectedCount: 1,
            observedCount: 1,
            expectedFigureCount: 0,
            observedFigureCount: 0,
            matched: true,
            figurePresentationsMatched: true
        }
    };
    assert.equal(validateDesktopHostResult(result, request), result);
    assert.throws(() => validateDesktopHostResult(result, { ...request, exampleKey: "input#two" }), /different example/u);
    assert.throws(
        () => validateDesktopHostResult({
            ...result,
            interactions: { ...result.interactions, observedCount: 0 }
        }, request),
        /counts disagree/u
    );
    assert.throws(
        () => validateDesktopHostResult({
            ...result,
            interactions: { ...result.interactions, expectedCount: 0, observedCount: 0 }
        }, request),
        /expected count does not match/u
    );
    assert.throws(
        () => validateDesktopHostRequest({ ...request, workspaceRoot: "relative/workspace" }),
        /must be absolute/u
    );
});
