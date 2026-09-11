import assert from "node:assert/strict";
import test from "node:test";

import { mergeLaneResults, usesBrowserLane, usesNativeLane } from "./lanes.mjs";
import { parseNativeArtifactManifest } from "./native.mjs";

const result = (id, errorText = "", errorIdentifier = "") => ({
    id,
    stdoutText: "",
    valueText: "",
    errorText,
    errorIdentifier
});

test("harnesses select only their executable lanes", () => {
    assert.equal(usesBrowserLane("Portable"), true);
    assert.equal(usesNativeLane("Portable"), true);
    assert.equal(usesBrowserLane("Wgpu"), true);
    assert.equal(usesNativeLane("Wgpu"), false);
    assert.equal(usesBrowserLane("NativeFilesystem"), false);
    assert.equal(usesNativeLane("NativeFilesystem"), true);
    assert.equal(usesNativeLane("NativeLoopbackNetwork"), true);
    assert.equal(usesBrowserLane("InteractiveHost"), false);
    assert.equal(usesNativeLane("InteractiveHost"), true);
});

test("portable success requires both lanes", () => {
    const cases = [{ id: 1, harness: "Portable", verification: { Assertions: { source: "assert(true)" } } }];
    assert.equal(mergeLaneResults(cases, [result(1)], [result(1)], [])[0].errorText, "");
    assert.match(
        mergeLaneResults(cases, [result(1)], [result(1, "native failed")], [])[0].errorText,
        /portable example failed/
    );
});

test("portable expected errors must agree by stable identifier", () => {
    const cases = [{
        id: 2,
        harness: "Portable",
        verification: { ExpectedError: { identifier: "RunMat:test:Expected" } }
    }];
    const expected = result(2, "expected", "RunMat:test:Expected");
    assert.equal(mergeLaneResults(cases, [expected], [expected], [])[0].errorIdentifier, "RunMat:test:Expected");
    assert.match(
        mergeLaneResults(cases, [expected], [result(2, "other", "RunMat:test:Other")], [])[0].errorText,
        /error identifiers differ/
    );
});

test("an unsupported harness produces an explicit failing result", () => {
    const testCase = { id: 3, harness: "Unimplemented", verification: "Succeeds" };
    assert.match(mergeLaneResults([testCase], [], [], [testCase])[0].errorText, /no verifier adapter/);
});

test("native results use the structured run artifact contract", () => {
    assert.deepEqual(
        parseNativeArtifactManifest(JSON.stringify({
            schema_version: "runmat.artifacts.v1",
            success: false,
            used_jit: false,
            error_identifier: "RunMat:test:Expected"
        })),
        { success: false, errorIdentifier: "RunMat:test:Expected", usedJit: false }
    );
    assert.throws(
        () => parseNativeArtifactManifest(JSON.stringify({
            schema_version: "runmat.artifacts.v0",
            success: true,
            used_jit: false,
            error_identifier: null
        })),
        /unsupported or missing/
    );
    assert.throws(
        () => parseNativeArtifactManifest(JSON.stringify({
            schema_version: "runmat.artifacts.v1",
            success: "yes",
            used_jit: false,
            error_identifier: null
        })),
        /success must be a boolean/
    );
});
