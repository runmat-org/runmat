import assert from "node:assert/strict";
import test from "node:test";

import { nativeEngineArguments, parseNativeArtifactManifest } from "./native.mjs";

test("native execution requirements select the actual engine", () => {
    assert.deepEqual(nativeEngineArguments("Default"), []);
    assert.deepEqual(nativeEngineArguments("Interpreter"), ["--no-jit"]);
    assert.deepEqual(nativeEngineArguments("Jit"), []);
    assert.throws(() => nativeEngineArguments("Aot"), /native compile adapter/u);
    assert.throws(() => nativeEngineArguments("unknown"), /unsupported native execution engine/u);
});

test("native artifact manifests carry observed JIT execution", () => {
    assert.deepEqual(parseNativeArtifactManifest(JSON.stringify({
        schema_version: "runmat.artifacts.v1",
        success: true,
        used_jit: true,
        error_identifier: null
    })), { success: true, usedJit: true, errorIdentifier: "" });
    assert.throws(() => parseNativeArtifactManifest(JSON.stringify({
        schema_version: "runmat.artifacts.v1",
        success: true,
        error_identifier: null
    })), /used_jit/u);
});
