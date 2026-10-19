import assert from "node:assert/strict";
import test from "node:test";

import { nativeAccelerationConfig, nativeEngineArguments, parseNativeArtifactManifest } from "./native.mjs";

test("WGPU examples require the declared backend without compatibility fallback", () => {
    const config = nativeAccelerationConfig("Wgpu");
    assert.match(config, /\[runtime\.accelerate\]\nenabled = true\nprovider = "wgpu"/u);
    assert.match(config, /allow_inprocess_fallback = false/u);
    assert.match(config, /wgpu_force_fallback_adapter = false/u);
    assert.match(config, /\[runtime\.accelerate\.auto_offload\]\nenabled = false/u);
});

test("host examples keep acceleration disabled independently of their engine", () => {
    for (const harness of ["Portable", "NativeFilesystem", "NativeInterop"]) {
        const config = nativeAccelerationConfig(harness);
        assert.match(config, /\[runtime\.accelerate\]\nenabled = false/u);
        assert.match(config, /allow_inprocess_fallback = true/u);
    }
});

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
