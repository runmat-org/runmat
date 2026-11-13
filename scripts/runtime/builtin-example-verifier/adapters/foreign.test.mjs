import assert from "node:assert/strict";
import test from "node:test";

import {
    buildJavaCompileArguments,
    buildMexArguments,
    javaArtifactConfiguration,
    mexExtension,
    pythonExecutionConfiguration,
    pythonProbeIncompatibility
} from "./foreign.mjs";

test("MEX preparation derives one closed command from typed catalog fields", () => {
    assert.deepEqual(buildMexArguments({
        module_name: "fixture_mex",
        api: "R2017b",
        translation_units: [
            { relative_path: "gateway.c", language: "C" },
            { relative_path: "support.cpp", language: "Cxx" }
        ],
        include_directories: ["include"],
        definitions: [{ name: "FEATURE", value: "1" }]
    }), [
        "--color=never", "mex", "--R2017b", "--output", "fixture_mex", "--out-dir", ".",
        "-Iinclude", "-DFEATURE=1", "gateway.c", "support.cpp"
    ]);
});

test("MEX artifact extensions are derived from the exact native target", () => {
    assert.equal(mexExtension("darwin", "arm64"), "mexmaca64");
    assert.equal(mexExtension("darwin", "x64"), "mexmaci64");
    assert.equal(mexExtension("linux", "x64"), "mexa64");
    assert.equal(mexExtension("win32", "x64"), "mexw64");
    assert.throws(() => mexExtension("linux", "arm64"), /unavailable/u);
});

test("CUDA translation units select the dedicated MEX build command", () => {
    assert.equal(buildMexArguments({
        module_name: "fixture_gpu",
        api: "R2018a",
        translation_units: [{ relative_path: "gateway.cu", language: "Cuda" }],
        include_directories: [],
        definitions: []
    })[1], "mexcuda");
});

test("Java compilation and artifact configuration use only declared inputs", () => {
    assert.deepEqual(buildJavaCompileArguments("/work", "/work/.classes", {
        release: 17,
        compile_classpath: ["vendor/a.jar", "vendor/b.jar"],
        source_files: ["src/Fixture.java"]
    }), [
        "--release", "17", "-d", "/work/.classes",
        "-classpath", ["/work/vendor/a.jar", "/work/vendor/b.jar"].join(process.platform === "win32" ? ";" : ":"),
        "/work/src/Fixture.java"
    ]);
    assert.equal(javaArtifactConfiguration("fixture", ".runmat-example/java/fixture.jar"),
        '[java-artifacts."fixture"]\npath = ".runmat-example/java/fixture.jar"\n');
});

test("Python source-tree preparation preserves exact interpreter and isolation facts", () => {
    const [config, environment] = pythonExecutionConfiguration("/work", {
        artifact: { SourceTree: { module_root: "python", modules: ["fixture"] } }
    }, "OutOfProcess", { executable: "/usr/bin/python3", major: 3, minor: 12 }, "/existing");
    assert.equal(config,
        '[runtime.foreign.python]\nexecutable = "/usr/bin/python3"\nversion = "3.12"\nexecution_mode = "out_of_process"\n');
    assert.equal(environment.PYTHONPATH,
        ["/work/python", "/existing"].join(process.platform === "win32" ? ";" : ":"));
});

test("Python wheel preparation emits the declared immutable artifact identity", () => {
    const [config, environment] = pythonExecutionConfiguration("/work", {
        artifact: { Wheel: { artifact_name: "fixture", relative_path: "dist/fixture.whl", module: "fixture" } }
    }, "InProcess", { executable: "/python", major: 3, minor: 11 });
    assert.equal(config,
        '[runtime.foreign.python]\nexecutable = "/python"\nversion = "3.11"\nexecution_mode = "in_process"\n\n' +
        '[python-artifacts."fixture"]\npath = "dist/fixture.whl"\nmodule = "fixture"\n');
    assert.deepEqual(environment, {});
});

test("native Python wheels require the exact interpreter ABI and platform", () => {
    const nativeWheel = {
        environment: { implementation: "Cpython", major: 3, minor: 12 },
        artifact: { Wheel: {
            artifact_name: "fixture", relative_path: "fixture.whl", module: "fixture",
            compatibility: { Native: { abi_tag: "cpython-312-darwin", platform_tag: "macosx-15-arm64" } }
        } }
    };
    const observed = {
        executable: "/python", implementation: "cpython", major: 3, minor: 12,
        abi_tag: "cpython-312-darwin", platform_tag: "macosx-15-arm64"
    };
    assert.equal(pythonProbeIncompatibility(nativeWheel, observed), "");
    assert.match(pythonProbeIncompatibility(nativeWheel, { ...observed, abi_tag: "cpython-311-darwin" }), /does not match/u);
    assert.match(pythonProbeIncompatibility(nativeWheel, { ...observed, implementation: "pypy" }), /required CPython/u);
});
