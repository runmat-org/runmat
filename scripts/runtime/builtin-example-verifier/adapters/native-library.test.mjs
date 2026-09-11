import assert from "node:assert/strict";
import test from "node:test";

import {
    buildNativeInterfaceArguments,
    nativeInterfaceConfiguration,
    nativeLibraryBuildPlan,
    nativeLibraryTarget
} from "./native-library.mjs";

const preparation = {
    library_name: "fixture",
    translation_units: [
        { relative_path: "src/fixture.c", language: "C" },
        { relative_path: "src/support.cpp", language: "Cxx" }
    ],
    include_directories: ["include"],
    definitions: [{ name: "BUILDING_FIXTURE", value: null }],
    interface: {
        interface_name: "fixture_api",
        primary_header: "include/fixture.h",
        additional_headers: ["include/types.h"],
        include_directories: ["include", "vendor/include"],
        definitions: [{ name: "PUBLIC_LEVEL", value: "2" }]
    }
};

test("native interface preparation argv is derived only from typed fields", () => {
    assert.deepEqual(buildNativeInterfaceArguments(
        preparation,
        ".runmat-example/native/libfixture.so",
        ".runmat-example/native/libfixture.so.runmat.json"
    ), [
        "--color=never", "native-interface", "prepare",
        "--library", ".runmat-example/native/libfixture.so",
        "--library-name", "fixture",
        "--header", "include/fixture.h",
        "--interface-name", "fixture_api",
        "--frontend", "clang",
        "--output", ".runmat-example/native/libfixture.so.runmat.json",
        "--add-header", "include/types.h",
        "-I", "include", "-I", "vendor/include",
        "-D", "PUBLIC_LEVEL=2"
    ]);
});

test("native interface configuration preserves the declared isolation policy", () => {
    assert.equal(nativeInterfaceConfiguration("fixture_api", "native/libfixture.so", "native/libfixture.json", "OutOfProcess"),
        '[runtime.foreign.native]\nisolation = "process"\n\n' +
        '[native-interfaces."fixture_api"]\nmanifest = "native/libfixture.json"\nlibrary = "native/libfixture.so"\n');
    assert.match(nativeInterfaceConfiguration("fixture_api", "native/libfixture.so", "native/libfixture.json", "InProcess"),
        /isolation = "in_process"/u);
});

test("GNU-like native library plans compile declared units and link with the owning driver", () => {
    const target = nativeLibraryTarget("linux", "x64", "fixture");
    const plan = nativeLibraryBuildPlan(preparation, target, { CC: "clang", CXX: "clang++" });
    assert.equal(plan.available, true);
    assert.deepEqual(plan.commands, [
        {
            label: "C compilation", executable: "clang",
            arguments: ["-x", "c", "-fPIC", "-c", "src/fixture.c", "-o", ".runmat-example/native/unit-0000.o", "-I", "include", "-DBUILDING_FIXTURE"]
        },
        {
            label: "Cxx compilation", executable: "clang++",
            arguments: ["-x", "c++", "-fPIC", "-c", "src/support.cpp", "-o", ".runmat-example/native/unit-0001.o", "-I", "include", "-DBUILDING_FIXTURE"]
        },
        {
            label: "native library link", executable: "clang++",
            arguments: ["-shared", ".runmat-example/native/unit-0000.o", ".runmat-example/native/unit-0001.o", "-o", ".runmat-example/native/libfixture.so"]
        }
    ]);
});

test("native library target names are exact and unsupported targets stay unavailable", () => {
    assert.equal(nativeLibraryTarget("darwin", "arm64", "fixture").relativePath, ".runmat-example/native/libfixture.dylib");
    assert.equal(nativeLibraryTarget("linux", "x64", "fixture").relativePath, ".runmat-example/native/libfixture.so");
    assert.equal(nativeLibraryTarget("win32", "x64", "fixture").relativePath, ".runmat-example/native/fixture.dll");
    assert.equal(nativeLibraryTarget("linux", "arm64", "fixture").available, false);
});

test("Windows rejects native source languages without a declared supported toolchain path", () => {
    const target = nativeLibraryTarget("win32", "x64", "fixture");
    const plan = nativeLibraryBuildPlan({ ...preparation, translation_units: [{ relative_path: "fixture.f90", language: "Fortran" }] }, target, {});
    assert.equal(plan.available, false);
    assert.match(plan.reason, /C or C\+\+/u);
});
