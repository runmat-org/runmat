import assert from "node:assert/strict";
import test from "node:test";

import {
  NO_EXAMPLE_REQUIREMENTS,
  validateBuiltinExampleFixture,
  validateBuiltinExampleRequirements
} from "./BuiltinExampleFixtureSchema.mjs";

const nativeRequirements = {
  host: "NativeOnly",
  engine: "Default",
  compiler: [],
  runtime: [],
  toolchain: []
};

function loopbackFixture() {
  return {
    Loopback: {
      id: { local_name: "http-value" },
      scenario: { Http: { exchanges: [{
        request: { method: "Get", path: "/value", body: null },
        response: { status: 200, headers: [{ name: "content-type", value: "text/plain" }], body: [111, 107] }
      }] } },
      endpoint_substitutions: ["HttpBaseUrl"]
    }
  };
}

test("accepts a closed loopback fixture with a declared fixed token", () => {
  assert.equal(validateBuiltinExampleRequirements(NO_EXAMPLE_REQUIREMENTS), NO_EXAMPLE_REQUIREMENTS);
  const fixture = loopbackFixture();
  assert.equal(validateBuiltinExampleFixture(fixture, "fixture", {
    program: 'url = "__RUNMAT_HTTP_BASE_URL__/value";',
    harness: "NativeLoopbackNetwork",
    requirements: nativeRequirements
  }), fixture);
});

test("rejects unsafe filesystem paths and file descendants", () => {
  const base = { id: { local_name: "files" }, root: "IsolatedWorkspace" };
  assert.throws(() => validateBuiltinExampleFixture({ Filesystem: { ...base, entries: [{ Directory: { relative_path: "../escape" } }] } }), /relative path/);
  assert.throws(() => validateBuiltinExampleFixture({ Filesystem: { ...base, entries: [{ Directory: { relative_path: "CON" } }] } }), /relative path/);
  assert.throws(() => validateBuiltinExampleFixture({ Filesystem: { ...base, entries: [
    { File: { relative_path: "a", content: { Utf8: "file" } } },
    { File: { relative_path: "a/b", content: { Utf8: "child" } } }
  ] } }), /below a file/);
});

test("rejects duplicate capabilities and endpoint declaration mismatches", () => {
  assert.throws(() => validateBuiltinExampleRequirements({ ...nativeRequirements, runtime: ["Python", "Python"] }), /sorted and unique/);
  assert.throws(() => validateBuiltinExampleFixture(loopbackFixture(), "fixture", {
    program: 'host = "__RUNMAT_LOOPBACK_HOST__";',
    harness: "NativeLoopbackNetwork",
    requirements: nativeRequirements
  }), /absent|undeclared/);
});

test("rejects incompatible harnesses and unterminated CLI transcripts", () => {
  assert.throws(() => validateBuiltinExampleFixture(loopbackFixture(), "fixture", {
    program: 'url = "__RUNMAT_HTTP_BASE_URL__";',
    harness: "Portable",
    requirements: nativeRequirements
  }), /harness/);
  assert.throws(() => validateBuiltinExampleFixture({ CliInteraction: {
    id: { local_name: "prompt" },
    transcript: [{ SendLine: "4" }]
  } }, "fixture", {
    program: "value = input('value: ');",
    harness: "InteractiveHost",
    requirements: nativeRequirements
  }), /terminal action/);
});

test("rejects foreign fixtures without adapter capabilities", () => {
  const fixture = { ForeignAdapter: {
    id: { local_name: "native" },
    files: {
      id: { local_name: "files" }, root: "IsolatedWorkspace",
      entries: [{ File: { relative_path: "entry.f90", content: { Utf8: "end" } } }]
    },
    preparation: { Mex: {
      module_name: "native",
      api: "R2017b",
      translation_units: [{ relative_path: "entry.f90", language: "Fortran" }],
      include_directories: [],
      definitions: []
    } }
  } };
  assert.throws(() => validateBuiltinExampleFixture(fixture, "fixture", {
    program: "value = native();",
    harness: "NativeForeignRuntime",
    requirements: nativeRequirements
  }), /adapter runtime/);
});

test("rejects isolation at the foreign-fixture level", () => {
  const fixture = { ForeignAdapter: {
    id: { local_name: "java" },
    isolation: "OutOfProcess",
    files: {
      id: { local_name: "files" }, root: "IsolatedWorkspace",
      entries: [{ File: { relative_path: "Fixture.java", content: { Utf8: "class Fixture {}" } } }]
    },
    preparation: { Java: {
      artifact_name: "fixture",
      release: 17,
      source_files: ["Fixture.java"],
      resources: [],
      compile_classpath: []
    } }
  } };
  assert.throws(() => validateBuiltinExampleFixture(fixture, "fixture", {
    program: "java.lang.String('value');",
    harness: "NativeForeignRuntime",
    requirements: {
      host: "NativeOnly", engine: "Default", compiler: ["JavaBytecode"],
      runtime: ["JavaVirtualMachine"], toolchain: ["JavaDevelopmentKit"]
    }
  }), /fields/u);
});
