import assert from "node:assert/strict";
import test from "node:test";
import { validateBuiltinExampleFixture } from "./BuiltinExampleFixtureSchema.mjs";

function fixture(request) {
  return { DesktopHostOnly: {
    id: { local_name: "editor" },
    entries: [{ File: { relative_path: "editor.m", content: { Utf8: "x=1;\n" } } }],
    interactions: [{ Document: request }]
  } };
}

test("validates typed document requests and positive positions", () => {
  for (const request of [
    { OpenFile: { relative_path: "editor.m" } },
    { OpenVariable: { name: "results" } },
    { Navigate: { relative_path: "editor.m", line: 12, column: 4, selection: "Select" } }
  ]) assert.doesNotThrow(() => validateBuiltinExampleFixture(fixture(request)));

  for (const relative_path of ["../editor.m", "/editor.m", "missing.m"]) {
    assert.throws(() => validateBuiltinExampleFixture(fixture({ OpenFile: { relative_path } })));
  }
  for (const name of ["", "a.b", "../x", "a\0b"]) {
    assert.throws(() => validateBuiltinExampleFixture(fixture({ OpenVariable: { name } })));
  }
  for (const line of [0, -1, 0.5, Number.MAX_SAFE_INTEGER + 1]) {
    assert.throws(() => validateBuiltinExampleFixture(fixture({ Navigate: {
      relative_path: "editor.m", line, column: 1, selection: "Caret"
    } })));
  }
  assert.throws(() => validateBuiltinExampleFixture(fixture({ Navigate: {
    relative_path: "editor.m", line: 1, column: 1, selection: "anything"
  } })));
});
