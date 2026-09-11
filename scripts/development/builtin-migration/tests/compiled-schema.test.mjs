import assert from "node:assert/strict";
import test from "node:test";

import { legacyFunction } from "../compiled-schema.mjs";

test("compiled type schema accepts each closed unit and structured variant", () => {
  const value = legacyFunctionFixture([
    "Int", "Num", "Bool", "String", "Symbolic", "Void", "Unknown",
    { Logical: { shape: [2, null] } },
    { Tensor: { shape: null } },
    { SymbolicArray: { shape: [] } },
    { Object: { class_name: "table", shape: [null, 3] } },
    { Cell: { element_type: { Struct: { known_fields: ["a", "b"] } }, length: 2 } },
    { Function: { params: ["Num"], returns: { OutputList: ["Num", "Bool"] } } },
    { Union: ["String", { Cell: { element_type: null, length: null } }] },
  ]);
  assert.doesNotThrow(() => legacyFunction(value));
});

test("compiled type schema rejects unknown variants and nested unknown fields", () => {
  const unknown = legacyFunctionFixture([{ Guessed: {} }]);
  assert.throws(() => legacyFunction(unknown), /unsupported type variant Guessed/);

  const widened = legacyFunctionFixture([{ Tensor: { shape: [2], nearby_shape: [2] } }]);
  assert.throws(() => legacyFunction(widened), /Tensor type fields must be exactly/);

  const ambiguous = legacyFunctionFixture([{ Tensor: { shape: [2] }, Logical: { shape: [2] } }]);
  assert.throws(() => legacyFunction(ambiguous), /exactly one variant/);
});

function legacyFunctionFixture(parameterTypes) {
  return {
    name: "fixture", description: "Fixture", category: "test", parameter_types: parameterTypes,
    return_type: "Unknown", resolver: "none", semantic_authority: "derived",
    semantics: {
      compatibility: "Matlab", async_behavior: "NeverSuspends",
      effects: { workspace: false, environment: false, filesystem: false, network: false, ui: false, random: false, time: false, host_callback: false, unknown: false },
      workspace_effect: null, environment_effect: null, purity: "Pure", semantic_kind: "General",
    },
    accelerator_tags: [], is_sink: false, suppress_auto_output: false, execution_stack: "Any",
    required_capabilities: [], descriptor: null, extensions: [], integer_capabilities: [], integer_audit: null,
  };
}
