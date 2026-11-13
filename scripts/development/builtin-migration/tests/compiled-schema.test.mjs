import assert from "node:assert/strict";
import test from "node:test";

import { catalogEntry, constant, legacyFunction } from "../compiled-schema.mjs";
import { compiledInventoryFixture } from "./helpers.mjs";
import { parseCompiledInventory } from "../compiled-inventory.mjs";
import { contentDigest } from "../evidence.mjs";

test("compiled inventory accepts the actual declared feature set without widening to unknown features", () => {
  const fixture = compiledInventoryFixture();
  assert.ok(fixture.snapshot.build.crate_feature_inventory.known_features.includes("plot-core"));
  assert.doesNotThrow(() => parseCompiledInventory(fixture));
  fixture.snapshot.build.crate_feature_inventory.known_features.push("guessed-feature");
  fixture.digest.value = contentDigest(Buffer.from(JSON.stringify(fixture.snapshot))).slice("sha256:".length);
  assert.throws(() => parseCompiledInventory(fixture), /known_features must exactly match/);
});

test("compiled catalog parses typed audience and operand preparation forms", () => {
  const entry = compiledInventoryFixture().snapshot.declared.catalog_entries[0];
  assert.doesNotThrow(() => catalogEntry(entry));
  entry.placement.preparation = { Forms: [{
    id: 1, binding_variant: "default", signatures: [0], data_arguments: [0],
    operation: { FloatingCast: "Double" }, provider_requirement: "F64",
  }] };
  assert.doesNotThrow(() => catalogEntry(entry));
  entry.placement.preparation.Forms[0].operation = { ElementwiseNary: { data_arity: 2 } };
  assert.doesNotThrow(() => catalogEntry(entry));
  entry.placement.preparation.Forms[0].operation = { RealUnary: "closed_unit_interval" };
  assert.doesNotThrow(() => catalogEntry(entry));
  entry.placement.preparation.Forms[0].operation = { FloatingRemainder: "modulus" };
  assert.doesNotThrow(() => catalogEntry(entry));
  entry.placement.preparation.Forms[0].operation = { Sink: "Gather" };
  assert.doesNotThrow(() => catalogEntry(entry));
  entry.placement.preparation.Forms[0].operation = { FloatingRemainder: "Modulus" };
  assert.throws(() => catalogEntry(entry), /floating remainder kind/);
  entry.placement.preparation.Forms[0].operation = "Unary";
  entry.placement.preparation.Forms[0].data_arguments = [1, 0];
  assert.throws(() => catalogEntry(entry), /strictly increasing/);
  entry.placement.preparation = { Forms: [], guessed: true };
  assert.throws(() => catalogEntry(entry), /operand preparation fields/);
});

test("foreign runtime link policy retains its actual typed adapter family", () => {
  const entry = compiledInventoryFixture().snapshot.declared.catalog_entries[0];
  for (const family of ["NativeFfi", "Mex", "Java", "Python"]) {
    entry.link.policy = { ForeignRuntime: family };
    assert.doesNotThrow(() => catalogEntry(entry));
  }
  entry.link.policy = "ForeignRuntime";
  assert.throws(() => catalogEntry(entry), /link policy/);
  entry.link.policy = { ForeignRuntime: "guessed-adapter" };
  assert.throws(() => catalogEntry(entry), /foreign adapter family/);
});

test("catalog and legacy semantics retain declared effect and suspension identities", () => {
  const entry = compiledInventoryFixture().snapshot.declared.catalog_entries[0];
  entry.contract.workspace_effect = "DynamicEvalCurrentWorkspace";
  assert.doesNotThrow(() => catalogEntry(entry));
  const legacy = legacyFunctionFixture([]);
  legacy.semantics.workspace_effect = "DynamicEvalCurrentWorkspace";
  assert.doesNotThrow(() => legacyFunction(legacy));
  for (const behavior of ["NeverSuspends", "MaySuspend", "MaySuspendLocally", "RequiresAsyncRuntime"]) {
    entry.contract.async_behavior = behavior;
    legacy.semantics.async_behavior = behavior;
    assert.doesNotThrow(() => catalogEntry(entry));
    assert.doesNotThrow(() => legacyFunction(legacy));
  }
  entry.contract.async_behavior = "guessed-suspension";
  legacy.semantics.async_behavior = "guessed-suspension";
  assert.throws(() => catalogEntry(entry), /async behavior/);
  assert.throws(() => legacyFunction(legacy), /semantic async/);
  legacy.semantics.async_behavior = "NeverSuspends";
  legacy.semantics.workspace_effect = "guessed-workspace-effect";
  assert.throws(() => legacyFunction(legacy), /workspace effect/);
});

test("extension schema distinguishes rejected forms and profile-dependent behavior", () => {
  const fixture = legacyFunctionFixture([]);
  for (const [mode, error_identifier, valid] of [
    ["RunMatOnly", "RunMat:compatibility:Fixture", true],
    ["RunMatOnly", null, false],
    ["RunMatOnly", "RunMat:Fixture", false],
    ["ProfileDependentBehavior", null, true],
    ["ProfileDependentBehavior", "RunMat:compatibility:Fixture", false],
    [{ AvailableSince: { year: 2026, half: "B" } }, "RunMat:compatibility:Fixture", true],
    [{ AvailableSince: { year: 2026, half: "B" } }, null, false],
    [{ AvailableSince: { year: 0, half: "B" } }, "RunMat:compatibility:Fixture", false],
    [{ AvailableSince: { year: 65535, half: "A" } }, "RunMat:compatibility:Fixture", true],
    [{ AvailableSince: { year: 65536, half: "B" } }, "RunMat:compatibility:Fixture", false],
    [{ AvailableSince: { year: 2026.5, half: "B" } }, "RunMat:compatibility:Fixture", false],
    [{ AvailableSince: { year: "2026", half: "B" } }, "RunMat:compatibility:Fixture", false],
    [{ AvailableSince: { year: 2026, half: "C" } }, "RunMat:compatibility:Fixture", false],
    [{ AvailableSince: { year: 2026, half: "B", inferred: true } }, "RunMat:compatibility:Fixture", false],
    [{ AvailableSince: { year: 2026 } }, "RunMat:compatibility:Fixture", false],
    [{ AvailableSince: { year: 2026, half: "B" }, RunMatOnly: {} }, "RunMat:compatibility:Fixture", false],
    ["Unknown", null, false],
  ]) {
    fixture.extensions = [{ id: "fixture", mode, description: "Fixture", error_identifier }];
    if (valid) assert.doesNotThrow(() => legacyFunction(fixture));
    else assert.throws(() => legacyFunction(fixture));
  }
});

test("constant documentation is optional, canonical, and strictly shaped", () => {
  const value = {
    name: "pi", kind: "real_double",
    provenance: { source_file: "crates/runmat-builtins/src/catalog/constant.rs", module_path: "runmat_builtins::catalog::constant" },
  };
  assert.doesNotThrow(() => constant(value));
  value.documentation = {
    authority: "Catalog", title: null, slug: null, summary: "Circle constant.",
    description: "A real double scalar.", keywords: ["constant"], related: [], sections: [],
    examples: [], example_exemption: "Schema fixture; executable examples are tested in the exporter.",
    faqs: [], links: [], media: [], evidence: { implementation: [], verification: [], notes: [] },
    introduced: null, status: null,
  };
  assert.doesNotThrow(() => constant(value));
  assert.throws(() => constant({ ...value, documentation: { ...value.documentation, guessed: true } }), /fields must be exactly/);
  assert.throws(() => constant({ ...value, documentation: { ...value.documentation, authority: "LegacySidecar" } }), /catalog-owned/);
  assert.throws(() => constant({ ...value, documentation: null }));
  assert.throws(() => constant({ ...value, callable: true }), /fields must be exactly/);
});

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
