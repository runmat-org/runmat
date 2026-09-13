import assert from "node:assert/strict";
import test from "node:test";

import {
  applyModuleCompositionTransitions, generateModuleCompositionProducts, generatedHeader,
  parseGeneratedModuleComposition, parseModuleCompositionProjection,
  parseModuleCompositionTransition, renderModuleCompositionProduct,
  validateModuleCompositionControl, verifyModuleCompositionProduct,
} from "../index.mjs";

test("renders and independently verifies the closed catalog and runtime grammar", () => {
  const projection = fixtureProjection();
  const generated = generateModuleCompositionProducts(projection);
  assert.deepEqual(generated.map(({ product_id, path }) => ({ product_id, path })), [
    { product_id: "catalog-math", path: "crates/runmat-builtins/src/catalog/entries/math/mod.rs" },
    { product_id: "runtime-math", path: "crates/runmat-runtime/src/builtins/math/mod.rs" },
  ]);
  const catalog = generated[0].content;
  assert.match(catalog, /pub\(in crate::catalog\) mod arithmetic;/);
  assert.match(catalog, /pub use arithmetic::\{ADD_CATALOG_ENTRY, SUB_CATALOG_ENTRY\};/);
  assert.match(catalog, /values\.extend\(arithmetic::ENTRIES\.iter\(\)\.copied\(\)\);/);
  assert.match(catalog, /#\[cfg\(feature = "plot-core"\)\][\s\S]*values\.extend\(plotting::ALIASES\.iter\(\)\.copied\(\)\);/);
  assert.match(catalog, /values\.extend\(arithmetic::CONSTANTS\.iter\(\)\.copied\(\)\);/);
  assert.doesNotMatch(generated[1].content, /extend_(entries|aliases|constants)/);
  for (const [index, product] of projection.products.entries()) {
    assert.deepEqual(verifyModuleCompositionProduct(product, generated[index].content), {
      product_id: product.product_id, path: product.path, result: "pass",
    });
  }
});

test("generation is deterministic while declaration order remains explicit", () => {
  const projection = fixtureProjection();
  const first = generateModuleCompositionProducts(projection);
  const second = generateModuleCompositionProducts(structuredClone(projection));
  assert.deepEqual(first, second);
  assert.deepEqual(first.map((entry) => Buffer.from(entry.content)), second.map((entry) => Buffer.from(entry.content)));
  assert.ok(first[0].content.indexOf("mod arithmetic") < first[0].content.indexOf("mod plotting"));
});

test("an empty product retains the complete canonical generated envelope", () => {
  const product = fixtureProjection().products[1];
  product.children = [];
  const source = renderModuleCompositionProduct(product);
  assert.equal(source, `${generatedHeader()}\n`);
  assert.doesNotThrow(() => verifyModuleCompositionProduct(product, source));
});

test("product presence is explicit and independent from an empty child set", () => {
  const projection = fixtureProjection();
  projection.products[1].children = [];
  const present = parseModuleCompositionProjection(projection);
  assert.equal(present.products[1].state, "present");
  assert.deepEqual(
    generateModuleCompositionProducts(present).map((entry) => entry.product_id),
    ["catalog-math", "runtime-math"],
  );

  const absent = structuredClone(projection);
  absent.products[1].state = "absent";
  const parsedAbsent = parseModuleCompositionProjection(absent);
  assert.deepEqual(
    generateModuleCompositionProducts(parsedAbsent).map((entry) => entry.product_id),
    ["catalog-math"],
  );
  assert.throws(
    () => renderModuleCompositionProduct(parsedAbsent.products[1]),
    /absent product cannot be rendered/,
  );

  const missing = structuredClone(projection);
  delete missing.products[1].state;
  assert.throws(() => parseModuleCompositionProjection(missing), /fields must be exactly/);
  const invalid = structuredClone(projection);
  invalid.products[1].state = "reserved";
  assert.throws(() => parseModuleCompositionProjection(invalid), /must be one of absent, present/);
  absent.products[1].children = fixtureProjection().products[1].children;
  const absentWithChildren = parseModuleCompositionProjection(absent);
  assert.equal(absentWithChildren.products[1].state, "absent");
  assert.deepEqual(
    generateModuleCompositionProducts(absentWithChildren).map((entry) => entry.product_id),
    ["catalog-math"],
  );
});

test("canonical child storage does not reorder macro-bearing runtime declarations", () => {
  const product = fixtureProjection().products[1];
  product.children = [
    child({
      module: "acceleration", source_path: "crates/runmat-runtime/src/builtins/math/acceleration/mod.rs",
      role: "group", visibility: "public", declarationOrder: 1,
    }),
    child({
      module: "common", source_path: "crates/runmat-runtime/src/builtins/math/common/mod.rs",
      role: "support", visibility: "public", macroUse: true, declarationOrder: 0,
    }),
  ];
  const parsed = parseModuleCompositionProjection({
    schema_version: 5, kind: "runmat-builtin-module-composition-projection", products: [product],
  });
  assert.deepEqual(parsed.products[0].children.map((entry) => entry.module), ["acceleration", "common"]);
  const source = renderModuleCompositionProduct(parsed.products[0]);
  assert.ok(source.indexOf("mod common") < source.indexOf("mod acceleration"));
  assert.match(source, /#\[macro_use\]\npub mod common;/);
  assert.doesNotThrow(() => verifyModuleCompositionProduct(parsed.products[0], source));
});

test("catalog composition models slice, grouped-slice, function, and empty aggregators", () => {
  const product = fixtureProjection().products[0];
  product.aggregations = ["entries"];
  product.children[0].aggregation_sources = [{ role: "entries", kind: "function", order: 0, condition: { kind: "always" } }];
  product.children[1].declaration_condition = { kind: "always" };
  product.children[1].aggregation_sources = [{ role: "entries", kind: "groups", order: 1, condition: { kind: "always" } }];
  const source = renderModuleCompositionProduct(product);
  assert.match(source, /arithmetic::extend_entries\(values\);/);
  assert.match(source, /plotting::ENTRY_GROUPS[\s\S]*\.iter\(\)[\s\S]*\.flat_map/);
  assert.doesNotThrow(() => verifyModuleCompositionProduct(product, source));

  const aliases = {
    ...structuredClone(product),
    product_id: "catalog-aliases",
    path: "crates/runmat-builtins/src/catalog/aliases/mod.rs",
    module_path: "crate::catalog::aliases",
    aggregations: ["aliases"],
    children: [],
  };
  const empty = renderModuleCompositionProduct(aliases);
  assert.match(empty, /fn extend_aliases[^]*\{\}/);
  assert.doesNotThrow(() => verifyModuleCompositionProduct(aliases, empty));
});

test("catalog aggregation implementations may be exact typed child exports", () => {
  const product = fixtureProjection().products[0];
  product.aggregations = ["entries"];
  product.aggregation_exports = [{
    role: "entries", module: "registry", visibility: "super",
    condition: { kind: "always" }, doc_hidden: false,
  }];
  product.children = [child({
    module: "registry", source_path: "crates/runmat-builtins/src/catalog/entries/math/registry.rs",
    sourceKind: "file", role: "support", visibility: "private", declarationOrder: 0,
  })];
  const source = renderModuleCompositionProduct(product);
  assert.match(source, /pub\(super\) use registry::extend_entries;/);
  assert.doesNotMatch(source, /fn extend_entries/);
  assert.doesNotThrow(() => verifyModuleCompositionProduct(product, source));

  const wrongModule = structuredClone(product);
  wrongModule.aggregation_exports[0].module = "missing";
  assert.throws(() => renderModuleCompositionProduct(wrongModule), /references undeclared child/);
  const danglingEmpty = structuredClone(product);
  danglingEmpty.children = [];
  assert.throws(() => renderModuleCompositionProduct(danglingEmpty), /references undeclared child/);
  assert.throws(
    () => verifyModuleCompositionProduct(product, source.replace("extend_entries", "extend_aliases")),
    /differs from its typed projection|canonical/,
  );
});

test("projection rejects invalid parents, paths, keywords, enums, and collisions", () => {
  for (const version of [1, 2, 3, 4, 6]) {
    const wrongVersion = fixtureProjection();
    wrongVersion.schema_version = version;
    assert.throws(() => parseModuleCompositionProjection(wrongVersion), /schema_version 5/);
  }
  const cases = [
    [mutate((value) => { value.products[0].path = "crates/runmat-builtins/src/catalog/entries/other/mod.rs"; }), /does not match its logical parent/],
    [mutate((value) => { value.products[0].children[0].source_path = "crates/runmat-builtins/src/catalog/entries/io/arithmetic/mod.rs"; }), /outside its reviewed parent/],
    [mutate((value) => { value.products[0].children[0].module = "struct"; }), /must use r# exactly/],
    [mutate((value) => { value.products[0].children[0].module = "r#ordinary"; }), /must use r# exactly/],
    [mutate((value) => { value.products[0].children[0].module = "r#super"; }), /Rust reserves/],
    [mutate((value) => { value.products[0].children[0].module = "_"; }), /Rust reserves/],
    [mutate((value) => { value.products[0].children[0].role = "builtin"; }), /must be one of/],
    [mutate((value) => { value.products[0].children[0].visibility = "workspace"; }), /must be one of/],
    [mutate((value) => { value.products[0].children[0].declaration_condition = { kind: "cfg", feature: "x" }; }), /unsupported kind/],
    [mutate((value) => { value.products[0].children[0].aggregation_sources = [{ role: "bindings", kind: "slice", order: 0, condition: { kind: "always" } }]; }), /must be one of/],
    [mutate((value) => { value.products[0].children[0].aggregation_sources = [{ role: "entries", kind: "unknown", order: 0, condition: { kind: "always" } }]; }), /must be one of/],
    [mutate((value) => { value.products[0].children[0].aggregation_sources = [{ role: "aliases", kind: "groups", order: 0, condition: { kind: "always" } }]; }), /grouped aggregation/],
    [mutate((value) => { value.products[1].children[0].aggregation_sources = [{ role: "entries", kind: "slice", order: 0, condition: { kind: "always" } }]; }), /is not emitted by its parent|runtime children cannot/],
    [mutate((value) => { value.products[1].children[0].visibility = "catalog"; }), /cannot use catalog visibility/],
    [mutate((value) => { value.products[0].children.push({ ...value.products[0].children[0], module: "Arithmetic", source_path: "crates/runmat-builtins/src/catalog/entries/math/Arithmetic/mod.rs" }); }), /collide case-insensitively/],
    [mutate((value) => { value.products[1].path = value.products[0].path.toUpperCase(); }), /product path does not match|outside its crate role/],
    [mutate((value) => { value.products[0].children[1].source_path = value.products[0].children[0].source_path; }), /source paths collide case-insensitively/],
  ];
  for (const [value, error] of cases) assert.throws(() => parseModuleCompositionProjection(value), error);
});

test("raw keyword modules are explicit and path-checked", () => {
  const projection = fixtureProjection();
  projection.products[1].children = [child({
    module: "r#struct", source_path: "crates/runmat-runtime/src/builtins/math/struct/mod.rs",
    visibility: "crate", role: "group",
  })];
  const parsed = parseModuleCompositionProjection(projection);
  assert.match(renderModuleCompositionProduct(parsed.products[1]), /pub\(crate\) mod r#struct;/);
});

test("parent-scoped declarations and reexports remain closed and verifiable", () => {
  const product = fixtureProjection().products[0];
  product.children = [child({
    module: "inference", source_path: "crates/runmat-builtins/src/catalog/entries/math/inference.rs",
    sourceKind: "file", role: "support", visibility: "super", reexportItems: ["infer"],
    reexportVisibility: "super",
  })];
  const source = renderModuleCompositionProduct(product);
  assert.match(source, /pub\(super\) mod inference;/);
  assert.match(source, /pub\(super\) use inference::\{infer\};/);
  assert.doesNotThrow(() => verifyModuleCompositionProduct(product, source));
});

test("case-distinct Rust item identities survive named reexport composition", () => {
  const product = fixtureProjection().products[1];
  product.children = [child({
    module: "definitions",
    source_path: "crates/runmat-runtime/src/builtins/math/definitions.rs",
    sourceKind: "file",
    role: "support",
    reexportItems: ["Inf", "NaN", "inf", "nan"],
    reexportVisibility: "crate",
  })];
  const parsed = parseModuleCompositionProjection({
    schema_version: 5,
    kind: "runmat-builtin-module-composition-projection",
    products: [product],
  });
  const source = renderModuleCompositionProduct(parsed.products[0]);
  assert.match(source, /definitions::\{Inf, NaN, inf, nan\}/);
  assert.doesNotThrow(() => verifyModuleCompositionProduct(parsed.products[0], source));
});

test("conditions, hidden reexports, aliases, and independent reexport conditions round-trip", () => {
  const product = fixtureProjection().products[1];
  product.children = [child({
    module: "plotting", source_path: "crates/runmat-runtime/src/builtins/math/plotting/mod.rs",
    role: "group", visibility: "public",
  })];
  product.children[0].declaration_condition = {
    kind: "all",
    conditions: [
      { kind: "target-architecture", architecture: "wasm32" },
      { kind: "cargo-feature", feature: "plot-web" },
    ],
  };
  product.children[0].reexports = [
    { kind: "glob", visibility: "crate", condition: product.children[0].declaration_condition, doc_hidden: false },
    { kind: "named", visibility: "public", condition: { kind: "all", conditions: [{ kind: "target-architecture", architecture: "wasm32" }, { kind: "cargo-feature", feature: "plot-web" }, { kind: "test" }] }, doc_hidden: true, items: [{ name: "evaluate", alias: "evaluate_plot" }] },
  ];
  const source = renderModuleCompositionProduct(product);
  assert.match(source, /#\[cfg\(all\(target_arch = "wasm32", feature = "plot-web"\)\)\]\npub mod plotting;/);
  assert.match(source, /#\[cfg\(all\(target_arch = "wasm32", feature = "plot-web", test\)\)\]\n#\[doc\(hidden\)\]\npub use plotting::\{evaluate as evaluate_plot\};/);
  assert.deepEqual(parseGeneratedModuleComposition(source), parseGeneratedModuleComposition(renderModuleCompositionProduct(product)));
  assert.doesNotThrow(() => verifyModuleCompositionProduct(product, source));
});

test("the v5 grammar rejects raw cfg, malformed conjunctions, aliases, and aggregation order", () => {
  const invalid = [
    [mutate((value) => { value.products[1].children[0].declaration_condition = { kind: "any", conditions: [] }; }), /unsupported kind/],
    [mutate((value) => { value.products[1].children[0].declaration_condition = { kind: "target-architecture", architecture: "x86_64" }; }), /unsupported target architecture/],
    [mutate((value) => { value.products[1].children[0].declaration_condition = { kind: "all", conditions: [{ kind: "cargo-feature", feature: "plot-web" }] }; }), /at least two/],
    [mutate((value) => { value.products[1].children[0].declaration_condition = { kind: "all", conditions: [{ kind: "cargo-feature", feature: "plot-web" }, { kind: "target-architecture", architecture: "wasm32" }] }; }), /canonical condition order/],
    [mutate((value) => { value.products[0].children[0].declaration_order = 2; }), /declaration orders.*contiguous/],
    [mutate((value) => { delete value.products[0].children[0].declaration_order; }), /fields must be exactly/],
    [mutate((value) => { delete value.products[0].children[0].aggregation_sources[0].condition; }), /fields must be exactly/],
    [mutate((value) => {
      value.products[0].children[0].declaration_condition = { kind: "cargo-feature", feature: "catalog-a" };
      for (const reexport of value.products[0].children[0].reexports) reexport.condition = { kind: "cargo-feature", feature: "catalog-a" };
      value.products[0].children[0].aggregation_sources[0].condition = { kind: "test" };
    }), /aggregation condition must imply/],
    [mutate((value) => {
      value.products[0].children[0].declaration_condition = { kind: "cargo-feature", feature: "catalog-a" };
      value.products[0].children[0].aggregation_sources[0].condition = { kind: "cargo-feature", feature: "catalog-a" };
    }), /reexport condition must imply/],
    [mutate((value) => { value.products[0].children[0].reexports[0].items[0].alias = "not-an-identifier"; }), /Rust item identifier/],
    [mutate((value) => { value.products[0].children[0].reexports[0].items[1].alias = "ADD_CATALOG_ENTRY"; }), /exported item names collide/],
    [mutate((value) => { value.products[0].children[0].reexports.push(structuredClone(value.products[0].children[0].reexports[0])); }), /reexports must be unique/],
    [mutate((value) => { value.products[0].children[0].aggregation_sources[0].order = 2; }), /contiguous from zero/],
    [mutate((value) => { value.products[0].children[1].aggregation_sources[0].order = -1; }), /nonnegative integer/],
  ];
  for (const [projection, error] of invalid) assert.throws(() => parseModuleCompositionProjection(projection), error);
  const source = renderModuleCompositionProduct(fixtureProjection().products[1]);
  assert.throws(
    () => parseGeneratedModuleComposition(source.replace("pub mod arithmetic;", "#[cfg(any())]\npub mod arithmetic;")),
    /unsupported composition attribute/,
  );
});

test("aggregation contributor order is independent of child declaration order", () => {
  const product = fixtureProjection().products[0];
  product.aggregations = ["entries"];
  product.children[0].aggregation_sources = [{ role: "entries", kind: "slice", order: 1, condition: { kind: "always" } }];
  product.children[1].aggregation_sources = [{ role: "entries", kind: "groups", order: 0, condition: product.children[1].declaration_condition }];
  const source = renderModuleCompositionProduct(product);
  assert.ok(source.indexOf("plotting::ENTRY_GROUPS") < source.indexOf("arithmetic::ENTRIES"));
  assert.doesNotThrow(() => verifyModuleCompositionProduct(product, source));
});

test("aggregation conditions remain independent from declaration conditions", () => {
  const product = fixtureProjection().products[0];
  product.aggregations = ["entries"];
  product.children[0].aggregation_sources = [{
    role: "entries", kind: "slice", order: 0, condition: { kind: "test" },
  }];
  product.children[1].aggregation_sources = [];
  const source = renderModuleCompositionProduct(product);
  assert.match(source, /pub\(in crate::catalog\) mod arithmetic;[\s\S]*#\[cfg\(test\)\]\n    values\.extend\(arithmetic::ENTRIES/);
  assert.doesNotThrow(() => verifyModuleCompositionProduct(product, source));
});

test("closed path and macro-use attributes remain independent of semantic child roles", () => {
  const product = fixtureProjection().products[1];
  product.children = [child({
    module: "common", source_path: "crates/runmat-runtime/src/builtins/math/helpers/common.rs",
    sourceKind: "file", role: "group", visibility: "public", macroUse: true, declarationOrder: 0,
  }), child({
    module: "tests", source_path: "crates/runmat-runtime/src/builtins/math/tests.rs",
    sourceKind: "file", role: "support", feature: "test", declarationOrder: 1,
  })];
  const source = renderModuleCompositionProduct(product);
  assert.match(source, /#\[path = "helpers\/common.rs"\]\n#\[macro_use\]\npub mod common;/);
  assert.match(source, /#\[cfg\(test\)\]\nmod tests;/);
  assert.doesNotThrow(() => verifyModuleCompositionProduct(product, source));
  product.children[0].role = "identity";
  assert.doesNotThrow(() => verifyModuleCompositionProduct(product, renderModuleCompositionProduct(product)));
});

test("the verifier rejects extra Rust, altered topology, and malformed aggregation", () => {
  const product = fixtureProjection().products[0];
  const source = renderModuleCompositionProduct(product);
  assert.throws(
    () => verifyModuleCompositionProduct(product, source.replace("mod arithmetic;", "mod arithmetic;\nfn injected() {}")),
    /unsupported handwritten Rust syntax/,
  );
  assert.throws(() => verifyModuleCompositionProduct(product, source.replace("mod arithmetic;", "mod replacement;")), /(differs from its typed projection|canonical module order)/);
  assert.throws(
    () => parseGeneratedModuleComposition(source.replace("arithmetic::ENTRIES", "arithmetic::CONSTANTS")),
    /aggregation statement is not representable/,
  );
  assert.throws(() => parseGeneratedModuleComposition(source.replace("// @generated", "// handwritten")), /header is invalid/);
  assert.throws(() => verifyModuleCompositionProduct(product, source.replaceAll("\n", "\r\n")), /canonical LF-terminated/);
});

test("sequential transitions retain prior children and leave other products unchanged", () => {
  const baseline = fixtureProjection();
  baseline.products[0].children = [];
  const runtimeBefore = renderModuleCompositionProduct(baseline.products[1]);
  const arithmetic = fixtureProjection().products[0].children[0];
  const plotting = fixtureProjection().products[0].children[1];
  const first = transition("bundle-one", [{ product_id: "catalog-math", operation: "add", before: null, after: arithmetic }]);
  const afterFirst = applyModuleCompositionTransitions(baseline, [first]);
  assert.deepEqual(afterFirst.products[0].children.map((entry) => entry.module), ["arithmetic"]);
  const second = transition("bundle-two", [{ product_id: "catalog-math", operation: "add", before: null, after: plotting }]);
  const afterSecond = applyModuleCompositionTransitions(baseline, [first, second]);
  assert.deepEqual(afterSecond.products[0].children.map((entry) => entry.module), ["arithmetic", "plotting"]);
  assert.deepEqual(afterSecond.products[1], baseline.products[1]);
  assert.equal(renderModuleCompositionProduct(afterSecond.products[1]), runtimeBefore);
  assert.throws(() => applyModuleCompositionTransitions(baseline, [first, first]), /duplicate module composition transition/);
});

test("replacement and removal require the exact effective prior child", () => {
  const baseline = fixtureProjection();
  const before = baseline.products[0].children[0];
  const after = { ...structuredClone(before), visibility: "crate" };
  const replace = transition("replace-one", [{ product_id: "catalog-math", operation: "replace", before, after }]);
  assert.doesNotThrow(() => parseModuleCompositionTransition(replace, baseline));
  for (const version of [1, 2, 3, 4, 6]) {
    assert.throws(
      () => parseModuleCompositionTransition({ ...replace, schema_version: version }, baseline),
      /schema_version 5/,
    );
  }
  const projected = applyModuleCompositionTransitions(baseline, [replace]);
  assert.equal(projected.products[0].children[0].visibility, "crate");
  const staleRemove = transition("remove-stale", [{ product_id: "catalog-math", operation: "remove", before, after: null }]);
  assert.throws(() => applyModuleCompositionTransitions(baseline, [replace, staleRemove]), /prior state differs/);
  const plotting = baseline.products[0].children[1];
  const reorderedPlotting = { ...structuredClone(plotting), declaration_order: 0 };
  const remove = transition("remove-current", [
    { product_id: "catalog-math", operation: "remove", before: after, after: null },
    { product_id: "catalog-math", operation: "replace", before: plotting, after: reorderedPlotting },
  ]);
  assert.equal(applyModuleCompositionTransitions(baseline, [replace, remove]).products[0].children.length, 1);
});

test("state transitions activate, retain, and deactivate parents atomically", () => {
  const baseline = fixtureProjection();
  const runtime = baseline.products[1];
  const child = structuredClone(runtime.children[0]);
  runtime.state = "absent";
  runtime.children = [];
  const activate = transition("activate-runtime", [{
    product_id: runtime.product_id, operation: "add", before: null, after: child,
  }], [{
    product_id: runtime.product_id, before_state: "absent", after_state: "present",
  }]);
  const active = applyModuleCompositionTransitions(baseline, [activate]);
  assert.equal(active.products[1].state, "present");
  assert.deepEqual(active.products[1].children.map((entry) => entry.module), ["arithmetic"]);

  const retain = transition("retain-runtime", [{
    product_id: runtime.product_id, operation: "remove", before: child, after: null,
  }]);
  const retained = applyModuleCompositionTransitions(active, [retain]);
  assert.equal(retained.products[1].state, "present");
  assert.deepEqual(retained.products[1].children, []);
  assert.doesNotThrow(() => renderModuleCompositionProduct(retained.products[1]));

  const deactivate = transition("deactivate-runtime", [], [{
    product_id: runtime.product_id, before_state: "present", after_state: "absent",
  }]);
  const inactive = applyModuleCompositionTransitions(retained, [deactivate]);
  assert.equal(inactive.products[1].state, "absent");
  assert.deepEqual(inactive.products[1].children, []);

  const stale = structuredClone(deactivate);
  stale.product_states[0].before_state = "absent";
  assert.throws(
    () => applyModuleCompositionTransitions(retained, [stale]),
    /prior product state differs/,
  );
  const implicit = structuredClone(activate);
  implicit.product_states = [];
  assert.throws(
    () => applyModuleCompositionTransitions(baseline, [implicit]),
    /product states must be a nonempty array/,
  );
  const noop = transition("noop-runtime", [], [{
    product_id: runtime.product_id, before_state: "absent", after_state: "absent",
  }]);
  assert.throws(
    () => applyModuleCompositionTransitions(baseline, [noop]),
    /unchanged product state requires a child change/,
  );
});

test("reviewed bundle transitions exactly cover products, scopes, and disjoint child keys", () => {
  const baseline = fixtureProjection();
  baseline.products[0].children = [];
  const products = new Map(baseline.products.map((product) => [product.product_id, {
    product_id: product.product_id,
    path: product.path,
    lifecycle: { kind: "bundle-referenced" },
    verification: {
      kind: "rust_module_composition",
      crate_role: product.crate_role,
      module_path: product.module_path,
    },
  }]));
  const arithmetic = fixtureProjection().products[0].children[0];
  const bundle = (id, after = arithmetic) => ({
    integration_product_refs: ["catalog-math"],
    module_composition_transition: transition(id, [{
      product_id: "catalog-math", operation: "add", before: null, after,
    }]),
    authored_write_set: [{ kind: "tree", path: after.source_path.slice(0, -"/mod.rs".length) }],
  });
  assert.doesNotThrow(() => validateModuleCompositionControl(
    baseline, products, new Map([["bundle-one", bundle("bundle-one")]]),
  ));
  assert.throws(() => validateModuleCompositionControl(
    baseline, products, new Map([["bundle-one", {
      ...bundle("bundle-one"), integration_product_refs: [],
    }]]),
  ), /without composition products/);
  assert.throws(() => validateModuleCompositionControl(
    baseline, products, new Map([["bundle-one", {
      ...bundle("bundle-one"), authored_write_set: [{ kind: "file", path: "elsewhere.rs" }],
    }]]),
  ), /outside its authored scope/);
  assert.throws(() => validateModuleCompositionControl(
    baseline, products, new Map([
      ["bundle-one", bundle("bundle-one")],
      ["bundle-two", bundle("bundle-two")],
    ]),
  ), /also changed by bundle-one/);

  const unchangedBaseline = fixtureProjection();
  const unchangedProducts = new Map(unchangedBaseline.products.map((product) => [product.product_id, {
    product_id: product.product_id,
    path: product.path,
    lifecycle: { kind: "bundle-referenced" },
    verification: {
      kind: "rust_module_composition",
      crate_role: product.crate_role,
      module_path: product.module_path,
    },
  }]));
  const unchanged = unchangedBaseline.products[0].children[0];
  const unchangedBundle = {
    integration_product_refs: ["catalog-math"],
    module_composition_transition: transition("bundle-one", [{
      product_id: "catalog-math", operation: "replace", before: unchanged,
      after: structuredClone(unchanged),
    }]),
    authored_write_set: [{ kind: "tree", path: unchanged.source_path.slice(0, -"/mod.rs".length) }],
  };
  assert.throws(() => validateModuleCompositionControl(
    unchangedBaseline, unchangedProducts,
    new Map([["bundle-one", unchangedBundle]]),
  ), /must change the reviewed child/);
  assert.throws(
    () => parseModuleCompositionTransition(unchangedBundle.module_composition_transition, unchangedBaseline),
    /must change the reviewed child/,
  );
});

test("reviewed baseline-only composition products require no bundle transition", () => {
  const baseline = fixtureProjection();
  const products = new Map(baseline.products.map((product) => [product.product_id, {
    product_id: product.product_id,
    path: product.path,
    lifecycle: { kind: "reviewed-baseline-only" },
    verification: {
      kind: "rust_module_composition",
      crate_role: product.crate_role,
      module_path: product.module_path,
    },
  }]));
  assert.doesNotThrow(() => validateModuleCompositionControl(baseline, products, new Map()));

  const child = baseline.products[0].children[0];
  const bundle = {
    integration_product_refs: ["catalog-math"],
    module_composition_transition: transition("bundle-one", [{
      product_id: "catalog-math", operation: "replace", before: child,
      after: structuredClone(child),
    }]),
    authored_write_set: [{ kind: "tree", path: child.source_path.slice(0, -"/mod.rs".length) }],
  };
  assert.throws(
    () => validateModuleCompositionControl(
      baseline, products, new Map([["bundle-one", bundle]]),
    ),
    /reviewed-baseline-only composition product catalog-math cannot be bundle referenced/,
  );
});

test("a parent may reference an exact integration-owned child product", () => {
  const baseline = fixtureProjection();
  const parent = baseline.products[0];
  parent.children = [];
  const childProduct = {
    product_id: "catalog-math-arithmetic",
    crate_role: "catalog",
    path: "crates/runmat-builtins/src/catalog/entries/math/arithmetic/mod.rs",
    module_path: "crate::catalog::entries::math::arithmetic",
    state: "absent",
    aggregations: ["entries"],
    aggregation_exports: [],
    children: [],
  };
  baseline.products.push(childProduct);
  baseline.products.sort((left, right) =>
    left.product_id < right.product_id ? -1 : left.product_id > right.product_id ? 1 : 0);
  const products = new Map(baseline.products.map((product) => [product.product_id, {
    product_id: product.product_id,
    path: product.path,
    baseline_digest: product.state === "absent" ? null : `sha256:${"1".repeat(64)}`,
    lifecycle: { kind: "bundle-referenced" },
    verification: {
      kind: "rust_module_composition",
      crate_role: product.crate_role,
      module_path: product.module_path,
    },
  }]));
  const parentChild = child({
    module: "arithmetic",
    source_path: childProduct.path,
    role: "group",
    visibility: "private",
    aggregations: ["entries"],
  });
  const leaf = child({
    module: "add",
    source_path: "crates/runmat-builtins/src/catalog/entries/math/arithmetic/add.rs",
    sourceKind: "file",
  });
  const bundle = {
    prerequisites: [],
    integration_product_refs: ["catalog-math", "catalog-math-arithmetic"],
    module_composition_transition: transition("bundle-one", [
      { product_id: "catalog-math", operation: "add", before: null, after: parentChild },
      { product_id: "catalog-math-arithmetic", operation: "add", before: null, after: leaf },
    ], [
      { product_id: "catalog-math", before_state: "present", after_state: "present" },
      { product_id: "catalog-math-arithmetic", before_state: "absent", after_state: "present" },
    ]),
    authored_write_set: [{ kind: "file", path: leaf.source_path }],
  };
  assert.doesNotThrow(() => validateModuleCompositionControl(
    baseline, products, new Map([["bundle-one", bundle]]),
  ));

  const unreviewed = structuredClone(bundle);
  unreviewed.module_composition_transition.changes[0].after.source_path =
    "crates/runmat-builtins/src/catalog/entries/math/unreviewed/mod.rs";
  assert.throws(
    () => validateModuleCompositionControl(
      baseline, products, new Map([["bundle-one", unreviewed]]),
    ),
    /outside its authored scope/,
  );
});

test("prerequisite order validates sequential additions after parent activation", () => {
  const baseline = fixtureProjection();
  const catalog = baseline.products[0];
  catalog.state = "absent";
  catalog.children = [];
  const products = new Map(baseline.products.map((product) => [product.product_id, {
    product_id: product.product_id,
    path: product.path,
    baseline_digest: product.state === "absent" ? null : `sha256:${"1".repeat(64)}`,
    lifecycle: {
      kind: product.product_id === catalog.product_id
        ? "bundle-referenced"
        : "reviewed-baseline-only",
    },
    verification: {
      kind: "rust_module_composition",
      crate_role: product.crate_role,
      module_path: product.module_path,
    },
  }]));
  const alpha = child({
    module: "alpha",
    source_path: "crates/runmat-builtins/src/catalog/entries/math/alpha/mod.rs",
    role: "group",
    visibility: "private",
    declarationOrder: 0,
    aggregations: ["entries"],
  });
  const beta = child({
    module: "beta",
    source_path: "crates/runmat-builtins/src/catalog/entries/math/beta/mod.rs",
    role: "group",
    visibility: "private",
    declarationOrder: 1,
    aggregations: ["entries"],
  });
  beta.aggregation_sources[0].order = 1;
  const activating = {
    prerequisites: [],
    integration_product_refs: [catalog.product_id],
    module_composition_transition: transition("z-activate", [{
      product_id: catalog.product_id,
      operation: "add",
      before: null,
      after: alpha,
    }], [{
      product_id: catalog.product_id,
      before_state: "absent",
      after_state: "present",
    }]),
    authored_write_set: [{ kind: "tree", path: alpha.source_path.slice(0, -"/mod.rs".length) }],
  };
  const extending = {
    prerequisites: [{ bundle_id: "z-activate", kind: "infrastructure" }],
    integration_product_refs: [catalog.product_id],
    module_composition_transition: transition("a-extend", [{
      product_id: catalog.product_id,
      operation: "add",
      before: null,
      after: beta,
    }]),
    authored_write_set: [{ kind: "tree", path: beta.source_path.slice(0, -"/mod.rs".length) }],
  };
  assert.doesNotThrow(() => validateModuleCompositionControl(
    baseline,
    products,
    new Map([["a-extend", extending], ["z-activate", activating]]),
  ));

  const unordered = structuredClone(extending);
  unordered.prerequisites = [];
  assert.throws(
    () => validateModuleCompositionControl(
      baseline,
      products,
      new Map([["a-extend", unordered], ["z-activate", activating]]),
    ),
    /composition transitions a-extend and z-activate must be prerequisite-ordered/,
  );
});

function fixtureProjection() {
  return {
    schema_version: 5,
    kind: "runmat-builtin-module-composition-projection",
    products: [
      {
        product_id: "catalog-math", crate_role: "catalog",
        path: "crates/runmat-builtins/src/catalog/entries/math/mod.rs",
        module_path: "crate::catalog::entries::math",
        state: "present",
        aggregations: ["entries", "aliases", "constants"],
        aggregation_exports: [],
        children: [
          child({
            module: "arithmetic", source_path: "crates/runmat-builtins/src/catalog/entries/math/arithmetic/mod.rs",
            role: "group", visibility: "catalog", declarationOrder: 0, aggregations: ["entries", "constants"], reexportItems: ["ADD_CATALOG_ENTRY", "SUB_CATALOG_ENTRY"],
          }),
          child({
            module: "plotting", source_path: "crates/runmat-builtins/src/catalog/entries/math/plotting/mod.rs",
            role: "group", visibility: "private", feature: "plot-core", declarationOrder: 1, aggregations: ["aliases"],
          }),
        ],
      },
      {
        product_id: "runtime-math", crate_role: "runtime",
        path: "crates/runmat-runtime/src/builtins/math/mod.rs",
        module_path: "crate::builtins::math",
        state: "present",
        aggregations: [],
        aggregation_exports: [],
        children: [child({
          module: "arithmetic", source_path: "crates/runmat-runtime/src/builtins/math/arithmetic/mod.rs",
          role: "group", visibility: "public",
        })],
      },
    ],
  };
}

function child({ module, source_path, sourceKind = "directory", role = "identity", visibility = "private", feature = null, declarationOrder = 0, aggregations = [], reexportItems = null, reexportVisibility = "public", macroUse = false }) {
  const declarationCondition = feature === null ? { kind: "always" }
    : feature === "test" ? { kind: "test" } : { kind: "cargo-feature", feature };
  return {
    module, source_kind: sourceKind, source_path, role, visibility,
    declaration_condition: declarationCondition, declaration_order: declarationOrder,
    macro_use: macroUse,
    reexports: reexportItems === null ? [] : [{ kind: "named", visibility: reexportVisibility, condition: { kind: "always" }, doc_hidden: false, items: reexportItems.map((name) => ({ name, alias: null })) }],
    aggregation_sources: aggregations.map((role) => ({ role, kind: "slice", order: 0, condition: declarationCondition })),
  };
}

function transition(transition_id, changes, productStates = null) {
  const productIds = [...new Set(changes.map((entry) => entry.product_id))].sort();
  return {
    schema_version: 5,
    kind: "runmat-builtin-module-composition-transition",
    transition_id,
    product_states: productStates ?? productIds.map((product_id) => ({
      product_id, before_state: "present", after_state: "present",
    })),
    changes,
  };
}

function mutate(callback) {
  const value = fixtureProjection();
  callback(value);
  return value;
}
