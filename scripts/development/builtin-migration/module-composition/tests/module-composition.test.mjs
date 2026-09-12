import assert from "node:assert/strict";
import test from "node:test";

import {
  applyModuleCompositionTransitions, generateModuleCompositionProducts,
  parseGeneratedModuleComposition, parseModuleCompositionProjection,
  parseModuleCompositionTransition, renderModuleCompositionProduct,
  verifyModuleCompositionProduct,
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

test("generation is deterministic and uses canonical code-point order", () => {
  const projection = fixtureProjection();
  const first = generateModuleCompositionProducts(projection);
  const second = generateModuleCompositionProducts(structuredClone(projection));
  assert.deepEqual(first, second);
  assert.deepEqual(first.map((entry) => Buffer.from(entry.content)), second.map((entry) => Buffer.from(entry.content)));
  assert.ok(first[0].content.indexOf("mod arithmetic") < first[0].content.indexOf("mod plotting"));
});

test("projection rejects invalid parents, paths, keywords, enums, and collisions", () => {
  const cases = [
    [mutate((value) => { value.products[0].path = "crates/runmat-builtins/src/catalog/entries/other/mod.rs"; }), /does not match its logical parent/],
    [mutate((value) => { value.products[0].children[0].source_path = "crates/runmat-builtins/src/catalog/entries/io/arithmetic/mod.rs"; }), /outside its reviewed parent/],
    [mutate((value) => { value.products[0].children[0].module = "struct"; }), /must use r# exactly/],
    [mutate((value) => { value.products[0].children[0].module = "r#ordinary"; }), /must use r# exactly/],
    [mutate((value) => { value.products[0].children[0].module = "r#super"; }), /Rust reserves/],
    [mutate((value) => { value.products[0].children[0].module = "_"; }), /Rust reserves/],
    [mutate((value) => { value.products[0].children[0].role = "builtin"; }), /must be one of/],
    [mutate((value) => { value.products[0].children[0].visibility = "workspace"; }), /must be one of/],
    [mutate((value) => { value.products[0].children[0].feature_policy = { kind: "cfg", feature: "x" }; }), /unsupported kind/],
    [mutate((value) => { value.products[0].children[0].aggregation_roles = ["bindings"]; }), /must be one of/],
    [mutate((value) => { value.products[1].children[0].aggregation_roles = ["entries"]; }), /runtime children cannot/],
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

test("closed path and macro-use attributes reproduce safe noncanonical module placement", () => {
  const product = fixtureProjection().products[1];
  product.children = [child({
    module: "common", source_path: "crates/runmat-runtime/src/builtins/math/helpers/common.rs",
    sourceKind: "file", role: "support", visibility: "public", macroUse: true,
  })];
  const source = renderModuleCompositionProduct(product);
  assert.match(source, /#\[path = "helpers\/common.rs"\]\n#\[macro_use\]\npub mod common;/);
  assert.doesNotThrow(() => verifyModuleCompositionProduct(product, source));
  const invalid = structuredClone(product);
  invalid.children[0].role = "identity";
  assert.throws(() => renderModuleCompositionProduct(invalid), /restricted to support/);
});

test("the verifier rejects extra Rust, altered topology, and malformed aggregation", () => {
  const product = fixtureProjection().products[0];
  const source = renderModuleCompositionProduct(product);
  assert.throws(() => verifyModuleCompositionProduct(product, source.replace("mod arithmetic;", "mod arithmetic;\nfn injected() {}")), /unsupported generated Rust syntax/);
  assert.throws(() => verifyModuleCompositionProduct(product, source.replace("mod arithmetic;", "mod replacement;")), /(differs from its typed projection|canonical code-point order)/);
  assert.throws(() => parseGeneratedModuleComposition(source.replace("arithmetic::ENTRIES", "arithmetic::CONSTANTS")), /aggregation child is invalid/);
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
  const projected = applyModuleCompositionTransitions(baseline, [replace]);
  assert.equal(projected.products[0].children[0].visibility, "crate");
  const staleRemove = transition("remove-stale", [{ product_id: "catalog-math", operation: "remove", before, after: null }]);
  assert.throws(() => applyModuleCompositionTransitions(baseline, [replace, staleRemove]), /prior state differs/);
  const remove = transition("remove-current", [{ product_id: "catalog-math", operation: "remove", before: after, after: null }]);
  assert.equal(applyModuleCompositionTransitions(baseline, [replace, remove]).products[0].children.length, 1);
});

function fixtureProjection() {
  return {
    schema_version: 1,
    kind: "runmat-builtin-module-composition-projection",
    products: [
      {
        product_id: "catalog-math", crate_role: "catalog",
        path: "crates/runmat-builtins/src/catalog/entries/math/mod.rs",
        module_path: "crate::catalog::entries::math",
        children: [
          child({
            module: "arithmetic", source_path: "crates/runmat-builtins/src/catalog/entries/math/arithmetic/mod.rs",
            role: "group", visibility: "catalog", aggregations: ["constants", "entries"], reexportItems: ["ADD_CATALOG_ENTRY", "SUB_CATALOG_ENTRY"],
          }),
          child({
            module: "plotting", source_path: "crates/runmat-builtins/src/catalog/entries/math/plotting/mod.rs",
            role: "group", visibility: "private", feature: "plot-core", aggregations: ["aliases"],
          }),
        ],
      },
      {
        product_id: "runtime-math", crate_role: "runtime",
        path: "crates/runmat-runtime/src/builtins/math/mod.rs",
        module_path: "crate::builtins::math",
        children: [child({
          module: "arithmetic", source_path: "crates/runmat-runtime/src/builtins/math/arithmetic/mod.rs",
          role: "group", visibility: "public",
        })],
      },
    ],
  };
}

function child({ module, source_path, sourceKind = "directory", role = "identity", visibility = "private", feature = null, aggregations = [], reexportItems = null, macroUse = false }) {
  return {
    module, source_kind: sourceKind, source_path, role, visibility,
    feature_policy: feature === null ? { kind: "always" } : { kind: "cargo-feature", feature }, macro_use: macroUse,
    reexport: reexportItems === null ? { kind: "none" } : { kind: "named", visibility: "public", items: reexportItems },
    aggregation_roles: aggregations,
  };
}

function transition(transition_id, changes) {
  return { schema_version: 1, kind: "runmat-builtin-module-composition-transition", transition_id, changes };
}

function mutate(callback) {
  const value = fixtureProjection();
  callback(value);
  return value;
}
