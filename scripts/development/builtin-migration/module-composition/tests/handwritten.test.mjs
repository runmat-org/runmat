import assert from "node:assert/strict";
import test from "node:test";

import { auditHandwrittenComposition } from "../handwritten.mjs";

const ALWAYS = Object.freeze({ kind: "always" });
const TEST = Object.freeze({ kind: "test" });

test("handwritten audit accepts the complete typed declaration and reexport surface", () => {
  const product = runtimeProduct();
  const source = `
#[cfg(test)]
#[path = "alpha/source.rs"]
#[macro_use]
pub(crate) mod alpha;
pub mod beta;

#[cfg(target_arch = "wasm32")]
#[doc(hidden)]
pub use alpha::{
    evaluate as evaluate_alpha,
    Alpha,
};
pub(super) use beta::*;
`;
  assert.equal(auditHandwrittenComposition(source, product).length, 2);
});

test("handwritten audit accepts closed legacy aggregation spellings with exact semantic order", () => {
  const product = catalogProduct();
  const source = `
pub mod alpha;
pub mod beta;

pub(super) fn extend_entries(entries: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    super::extend_groups(entries, beta::ENTRY_GROUPS,);
    entries.extend_from_slice(alpha::ENTRIES);
}
`;
  assert.equal(auditHandwrittenComposition(source, product).length, 2);
});

test("handwritten audit preserves declaration order independently from canonical child storage", () => {
  const product = {
    product_id: "runtime-example", crate_role: "runtime",
    path: "crates/runmat-runtime/src/builtins/example/mod.rs", aggregations: [], aggregation_exports: [],
    children: [
      child("acceleration", { declarationOrder: 1 }),
      child("common", { declarationOrder: 0, macroUse: true }),
    ],
  };
  assert.equal(auditHandwrittenComposition(
    "#[macro_use]\npub mod common;\npub mod acceleration;\n", product,
  ).length, 2);
  assert.throws(() => auditHandwrittenComposition(
    "pub mod acceleration;\n#[macro_use]\npub mod common;\n", product,
  ), /surface differs/);
});

test("handwritten audit keeps aggregation cfg independent from declaration cfg", () => {
  const product = catalogProduct();
  product.children[0].aggregation_sources[0].condition = TEST;
  const source = `
pub mod alpha;
pub mod beta;
pub(super) fn extend_entries(entries: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    super::extend_groups(entries, beta::ENTRY_GROUPS);
    #[cfg(test)]
    entries.extend_from_slice(alpha::ENTRIES);
}
`;
  assert.equal(auditHandwrittenComposition(source, product).length, 2);
  assert.throws(() => auditHandwrittenComposition(
    source.replace("    #[cfg(test)]\n", ""), product,
  ), /surface differs/);
});

test("handwritten audit treats a child aggregation export as typed product composition", () => {
  const product = {
    product_id: "catalog-example", crate_role: "catalog",
    path: "crates/runmat-builtins/src/catalog/entries/example/mod.rs",
    aggregations: ["entries"],
    aggregation_exports: [{
      role: "entries", module: "registry", visibility: "super",
      condition: ALWAYS, doc_hidden: false,
    }],
    children: [child("registry", {
      declarationOrder: 0,
      sourcePath: "crates/runmat-builtins/src/catalog/entries/example/registry/mod.rs",
    })],
  };
  const source = "pub mod registry;\npub(super) use registry::extend_entries;\n";
  assert.equal(auditHandwrittenComposition(source, product).length, 1);
  assert.throws(
    () => auditHandwrittenComposition(source.replace("extend_entries", "extend_aliases"), product),
    /surface differs/,
  );
});

test("handwritten audit normalizes named-item order and rejects semantic reexport drift", () => {
  const valid = `#[cfg(test)]\n#[path = "alpha/source.rs"]\n#[macro_use]\npub(crate) mod alpha;\npub mod beta;\n#[cfg(target_arch = "wasm32")]\n#[doc(hidden)]\npub use alpha::{ evaluate as evaluate_alpha, Alpha, };\npub(super) use beta::*;\n`;
  assert.equal(auditHandwrittenComposition(
    valid.replace("evaluate as evaluate_alpha, Alpha", "Alpha, evaluate as evaluate_alpha"),
    runtimeProduct(),
  ).length, 2);
  const reversedStatements = valid.replace(
    /#\[cfg\(target_arch = "wasm32"\)\][\s\S]*?\};\npub\(super\) use beta::\*;/,
    "pub(super) use beta::*;\n#[cfg(target_arch = \"wasm32\")]\n#[doc(hidden)]\npub use alpha::{ evaluate as evaluate_alpha, Alpha, };",
  );
  assert.equal(auditHandwrittenComposition(reversedStatements, runtimeProduct()).length, 2);
  const changes = [
    ["pub use alpha", "pub(crate) use alpha"],
    ["evaluate_alpha", "evaluate_beta"],
    ["#[cfg(target_arch = \"wasm32\")]\n", ""],
    ["#[doc(hidden)]\n", ""],
  ];
  for (const [before, after] of changes) assert.throws(() => auditHandwrittenComposition(valid.replace(before, after), runtimeProduct()), /surface differs/);
});

test("handwritten audit accepts reviewed private child reexports and rejects unrepresentable imports", () => {
  const privateChild = oneChild();
  privateChild.children[0].reexports = [{ visibility: "private", condition: ALWAYS, doc_hidden: false, kind: "glob" }];
  assert.equal(auditHandwrittenComposition("pub mod alpha;\nuse alpha::*;\n", privateChild).length, 1);
  assert.throws(() => auditHandwrittenComposition("pub mod alpha;\npub use crate::alpha::*;\n", oneChild()), /unrepresentable path/);
  assert.throws(() => auditHandwrittenComposition("pub mod alpha;\nuse crate::BuiltinCatalogEntry;\n", oneChild()), /not a declared direct child/);
  assert.throws(() => auditHandwrittenComposition("pub mod alpha;\npub(in crate::other) use alpha::*;\n", oneChild()), /visibility is not representable/);
});

test("handwritten audit rejects duplicate and context-invalid attributes", () => {
  const values = [
    "#[cfg(test)]\n#[cfg(test)]\npub mod alpha;\n",
    "#[doc(hidden)]\npub mod alpha;\n",
    "#[path = \"alpha.rs\"]\npub use alpha::*;\n",
    "#[macro_use]\npub use alpha::*;\n",
    "#[doc(hidden)]\n#[doc(hidden)]\npub use alpha::*;\n",
  ];
  for (const source of values) assert.throws(() => auditHandwrittenComposition(source, oneChild()), /duplicate|invalid/);
});

test("handwritten audit rejects aggregation semantic, signature, condition, and body drift", () => {
  const valid = "pub mod alpha;\npub mod beta;\npub(super) fn extend_entries(entries: &mut Vec<&'static crate::BuiltinCatalogEntry>) { super::extend_groups(entries, beta::ENTRY_GROUPS); entries.extend_from_slice(alpha::ENTRIES); }\n";
  const changes = [
    ["beta::ENTRY_GROUPS); entries.extend_from_slice(alpha::ENTRIES)", "alpha::ENTRIES); super::extend_groups(entries, beta::ENTRY_GROUPS)"],
    ["beta::ENTRY_GROUPS", "alpha::ENTRY_GROUPS"],
    ["extend_from_slice(alpha::ENTRIES)", "extend(alpha::ENTRIES.iter().cloned())"],
    ["BuiltinCatalogEntry", "BuiltinCatalogAlias"],
    ["super::extend_groups", "#[cfg(test)] super::extend_groups"],
  ];
  for (const [before, after] of changes) assert.throws(() => auditHandwrittenComposition(valid.replace(before, after), catalogProduct()), /surface differs|not representable|value type/);
  assert.throws(() => auditHandwrittenComposition(valid.replace("entries.extend_from_slice(alpha::ENTRIES);", "let hidden = 1;"), catalogProduct()), /not representable/);
});

test("handwritten audit rejects extra functions and duplicate direct modules", () => {
  assert.throws(() => auditHandwrittenComposition("pub mod alpha;\nfn hidden() {}\n", oneChild()), /unsupported handwritten Rust syntax/);
  assert.throws(() => auditHandwrittenComposition("pub mod alpha;\npub mod alpha;\n", oneChild()), /collide or repeat/);
});

function runtimeProduct() {
  return {
    product_id: "runtime-example", crate_role: "runtime", path: "crates/runmat-runtime/src/builtins/example/mod.rs", aggregations: [], aggregation_exports: [],
    children: [
      child("alpha", { declarationOrder: 0, visibility: "crate", condition: TEST, sourcePath: "crates/runmat-runtime/src/builtins/example/alpha/source.rs", macroUse: true, reexports: [{ visibility: "public", condition: { kind: "target-architecture", architecture: "wasm32" }, doc_hidden: true, kind: "named", items: [{ name: "Alpha", alias: null }, { name: "evaluate", alias: "evaluate_alpha" }] }] }),
      child("beta", { declarationOrder: 1, reexports: [{ visibility: "super", condition: ALWAYS, doc_hidden: false, kind: "glob" }] }),
    ],
  };
}

function catalogProduct() {
  return {
    product_id: "catalog-example", crate_role: "catalog", path: "crates/runmat-builtins/src/catalog/entries/example/mod.rs", aggregations: ["entries"], aggregation_exports: [],
    children: [
      child("alpha", { declarationOrder: 0, sourcePath: "crates/runmat-builtins/src/catalog/entries/example/alpha/mod.rs", aggregation_sources: [{ role: "entries", kind: "slice", order: 1, condition: ALWAYS }] }),
      child("beta", { declarationOrder: 1, sourcePath: "crates/runmat-builtins/src/catalog/entries/example/beta/mod.rs", aggregation_sources: [{ role: "entries", kind: "groups", order: 0, condition: ALWAYS }] }),
    ],
  };
}

function oneChild() {
  return { product_id: "one-child", crate_role: "runtime", path: "crates/runmat-runtime/src/builtins/example/mod.rs", aggregations: [], aggregation_exports: [], children: [child("alpha")] };
}

function child(module, options = {}) {
  return {
    module, visibility: options.visibility ?? "public", declaration_condition: options.condition ?? ALWAYS,
    declaration_order: options.declarationOrder ?? 0,
    source_kind: "directory", source_path: options.sourcePath ?? `crates/runmat-runtime/src/builtins/example/${module}/mod.rs`,
    macro_use: options.macroUse ?? false, reexports: options.reexports ?? [], aggregation_sources: options.aggregation_sources ?? [],
  };
}
