import assert from "node:assert/strict";
import test from "node:test";

import {
  moduleCompositionProductRegistry, validateReviewedModuleCompositionAuthority,
} from "../index.mjs";

test("fixed registry is the exact reviewed module-composition authority", () => {
  const { products, projection } = authorityFixture();
  products.set("unrelated-product", {
    product_id: "unrelated-product",
    verification: { kind: "content_identity" },
  });
  assert.equal(validateReviewedModuleCompositionAuthority(products, projection), projection);
});

test("fixed registry rejects omitted and additional composition products", () => {
  const omitted = authorityFixture();
  omitted.products.delete("catalog-aliases");
  omitted.projection.products = omitted.projection.products
    .filter((entry) => entry.product_id !== "catalog-aliases");
  assert.throws(
    () => validateReviewedModuleCompositionAuthority(omitted.products, omitted.projection),
    /missing catalog-aliases/,
  );

  const additional = authorityFixture();
  additional.products.set("runtime-unreviewed", {
    product_id: "runtime-unreviewed",
    path: "crates/runmat-runtime/src/builtins/unreviewed/mod.rs",
    verification: {
      kind: "rust_module_composition",
      crate_role: "runtime",
      module_path: "crate::builtins::unreviewed",
    },
  });
  assert.throws(
    () => validateReviewedModuleCompositionAuthority(additional.products, additional.projection),
    /unexpected runtime-unreviewed/,
  );
});

test("fixed registry rejects reviewed product field drift", () => {
  for (const [field, mutate, message] of [
    ["path", (entry) => { entry.path = "crates/runmat-runtime/src/builtins/drift/mod.rs"; }, /path does not match/],
    ["crate role", (entry) => { entry.verification.crate_role = "catalog"; }, /crate role does not match/],
    ["module path", (entry) => { entry.verification.module_path = "crate::builtins::drift"; }, /module path does not match/],
  ]) {
    const fixture = authorityFixture();
    mutate(fixture.products.get("runtime-root"));
    assert.throws(
      () => validateReviewedModuleCompositionAuthority(fixture.products, fixture.projection),
      message,
      field,
    );
  }
});

test("fixed registry rejects projection coverage, fields, and aggregation drift", () => {
  const omitted = authorityFixture();
  omitted.projection.products = omitted.projection.products.slice(1);
  assert.throws(
    () => validateReviewedModuleCompositionAuthority(omitted.products, omitted.projection),
    /projection must exactly cover.*missing catalog-acceleration/,
  );

  const additional = authorityFixture();
  additional.projection.products.push({
    product_id: "runtime-unreviewed",
    crate_role: "runtime",
    path: "crates/runmat-runtime/src/builtins/unreviewed/mod.rs",
    module_path: "crate::builtins::unreviewed",
    aggregations: [],
    aggregation_exports: [],
    children: [],
  });
  assert.throws(
    () => validateReviewedModuleCompositionAuthority(additional.products, additional.projection),
    /projection must exactly cover.*unexpected runtime-unreviewed/,
  );

  for (const [field, mutate, message] of [
    ["path", (entry) => { entry.path = "crates/runmat-builtins/src/catalog/entries/drift/mod.rs"; }, /projection path does not match/],
    ["crate role", (entry) => { entry.crate_role = "runtime"; }, /projection crate role does not match/],
    ["module path", (entry) => { entry.module_path = "crate::catalog::entries::drift"; }, /projection module path does not match/],
    ["aggregations", (entry) => { entry.aggregations = ["entries", "constants"]; }, /aggregation roles do not match/],
    ["aggregation exports", (entry) => { entry.aggregation_exports = [{ role: "entries" }]; }, /aggregation exports do not match/],
  ]) {
    const fixture = authorityFixture();
    mutate(fixture.projection.products.find((entry) => entry.product_id === "catalog-constants"));
    assert.throws(
      () => validateReviewedModuleCompositionAuthority(fixture.products, fixture.projection),
      message,
      field,
    );
  }
});

function authorityFixture() {
  const definitions = moduleCompositionProductRegistry();
  const products = new Map(definitions.map((definition) => [definition.product_id, {
    product_id: definition.product_id,
    path: definition.path,
    verification: {
      kind: "rust_module_composition",
      crate_role: definition.crate_role,
      module_path: definition.module_path,
    },
  }]));
  const projection = {
    products: definitions.map((definition) => ({
      ...structuredClone(definition), state: "absent", children: [],
    })),
  };
  return { products, projection };
}
