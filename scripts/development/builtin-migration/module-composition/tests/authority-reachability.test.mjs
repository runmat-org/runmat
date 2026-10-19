import assert from "node:assert/strict";
import test from "node:test";

import { validateAuthorityReachability } from "../authority-reachability.mjs";

const OWNER = "crates/runmat-runtime/src/builtins/math/core/alpha/mod.rs";
const CHILD = "crates/runmat-runtime/src/builtins/math/core/mod.rs";

test("identity authority must be reachable through the effective module composition", () => {
  const identities = new Map([["alpha", identity(OWNER)]]);
  assert.doesNotThrow(() => validateAuthorityReachability(composition(CHILD), identities));
  assert.throws(
    () => validateAuthorityReachability(composition(null), identities),
    /alpha: authority .* is unreachable .* runtime-math does not declare/,
  );
});

test("identity authority requires every present product and declaration in its ancestor chain", () => {
  const authority = "crates/runmat-runtime/src/builtins/logical/relational/eq/mod.rs";
  const identities = new Map([["eq", identity(authority)]]);
  const complete = nestedComposition();
  assert.doesNotThrow(() => validateAuthorityReachability(complete, identities));

  const missingRootDeclaration = structuredClone(complete);
  missingRootDeclaration.effective.products[0].children = [];
  assert.throws(
    () => validateAuthorityReachability(missingRootDeclaration, identities),
    /runtime-root does not declare a child owning .*builtins\/logical\/mod\.rs/,
  );

  const absentFamily = structuredClone(complete);
  absentFamily.effective.products[2].state = "absent";
  assert.throws(
    () => validateAuthorityReachability(absentFamily, identities),
    /runtime-logical-relational is absent/,
  );

  const missingIdentityLeaf = structuredClone(complete);
  missingIdentityLeaf.effective.products[2].children = [];
  assert.throws(
    () => validateAuthorityReachability(missingIdentityLeaf, identities),
    /runtime-logical-relational does not declare a child owning .*logical\/relational\/eq\/mod\.rs/,
  );
});

test("distinct bundles sharing a nested module parent require integration product authority", () => {
  const identities = new Map([
    ["eq", identity("crates/runmat-runtime/src/builtins/logical/relational/eq/mod.rs", "bundle-eq")],
    ["ne", identity("crates/runmat-runtime/src/builtins/logical/relational/ne/mod.rs", "bundle-ne")],
  ]);
  const unownedFamily = nestedComposition();
  unownedFamily.effective.products.pop();
  assert.throws(
    () => validateAuthorityReachability(unownedFamily, identities),
    /logical\/relational\/mod\.rs: shared authority parent used by bundles bundle-eq, bundle-ne is absent from integration product authority/,
  );
  const complete = nestedComposition();
  complete.effective.products[2].children.push({
    source_kind: "directory",
    source_path: "crates/runmat-runtime/src/builtins/logical/relational/ne/mod.rs",
  });
  assert.doesNotThrow(() => validateAuthorityReachability(complete, identities));
});

test("a declared file module owns both its source and logical submodule tree", () => {
  const owner = "crates/runmat-builtins/src/catalog/entries/constants/core/pi/mod.rs";
  const identities = new Map([["pi", catalogIdentity(owner)]]);
  const composition = { effective: { products: [{
    product_id: "catalog-constants",
    path: "crates/runmat-builtins/src/catalog/entries/constants/mod.rs",
    state: "present",
    children: [{
      source_kind: "file",
      source_path: "crates/runmat-builtins/src/catalog/entries/constants/core.rs",
    }],
  }] } };
  assert.doesNotThrow(() => validateAuthorityReachability(composition, identities));
  composition.effective.products[0].children[0].source_path =
    "crates/runmat-builtins/src/catalog/entries/constants/other.rs";
  assert.throws(
    () => validateAuthorityReachability(composition, identities),
    /catalog-constants does not declare a child owning/,
  );
});

test("a registered file-module product is an exact ancestor boundary", () => {
  const owner = "crates/runmat-builtins/src/catalog/entries/constants/core/pi/mod.rs";
  const identities = new Map([["pi", catalogIdentity(owner)]]);
  const composition = { effective: { products: [
    {
      product_id: "catalog-constants",
      path: "crates/runmat-builtins/src/catalog/entries/constants/mod.rs",
      state: "present",
      children: [{
        source_kind: "file",
        source_path: "crates/runmat-builtins/src/catalog/entries/constants/core.rs",
      }],
    },
    {
      product_id: "catalog-constants-core",
      path: "crates/runmat-builtins/src/catalog/entries/constants/core.rs",
      state: "present",
      children: [{ source_kind: "directory", source_path: owner }],
    },
  ] } };
  assert.doesNotThrow(() => validateAuthorityReachability(composition, identities));
  composition.effective.products[0].children[0].source_path =
    "crates/runmat-builtins/src/catalog/entries/constants/other.rs";
  assert.throws(
    () => validateAuthorityReachability(composition, identities),
    /catalog-constants does not declare a child owning .*constants\/core\.rs/,
  );
});

test("an authority equal to a registered file product requires that product to be present", () => {
  const owner = "crates/runmat-builtins/src/catalog/entries/constants/core.rs";
  const identities = new Map([["core", catalogIdentity(owner)]]);
  const composition = { effective: { products: [
    {
      product_id: "catalog-constants",
      path: "crates/runmat-builtins/src/catalog/entries/constants/mod.rs",
      state: "present",
      children: [{ source_kind: "file", source_path: owner }],
    },
    {
      product_id: "catalog-constants-core",
      path: owner,
      state: "absent",
      children: [],
    },
  ] } };
  assert.throws(
    () => validateAuthorityReachability(composition, identities),
    /catalog-constants-core is absent/,
  );
  composition.effective.products[1].state = "present";
  assert.doesNotThrow(() => validateAuthorityReachability(composition, identities));
});

function composition(childPath) {
  return { effective: { products: [{
    product_id: "runtime-math",
    path: "crates/runmat-runtime/src/builtins/math/mod.rs",
    state: "present",
    children: childPath === null ? [] : [{ source_kind: "directory", source_path: childPath }],
  }] } };
}

function identity(ownerPath, bundleId = "bundle-alpha") {
  return {
    bundle_id: bundleId,
    expected_authorities: {
      catalog_package: null,
      catalog_alias_package: null,
      catalog_constant_package: null,
    },
    implementation: {
      callable: { kind: "owned", owner_path: ownerPath },
      constant: { kind: "none", reason: "no-constant-form" },
    },
  };
}

function catalogIdentity(catalogPackage) {
  const value = identity("crates/runmat-runtime/src/builtins/constants.rs");
  value.expected_authorities.catalog_package = catalogPackage;
  value.implementation.callable = { kind: "none", reason: "no-callable-form" };
  return value;
}

function nestedComposition() {
  return { effective: { products: [
    product(
      "runtime-root",
      "crates/runmat-runtime/src/builtins/mod.rs",
      "crates/runmat-runtime/src/builtins/logical/mod.rs",
    ),
    product(
      "runtime-logical",
      "crates/runmat-runtime/src/builtins/logical/mod.rs",
      "crates/runmat-runtime/src/builtins/logical/relational/mod.rs",
    ),
    product(
      "runtime-logical-relational",
      "crates/runmat-runtime/src/builtins/logical/relational/mod.rs",
      "crates/runmat-runtime/src/builtins/logical/relational/eq/mod.rs",
    ),
  ] } };
}

function product(productId, productPath, childPath) {
  return {
    product_id: productId,
    path: productPath,
    state: "present",
    children: [{
      source_kind: childPath.endsWith("/mod.rs") ? "directory" : "file",
      source_path: childPath,
    }],
  };
}
