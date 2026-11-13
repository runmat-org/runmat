import assert from "node:assert/strict";
import test from "node:test";

import { validateSharedTargetPackageAuthority } from "../target-package-authority.mjs";

test("shared reviewed target packages require exact catalog and runtime product authority", () => {
  const topology = fixtureTopology();
  const products = fixtureProducts();
  assert.doesNotThrow(() => validateSharedTargetPackageAuthority(topology, products));

  const missing = new Map(products);
  missing.delete("runtime-logical-relational");
  assert.throws(
    () => validateSharedTargetPackageAuthority(topology, missing),
    /logical\/relational: shared target package used by bundles bundle-eq, bundle-ne requires integration product authority at .*logical\/relational\/mod\.rs/,
  );

  const mismatched = new Map(products);
  mismatched.get("runtime-logical-relational").verification.module_path =
    "crate::builtins::logical::rel";
  assert.throws(
    () => validateSharedTargetPackageAuthority(topology, mismatched),
    /shared target-package authority does not match runtime parent/,
  );
});

test("one bundle does not turn its target package into shared integration authority", () => {
  const topology = fixtureTopology();
  topology.bundles.delete("bundle-ne");
  assert.doesNotThrow(() => validateSharedTargetPackageAuthority(topology, new Map()));
});

function fixtureTopology() {
  const target = { domain: "logical", family: "relational" };
  return { bundles: new Map([
    ["bundle-ne", { composition: { target_packages: [target] } }],
    ["bundle-eq", { composition: { target_packages: [target] } }],
  ]) };
}

function fixtureProducts() {
  return new Map([
    product(
      "catalog-logical-relational", "catalog",
      "crates/runmat-builtins/src/catalog/entries/logical/relational/mod.rs",
      "crate::catalog::entries::logical::relational",
    ),
    product(
      "runtime-logical-relational", "runtime",
      "crates/runmat-runtime/src/builtins/logical/relational/mod.rs",
      "crate::builtins::logical::relational",
    ),
  ]);
}

function product(productId, crateRole, productPath, modulePath) {
  return [productId, {
    product_id: productId,
    path: productPath,
    verification: {
      kind: "rust_module_composition", crate_role: crateRole, module_path: modulePath,
    },
  }];
}
