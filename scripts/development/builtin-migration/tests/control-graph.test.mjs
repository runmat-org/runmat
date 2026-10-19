import assert from "node:assert/strict";
import test from "node:test";

import { validateBundleGraph } from "../control-graph.mjs";
import {
  deriveIntegrationProductExclusions, pathAllowed, scopesOverlap,
} from "../path-scope.mjs";

const PRODUCT = {
  product_id: "wasm-registry",
  path: "crates/runmat-runtime/src/builtins/generated_wasm_registry.rs",
  producer: "integration",
};

function fixture() {
  const bundles = new Map([
    ["alpha", { id: "alpha", identities: ["alpha"], prerequisites: [], authored_write_set: [{ kind: "file", path: "crates/alpha.rs" }], integration_outputs: [{ kind: "file", ...PRODUCT }] }],
    ["beta", { id: "beta", identities: ["beta"], prerequisites: [], authored_write_set: [{ kind: "file", path: "crates/beta.rs" }], integration_outputs: [{ kind: "file", ...PRODUCT }] }],
  ]);
  const identities = new Map([
    ["alpha", { identity: "alpha", bundle_id: "alpha", cohort: "C01" }],
    ["beta", { identity: "beta", bundle_id: "beta", cohort: "C01" }],
  ]);
  return { bundles, identities };
}

function products() {
  return new Map([[PRODUCT.product_id, PRODUCT]]);
}

test("identical integration-owned products may be required by independent bundles", () => {
  const { bundles, identities } = fixture();
  assert.doesNotThrow(() => validateBundleGraph(bundles, identities, products()));

  bundles.get("beta").integration_outputs[0] = {
    ...bundles.get("beta").integration_outputs[0], product_id: "other-product",
  };
  assert.throws(() => validateBundleGraph(bundles, identities, products()), /conflicting declarations/);
});

test("integration-owned products remain outside every authored lease", () => {
  const { bundles, identities } = fixture();
  bundles.get("beta").authored_write_set = [{ kind: "tree", path: "crates/runmat-runtime/src/builtins" }];
  assert.throws(() => validateBundleGraph(bundles, identities, products()), /exclusions are not exactly derived/);
});

test("an alias in another bundle has an explicit semantic dependency on its canonical identity", () => {
  const { bundles, identities } = fixture();
  identities.get("beta").public_identity = {
    kind: "alias",
    alias_spelling: { identity: "beta", spelling: "beta" },
    canonical_identity: "alpha",
  };
  assert.throws(
    () => validateBundleGraph(bundles, identities, products()),
    /canonical bundle as a semantic prerequisite/,
  );
  bundles.get("beta").prerequisites = [{ bundle_id: "alpha", kind: "semantic" }];
  assert.doesNotThrow(() => validateBundleGraph(bundles, identities, products()));
  bundles.get("beta").prerequisites[0].kind = "cohort";
  assert.throws(
    () => validateBundleGraph(bundles, identities, products()),
    /canonical bundle as a semantic prerequisite/,
  );
});

test("effective tree scopes deny exact integration products and allow sibling leaves", () => {
  const products = new Map([[PRODUCT.product_id, PRODUCT]]);
  const [scope] = deriveIntegrationProductExclusions([
    { kind: "tree", path: "crates/runmat-runtime/src/builtins" },
  ], products);
  assert.deepEqual(scope, {
    kind: "tree",
    path: "crates/runmat-runtime/src/builtins",
    excluded_files: [PRODUCT.path],
  });
  assert.equal(pathAllowed([scope], PRODUCT.path), false);
  assert.equal(pathAllowed([scope], "crates/runmat-runtime/src/builtins/math/foo.rs"), true);
  assert.equal(pathAllowed([scope], "CRATES/runmat-runtime/src/builtins/math/foo.rs"), false);
  assert.equal(pathAllowed([scope], "crates/runmat-runtime/src/builtins/../outside.rs"), false);
  assert.equal(pathAllowed([scope], PRODUCT.path.toUpperCase()), false);
  assert.equal(scopesOverlap(scope, { kind: "file", path: PRODUCT.path }), false);
  assert.equal(scopesOverlap(scope, {
    kind: "file", path: "crates/runmat-runtime/src/builtins/math/foo.rs",
  }), true);
});

test("bundle graph rejects undeclared, missing, and case-fold-colliding exclusions", () => {
  const { bundles, identities } = fixture();
  bundles.get("alpha").authored_write_set = [{
    kind: "tree",
    path: "crates/runmat-runtime/src/builtins",
    excluded_files: ["crates/runmat-runtime/src/builtins/not-a-product.rs"],
  }];
  assert.throws(() => validateBundleGraph(bundles, identities, products()), /exclusions are not exactly derived/);

  bundles.get("alpha").authored_write_set[0].excluded_files = [PRODUCT.path.toUpperCase()];
  assert.throws(() => validateBundleGraph(bundles, identities, products()), /exclusions are not exactly derived/);

  bundles.get("alpha").authored_write_set[0].excluded_files = [PRODUCT.path];
  assert.doesNotThrow(() => validateBundleGraph(bundles, identities, products()));

  bundles.get("beta").integration_outputs[0] = {
    ...bundles.get("beta").integration_outputs[0],
    product_id: "casefold-product",
    path: PRODUCT.path.toUpperCase(),
  };
  const caseFoldedProducts = products();
  caseFoldedProducts.set("casefold-product", bundles.get("beta").integration_outputs[0]);
  assert.throws(
    () => validateBundleGraph(bundles, identities, caseFoldedProducts),
    /case-fold/,
  );
});

test("graph validation derives exclusions from the complete reviewed product registry", () => {
  const { bundles, identities } = fixture();
  const baselineOnly = {
    product_id: "baseline-parent",
    path: "crates/runmat-runtime/src/builtins/math/mod.rs",
    producer: "integration",
  };
  const products = new Map([
    [PRODUCT.product_id, PRODUCT],
    [baselineOnly.product_id, baselineOnly],
  ]);
  bundles.get("alpha").authored_write_set = deriveIntegrationProductExclusions([
    { kind: "tree", path: "crates/runmat-runtime/src/builtins/math" },
  ], products);
  assert.doesNotThrow(() => validateBundleGraph(bundles, identities, products));
  delete bundles.get("alpha").authored_write_set[0].excluded_files;
  assert.throws(
    () => validateBundleGraph(bundles, identities, products),
    /exclusions are not exactly derived/,
  );
});
