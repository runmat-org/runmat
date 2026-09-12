import assert from "node:assert/strict";
import test from "node:test";

import { validateBundleGraph } from "../control-graph.mjs";

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

test("identical integration-owned products may be required by independent bundles", () => {
  const { bundles, identities } = fixture();
  assert.doesNotThrow(() => validateBundleGraph(bundles, identities));

  bundles.get("beta").integration_outputs[0] = {
    ...bundles.get("beta").integration_outputs[0], product_id: "other-product",
  };
  assert.throws(() => validateBundleGraph(bundles, identities), /conflicting declarations/);
});

test("integration-owned products remain outside every authored lease", () => {
  const { bundles, identities } = fixture();
  bundles.get("beta").authored_write_set = [{ kind: "tree", path: "crates/runmat-runtime/src/builtins" }];
  assert.throws(() => validateBundleGraph(bundles, identities), /overlaps/);
});

test("an alias in another bundle has an explicit semantic dependency on its canonical identity", () => {
  const { bundles, identities } = fixture();
  identities.get("beta").public_identity = {
    kind: "alias",
    alias_spelling: { identity: "beta", spelling: "beta" },
    canonical_identity: "alpha",
  };
  assert.throws(
    () => validateBundleGraph(bundles, identities),
    /canonical bundle as a semantic prerequisite/,
  );
  bundles.get("beta").prerequisites = [{ bundle_id: "alpha", kind: "semantic" }];
  assert.doesNotThrow(() => validateBundleGraph(bundles, identities));
  bundles.get("beta").prerequisites[0].kind = "cohort";
  assert.throws(
    () => validateBundleGraph(bundles, identities),
    /canonical bundle as a semantic prerequisite/,
  );
});
