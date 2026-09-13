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

function composition(childPath) {
  return { effective: { products: [{
    product_id: "runtime-math",
    path: "crates/runmat-runtime/src/builtins/math/mod.rs",
    state: "present",
    children: childPath === null ? [] : [{ source_path: childPath }],
  }] } };
}

function identity(ownerPath) {
  return {
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
