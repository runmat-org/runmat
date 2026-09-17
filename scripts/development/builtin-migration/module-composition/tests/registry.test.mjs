import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  MODULE_COMPOSITION_SUFFIXES, moduleCompositionProductRegistry,
} from "../index.mjs";
import {
  observeWorkingTreeModuleCompositionProducts,
} from "../baseline-observation.mjs";

const REPOSITORY = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)), "../../../../..",
);

const EXPECTED_SUFFIXES = [
  "acceleration", "acceleration/gpu", "argument_validation", "array", "array/creation", "array/shape",
  "array/sorting_sets", "cells", "common", "comms", "constants", "containers", "containers/dictionary",
  "control", "datetime", "datetime/arithmetic",
  "datetime/business_calendar", "datetime/calendar_duration", "datetime/components",
  "datetime/construction", "datetime/conversion", "datetime/core", "deep_learning", "deep_learning/autodiff", "diagnostics",
  "fea", "finance", "function_handles", "geometry", "geometry/triangulation", "graph", "image", "interop",
  "introspection", "io", "io/archive", "io/filetext", "io/repl_fs", "io/tabular_reading", "logical",
  "logical/relational", "math", "math/elementwise", "math/fft", "math/linalg", "math/linalg/factor",
  "math/reduction", "math/signal", "math/symbolic", "math/trigonometry", "objects", "objects/test_support",
  "parallel", "plotting", "plotting/animation", "plotting/axes", "plotting/colormaps",
  "plotting/figure_lifecycle", "plotting/properties", "plotting/ui",
  "stats", "stats/hist", "stats/ml", "stats/random", "stats/summary", "strings", "strings/core", "strings/queries",
  "strings/search", "strings/text_analytics", "strings/transform", "structs", "table", "table/timetable", "testing",
  "testing/plugins", "testing/runner", "timing", "timing/timer",
];

const REVIEWED_SHARED_TARGET_PACKAGES = [
  "acceleration/gpu", "array/creation", "array/shape", "array/sorting_sets", "containers/dictionary",
  "datetime/arithmetic", "datetime/calendar_duration",
  "datetime/components", "datetime/core", "io/archive", "io/filetext", "io/repl_fs",
  "io/tabular_reading", "logical/relational", "math/fft", "math/linalg/factor", "math/reduction", "math/signal",
  "math/symbolic", "objects/test_support", "plotting/animation", "plotting/axes", "plotting/colormaps",
  "plotting/figure_lifecycle", "plotting/ui", "stats/hist", "stats/ml", "stats/random", "stats/summary",
  "strings/core", "strings/queries", "strings/search", "strings/text_analytics",
  "strings/transform", "table/timetable", "testing/plugins", "testing/runner",
];

test("registry declares the exact canonical 162-product census", () => {
  assert.deepEqual(MODULE_COMPOSITION_SUFFIXES, EXPECTED_SUFFIXES);
  const products = moduleCompositionProductRegistry();
  assert.equal(products.length, 162);
  assert.equal(products.filter((entry) => entry.crate_role === "catalog").length, 81);
  assert.equal(products.filter((entry) => entry.crate_role === "runtime").length, 81);

  const ids = new Set(products.map((entry) => entry.product_id));
  assert.ok(ids.has("catalog-aliases"));
  assert.ok(ids.has("catalog-root"));
  assert.ok(ids.has("runtime-root"));
  assert.ok(ids.has("runtime-logical-rel"));
  for (const suffix of EXPECTED_SUFFIXES) {
    const slug = suffix.replaceAll("/", "-").replaceAll("_", "-");
    assert.ok(ids.has(`catalog-${slug}`));
    assert.ok(ids.has(`runtime-${slug}`));
  }
});

test("registry preserves each catalog domain's exact aggregation roles", () => {
  const products = new Map(moduleCompositionProductRegistry()
    .map((entry) => [entry.product_id, entry]));
  for (const productId of ["catalog-array", "catalog-array-creation"]) {
    assert.deepEqual(products.get(productId).aggregations, ["entries", "constants"]);
  }
  assert.deepEqual(products.get("catalog-constants").aggregations, ["constants"]);
  assert.deepEqual(products.get("catalog-root").aggregations, ["entries", "constants"]);
  assert.deepEqual(products.get("catalog-io-repl-fs").aggregation_exports, [{
    role: "entries", module: "registry", visibility: "super",
    condition: { kind: "always" }, doc_hidden: false,
  }]);
  assert.ok([...products.values()]
    .filter((entry) => entry.product_id !== "catalog-io-repl-fs")
    .every((entry) => entry.aggregation_exports.length === 0));
});

test("logical transition boundaries retain their exact catalog and runtime identities", () => {
  const products = new Map(moduleCompositionProductRegistry()
    .map((entry) => [entry.product_id, entry]));
  assert.deepEqual(products.get("catalog-logical-relational"), {
    product_id: "catalog-logical-relational",
    crate_role: "catalog",
    path: "crates/runmat-builtins/src/catalog/entries/logical/relational/mod.rs",
    module_path: "crate::catalog::entries::logical::relational",
    aggregations: ["entries"],
    aggregation_exports: [],
  });
  assert.deepEqual(products.get("runtime-logical-rel"), {
    product_id: "runtime-logical-rel",
    crate_role: "runtime",
    path: "crates/runmat-runtime/src/builtins/logical/rel/mod.rs",
    module_path: "crate::builtins::logical::rel",
    aggregations: [],
    aggregation_exports: [],
  });
  assert.deepEqual(products.get("runtime-logical-relational"), {
    product_id: "runtime-logical-relational",
    crate_role: "runtime",
    path: "crates/runmat-runtime/src/builtins/logical/relational/mod.rs",
    module_path: "crate::builtins::logical::relational",
    aggregations: [],
    aggregation_exports: [],
  });
});

test("reviewed file-to-directory promotions have first-class product authority", () => {
  const products = new Map(moduleCompositionProductRegistry()
    .map((entry) => [entry.product_id, entry]));
  for (const [productId, productPath, modulePath] of [
    [
      "runtime-datetime-construction",
      "crates/runmat-runtime/src/builtins/datetime/construction/mod.rs",
      "crate::builtins::datetime::construction",
    ],
    [
      "runtime-plotting-properties",
      "crates/runmat-runtime/src/builtins/plotting/properties/mod.rs",
      "crate::builtins::plotting::properties",
    ],
  ]) {
    assert.deepEqual(products.get(productId), {
      product_id: productId,
      crate_role: "runtime",
      path: productPath,
      module_path: modulePath,
      aggregations: [],
      aggregation_exports: [],
    });
  }
});

test("registry covers every family parent shared by the reviewed pilot and residual topology", () => {
  const paths = new Set(moduleCompositionProductRegistry().map((entry) => entry.path));
  for (const targetPackage of REVIEWED_SHARED_TARGET_PACKAGES) {
    assert.ok(paths.has(
      `crates/runmat-builtins/src/catalog/entries/${targetPackage}/mod.rs`,
    ), `${targetPackage} catalog parent`);
    assert.ok(paths.has(
      `crates/runmat-runtime/src/builtins/${targetPackage}/mod.rs`,
    ), `${targetPackage} runtime parent`);
  }
  assert.ok(paths.has("crates/runmat-runtime/src/builtins/logical/rel/mod.rs"));
});

test("registry identifiers and paths are globally unique", () => {
  const products = moduleCompositionProductRegistry();
  for (const field of ["product_id", "path", "module_path"]) {
    const values = products.map((entry) => entry[field]);
    assert.equal(new Set(values).size, 162, `${field} must be unique`);
    assert.equal(new Set(values.map((entry) => entry.toLowerCase())).size, 162,
      `${field} must be case-fold unique`);
  }
  assert.deepEqual(products.map((entry) => entry.product_id),
    [...products.map((entry) => entry.product_id)].sort());
  assert.ok(products.every((entry) => Object.keys(entry).join(",")
    === "product_id,crate_role,path,module_path,aggregations,aggregation_exports"));
});

test("registry values are deeply immutable and never discover the filesystem", () => {
  const products = moduleCompositionProductRegistry();
  assert.throws(() => products.push({}), TypeError);
  assert.throws(() => { products[0].path = "other"; }, TypeError);
  assert.throws(() => products[0].aggregations.push("entries"), TypeError);
  assert.throws(() => products.find((entry) => entry.product_id === "catalog-io-repl-fs")
    .aggregation_exports.push({}), TypeError);

  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const source = fs.readFileSync(path.join(root, "registry.mjs"), "utf8");
  assert.doesNotMatch(source,
    /node:fs|node:child_process|\breaddir(?:Sync)?\s*\(|\bglob(?:Sync)?\s*\(|\bwalk(?:Dir|Sync)?\s*\(/);
});

test("every present registered parent is composition-only and fully representable", () => {
  const observed = observeWorkingTreeModuleCompositionProducts(REPOSITORY);
  const registry = moduleCompositionProductRegistry();
  assert.equal(observed.length, registry.length);
  assert.deepEqual(
    observed.map((product) => product.product_id),
    registry.map((product) => product.product_id),
  );
  assert.ok(observed.some((product) => product.state === "present"));
  for (const product of observed) {
    assert.ok(product.state === "present" || product.state === "absent");
    if (product.state === "present") assert.ok(product.parent_evidence !== null);
    else assert.deepEqual(product.children, []);
  }
});
