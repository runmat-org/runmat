import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  MODULE_COMPOSITION_SUFFIXES, moduleCompositionProductRegistry,
} from "../index.mjs";

const EXPECTED_SUFFIXES = [
  "acceleration", "argument_validation", "array", "array/creation", "cells", "common",
  "comms", "constants", "containers", "control", "datetime", "deep_learning", "diagnostics",
  "fea", "finance", "function_handles", "geometry", "graph", "image", "interop",
  "introspection", "io", "io/repl_fs", "logical", "math", "math/elementwise", "math/linalg",
  "math/reduction", "math/signal", "math/trigonometry", "objects", "objects/test_support",
  "parallel", "plotting", "stats", "stats/ml", "stats/summary", "strings", "strings/queries",
  "strings/search", "strings/text_analytics", "strings/transform", "structs", "table", "testing",
  "testing/plugins", "testing/runner", "timing",
];

test("registry declares the exact canonical 99-product census", () => {
  assert.deepEqual(MODULE_COMPOSITION_SUFFIXES, EXPECTED_SUFFIXES);
  const products = moduleCompositionProductRegistry();
  assert.equal(products.length, 99);
  assert.equal(products.filter((entry) => entry.crate_role === "catalog").length, 50);
  assert.equal(products.filter((entry) => entry.crate_role === "runtime").length, 49);

  const ids = new Set(products.map((entry) => entry.product_id));
  assert.ok(ids.has("catalog-aliases"));
  assert.ok(ids.has("catalog-root"));
  assert.ok(ids.has("runtime-root"));
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

test("registry identifiers and paths are globally unique", () => {
  const products = moduleCompositionProductRegistry();
  for (const field of ["product_id", "path", "module_path"]) {
    const values = products.map((entry) => entry[field]);
    assert.equal(new Set(values).size, 99, `${field} must be unique`);
    assert.equal(new Set(values.map((entry) => entry.toLowerCase())).size, 99,
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
