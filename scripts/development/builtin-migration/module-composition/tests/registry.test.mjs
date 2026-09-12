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

  const expected = [
    fixed("catalog-aliases", "catalog", "catalog/aliases", ["aliases"]),
    fixed("catalog-root", "catalog", "catalog/entries", ["entries", "constants"]),
    fixed("runtime-root", "runtime", "builtins", []),
    ...EXPECTED_SUFFIXES.flatMap((suffix) => {
      const slug = suffix.replaceAll("/", "-").replaceAll("_", "-");
      return [
        fixed(`catalog-${slug}`, "catalog", `catalog/entries/${suffix}`,
          ["array", "array/creation", "constants"].includes(suffix)
            ? ["entries", "constants"]
            : ["entries"]),
        fixed(`runtime-${slug}`, "runtime", `builtins/${suffix}`, []),
      ];
    }),
  ].sort((left, right) => left.product_id < right.product_id ? -1 : left.product_id > right.product_id ? 1 : 0);
  assert.deepEqual(products, expected);
});

test("registry preserves constants in each owning catalog domain", () => {
  const products = new Map(moduleCompositionProductRegistry()
    .map((entry) => [entry.product_id, entry]));
  for (const productId of ["catalog-array", "catalog-array-creation", "catalog-constants"]) {
    assert.deepEqual(products.get(productId).aggregations, ["entries", "constants"]);
  }
  assert.deepEqual(products.get("catalog-root").aggregations, ["entries", "constants"]);
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
    === "product_id,crate_role,path,module_path,aggregations"));
});

test("registry values are deeply immutable and never discover the filesystem", () => {
  const products = moduleCompositionProductRegistry();
  assert.throws(() => products.push({}), TypeError);
  assert.throws(() => { products[0].path = "other"; }, TypeError);
  assert.throws(() => products[0].aggregations.push("entries"), TypeError);

  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const source = fs.readFileSync(path.join(root, "registry.mjs"), "utf8");
  assert.doesNotMatch(source,
    /node:fs|node:child_process|\breaddir(?:Sync)?\s*\(|\bglob(?:Sync)?\s*\(|\bwalk(?:Dir|Sync)?\s*\(/);
});

function fixed(productId, crateRole, relativePath, aggregations) {
  const crate = crateRole === "catalog" ? "runmat-builtins" : "runmat-runtime";
  return {
    product_id: productId,
    crate_role: crateRole,
    path: `crates/${crate}/src/${relativePath}/mod.rs`,
    module_path: `crate::${relativePath.replaceAll("/", "::")}`,
    aggregations,
  };
}
