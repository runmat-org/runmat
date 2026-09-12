import { compareCodePoint } from "../constants.mjs";
import { deepImmutable } from "../immutable.mjs";

export const MODULE_COMPOSITION_SUFFIXES = Object.freeze([
  "acceleration",
  "argument_validation",
  "array",
  "array/creation",
  "cells",
  "common",
  "comms",
  "constants",
  "containers",
  "control",
  "datetime",
  "deep_learning",
  "diagnostics",
  "fea",
  "finance",
  "function_handles",
  "geometry",
  "graph",
  "image",
  "interop",
  "introspection",
  "io",
  "io/repl_fs",
  "logical",
  "math",
  "math/elementwise",
  "math/linalg",
  "math/reduction",
  "math/signal",
  "math/trigonometry",
  "objects",
  "objects/test_support",
  "parallel",
  "plotting",
  "stats",
  "stats/ml",
  "stats/summary",
  "strings",
  "strings/queries",
  "strings/search",
  "strings/text_analytics",
  "strings/transform",
  "structs",
  "table",
  "testing",
  "testing/plugins",
  "testing/runner",
  "timing",
]);

const PRODUCTS = deepImmutable([
  fixedProduct("catalog-aliases", "catalog", "catalog/aliases", ["aliases"]),
  fixedProduct("catalog-root", "catalog", "catalog/entries", ["entries", "constants"]),
  fixedProduct("runtime-root", "runtime", "builtins", []),
  ...MODULE_COMPOSITION_SUFFIXES.flatMap(pairedProducts),
].sort((left, right) => compareCodePoint(left.product_id, right.product_id)));

export function moduleCompositionProductRegistry() {
  return PRODUCTS;
}

function pairedProducts(suffix) {
  const slug = suffix.replaceAll("/", "-").replaceAll("_", "-");
  const catalogAggregations = ["array", "array/creation"].includes(suffix)
    ? ["entries", "constants"]
    : suffix === "constants" ? ["constants"] : ["entries"];
  return [
    fixedProduct(`catalog-${slug}`, "catalog", `catalog/entries/${suffix}`, catalogAggregations),
    fixedProduct(`runtime-${slug}`, "runtime", `builtins/${suffix}`, []),
  ];
}

function fixedProduct(productId, crateRole, relativePath, aggregations) {
  const crate = crateRole === "catalog" ? "runmat-builtins" : "runmat-runtime";
  return {
    product_id: productId,
    crate_role: crateRole,
    path: `crates/${crate}/src/${relativePath}/mod.rs`,
    module_path: `crate::${relativePath.replaceAll("/", "::")}`,
    aggregations,
  };
}
