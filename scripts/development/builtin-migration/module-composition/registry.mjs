import { compareCodePoint } from "../constants.mjs";
import { deepImmutable } from "../immutable.mjs";

const PAIRED_PRODUCT_CONFIGURATIONS = deepImmutable(validatePairedConfigurations([
  paired("acceleration"),
  paired("acceleration/gpu"),
  paired("argument_validation"),
  paired("array", ["entries", "constants"]),
  paired("array/creation", ["entries", "constants"]),
  paired("array/shape"),
  paired("array/sorting_sets"),
  paired("cells"),
  paired("common"),
  paired("comms"),
  paired("constants", ["constants"]),
  paired("containers"),
  paired("containers/dictionary"),
  paired("control"),
  paired("datetime"),
  paired("datetime/arithmetic"),
  paired("datetime/business_calendar"),
  paired("datetime/calendar_duration"),
  paired("datetime/components"),
  paired("datetime/construction"),
  paired("datetime/conversion"),
  paired("datetime/core"),
  paired("deep_learning"),
  paired("deep_learning/autodiff"),
  paired("diagnostics"),
  paired("fea"),
  paired("finance"),
  paired("function_handles"),
  paired("geometry"),
  paired("geometry/triangulation"),
  paired("graph"),
  paired("image"),
  paired("interop"),
  paired("introspection"),
  paired("io"),
  paired("io/archive"),
  paired("io/filetext"),
  paired("io/repl_fs", ["entries"], [aggregationExport("entries", "registry")]),
  paired("io/tabular_reading"),
  paired("logical"),
  paired("logical/relational"),
  paired("math"),
  paired("math/bitwise", ["entries"]),
  paired("math/elementwise"),
  paired("math/fft"),
  paired("math/linalg"),
  paired("math/linalg/factor"),
  paired("math/reduction"),
  paired("math/signal"),
  paired("math/symbolic"),
  paired("math/trigonometry"),
  paired("objects"),
  paired("objects/test_support"),
  paired("parallel"),
  paired("plotting"),
  paired("plotting/animation"),
  paired("plotting/axes"),
  paired("plotting/colormaps"),
  paired("plotting/figures"),
  paired("plotting/graphics_objects"),
  paired("plotting/properties"),
  paired("plotting/ui"),
  paired("stats"),
  paired("stats/hist"),
  paired("stats/ml"),
  paired("stats/random"),
  paired("stats/summary"),
  paired("strings"),
  paired("strings/core"),
  paired("strings/queries"),
  paired("strings/search"),
  paired("strings/text_analytics"),
  paired("strings/transform"),
  paired("structs"),
  paired("table"),
  paired("table/timetable"),
  paired("testing"),
  paired("testing/plugins"),
  paired("testing/runner"),
  paired("timing"),
  paired("timing/timer"),
  paired("workspace"),
]));

export const MODULE_COMPOSITION_SUFFIXES = Object.freeze(
  PAIRED_PRODUCT_CONFIGURATIONS.map((entry) => entry.suffix),
);

const PRODUCTS = deepImmutable([
  fixedProduct("catalog-aliases", "catalog", "catalog/aliases", ["aliases"]),
  fixedProduct("catalog-root", "catalog", "catalog/entries", ["entries", "constants"]),
  fixedProduct("runtime-root", "runtime", "builtins", []),
  ...PAIRED_PRODUCT_CONFIGURATIONS.flatMap(pairedProducts),
  fixedProduct("runtime-logical-rel", "runtime", "builtins/logical/rel", []),
].sort((left, right) => compareCodePoint(left.product_id, right.product_id)));

export function moduleCompositionProductRegistry() {
  return PRODUCTS;
}

function pairedProducts(configuration) {
  const { suffix, catalog_aggregations: catalogAggregations, aggregation_exports: aggregationExports } = configuration;
  const slug = suffix.replaceAll("/", "-").replaceAll("_", "-");
  return [
    fixedProduct(`catalog-${slug}`, "catalog", `catalog/entries/${suffix}`, catalogAggregations, aggregationExports),
    fixedProduct(`runtime-${slug}`, "runtime", `builtins/${suffix}`, []),
  ];
}

function paired(suffix, catalogAggregations = ["entries"], aggregationExports = []) {
  return { suffix, catalog_aggregations: catalogAggregations, aggregation_exports: aggregationExports };
}

function aggregationExport(role, module) {
  return {
    role, module, visibility: "super", condition: { kind: "always" }, doc_hidden: false,
  };
}

function validatePairedConfigurations(values) {
  const suffixes = values.map((entry) => entry.suffix);
  if (new Set(suffixes).size !== suffixes.length
    || JSON.stringify(suffixes) !== JSON.stringify([...suffixes].sort(compareCodePoint))) {
    throw new Error("module composition paired-product suffixes must be unique and canonical");
  }
  for (const entry of values) {
    if (!/^[a-z][a-z0-9_]*(?:\/[a-z][a-z0-9_]*)*$/.test(entry.suffix)) {
      throw new Error(`invalid module composition suffix ${entry.suffix}`);
    }
    validateRoles(entry.catalog_aggregations, `${entry.suffix} catalog aggregations`);
    validateRoles(entry.aggregation_exports.map((item) => item.role), `${entry.suffix} aggregation exports`);
    for (const item of entry.aggregation_exports) {
      if (!entry.catalog_aggregations.includes(item.role)
        || !/^[a-z_][a-z0-9_]*$/.test(item.module)
        || JSON.stringify(Object.keys(item)) !== JSON.stringify(["role", "module", "visibility", "condition", "doc_hidden"])
        || item.visibility !== "super" || item.condition.kind !== "always" || item.doc_hidden !== false) {
        throw new Error(`${entry.suffix} has an invalid aggregation export`);
      }
    }
  }
  return values;
}

function validateRoles(values, label) {
  const expected = ["entries", "aliases", "constants"].filter((role) => values.includes(role));
  if (JSON.stringify(values) !== JSON.stringify(expected)) throw new Error(`${label} must be unique and canonical`);
}

function fixedProduct(productId, crateRole, relativePath, aggregations, aggregationExports = []) {
  const crate = crateRole === "catalog" ? "runmat-builtins" : "runmat-runtime";
  return {
    product_id: productId,
    crate_role: crateRole,
    path: `crates/${crate}/src/${relativePath}/mod.rs`,
    module_path: `crate::${relativePath.replaceAll("/", "::")}`,
    aggregations,
    aggregation_exports: aggregationExports,
  };
}
