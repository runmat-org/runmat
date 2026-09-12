import { compareCodePoint } from "../constants.mjs";
import {
  array, boolean, enumValue, exact, kind, repositoryPath, stableId,
} from "../schema.mjs";
import {
  canonicalChildSourcePath, childPathAttribute, rustItemIdentifier,
  rustModuleIdentifier, rustParentModule, validateChildPath, validateParentPath,
} from "./rust-schema.mjs";

export {
  canonicalChildSourcePath, childPathAttribute, rustItemIdentifier, rustModuleIdentifier,
} from "./rust-schema.mjs";

export const CRATE_ROLES = Object.freeze(["catalog", "runtime"]);
export const CHILD_ROLES = Object.freeze(["group", "identity", "support"]);
export const VISIBILITIES = Object.freeze(["private", "super", "crate", "catalog", "public"]);
export const FEATURE_POLICIES = Object.freeze(["always", "cargo-feature", "test"]);
export const AGGREGATION_ROLES = Object.freeze(["entries", "aliases", "constants"]);
export const AGGREGATION_SOURCES = Object.freeze(["slice", "groups", "function"]);

const FEATURE = /^[a-zA-Z0-9][a-zA-Z0-9_+.-]*$/;

export function parseModuleCompositionProjection(value) {
  kind(value, 2, "runmat-builtin-module-composition-projection", "module composition projection");
  exact(value, ["schema_version", "kind", "products"], "module composition projection");
  const products = array(value.products, "module composition products").map(parseCompositionProduct);
  canonicalUnique(products, (entry) => entry.product_id, "module composition product ids", true);
  caseFoldUnique(products, (entry) => entry.path, "module composition product paths");
  caseFoldUnique(products, (entry) => entry.module_path, "module composition parent modules");
  return { ...value, products };
}

export function parseCompositionProduct(value) {
  exact(value, ["product_id", "crate_role", "path", "module_path", "aggregations", "children"], "module composition product");
  const productId = stableId(value.product_id, "module composition product id");
  const crateRole = enumValue(value.crate_role, CRATE_ROLES, `${productId} crate role`);
  const productPath = repositoryPath(value.path, `${productId} product path`);
  const modulePath = rustParentModule(value.module_path, `${productId} parent module`);
  validateParentPath(crateRole, productPath, modulePath, productId);
  const aggregations = array(value.aggregations, `${productId} aggregations`, { empty: true })
    .map((entry) => enumValue(entry, AGGREGATION_ROLES, `${productId} aggregation`));
  aggregationOrder(aggregations, `${productId} aggregations`);
  if (crateRole === "runtime" && aggregations.length) {
    throw new Error(`${productId}: runtime products cannot declare catalog aggregation`);
  }
  const children = array(value.children, `${productId} children`, { empty: true })
    .map((entry) => parseCompositionChild(entry, {
      productId, crateRole, productPath, modulePath, aggregations,
    }));
  canonicalUnique(children, (entry) => entry.module, `${productId} child modules`, true);
  caseFoldUnique(children, (entry) => entry.source_path, `${productId} child source paths`);
  return {
    ...value, product_id: productId, crate_role: crateRole, path: productPath,
    module_path: modulePath, aggregations, children,
  };
}

export function parseModuleCompositionContract(value, productPath, productId) {
  exact(value, ["kind", "crate_role", "module_path"], `${productId} module composition contract`);
  if (value.kind !== "rust_module_composition") throw new Error(`${productId}: module composition contract has an invalid kind`);
  const crateRole = enumValue(value.crate_role, CRATE_ROLES, `${productId} crate role`);
  const modulePath = rustParentModule(value.module_path, `${productId} parent module`);
  validateParentPath(crateRole, productPath, modulePath, productId);
  return { kind: value.kind, crate_role: crateRole, module_path: modulePath };
}

export function parseCompositionChild(value, parent) {
  exact(value, ["module", "source_kind", "source_path", "role", "visibility", "feature_policy", "macro_use", "reexport", "aggregation_sources"], `${parent.productId} child`);
  const module = rustModuleIdentifier(value.module, `${parent.productId} child module`);
  const sourceKind = enumValue(value.source_kind, ["file", "directory"], `${module} source kind`);
  const sourcePath = repositoryPath(value.source_path, `${module} source path`);
  const role = enumValue(value.role, CHILD_ROLES, `${module} child role`);
  const visibility = parseVisibility(value.visibility, parent.crateRole, `${module} visibility`);
  const featurePolicy = parseFeaturePolicy(value.feature_policy, `${module} feature policy`);
  const macroUse = boolean(value.macro_use, `${module} macro use`);
  const reexport = parseReexport(value.reexport, parent.crateRole, module);
  const aggregationSources = array(value.aggregation_sources, `${module} aggregation sources`, { empty: true })
    .map((entry) => parseAggregationSource(entry, module, parent.aggregations));
  aggregationOrder(aggregationSources.map((entry) => entry.role), `${module} aggregation sources`);
  validateChildPath(sourcePath, sourceKind, module, parent.productPath);
  if (parent.crateRole === "runtime" && aggregationSources.length) throw new Error(`${module}: runtime children cannot declare catalog aggregation`);
  if (role === "support" && aggregationSources.length) throw new Error(`${module}: support children cannot contribute catalog aggregation`);
  if (macroUse && role !== "support") throw new Error(`${module}: macro use is restricted to support children`);
  return { ...value, module, source_kind: sourceKind, source_path: sourcePath, role, visibility, feature_policy: featurePolicy, macro_use: macroUse, reexport, aggregation_sources: aggregationSources };
}

function parseAggregationSource(value, module, parentAggregations) {
  exact(value, ["role", "kind"], `${module} aggregation source`);
  const role = enumValue(value.role, AGGREGATION_ROLES, `${module} aggregation role`);
  const sourceKind = enumValue(value.kind, AGGREGATION_SOURCES, `${module} aggregation source kind`);
  if (!parentAggregations.includes(role)) throw new Error(`${module}: ${role} is not emitted by its parent`);
  if (sourceKind === "groups" && role !== "entries") {
    throw new Error(`${module}: grouped aggregation is supported only for catalog entries`);
  }
  return { role, kind: sourceKind };
}

function parseVisibility(value, crateRole, label) {
  const visibility = enumValue(value, VISIBILITIES, label);
  if (crateRole === "runtime" && visibility === "catalog") throw new Error(`${label} cannot use catalog visibility in the runtime crate`);
  return visibility;
}

function parseFeaturePolicy(value, label) {
  if (value?.kind === "always") exact(value, ["kind"], label);
  else if (value?.kind === "test") exact(value, ["kind"], label);
  else if (value?.kind === "cargo-feature") {
    exact(value, ["kind", "feature"], label);
    if (typeof value.feature !== "string" || !FEATURE.test(value.feature)) throw new Error(`${label} has an invalid Cargo feature`);
  } else throw new Error(`${label} has an unsupported kind`);
  return value;
}

function parseReexport(value, crateRole, module) {
  if (value?.kind === "none") { exact(value, ["kind"], `${module} reexport`); return value; }
  exact(value, value?.kind === "glob" ? ["kind", "visibility"] : ["kind", "visibility", "items"], `${module} reexport`);
  if (!["glob", "named"].includes(value.kind)) throw new Error(`${module} reexport has an unsupported kind`);
  parseVisibility(value.visibility, crateRole, `${module} reexport visibility`);
  if (value.kind === "named") {
    const items = array(value.items, `${module} reexport items`).map((entry) => rustItemIdentifier(entry, `${module} reexport item`));
    canonicalUnique(items, (entry) => entry, `${module} reexport items`, true);
  }
  return value;
}

function canonicalUnique(values, key, label, caseFold = false) {
  const keys = values.map(key);
  if (new Set(keys).size !== keys.length) throw new Error(`${label} must be unique`);
  if (caseFold && new Set(keys.map((entry) => entry.toLowerCase())).size !== keys.length) throw new Error(`${label} collide case-insensitively`);
  if (JSON.stringify(keys) !== JSON.stringify([...keys].sort(compareCodePoint))) throw new Error(`${label} must use canonical code-point order`);
}

function caseFoldUnique(values, key, label) {
  const keys = values.map(key).map((entry) => entry.toLowerCase());
  if (new Set(keys).size !== keys.length) throw new Error(`${label} collide case-insensitively`);
}

function aggregationOrder(roles, label) {
  if (new Set(roles).size !== roles.length) throw new Error(`${label} must be unique`);
  const expected = AGGREGATION_ROLES.filter((role) => roles.includes(role));
  if (JSON.stringify(roles) !== JSON.stringify(expected)) {
    throw new Error(`${label} must use entries, aliases, constants order`);
  }
}
