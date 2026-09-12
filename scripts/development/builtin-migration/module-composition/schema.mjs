import { compareCodePoint } from "../constants.mjs";
import { array, boolean, enumValue, exact, kind, repositoryPath, stableId } from "../schema.mjs";
import { conditionKey, parseCompositionCondition } from "./condition.mjs";
import {
  canonicalChildSourcePath, childPathAttribute, rustItemIdentifier,
  rustModuleIdentifier, rustParentModule, validateChildPath, validateParentPath,
} from "./rust-schema.mjs";

export { canonicalChildSourcePath, childPathAttribute, rustItemIdentifier, rustModuleIdentifier } from "./rust-schema.mjs";
export const CRATE_ROLES = Object.freeze(["catalog", "runtime"]);
export const CHILD_ROLES = Object.freeze(["group", "identity", "support"]);
export const VISIBILITIES = Object.freeze(["private", "super", "crate", "catalog", "public"]);
export const AGGREGATION_ROLES = Object.freeze(["entries", "aliases", "constants"]);
export const AGGREGATION_SOURCES = Object.freeze(["slice", "groups", "function"]);

export function parseModuleCompositionProjection(value) {
  kind(value, 3, "runmat-builtin-module-composition-projection", "module composition projection");
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
  const aggregations = array(value.aggregations, `${productId} aggregations`, { empty: true }).map((entry) => enumValue(entry, AGGREGATION_ROLES, `${productId} aggregation`));
  aggregationRoleOrder(aggregations, `${productId} aggregations`);
  if (crateRole === "runtime" && aggregations.length) throw new Error(`${productId}: runtime products cannot declare catalog aggregation`);
  const children = array(value.children, `${productId} children`, { empty: true }).map((entry) => parseCompositionChild(entry, { productId, crateRole, productPath, modulePath, aggregations }));
  canonicalUnique(children, (entry) => entry.module, `${productId} child modules`, true);
  caseFoldUnique(children, (entry) => entry.source_path, `${productId} child source paths`);
  validateAggregationOrders(children, aggregations, productId);
  return { ...value, product_id: productId, crate_role: crateRole, path: productPath, module_path: modulePath, aggregations, children };
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
  exact(value, ["module", "source_kind", "source_path", "role", "visibility", "declaration_condition", "macro_use", "reexports", "aggregation_sources"], `${parent.productId} child`);
  const module = rustModuleIdentifier(value.module, `${parent.productId} child module`);
  const sourceKind = enumValue(value.source_kind, ["file", "directory"], `${module} source kind`);
  const sourcePath = repositoryPath(value.source_path, `${module} source path`);
  const role = enumValue(value.role, CHILD_ROLES, `${module} child role`);
  const visibility = parseVisibility(value.visibility, parent.crateRole, `${module} visibility`);
  const declarationCondition = parseCompositionCondition(value.declaration_condition, `${module} declaration condition`);
  const macroUse = boolean(value.macro_use, `${module} macro use`);
  const reexports = array(value.reexports, `${module} reexports`, { empty: true }).map((entry) => parseReexport(entry, parent.crateRole, module));
  canonicalUnique(reexports, reexportKey, `${module} reexports`);
  const aggregationSources = array(value.aggregation_sources, `${module} aggregation sources`, { empty: true }).map((entry) => parseAggregationSource(entry, module, parent.aggregations));
  aggregationRoleOrder(aggregationSources.map((entry) => entry.role), `${module} aggregation sources`);
  validateChildPath(sourcePath, sourceKind, module, parent.productPath);
  if (parent.crateRole === "runtime" && aggregationSources.length) throw new Error(`${module}: runtime children cannot declare catalog aggregation`);
  if (role === "support" && aggregationSources.length) throw new Error(`${module}: support children cannot contribute catalog aggregation`);
  if (macroUse && role !== "support") throw new Error(`${module}: macro use is restricted to support children`);
  return { ...value, module, source_kind: sourceKind, source_path: sourcePath, role, visibility, declaration_condition: declarationCondition, macro_use: macroUse, reexports, aggregation_sources: aggregationSources };
}

function parseAggregationSource(value, module, parentAggregations) {
  exact(value, ["role", "kind", "order"], `${module} aggregation source`);
  const role = enumValue(value.role, AGGREGATION_ROLES, `${module} aggregation role`);
  const sourceKind = enumValue(value.kind, AGGREGATION_SOURCES, `${module} aggregation source kind`);
  if (!parentAggregations.includes(role)) throw new Error(`${module}: ${role} is not emitted by its parent`);
  if (sourceKind === "groups" && role !== "entries") throw new Error(`${module}: grouped aggregation is supported only for catalog entries`);
  if (!Number.isSafeInteger(value.order) || value.order < 0) throw new Error(`${module} aggregation order must be a nonnegative integer`);
  return { role, kind: sourceKind, order: value.order };
}

function parseVisibility(value, crateRole, label) {
  const visibility = enumValue(value, VISIBILITIES, label);
  if (crateRole === "runtime" && visibility === "catalog") throw new Error(`${label} cannot use catalog visibility in the runtime crate`);
  return visibility;
}

function parseReexport(value, crateRole, module) {
  const fields = value?.kind === "glob" ? ["kind", "visibility", "condition", "doc_hidden"] : ["kind", "visibility", "condition", "doc_hidden", "items"];
  exact(value, fields, `${module} reexport`);
  if (!["glob", "named"].includes(value.kind)) throw new Error(`${module} reexport has an unsupported kind`);
  const visibility = parseVisibility(value.visibility, crateRole, `${module} reexport visibility`);
  const condition = parseCompositionCondition(value.condition, `${module} reexport condition`);
  const docHidden = boolean(value.doc_hidden, `${module} reexport doc-hidden`);
  if (value.kind === "glob") return { kind: "glob", visibility, condition, doc_hidden: docHidden };
  const items = array(value.items, `${module} reexport items`).map((entry) => parseReexportItem(entry, module));
  canonicalUnique(items, reexportItemKey, `${module} reexport items`, true);
  caseFoldUnique(items, (entry) => entry.alias ?? entry.name, `${module} exported item names`);
  return { kind: "named", visibility, condition, doc_hidden: docHidden, items };
}

function parseReexportItem(value, module) {
  exact(value, ["name", "alias"], `${module} reexport item`);
  return { name: rustItemIdentifier(value.name, `${module} reexport item name`), alias: value.alias === null ? null : rustItemIdentifier(value.alias, `${module} reexport item alias`) };
}

function reexportItemKey(value) { return `${value.name}\0${value.alias ?? ""}`; }
function reexportKey(value) {
  const items = value.kind === "named" ? value.items.map(reexportItemKey).join("\0") : "";
  return `${conditionKey(value.condition)}\0${value.visibility}\0${value.doc_hidden ? 1 : 0}\0${value.kind}\0${items}`;
}

function validateAggregationOrders(children, roles, productId) {
  for (const role of roles) {
    const orders = children.flatMap((child) => child.aggregation_sources.filter((source) => source.role === role).map((source) => source.order));
    const expected = Array.from({ length: orders.length }, (_, index) => index);
    if (new Set(orders).size !== orders.length || JSON.stringify([...orders].sort((a, b) => a - b)) !== JSON.stringify(expected)) throw new Error(`${productId} ${role} aggregation orders must be unique and contiguous from zero`);
  }
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

function aggregationRoleOrder(roles, label) {
  if (new Set(roles).size !== roles.length) throw new Error(`${label} must be unique`);
  const expected = AGGREGATION_ROLES.filter((role) => roles.includes(role));
  if (JSON.stringify(roles) !== JSON.stringify(expected)) throw new Error(`${label} must use entries, aliases, constants order`);
}
