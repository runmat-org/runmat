import { compareCodePoint } from "../constants.mjs";
import { array, boolean, enumValue, exact, kind, repositoryPath, stableId } from "../schema.mjs";
import { conditionImplies, conditionKey, parseCompositionCondition } from "./condition.mjs";
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
export const PRODUCT_STATES = Object.freeze(["absent", "present"]);

export function parseModuleCompositionProjection(value) {
  kind(value, 5, "runmat-builtin-module-composition-projection", "module composition projection");
  exact(value, ["schema_version", "kind", "products"], "module composition projection");
  const products = array(value.products, "module composition products").map(parseCompositionProduct);
  canonicalUnique(products, (entry) => entry.product_id, "module composition product ids", true);
  caseFoldUnique(products, (entry) => entry.path, "module composition product paths");
  caseFoldUnique(products, (entry) => entry.module_path, "module composition parent modules");
  return { ...value, products };
}

export function parseCompositionProduct(value) {
  exact(value, ["product_id", "crate_role", "path", "module_path", "state", "aggregations", "aggregation_exports", "children"], "module composition product");
  const productId = stableId(value.product_id, "module composition product id");
  const crateRole = enumValue(value.crate_role, CRATE_ROLES, `${productId} crate role`);
  const productPath = repositoryPath(value.path, `${productId} product path`);
  const modulePath = rustParentModule(value.module_path, `${productId} parent module`);
  const state = enumValue(value.state, PRODUCT_STATES, `${productId} product state`);
  validateParentPath(crateRole, productPath, modulePath, productId);
  const aggregations = array(value.aggregations, `${productId} aggregations`, { empty: true }).map((entry) => enumValue(entry, AGGREGATION_ROLES, `${productId} aggregation`));
  aggregationRoleOrder(aggregations, `${productId} aggregations`);
  if (crateRole === "runtime" && aggregations.length) throw new Error(`${productId}: runtime products cannot declare catalog aggregation`);
  const aggregationExports = array(value.aggregation_exports, `${productId} aggregation exports`, { empty: true }).map((entry) => parseAggregationExport(entry, crateRole, productId, aggregations));
  aggregationRoleOrder(aggregationExports.map((entry) => entry.role), `${productId} aggregation exports`);
  const localAggregations = aggregations.filter((role) => !aggregationExports.some((entry) => entry.role === role));
  const children = array(value.children, `${productId} children`, { empty: true }).map((entry) => parseCompositionChild(entry, { productId, crateRole, productPath, modulePath, aggregations: localAggregations }));
  canonicalUnique(children, (entry) => entry.module, `${productId} child modules`, true);
  caseFoldUnique(children, (entry) => entry.source_path, `${productId} child source paths`);
  validateDeclarationOrders(children, productId);
  if (state === "present") validateAggregationExports(aggregationExports, children, productId);
  validateAggregationOrders(children, localAggregations, productId);
  return { ...value, product_id: productId, crate_role: crateRole, path: productPath, module_path: modulePath, state, aggregations, aggregation_exports: aggregationExports, children };
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
  exact(value, ["module", "source_kind", "source_path", "role", "visibility", "declaration_condition", "declaration_order", "macro_use", "reexports", "aggregation_sources"], `${parent.productId} child`);
  const module = rustModuleIdentifier(value.module, `${parent.productId} child module`);
  const sourceKind = enumValue(value.source_kind, ["file", "directory"], `${module} source kind`);
  const sourcePath = repositoryPath(value.source_path, `${module} source path`);
  const role = enumValue(value.role, CHILD_ROLES, `${module} child role`);
  const visibility = parseVisibility(value.visibility, parent.crateRole, `${module} visibility`);
  const declarationCondition = parseCompositionCondition(value.declaration_condition, `${module} declaration condition`);
  if (!Number.isSafeInteger(value.declaration_order) || value.declaration_order < 0) throw new Error(`${module} declaration order must be a nonnegative integer`);
  const macroUse = boolean(value.macro_use, `${module} macro use`);
  const reexports = array(value.reexports, `${module} reexports`, { empty: true })
    .map((entry) => parseCompositionReexport(entry, parent.crateRole, module));
  canonicalUnique(reexports, compositionReexportKey, `${module} reexports`);
  const aggregationSources = array(value.aggregation_sources, `${module} aggregation sources`, { empty: true }).map((entry) => parseAggregationSource(entry, module, parent.aggregations));
  aggregationRoleOrder(aggregationSources.map((entry) => entry.role), `${module} aggregation sources`);
  validateChildPath(sourcePath, sourceKind, module, parent.productPath);
  if (parent.crateRole === "runtime" && aggregationSources.length) throw new Error(`${module}: runtime children cannot declare catalog aggregation`);
  if (role === "support" && aggregationSources.length) throw new Error(`${module}: support children cannot contribute catalog aggregation`);
  for (const reexport of reexports) if (!conditionImplies(reexport.condition, declarationCondition)) {
    throw new Error(`${module}: reexport condition must imply its declaration condition`);
  }
  for (const source of aggregationSources) if (!conditionImplies(source.condition, declarationCondition)) {
    throw new Error(`${module}: aggregation condition must imply its declaration condition`);
  }
  return { ...value, module, source_kind: sourceKind, source_path: sourcePath, role, visibility, declaration_condition: declarationCondition, declaration_order: value.declaration_order, macro_use: macroUse, reexports, aggregation_sources: aggregationSources };
}

function parseAggregationSource(value, module, parentAggregations) {
  exact(value, ["role", "kind", "order", "condition"], `${module} aggregation source`);
  const role = enumValue(value.role, AGGREGATION_ROLES, `${module} aggregation role`);
  const sourceKind = enumValue(value.kind, AGGREGATION_SOURCES, `${module} aggregation source kind`);
  if (!parentAggregations.includes(role)) throw new Error(`${module}: ${role} is not emitted by its parent`);
  if (sourceKind === "groups" && role !== "entries") throw new Error(`${module}: grouped aggregation is supported only for catalog entries`);
  if (!Number.isSafeInteger(value.order) || value.order < 0) throw new Error(`${module} aggregation order must be a nonnegative integer`);
  const condition = parseCompositionCondition(value.condition, `${module} ${role} aggregation condition`);
  return { role, kind: sourceKind, order: value.order, condition };
}

function parseAggregationExport(value, crateRole, productId, parentAggregations) {
  exact(value, ["role", "module", "visibility", "condition", "doc_hidden"], `${productId} aggregation export`);
  const role = enumValue(value.role, AGGREGATION_ROLES, `${productId} aggregation export role`);
  if (!parentAggregations.includes(role)) throw new Error(`${productId}: ${role} aggregation export is not declared by its parent`);
  return {
    role,
    module: rustModuleIdentifier(value.module, `${productId} aggregation export module`),
    visibility: parseVisibility(value.visibility, crateRole, `${productId} aggregation export visibility`),
    condition: parseCompositionCondition(value.condition, `${productId} aggregation export condition`),
    doc_hidden: boolean(value.doc_hidden, `${productId} aggregation export doc-hidden`),
  };
}

function parseVisibility(value, crateRole, label) {
  const visibility = enumValue(value, VISIBILITIES, label);
  if (crateRole === "runtime" && visibility === "catalog") throw new Error(`${label} cannot use catalog visibility in the runtime crate`);
  return visibility;
}

export function parseCompositionReexport(value, crateRole, module) {
  const fields = value?.kind === "glob" ? ["kind", "visibility", "condition", "doc_hidden"] : ["kind", "visibility", "condition", "doc_hidden", "items"];
  exact(value, fields, `${module} reexport`);
  if (!["glob", "named"].includes(value.kind)) throw new Error(`${module} reexport has an unsupported kind`);
  const visibility = parseVisibility(value.visibility, crateRole, `${module} reexport visibility`);
  const condition = parseCompositionCondition(value.condition, `${module} reexport condition`);
  const docHidden = boolean(value.doc_hidden, `${module} reexport doc-hidden`);
  if (value.kind === "glob") return { kind: "glob", visibility, condition, doc_hidden: docHidden };
  const items = array(value.items, `${module} reexport items`).map((entry) => parseReexportItem(entry, module));
  canonicalUnique(items, reexportItemKey, `${module} reexport items`);
  const exportedNames = items.map((entry) => entry.alias ?? entry.name);
  if (new Set(exportedNames).size !== exportedNames.length) {
    throw new Error(`${module} exported item names collide`);
  }
  return { kind: "named", visibility, condition, doc_hidden: docHidden, items };
}

function parseReexportItem(value, module) {
  exact(value, ["name", "alias"], `${module} reexport item`);
  return { name: rustItemIdentifier(value.name, `${module} reexport item name`), alias: value.alias === null ? null : rustItemIdentifier(value.alias, `${module} reexport item alias`) };
}

function reexportItemKey(value) { return `${value.name}\0${value.alias ?? ""}`; }
export function compositionReexportKey(value) {
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

function validateDeclarationOrders(children, productId) {
  const orders = children.map((child) => child.declaration_order);
  const expected = Array.from({ length: orders.length }, (_, index) => index);
  if (new Set(orders).size !== orders.length || JSON.stringify([...orders].sort((a, b) => a - b)) !== JSON.stringify(expected)) {
    throw new Error(`${productId} declaration orders must be unique and contiguous from zero`);
  }
}

function validateAggregationExports(exports, children, productId) {
  const byModule = new Map(children.map((child) => [child.module, child]));
  for (const entry of exports) {
    const child = byModule.get(entry.module);
    if (!child) throw new Error(`${productId}: ${entry.role} aggregation export references undeclared child ${entry.module}`);
    if (!conditionImplies(entry.condition, child.declaration_condition)) {
      throw new Error(`${productId}: ${entry.role} aggregation export condition must imply ${entry.module}'s declaration condition`);
    }
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
