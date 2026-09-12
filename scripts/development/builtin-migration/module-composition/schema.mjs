import path from "node:path";

import { compareCodePoint } from "../constants.mjs";
import {
  array, boolean, enumValue, exact, kind, repositoryPath, stableId,
} from "../schema.mjs";

export const CRATE_ROLES = Object.freeze(["catalog", "runtime"]);
export const CHILD_ROLES = Object.freeze(["group", "identity", "support"]);
export const VISIBILITIES = Object.freeze(["private", "crate", "catalog", "public"]);
export const FEATURE_POLICIES = Object.freeze(["always", "cargo-feature"]);
export const AGGREGATION_ROLES = Object.freeze(["entries", "aliases", "constants"]);

const RUST_KEYWORDS = new Set([
  "Self", "abstract", "as", "async", "await", "become", "box", "break", "const",
  "continue", "crate", "do", "dyn", "else", "enum", "extern", "false", "final",
  "fn", "for", "gen", "if", "impl", "in", "let", "loop", "macro", "match",
  "mod", "move", "mut", "override", "priv", "pub", "ref", "return", "self",
  "static", "struct", "super", "trait", "true", "try", "type", "typeof", "union",
  "unsafe", "unsized", "use", "virtual", "where", "while", "yield",
]);
const UNRAWABLE_IDENTIFIERS = new Set(["Self", "_", "crate", "self", "super"]);
const FEATURE = /^[a-zA-Z0-9][a-zA-Z0-9_+.-]*$/;

export function parseModuleCompositionProjection(value) {
  kind(value, 1, "runmat-builtin-module-composition-projection", "module composition projection");
  exact(value, ["schema_version", "kind", "products"], "module composition projection");
  const products = array(value.products, "module composition products").map(parseCompositionProduct);
  canonicalUnique(products, (entry) => entry.product_id, "module composition product ids", true);
  caseFoldUnique(products, (entry) => entry.path, "module composition product paths");
  caseFoldUnique(products, (entry) => entry.module_path, "module composition parent modules");
  return { ...value, products };
}

export function parseCompositionProduct(value) {
  exact(value, ["product_id", "crate_role", "path", "module_path", "children"], "module composition product");
  const productId = stableId(value.product_id, "module composition product id");
  const crateRole = enumValue(value.crate_role, CRATE_ROLES, `${productId} crate role`);
  const productPath = repositoryPath(value.path, `${productId} product path`);
  const modulePath = rustParentModule(value.module_path, `${productId} parent module`);
  validateParentPath(crateRole, productPath, modulePath, productId);
  const children = array(value.children, `${productId} children`, { empty: true })
    .map((entry) => parseCompositionChild(entry, { productId, crateRole, productPath, modulePath }));
  canonicalUnique(children, (entry) => entry.module, `${productId} child modules`, true);
  caseFoldUnique(children, (entry) => entry.source_path, `${productId} child source paths`);
  return { ...value, product_id: productId, crate_role: crateRole, path: productPath, module_path: modulePath, children };
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
  exact(value, ["module", "source_kind", "source_path", "role", "visibility", "feature_policy", "macro_use", "reexport", "aggregation_roles"], `${parent.productId} child`);
  const module = rustModuleIdentifier(value.module, `${parent.productId} child module`);
  const sourceKind = enumValue(value.source_kind, ["file", "directory"], `${module} source kind`);
  const sourcePath = repositoryPath(value.source_path, `${module} source path`);
  const role = enumValue(value.role, CHILD_ROLES, `${module} child role`);
  const visibility = parseVisibility(value.visibility, parent.crateRole, `${module} visibility`);
  const featurePolicy = parseFeaturePolicy(value.feature_policy, `${module} feature policy`);
  const macroUse = boolean(value.macro_use, `${module} macro use`);
  const reexport = parseReexport(value.reexport, parent.crateRole, module);
  const aggregationRoles = array(value.aggregation_roles, `${module} aggregation roles`, { empty: true })
    .map((entry) => enumValue(entry, AGGREGATION_ROLES, `${module} aggregation role`));
  canonicalUnique(aggregationRoles, (entry) => entry, `${module} aggregation roles`);
  validateChildPath(sourcePath, sourceKind, module, parent.productPath);
  if (parent.crateRole === "runtime" && aggregationRoles.length) throw new Error(`${module}: runtime children cannot declare catalog aggregation`);
  if (role === "support" && aggregationRoles.length) throw new Error(`${module}: support children cannot contribute catalog aggregation`);
  if (macroUse && role !== "support") throw new Error(`${module}: macro use is restricted to support children`);
  return { ...value, module, source_kind: sourceKind, source_path: sourcePath, role, visibility, feature_policy: featurePolicy, macro_use: macroUse, reexport, aggregation_roles: aggregationRoles };
}

export function rustModuleIdentifier(value, label) {
  if (typeof value !== "string" || !/^(?:r#)?[A-Za-z_][A-Za-z0-9_]*$/.test(value)) throw new Error(`${label} must be a canonical Rust module identifier`);
  const raw = value.startsWith("r#");
  const bare = raw ? value.slice(2) : value;
  if (UNRAWABLE_IDENTIFIERS.has(bare)) throw new Error(`${label} uses an identifier Rust reserves in module paths`);
  if (raw !== RUST_KEYWORDS.has(bare)) throw new Error(`${label} must use r# exactly for a Rust keyword`);
  return value;
}

export function rustItemIdentifier(value, label) {
  if (typeof value !== "string" || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(value) || RUST_KEYWORDS.has(value)) throw new Error(`${label} must be a non-keyword Rust item identifier`);
  return value;
}

function parseVisibility(value, crateRole, label) {
  const visibility = enumValue(value, VISIBILITIES, label);
  if (crateRole === "runtime" && visibility === "catalog") throw new Error(`${label} cannot use catalog visibility in the runtime crate`);
  return visibility;
}

function parseFeaturePolicy(value, label) {
  if (value?.kind === "always") exact(value, ["kind"], label);
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

function rustParentModule(value, label) {
  if (typeof value !== "string" || !value.startsWith("crate::")) throw new Error(`${label} must start with crate::`);
  value.split("::").slice(1).forEach((entry) => rustModuleIdentifier(entry, label));
  return value;
}

function validateParentPath(crateRole, productPath, modulePath, id) {
  const root = crateRole === "catalog" ? "crates/runmat-builtins/src" : "crates/runmat-runtime/src";
  const requiredRoot = crateRole === "catalog" ? "crate::catalog" : "crate::builtins";
  if (!(modulePath === requiredRoot || modulePath.startsWith(`${requiredRoot}::`))) throw new Error(`${id}: parent module is outside its crate role`);
  const relative = modulePath.split("::").slice(1).map(stripRaw).join("/");
  if (productPath !== `${root}/${relative}/mod.rs`) throw new Error(`${id}: product path does not match its logical parent module`);
}

function validateChildPath(sourcePath, sourceKind, module, productPath) {
  const directory = path.posix.dirname(productPath);
  if (sourcePath === productPath || !sourcePath.startsWith(`${directory}/`)) throw new Error(`${module}: child source is outside its reviewed parent`);
  const relative = sourcePath.slice(directory.length + 1);
  if (sourceKind === "file" ? (!relative.endsWith(".rs") || relative.endsWith("/mod.rs")) : !relative.endsWith("/mod.rs")) {
    throw new Error(`${module}: child source does not match its declared source kind`);
  }
}

function stripRaw(value) { return value.startsWith("r#") ? value.slice(2) : value; }

export function canonicalChildSourcePath(productPath, child) {
  const directory = path.posix.dirname(productPath);
  const leaf = stripRaw(child.module);
  return child.source_kind === "file" ? `${directory}/${leaf}.rs` : `${directory}/${leaf}/mod.rs`;
}

export function childPathAttribute(productPath, child) {
  if (child.source_path === canonicalChildSourcePath(productPath, child)) return null;
  return path.posix.relative(path.posix.dirname(productPath), child.source_path);
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
