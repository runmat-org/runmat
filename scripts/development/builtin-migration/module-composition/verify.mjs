import path from "node:path";

import { compareCodePoint } from "../constants.mjs";
import { conditionKey, parseConditionAttribute } from "./condition.mjs";
import { renderModuleCompositionProduct, generatedHeader } from "./generate.mjs";
import { parseCompositionProduct } from "./schema.mjs";
import { expectedCompositionSurface } from "./surface.mjs";

const IDENTIFIER = "(?:r#)?[A-Za-z_][A-Za-z0-9_]*";
const DECLARATION = new RegExp(`^(?:(pub\\(super\\)|pub\\(crate\\)|pub\\(in crate::catalog\\)|pub) )?mod (${IDENTIFIER});$`);
const GLOB_REEXPORT = new RegExp(`^(?:(pub\\(super\\)|pub\\(crate\\)|pub\\(in crate::catalog\\)|pub) )?use (${IDENTIFIER})::\\*;$`);
const NAMED_REEXPORT = new RegExp(`^(?:(pub\\(super\\)|pub\\(crate\\)|pub\\(in crate::catalog\\)|pub) )?use (${IDENTIFIER})::\\{$`);
const AGGREGATION_EXPORT = new RegExp(`^(?:(pub\\(super\\)|pub\\(crate\\)|pub\\(in crate::catalog\\)|pub) )?use (${IDENTIFIER})::(extend_entries|extend_aliases|extend_constants);$`);
const NAMED_ITEM = new RegExp(`^    (${IDENTIFIER})(?: as (${IDENTIFIER}))?,$`);
const AGGREGATION = /^pub\(super\) fn (extend_entries|extend_aliases|extend_constants)\(values: &mut Vec<(&'static )?crate::(BuiltinCatalogEntry|BuiltinCatalogAlias|BuiltinConstantCatalogEntry)>\) \{$/;
const AGGREGATION_CONFIG = Object.freeze({
  extend_entries: ["entries", "BuiltinCatalogEntry", "ENTRIES", true],
  extend_aliases: ["aliases", "BuiltinCatalogAlias", "ALIASES", true],
  extend_constants: ["constants", "BuiltinConstantCatalogEntry", "CONSTANTS", false],
});
const AGGREGATION_EXPORT_ROLES = Object.freeze({
  extend_entries: "entries", extend_aliases: "aliases", extend_constants: "constants",
});

export function verifyModuleCompositionProduct(productValue, source) {
  const product = parseCompositionProduct(productValue);
  const observed = parseGeneratedModuleComposition(source);
  const expected = expectedCompositionSurface(product);
  if (JSON.stringify(observed) !== JSON.stringify(expected)) throw new Error(`${product.product_id}: generated Rust topology differs from its typed projection`);
  if (source !== renderModuleCompositionProduct(product)) throw new Error(`${product.product_id}: generated Rust bytes are not canonical`);
  return { product_id: product.product_id, path: product.path, result: "pass" };
}

export function parseGeneratedModuleComposition(source) {
  if (typeof source !== "string" || source.includes("\r") || source.includes("\0") || !source.endsWith("\n")) throw new Error("generated module source must be canonical LF-terminated UTF-8 text");
  const lines = source.slice(0, -1).split("\n");
  const header = generatedHeader().split("\n");
  if (lines.shift() !== header[0] || lines.shift() !== header[1] || lines.shift() !== "") throw new Error("generated module header is invalid");
  const declarations = [];
  const reexports = [];
  const aggregationExports = [];
  const aggregations = [];
  let phase = "declarations";
  while (lines.length) {
    if (lines[0] === "") { lines.shift(); continue; }
    const condition = takeCondition(lines);
    const docHidden = takeDocHidden(lines);
    const pathAttribute = takePath(lines);
    const macroUse = takeMacroUse(lines);
    const line = lines.shift();
    const declaration = DECLARATION.exec(line);
    if (declaration) {
      if (phase !== "declarations" || docHidden) throw new Error("module declaration has invalid placement or attributes");
      declarations.push({ module: declaration[2], visibility: visibility(declaration[1]), declaration_condition: condition, path_attribute: pathAttribute, macro_use: macroUse });
      continue;
    }
    if (pathAttribute !== null || macroUse) throw new Error("path or macro-use attribute is not attached to a module declaration");
    const glob = GLOB_REEXPORT.exec(line);
    const named = NAMED_REEXPORT.exec(line);
    const aggregationExport = AGGREGATION_EXPORT.exec(line);
    if (aggregationExport) {
      if (phase === "aggregations") throw new Error("aggregation export appears after aggregation");
      phase = "reexports";
      aggregationExports.push({
        role: AGGREGATION_EXPORT_ROLES[aggregationExport[3]], module: aggregationExport[2],
        visibility: visibility(aggregationExport[1]), condition, doc_hidden: docHidden,
      });
      continue;
    }
    if (glob || named) {
      if (phase === "aggregations") throw new Error("module reexport appears after aggregation");
      phase = "reexports";
      reexports.push(parseReexport(lines, glob, named, condition, docHidden));
      continue;
    }
    if (docHidden || condition.kind !== "always") throw new Error("attribute is not attached to a module declaration or reexport");
    const start = AGGREGATION.exec(line);
    if (!start) throw new Error(`unsupported generated Rust syntax: ${line}`);
    phase = "aggregations";
    aggregations.push(parseAggregation(lines, start));
  }
  uniqueDeclarations(declarations);
  canonicalReexports(reexports);
  canonicalAggregationExports(aggregationExports);
  const roles = aggregations.map((entry) => entry.role);
  const expectedRoles = ["entries", "aliases", "constants"].filter((role) => roles.includes(role));
  if (new Set(roles).size !== roles.length || JSON.stringify(roles) !== JSON.stringify(expectedRoles)) throw new Error("generated module aggregations must use canonical role order");
  return { declarations, reexports, aggregation_exports: aggregationExports, aggregations };
}

function parseReexport(lines, glob, named, condition, docHidden) {
  if (glob) return { module: glob[2], visibility: visibility(glob[1]), condition, doc_hidden: docHidden, reexport: { kind: "glob" } };
  const items = [];
  while (lines[0] !== "};") {
    if (!lines.length) throw new Error("named reexport is unterminated");
    const match = NAMED_ITEM.exec(lines.shift());
    if (!match) throw new Error("named reexport item is invalid");
    items.push({ name: match[1], alias: match[2] ?? null });
  }
  lines.shift();
  const itemKeys = items.map((item) => `${item.name}\0${item.alias ?? ""}`);
  const exportedNames = items.map((item) => item.alias ?? item.name);
  if (new Set(itemKeys).size !== itemKeys.length
    || JSON.stringify(itemKeys) !== JSON.stringify([...itemKeys].sort(compareCodePoint))
    || new Set(exportedNames).size !== exportedNames.length) {
    throw new Error("named reexport items must be unique and use canonical order");
  }
  return { module: named[2], visibility: visibility(named[1]), condition, doc_hidden: docHidden, reexport: { kind: "named", items } };
}

function parseAggregation(lines, start) {
  const [role, type, constant, references] = AGGREGATION_CONFIG[start[1]];
  if (start[3] !== type || Boolean(start[2]) !== references) throw new Error(`${role} aggregation value type is invalid`);
  const children = [];
  while (lines[0] !== "}") {
    if (!lines.length) throw new Error(`${role} aggregation is unterminated`);
    const condition = takeIndentedCondition(lines);
    const statement = lines.shift();
    const patterns = [
      ["slice", new RegExp(`^    values\\.extend\\((${IDENTIFIER})::${constant}\\.iter\\(\\)\\.copied\\(\\)\\);$`)],
      ["groups", new RegExp(`^    values\\.extend\\((${IDENTIFIER})::ENTRY_GROUPS\\.iter\\(\\)\\.flat_map\\(\\|group\\| group\\.iter\\(\\)\\.copied\\(\\)\\)\\);$`)],
      ["function", new RegExp(`^    (${IDENTIFIER})::${start[1]}\\(values\\);$`)],
    ];
    const parsed = patterns.flatMap(([sourceKind, pattern]) => {
      const match = pattern.exec(statement);
      return match ? [{ source_kind: sourceKind, module: match[1] }] : [];
    });
    if (parsed.length !== 1 || (role !== "entries" && parsed[0].source_kind === "groups")) throw new Error(`${role} aggregation child is invalid`);
    children.push({ ...parsed[0], condition, order: children.length });
  }
  lines.shift();
  return { role, children };
}

function takeCondition(lines) {
  const parsed = parseConditionAttribute(lines[0]);
  if (parsed === null) return { kind: "always" };
  lines.shift();
  return parsed;
}

function takeIndentedCondition(lines) {
  const line = lines[0]?.startsWith("    ") ? lines[0].slice(4) : "";
  const parsed = parseConditionAttribute(line);
  if (parsed === null) return { kind: "always" };
  lines.shift();
  return parsed;
}

function takeDocHidden(lines) { if (lines[0] !== "#[doc(hidden)]") return false; lines.shift(); return true; }
function takePath(lines) {
  const match = /^#\[path = "([A-Za-z0-9_./-]+\.rs)"\]$/.exec(lines[0]);
  if (!match) return null;
  lines.shift();
  if (match[1].startsWith("/") || match[1].includes("//") || path.posix.normalize(match[1]) !== match[1] || match[1].split("/").includes("..")) throw new Error("generated module path attribute is not a canonical descendant");
  return match[1];
}
function takeMacroUse(lines) { if (lines[0] !== "#[macro_use]") return false; lines.shift(); return true; }

function visibility(value) { return { undefined: "private", "pub(super)": "super", "pub(crate)": "crate", "pub(in crate::catalog)": "catalog", pub: "public" }[String(value)]; }
function uniqueDeclarations(values) {
  const keys = values.map((entry) => entry.module);
  if (new Set(keys.map((entry) => entry.toLowerCase())).size !== keys.length) throw new Error("generated module declarations must be unique");
}
function canonicalReexports(values) {
  let priorModule = null;
  let priorKey = null;
  for (const value of values) {
    const key = `${conditionKey(value.condition)}\0${value.visibility}\0${value.doc_hidden ? 1 : 0}\0${value.reexport.kind}\0${JSON.stringify(value.reexport)}`;
    if (priorModule !== null && (value.module < priorModule || (value.module === priorModule && key <= priorKey))) throw new Error("generated module reexports must use unique canonical order");
    priorModule = value.module;
    priorKey = key;
  }
}

function canonicalAggregationExports(values) {
  const roles = values.map((entry) => entry.role);
  const expected = ["entries", "aliases", "constants"].filter((role) => roles.includes(role));
  if (new Set(roles).size !== roles.length || JSON.stringify(roles) !== JSON.stringify(expected)) {
    throw new Error("generated aggregation exports must use unique canonical role order");
  }
}
