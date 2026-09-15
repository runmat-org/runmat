import { compareCodePoint } from "../constants.mjs";
import { conditionKey, parseConditionAttribute } from "./condition.mjs";
import { parseHandwrittenAggregation } from "./handwritten-aggregation.mjs";

const IDENTIFIER = "(?:r#)?[A-Za-z_][A-Za-z0-9_]*";
const DECLARATION = new RegExp(`^(?:(pub\\(super\\)|pub\\(crate\\)|pub\\(in crate::catalog\\)|pub)\\s+)?mod\\s+(${IDENTIFIER})\\s*;$`);
const USE = /^(?:(pub(?:\([^)]*\))?)\s+)?use\s+([\s\S]+);$/;
const AGGREGATION_EXPORTS = Object.freeze({
  extend_entries: "entries", extend_aliases: "aliases", extend_constants: "constants",
});

export function parseHandwrittenComposition(source, productId) {
  if (typeof source !== "string" || source.includes("\0")) throw new Error(`${productId}: parent source contains invalid text`);
  const surface = { declarations: [], reexports: [], aggregation_exports: [], aggregations: [] };
  for (const item of topLevelItems(stripComments(source, productId), productId)) {
    const parsed = parseItem(item, productId);
    if (parsed.kind === "declaration") surface.declarations.push(parsed.value);
    else if (parsed.kind === "reexport") surface.reexports.push(parsed.value);
    else if (parsed.kind === "aggregation_export") surface.aggregation_exports.push(parsed.value);
    else surface.aggregations.push(parsed.value);
  }
  surface.reexports.sort((left, right) => compareCodePoint(reexportSurfaceKey(left), reexportSurfaceKey(right)));
  surface.aggregation_exports.sort((left, right) => compareCodePoint(
    aggregationExportKey(left), aggregationExportKey(right),
  ));
  return surface;
}

function reexportSurfaceKey(value) {
  const payload = value.reexport.kind === "named"
    ? value.reexport.items.map((item) => `${item.name}\0${item.alias ?? ""}`).join("\0")
    : value.reexport.kind === "module" ? value.reexport.alias ?? "" : "";
  return `${value.module}\0${conditionKey(value.condition)}\0${value.visibility}\0${value.doc_hidden ? 1 : 0}\0${value.reexport.kind}\0${payload}`;
}

function aggregationExportKey(value) {
  const roleOrder = { entries: 0, aliases: 1, constants: 2 }[value.role];
  return `${roleOrder}\0${value.module}\0${conditionKey(value.condition)}\0${value.visibility}\0${value.doc_hidden ? 1 : 0}`;
}

function parseItem(item, productId) {
  const { attributes, body } = leadingAttributes(item, productId);
  const declaration = DECLARATION.exec(body);
  if (declaration) return { kind: "declaration", value: parseDeclaration(declaration, attributes, productId) };
  const use = USE.exec(body);
  if (use) {
    const value = parseReexport(use, attributes, productId);
    const aggregationExport = asAggregationExport(value);
    return aggregationExport === null
      ? { kind: "reexport", value }
      : { kind: "aggregation_export", value: aggregationExport };
  }
  if (/^(?:pub(?:\([^)]*\))?\s+)?fn\s+extend_/.test(body)) {
    rejectAttributes(attributes, [], "aggregation function", productId);
    return { kind: "aggregation", value: parseHandwrittenAggregation(body, productId) };
  }
  throw new Error(`${productId}: unsupported handwritten Rust syntax: ${body.split("\n", 1)[0].slice(0, 120)}`);
}

function parseDeclaration(match, attributes, productId) {
  rejectAttributes(attributes, ["condition", "path", "macro_use"], "module declaration", productId);
  return {
    module: match[2], visibility: visibility(match[1]),
    declaration_condition: single(attributes, "condition", { kind: "always" }, productId),
    path_attribute: single(attributes, "path", null, productId),
    macro_use: single(attributes, "macro_use", false, productId),
  };
}

function parseReexport(match, attributes, productId) {
  const parsedVisibility = visibility(match[1]);
  if (parsedVisibility === undefined) throw new Error(`${productId}: module reexport visibility is not representable`);
  rejectAttributes(attributes, ["condition", "doc_hidden"], "module reexport", productId);
  const condition = single(attributes, "condition", { kind: "always" }, productId);
  const docHidden = single(attributes, "doc_hidden", false, productId);
  const target = match[2].trim().replace(/^self::/, "");
  let parsed = new RegExp(`^(${IDENTIFIER})::\\*$`).exec(target);
  if (parsed) return reexport(parsed[1], parsedVisibility, condition, docHidden, { kind: "glob" });
  parsed = new RegExp(`^(${IDENTIFIER})::\\{([\\s\\S]*)\\}$`).exec(target);
  if (parsed) return reexport(parsed[1], parsedVisibility, condition, docHidden, { kind: "named", items: namedItems(parsed[2], productId) });
  parsed = new RegExp(`^(${IDENTIFIER})::(${IDENTIFIER})(?:\\s+as\\s+(${IDENTIFIER}))?$`).exec(target);
  if (parsed) return reexport(parsed[1], parsedVisibility, condition, docHidden, { kind: "named", items: [{ name: parsed[2], alias: parsed[3] ?? null }] });
  parsed = new RegExp(`^(${IDENTIFIER})(?:\\s+as\\s+(${IDENTIFIER}))?$`).exec(target);
  if (parsed) return reexport(parsed[1], parsedVisibility, condition, docHidden, { kind: "module", alias: parsed[2] ?? null });
  throw new Error(`${productId}: module reexport uses an unrepresentable path or item form`);
}

function namedItems(body, productId) {
  const values = body.split(",").map((entry) => entry.trim()).filter(Boolean).map((entry) => {
    const match = new RegExp(`^(${IDENTIFIER})(?:\\s+as\\s+(${IDENTIFIER}))?$`).exec(entry);
    if (!match) throw new Error(`${productId}: named module reexport item is invalid`);
    return { name: match[1], alias: match[2] ?? null };
  });
  if (!values.length) throw new Error(`${productId}: named module reexport must contain an item`);
  return values.sort((left, right) => {
    const leftKey = `${left.name}\0${left.alias ?? ""}`;
    const rightKey = `${right.name}\0${right.alias ?? ""}`;
    return leftKey < rightKey ? -1 : leftKey > rightKey ? 1 : 0;
  });
}

function reexport(module, parsedVisibility, condition, docHidden, value) {
  return { module, visibility: parsedVisibility, condition, doc_hidden: docHidden, reexport: value };
}

function asAggregationExport(value) {
  if (value.reexport.kind !== "named" || value.reexport.items.length !== 1) return null;
  const item = value.reexport.items[0];
  if (item.alias !== null || !Object.hasOwn(AGGREGATION_EXPORTS, item.name)) return null;
  return {
    role: AGGREGATION_EXPORTS[item.name], module: value.module,
    visibility: value.visibility, condition: value.condition, doc_hidden: value.doc_hidden,
  };
}

function leadingAttributes(item, productId) {
  const attributes = [];
  let body = item.trim();
  while (body.startsWith("#[")) {
    const end = body.indexOf("]");
    if (end < 0) throw new Error(`${productId}: unterminated composition attribute`);
    attributes.push(parseAttribute(body.slice(0, end + 1), productId));
    body = body.slice(end + 1).trim();
  }
  return { attributes, body };
}

function parseAttribute(value, productId) {
  const condition = parseConditionAttribute(value);
  if (condition !== null) return { kind: "condition", value: condition };
  if (value === "#[macro_use]") return { kind: "macro_use", value: true };
  if (value === "#[doc(hidden)]") return { kind: "doc_hidden", value: true };
  const path = /^#\[path = "([A-Za-z0-9_./-]+\.rs)"\]$/.exec(value);
  if (path) return { kind: "path", value: path[1] };
  throw new Error(`${productId}: unsupported composition attribute ${value}`);
}

function rejectAttributes(attributes, allowed, context, productId) {
  for (const attribute of attributes) if (!allowed.includes(attribute.kind)) throw new Error(`${productId}: ${attribute.kind} attribute is invalid on ${context}`);
  const kinds = attributes.map((entry) => entry.kind);
  if (new Set(kinds).size !== kinds.length) throw new Error(`${productId}: ${context} has duplicate attributes`);
}

function single(attributes, kind, fallback, productId) {
  const values = attributes.filter((entry) => entry.kind === kind);
  if (values.length > 1) throw new Error(`${productId}: duplicate ${kind} attribute`);
  return values[0]?.value ?? fallback;
}

function topLevelItems(source, productId) {
  const items = [];
  let start = 0; let braces = 0; let string = false; let escape = false;
  for (let index = 0; index < source.length; index += 1) {
    const value = source[index];
    if (string) { if (escape) escape = false; else if (value === "\\") escape = true; else if (value === '"') string = false; continue; }
    if (value === '"') { string = true; continue; }
    if (value === "{") braces += 1;
    else if (value === "}") {
      braces -= 1;
      if (braces < 0) throw new Error(`${productId}: unmatched closing brace`);
      const next = source.slice(index + 1).match(/\S/)?.[0] ?? null;
      if (braces === 0 && next !== ";") { take(items, source.slice(start, index + 1)); start = index + 1; }
    }
    else if (value === ";" && braces === 0) { take(items, source.slice(start, index + 1)); start = index + 1; }
  }
  if (braces || string) throw new Error(`${productId}: unterminated handwritten Rust syntax`);
  if (source.slice(start).trim()) throw new Error(`${productId}: unsupported trailing handwritten Rust syntax`);
  return items;
}

function stripComments(source, productId) {
  let output = ""; let string = false; let escape = false;
  for (let index = 0; index < source.length;) {
    if (string) { const value = source[index]; output += value; index += 1; if (escape) escape = false; else if (value === "\\") escape = true; else if (value === '"') string = false; }
    else if (source[index] === '"') { string = true; output += source[index]; index += 1; }
    else if (source.startsWith("//", index)) { const end = source.indexOf("\n", index); if (end < 0) break; output += "\n"; index = end + 1; }
    else if (source.startsWith("/*", index)) { const end = source.indexOf("*/", index + 2); if (end < 0) throw new Error(`${productId}: unterminated block comment`); output += source.slice(index, end + 2).replace(/[^\n]/g, ""); index = end + 2; }
    else { output += source[index]; index += 1; }
  }
  return output;
}

function visibility(value) { return { undefined: "private", "pub(super)": "super", "pub(crate)": "crate", "pub(in crate::catalog)": "catalog", pub: "public" }[String(value)]; }
function take(items, value) { if (value.trim()) items.push(value.trim()); }
