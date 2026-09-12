import path from "node:path";

import { compareCodePoint } from "../constants.mjs";
import { renderModuleCompositionProduct, generatedHeader } from "./generate.mjs";
import { childPathAttribute, parseCompositionProduct } from "./schema.mjs";

const CFG = /^#\[cfg\(feature = "([a-zA-Z0-9][a-zA-Z0-9_+.-]*)"\)\]$/;
const DECLARATION = /^(?:(pub\(crate\)|pub\(in crate::catalog\)|pub) )?mod ((?:r#)?[A-Za-z_][A-Za-z0-9_]*);$/;
const REEXPORT = /^(?:(pub\(crate\)|pub\(in crate::catalog\)|pub) )?use ((?:r#)?[A-Za-z_][A-Za-z0-9_]*)::(\*|\{[A-Za-z_][A-Za-z0-9_]*(?:, [A-Za-z_][A-Za-z0-9_]*)*\});$/;
const AGGREGATION = /^pub\(super\) fn (extend_entries|extend_aliases|extend_constants)\(values: &mut Vec<&'static crate::(BuiltinCatalogEntry|BuiltinCatalogAlias|BuiltinConstantCatalogEntry)>\) \{$/;
const AGGREGATION_CONFIG = Object.freeze({
  extend_entries: ["entries", "BuiltinCatalogEntry", "ENTRIES"],
  extend_aliases: ["aliases", "BuiltinCatalogAlias", "ALIASES"],
  extend_constants: ["constants", "BuiltinConstantCatalogEntry", "CONSTANTS"],
});

export function verifyModuleCompositionProduct(productValue, source) {
  const product = parseCompositionProduct(productValue);
  const observed = parseGeneratedModuleComposition(source);
  const expected = expectedSurface(product);
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
  const aggregations = [];
  let phase = "declarations";
  while (lines.length) {
    if (lines[0] === "") { lines.shift(); continue; }
    const feature = takeFeature(lines);
    const pathAttribute = takePath(lines);
    const macroUse = takeMacroUse(lines);
    const line = lines.shift();
    const declaration = DECLARATION.exec(line);
    if (declaration) {
      if (phase !== "declarations") throw new Error("module declaration appears after another generated section");
      declarations.push({ module: declaration[2], visibility: visibility(declaration[1]), feature_policy: feature, path_attribute: pathAttribute, macro_use: macroUse });
      continue;
    }
    if (pathAttribute !== null || macroUse) throw new Error("path or macro-use attribute is not attached to a module declaration");
    const reexport = REEXPORT.exec(line);
    if (reexport) {
      if (phase === "aggregations") throw new Error("module reexport appears after aggregation");
      phase = "reexports";
      reexports.push(parseReexport(reexport, feature));
      continue;
    }
    if (feature.kind !== "always") throw new Error("feature policy is not attached to a module declaration or reexport");
    const start = AGGREGATION.exec(line);
    if (!start) throw new Error(`unsupported generated Rust syntax: ${line}`);
    phase = "aggregations";
    aggregations.push(parseAggregation(lines, start));
  }
  canonicalUnique(declarations, (entry) => entry.module, "generated module declarations");
  canonicalUnique(reexports, (entry) => entry.module, "generated module reexports");
  const aggregationRoles = aggregations.map((entry) => entry.role);
  const expectedRoles = ["entries", "aliases", "constants"].filter((role) => aggregationRoles.includes(role));
  if (new Set(aggregationRoles).size !== aggregationRoles.length
    || JSON.stringify(aggregationRoles) !== JSON.stringify(expectedRoles)) {
    throw new Error("generated module aggregations must use canonical role order");
  }
  return { declarations, reexports, aggregations };
}

function parseAggregation(lines, start) {
  const [role, type, constant] = AGGREGATION_CONFIG[start[1]];
  if (start[2] !== type) throw new Error(`${role} aggregation value type is invalid`);
  const children = [];
  while (lines[0] !== "}") {
    if (!lines.length) throw new Error(`${role} aggregation is unterminated`);
    let feature = { kind: "always" };
    const match = /^    #\[cfg\(feature = "([a-zA-Z0-9][a-zA-Z0-9_+.-]*)"\)\]$/.exec(lines[0]);
    if (match) { lines.shift(); feature = { kind: "cargo-feature", feature: match[1] }; }
    const child = new RegExp(`^    values\\.extend\\(((?:r#)?[A-Za-z_][A-Za-z0-9_]*)::${constant}\\.iter\\(\\)\\.copied\\(\\)\\);$`).exec(lines.shift());
    if (!child) throw new Error(`${role} aggregation child is invalid`);
    children.push({ module: child[1], feature_policy: feature });
  }
  lines.shift();
  canonicalUnique(children, (entry) => entry.module, `${role} aggregation children`);
  return { role, children };
}

function parseReexport(match, featurePolicy) {
  const target = match[3];
  return {
    module: match[2], visibility: visibility(match[1]), feature_policy: featurePolicy,
    reexport: target === "*" ? { kind: "glob" } : { kind: "named", items: target.slice(1, -1).split(", ") },
  };
}

function takeFeature(lines) {
  const match = CFG.exec(lines[0]);
  if (!match) return { kind: "always" };
  lines.shift();
  return { kind: "cargo-feature", feature: match[1] };
}

function takePath(lines) {
  const match = /^#\[path = "([A-Za-z0-9_./-]+\.rs)"\]$/.exec(lines[0]);
  if (!match) return null;
  lines.shift();
  if (match[1].startsWith("/") || match[1].includes("//") || path.posix.normalize(match[1]) !== match[1]
    || match[1].split("/").includes("..")) throw new Error("generated module path attribute is not a canonical descendant");
  return match[1];
}

function takeMacroUse(lines) {
  if (lines[0] !== "#[macro_use]") return false;
  lines.shift();
  return true;
}

function expectedSurface(product) {
  return {
    declarations: product.children.map((entry) => ({
      module: entry.module, visibility: entry.visibility, feature_policy: entry.feature_policy,
      path_attribute: childPathAttribute(product.path, entry), macro_use: entry.macro_use,
    })),
    reexports: product.children.filter((entry) => entry.reexport.kind !== "none").map((entry) => ({
      module: entry.module, visibility: entry.reexport.visibility, feature_policy: entry.feature_policy,
      reexport: entry.reexport.kind === "glob" ? { kind: "glob" } : { kind: "named", items: entry.reexport.items },
    })),
    aggregations: product.crate_role === "runtime" ? [] : ["entries", "aliases", "constants"].flatMap((role) => {
      const children = product.children.filter((entry) => entry.aggregation_roles.includes(role)).map((entry) => ({ module: entry.module, feature_policy: entry.feature_policy }));
      return children.length ? [{ role, children }] : [];
    }),
  };
}

function visibility(value) { return { undefined: "private", "pub(crate)": "crate", "pub(in crate::catalog)": "catalog", pub: "public" }[String(value)]; }

function canonicalUnique(values, key, label) {
  const keys = values.map(key);
  if (new Set(keys).size !== keys.length || new Set(keys.map((entry) => entry.toLowerCase())).size !== keys.length) throw new Error(`${label} must be unique without case-fold collisions`);
  if (JSON.stringify(keys) !== JSON.stringify([...keys].sort(compareCodePoint))) throw new Error(`${label} must use canonical code-point order`);
}
