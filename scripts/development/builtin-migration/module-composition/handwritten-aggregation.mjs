import { parseConditionAttribute } from "./condition.mjs";

const IDENTIFIER = "(?:r#)?[A-Za-z_][A-Za-z0-9_]*";
const START = new RegExp(`^pub\\(super\\)\\s+fn\\s+(extend_entries|extend_aliases|extend_constants)\\s*\\(\\s*(${IDENTIFIER})\\s*:\\s*&mut\\s+Vec<(&'static\\s+)?crate::(BuiltinCatalogEntry|BuiltinCatalogAlias|BuiltinConstantCatalogEntry)>\\s*\\)\\s*\\{([\\s\\S]*)\\}$`);
const CONFIG = Object.freeze({ extend_entries: ["entries", "BuiltinCatalogEntry", "ENTRIES", true], extend_aliases: ["aliases", "BuiltinCatalogAlias", "ALIASES", true], extend_constants: ["constants", "BuiltinConstantCatalogEntry", "CONSTANTS", false] });

export function parseHandwrittenAggregation(body, productId) {
  const match = START.exec(body);
  if (!match) throw new Error(`${productId}: aggregation signature is not representable`);
  const [role, type, constant, references] = CONFIG[match[1]];
  if (match[4] !== type || Boolean(match[3]) !== references) throw new Error(`${productId}: ${role} aggregation value type is invalid`);
  const children = [];
  for (const raw of statements(match[5], productId)) {
    const { condition, statement } = statementCondition(raw, productId);
    for (const addition of contribution(statement, match[2], match[1], constant, role, productId)) children.push({ ...addition, condition, order: children.length });
  }
  return { role, children };
}

function contribution(statement, parameter, functionName, constant, role, productId) {
  const escaped = escapeRegExp(parameter);
  let match = new RegExp(`^${escaped}\\.extend\\((${IDENTIFIER})::${constant}\\.iter\\(\\)\\.copied\\(\\)\\)$`).exec(statement);
  if (match) return [{ source_kind: "slice", module: match[1] }];
  match = new RegExp(`^${escaped}\\.extend_from_slice\\((${IDENTIFIER})::${constant}\\)$`).exec(statement);
  if (match) return [{ source_kind: "slice", module: match[1] }];
  match = new RegExp(`^${escaped}\\.extend\\((${IDENTIFIER})::ENTRY_GROUPS\\.iter\\(\\)\\.flat_map\\(\\|group\\|group\\.iter\\(\\)\\.copied\\(\\)\\)\\)$`).exec(statement);
  if (match && role === "entries") return [{ source_kind: "groups", module: match[1] }];
  match = new RegExp(`^(${IDENTIFIER})::${functionName}\\(${escaped},?\\)$`).exec(statement);
  if (match) return [{ source_kind: "function", module: match[1] }];
  match = new RegExp(`^super::extend_groups\\(${escaped},(${IDENTIFIER})::ENTRY_GROUPS,?\\)$`).exec(statement);
  if (match && role === "entries") return [{ source_kind: "groups", module: match[1] }];
  match = new RegExp(`^super::extend_groups\\(${escaped},&\\[([\\s\\S]*)\\],?\\)$`).exec(statement);
  if (match && role === "entries") {
    const modules = match[1].split(",").map((entry) => entry.trim()).filter(Boolean).map((entry) => {
      const child = new RegExp(`^(${IDENTIFIER})::ENTRIES$`).exec(entry);
      if (!child) throw new Error(`${productId}: grouped entries aggregation child is invalid`);
      return { source_kind: "slice", module: child[1] };
    });
    if (!modules.length) throw new Error(`${productId}: grouped entries aggregation must contain a child`);
    return modules;
  }
  throw new Error(`${productId}: ${role} aggregation statement is not representable: ${statement.slice(0, 160)}`);
}

function statements(body, productId) {
  const result = []; let start = 0; let brackets = 0; let parentheses = 0;
  for (let index = 0; index < body.length; index += 1) {
    if (body[index] === "[") brackets += 1; else if (body[index] === "]") brackets -= 1; else if (body[index] === "(") parentheses += 1; else if (body[index] === ")") parentheses -= 1;
    else if (body[index] === ";" && brackets === 0 && parentheses === 0) { const value = body.slice(start, index).trim(); if (value) result.push(value); start = index + 1; }
    if (brackets < 0 || parentheses < 0) throw new Error(`${productId}: malformed aggregation body`);
  }
  if (brackets || parentheses || body.slice(start).trim()) throw new Error(`${productId}: aggregation body contains unterminated or non-statement syntax`);
  return result;
}

function statementCondition(value, productId) {
  if (!value.startsWith("#[")) return { condition: { kind: "always" }, statement: compact(value) };
  const end = value.indexOf("]"); const attribute = value.slice(0, end + 1); const condition = parseConditionAttribute(attribute);
  if (condition === null) throw new Error(`${productId}: aggregation statement has unsupported attribute`);
  const statement = value.slice(end + 1).trim();
  if (statement.startsWith("#[")) throw new Error(`${productId}: aggregation statement has duplicate attributes`);
  return { condition, statement: compact(statement) };
}

function compact(value) {
  return value.replace(/\s+/g, " ").replace(/\s*([(),.\[\]&|])\s*/g, "$1").replace(/,\)/g, ")");
}
function escapeRegExp(value) { return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }
