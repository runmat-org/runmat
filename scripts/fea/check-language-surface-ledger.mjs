#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const ledgerPath = path.join(repo, "scripts/fea/language-surface-ledger.json");
const ledger = JSON.parse(fs.readFileSync(ledgerPath, "utf8"));
let failed = false;

function fail(message) {
  console.error(`FEA language surface ledger check failed: ${message}`);
  failed = true;
}

function itemBody(source, kind, name) {
  const pattern = new RegExp(`\\b${kind}\\s+${name}\\b[^\\{]*\\{`, "m");
  const match = pattern.exec(source);
  if (!match) throw new Error(`cannot find ${kind} ${name}`);
  const open = source.indexOf("{", match.index);
  let depth = 1;
  for (let cursor = open + 1; cursor < source.length; cursor += 1) {
    if (source[cursor] === "{") depth += 1;
    else if (source[cursor] === "}") depth -= 1;
    if (depth === 0) return source.slice(open + 1, cursor);
  }
  throw new Error(`unterminated ${kind} ${name}`);
}

function topLevelLines(body) {
  const lines = [];
  let current = "";
  let depth = 0;
  for (const rawLine of body.split("\n")) {
    const line = rawLine.trim();
    if (depth === 0 && (line.startsWith("#[") || line.startsWith("//"))) continue;
    current += `${current ? " " : ""}${line}`;
    for (const char of line) {
      if (["(", "[", "{"].includes(char)) depth += 1;
      else if ([")", "]", "}"].includes(char)) depth -= 1;
    }
    if (depth === 0 && current.endsWith(",")) {
      lines.push(current.slice(0, -1).trim());
      current = "";
    }
  }
  if (current.trim()) lines.push(current.trim());
  return lines;
}

function declaredMembers(source, kind, name) {
  const lines = topLevelLines(itemBody(source, kind, name));
  if (kind === "struct") {
    return lines
      .map((line) => /^(?:pub(?:\([^)]*\))?\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*:/.exec(line)?.[1])
      .filter(Boolean);
  }
  return lines
    .map((line) => /^([A-Za-z_][A-Za-z0-9_]*)\b/.exec(line)?.[1])
    .filter(Boolean);
}

const dispositions = new Set([
  "authored_dsl",
  "compiler_derived",
  "execution_policy",
  "internal_only",
  "removed",
]);
const requiredFacts = [
  "disposition",
  "cardinality",
  "default",
  "target",
  "identity",
  "validation",
  "positive_coverage",
  "negative_coverage",
];

if (ledger.schema !== 1) fail(`unsupported ledger schema ${ledger.schema}`);
const seen = new Set();
for (const surface of ledger.surfaces ?? []) {
  const absolute = path.join(repo, surface.source);
  if (!fs.existsSync(absolute)) {
    fail(`${surface.name} source does not exist: ${surface.source}`);
    continue;
  }
  const source = fs.readFileSync(absolute, "utf8");
  let declared;
  try {
    declared = declaredMembers(source, surface.kind, surface.name);
  } catch (error) {
    fail(error.message);
    continue;
  }
  const recorded = Object.keys(surface.members ?? {});
  for (const member of declared.filter((name) => !recorded.includes(name))) {
    fail(`${surface.name}.${member} has no reviewed disposition`);
  }
  for (const member of recorded.filter((name) => !declared.includes(name))) {
    fail(`${surface.name}.${member} is stale or misspelled`);
  }
  for (const [member, overrides] of Object.entries(surface.members ?? {})) {
    const id = `${surface.name}.${member}`;
    if (seen.has(id)) fail(`duplicate entry ${id}`);
    seen.add(id);
    const facts = { ...(surface.defaults ?? {}), ...overrides };
    for (const fact of requiredFacts) {
      if (facts[fact] === undefined || facts[fact] === "") fail(`${id} omits ${fact}`);
    }
    if (!dispositions.has(facts.disposition)) {
      fail(`${id} has unknown or unresolved disposition ${facts.disposition}`);
    }
    if (facts.positive_coverage === facts.negative_coverage) {
      fail(`${id} must name distinct positive and negative coverage`);
    }
  }
}

for (const concentration of ledger.concentration_limits ?? []) {
  const absolute = path.join(repo, concentration.source);
  const lines = fs.readFileSync(absolute, "utf8").split("\n").length;
  if (lines > concentration.maximum_lines) {
    fail(`${concentration.source} grew to ${lines} lines (limit ${concentration.maximum_lines})`);
  }
}

if ((ledger.surfaces ?? []).length === 0 || seen.size === 0) {
  fail("ledger contains no checked surfaces");
}

if (failed) process.exit(1);
console.log(`FEA language surface ledger is complete for ${seen.size} current members`);
