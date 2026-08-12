#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const manifestPath = path.join(repo, "docs/development/rm1064-value-cutover.json");
const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));

function fail(message) {
  console.error(`RM-1064 value cutover guard failed: ${message}`);
  process.exitCode = 1;
}

function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function walk(directory) {
  const files = [];
  if (!fs.existsSync(directory)) return files;
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    if (["target", ".git", "node_modules"].includes(entry.name)) continue;
    const absolute = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...walk(absolute));
    else if (entry.isFile() && entry.name.endsWith(".rs")) files.push(absolute);
  }
  return files.sort();
}

function relative(file) {
  return path.relative(repo, file).split(path.sep).join("/");
}

function git(...args) {
  return spawnSync("git", args, { cwd: repo, encoding: "utf8" });
}

if (manifest.schema_version !== 1) fail(`unsupported schema_version ${manifest.schema_version}`);
if (!["structural-baseline", "extracted", "catalog-separated"].includes(manifest.stage)) fail(`unsupported stage ${manifest.stage}`);

for (const field of ["integer_commit", "qualified_merge", "current_owner", "target_owner"]) {
  if (typeof manifest.source?.[field] !== "string" || manifest.source[field].length === 0) {
    fail(`source.${field} must be a non-empty string`);
  }
}

for (const commit of [manifest.source.integer_commit, manifest.source.qualified_merge]) {
  const result = git("merge-base", "--is-ancestor", commit, "HEAD");
  if (result.status !== 0) fail(`${commit} is not an ancestor of HEAD`);
}

const rustFiles = walk(path.join(repo, "crates"));
const rustSources = rustFiles.map((file) => ({ file: relative(file), text: fs.readFileSync(file, "utf8") }));
const declarationKeys = new Set();
const declaredSymbols = new Set();
const intentionallyPrivate = new Set(["TensorStorage", "SparseValueStorage", "SparseCscParts"]);

for (const declaration of manifest.declarations ?? []) {
  for (const field of ["symbol", "kind", "source", "target", "family"]) {
    if (typeof declaration[field] !== "string" || declaration[field].length === 0) {
      fail(`declaration ${JSON.stringify(declaration)} has invalid ${field}`);
    }
  }
  if (!["enum", "struct", "type"].includes(declaration.kind)) {
    fail(`${declaration.symbol} has unsupported declaration kind ${declaration.kind}`);
    continue;
  }
  const key = `${declaration.kind}:${declaration.symbol}`;
  if (declarationKeys.has(key)) fail(`duplicate declaration entry ${key}`);
  declarationKeys.add(key);
  if (declaredSymbols.has(declaration.symbol)) fail(`symbol ${declaration.symbol} is assigned more than once`);
  declaredSymbols.add(declaration.symbol);

  const expectedPath = manifest.stage === "structural-baseline" ? declaration.source : declaration.target;
  const visibility = intentionallyPrivate.has(declaration.symbol) ? "" : "pub(?:\\([^)]*\\))?\\s+";
  const pattern = new RegExp(`(?:^|\\n)\\s*${visibility}${declaration.kind}\\s+${escapeRegex(declaration.symbol)}(?:\\s*<|\\s|=|\\{)`);
  const matches = rustSources.filter(({ text }) => pattern.test(text)).map(({ file }) => file);
  if (matches.length !== 1) {
    fail(`${declaration.kind} ${declaration.symbol} must have exactly one declaration; found ${matches.length}: ${matches.join(", ")}`);
  } else if (matches[0] !== expectedPath) {
    fail(`${declaration.kind} ${declaration.symbol} is in ${matches[0]}; ${manifest.stage} expects ${expectedPath}`);
  }
}

const nonValueSymbols = new Set();
for (const relocation of manifest.non_value_relocations ?? []) {
  for (const field of ["source", "target", "slice", "rule"]) {
    if (typeof relocation[field] !== "string" || relocation[field].length === 0) {
      fail(`non-value relocation ${JSON.stringify(relocation)} has invalid ${field}`);
    }
  }
  if (!Array.isArray(relocation.symbols) || relocation.symbols.length === 0) {
    fail(`non-value relocation for ${relocation.target} has no symbols`);
    continue;
  }
  const source = path.join(repo, relocation.source);
  if (!fs.existsSync(source)) fail(`non-value source does not exist: ${relocation.source}`);
  const sourceText = fs.existsSync(source) ? fs.readFileSync(source, "utf8") : "";
  for (const symbol of relocation.symbols) {
    if (nonValueSymbols.has(symbol)) fail(`non-value symbol ${symbol} is classified more than once`);
    nonValueSymbols.add(symbol);
    if (!sourceText.includes(symbol)) fail(`non-value symbol ${symbol} is not present in ${relocation.source}`);
  }
}

const catalog = JSON.parse(fs.readFileSync(path.join(repo, "docs/builtins/meta.json"), "utf8"));
const catalogNames = new Set(catalog.builtins.map((builtin) => builtin.name));
const pilotIdentities = new Set();
for (const pilot of manifest.representative_pilots ?? []) {
  if (typeof pilot.identity !== "string" || pilot.identity.length === 0) fail("pilot identity must be non-empty");
  if (pilotIdentities.has(pilot.identity)) fail(`duplicate pilot identity ${pilot.identity}`);
  pilotIdentities.add(pilot.identity);
  if (!Array.isArray(pilot.seams) || pilot.seams.length === 0) fail(`pilot ${pilot.identity} has no seams`);
  if (!catalogNames.has(pilot.identity)) fail(`pilot ${pilot.identity} is absent from the generated builtin catalog`);
}

if (manifest.stage === "structural-baseline" && fs.existsSync(path.join(repo, "crates/runmat-value"))) {
  fail("runmat-value exists while the manifest still declares structural-baseline ownership; R03 must update the manifest atomically");
}

if (["extracted", "catalog-separated"].includes(manifest.stage)) {
  const builtins = rustSources.filter(({ file }) => file.startsWith("crates/runmat-builtins/")).map(({ text }) => text).join("\n");
  for (const symbol of declaredSymbols) {
    const reexport = new RegExp(`\\bpub\\s+use\\b[^;]*\\b${escapeRegex(symbol)}\\b`);
    if (reexport.test(builtins)) fail(`runmat-builtins re-exports extracted live-value symbol ${symbol}`);
  }
}

if (manifest.stage === "catalog-separated") {
  const builtinsManifest = fs.readFileSync(path.join(repo, "crates/runmat-builtins/Cargo.toml"), "utf8");
  if (/^runmat-value\s*=/m.test(builtinsManifest)) fail("runmat-builtins depends on runmat-value after catalog separation");
}

if (process.exitCode) process.exit(process.exitCode);
console.log(`RM-1064 value cutover manifest is current (${manifest.stage}; ${manifest.declarations.length} declarations; ${manifest.non_value_relocations.length} non-value relocations; ${manifest.representative_pilots.length} pilots)`);
