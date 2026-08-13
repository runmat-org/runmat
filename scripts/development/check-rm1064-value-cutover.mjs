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
  for (const symbol of relocation.symbols) {
    if (nonValueSymbols.has(symbol)) fail(`non-value symbol ${symbol} is classified more than once`);
    nonValueSymbols.add(symbol);
  }
  if (relocation.status === "complete") {
    if (!Array.isArray(relocation.resolved_symbols) || relocation.resolved_symbols.length !== relocation.symbols.length) {
      fail(`completed relocation for ${relocation.target} must resolve every source symbol`);
      continue;
    }
    const resolvedSources = new Set();
    for (const resolution of relocation.resolved_symbols) {
      for (const field of ["source", "target", "path"]) {
        if (typeof resolution[field] !== "string" || resolution[field].length === 0) {
          fail(`completed relocation ${JSON.stringify(resolution)} has invalid ${field}`);
        }
      }
      resolvedSources.add(resolution.source);
      const targetPath = path.join(repo, resolution.path);
      if (!fs.existsSync(targetPath)) {
        fail(`completed relocation target does not exist: ${resolution.path}`);
      } else if (!fs.readFileSync(targetPath, "utf8").includes(resolution.target)) {
        fail(`completed relocation target ${resolution.target} is absent from ${resolution.path}`);
      }
    }
    for (const symbol of relocation.symbols) {
      if (!resolvedSources.has(symbol)) fail(`completed relocation does not resolve ${symbol}`);
    }
    continue;
  }
  const source = path.join(repo, relocation.source);
  if (!fs.existsSync(source)) fail(`non-value source does not exist: ${relocation.source}`);
  const sourceText = fs.existsSync(source) ? fs.readFileSync(source, "utf8") : "";
  for (const symbol of relocation.symbols) {
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
  const legacyQualifiedNames = [...declaredSymbols]
    .map(escapeRegex)
    .sort((lhs, rhs) => rhs.length - lhs.length)
    .join("|");
  for (const symbol of declaredSymbols) {
    const reexport = new RegExp(`\\bpub\\s+use\\b[^;]*\\b${escapeRegex(symbol)}\\b`);
    if (reexport.test(builtins)) fail(`runmat-builtins re-exports extracted live-value symbol ${symbol}`);
  }
  const legacyQualified = new RegExp(`\\brunmat_builtins::(?:${legacyQualifiedNames})\\b`);
  for (const { file, text } of rustSources) {
    if (legacyQualified.test(text)) {
      fail(`${file} still addresses an extracted live-value symbol through runmat_builtins`);
    }
  }

  const valueManifest = fs.readFileSync(path.join(repo, "crates/runmat-value/Cargo.toml"), "utf8");
  const forbiddenDependencies = [
    "runmat-builtins", "runmat-runtime", "runmat-hir", "runmat-mir", "runmat-vm",
    "runmat-core", "runmat-accelerate", "runmat-filesystem", "runmat-process-host",
  ];
  for (const dependency of forbiddenDependencies) {
    if (new RegExp(`^${escapeRegex(dependency)}\\s*=`, "m").test(valueManifest)) {
      fail(`runmat-value has forbidden upward dependency ${dependency}`);
    }
  }
}

if ((manifest.completed_slices ?? []).includes("R04")) {
  const runtimeManifest = fs.readFileSync(path.join(repo, "crates/runmat-runtime/Cargo.toml"), "utf8");
  const hirManifest = fs.readFileSync(path.join(repo, "crates/runmat-hir/Cargo.toml"), "utf8");
  const typesManifest = fs.readFileSync(path.join(repo, "crates/runmat-types/Cargo.toml"), "utf8");
  for (const [owner, manifestText, forbidden] of [
    ["runmat-runtime", runtimeManifest, ["runmat-hir"]],
    ["runmat-hir", hirManifest, ["runmat-runtime", "runmat-value"]],
    ["runmat-types", typesManifest, ["runmat-value", "runmat-builtins", "runmat-runtime", "runmat-hir", "runmat-mir", "runmat-vm", "runmat-execution", "runmat-accelerate-api", "runmat-gc-api"]],
  ]) {
    for (const dependency of forbidden) {
      if (new RegExp(`^${escapeRegex(dependency)}\\s*=`, "m").test(manifestText)) {
        fail(`${owner} has forbidden R04 dependency ${dependency}`);
      }
    }
  }
  for (const { file, text } of rustSources) {
    if (file.startsWith("crates/runmat-runtime/") && /\brunmat_hir\b/.test(text)) {
      fail(`${file} imports frontend HIR vocabulary after R04`);
    }
    if (file.startsWith("crates/runmat-hir/") && /\brunmat_runtime\b/.test(text)) {
      fail(`${file} imports runtime session state after R04`);
    }
  }
  const builtins = rustSources
    .filter(({ file }) => file.startsWith("crates/runmat-builtins/"))
    .map(({ text }) => text)
    .join("\n");
  for (const symbol of ["ClassDef", "PropertyDef", "MethodDef", "CLASS_REGISTRY", "STATIC_VALUES", "ENUMERATION_REGISTRY"]) {
    if (new RegExp(`\\b${escapeRegex(symbol)}\\b`).test(builtins)) {
      fail(`runmat-builtins retains R04 class/session authority ${symbol}`);
    }
  }
  const valueSources = rustSources
    .filter(({ file }) => file.startsWith("crates/runmat-value/"))
    .map(({ text }) => text)
    .join("\n");
  if (/\b(?:type|enum)\s+Access\b/.test(valueSources) || /\bpub\s+use\b[^;]*\bMemberAccess\b/.test(valueSources)) {
    fail("runmat-value retains a member-access declaration alias or re-export after R04");
  }
}

if ((manifest.completed_slices ?? []).includes("R05")) {
  const requiredModules = [
    "crates/runmat-types/src/identity/operation.rs",
    "crates/runmat-types/src/contract/call.rs",
    "crates/runmat-types/src/contract/diagnostic.rs",
    "crates/runmat-types/src/contract/effects.rs",
    "crates/runmat-types/src/contract/indexing.rs",
    "crates/runmat-types/src/contract/literal.rs",
    "crates/runmat-types/src/contract/mutation.rs",
    "crates/runmat-types/src/contract/output.rs",
    "crates/runmat-types/src/rules/aggregate.rs",
    "crates/runmat-types/src/rules/call.rs",
    "crates/runmat-types/src/rules/indexing.rs",
    "crates/runmat-types/src/rules/literal.rs",
    "crates/runmat-types/src/rules/member.rs",
    "crates/runmat-types/src/rules/mutation.rs",
    "crates/runmat-types/src/rules/operator.rs",
    "crates/runmat-types/src/rules/shape.rs",
  ];
  for (const module of requiredModules) {
    if (!fs.existsSync(path.join(repo, module))) fail(`canonical R05 module is absent: ${module}`);
  }

  const canonicalDeclarations = new Map([
    ["OperatorKind", "crates/runmat-types/src/identity/operation.rs"],
    ["RequestedOutputCount", "crates/runmat-types/src/contract/output.rs"],
    ["IndexKind", "crates/runmat-types/src/contract/indexing.rs"],
    ["IndexResultContext", "crates/runmat-types/src/contract/indexing.rs"],
    ["LiteralValue", "crates/runmat-types/src/contract/literal.rs"],
    ["PlaceMutationKind", "crates/runmat-types/src/contract/mutation.rs"],
    ["AssignmentCreationPolicy", "crates/runmat-types/src/contract/mutation.rs"],
    ["AssignmentShapePolicy", "crates/runmat-types/src/contract/mutation.rs"],
  ]);
  for (const [symbol, owner] of canonicalDeclarations) {
    const declaration = new RegExp(`(?:^|\\n)\\s*pub(?:\\([^)]*\\))?\\s+enum\\s+${escapeRegex(symbol)}(?:\\s*<|\\s|\\{)`);
    const matches = rustSources.filter(({ text }) => declaration.test(text)).map(({ file }) => file);
    if (matches.length !== 1 || matches[0] !== owner) {
      fail(`R05 enum ${symbol} must be declared only by ${owner}; found ${matches.join(", ") || "none"}`);
    }
  }

  const builtins = rustSources
    .filter(({ file }) => file.startsWith("crates/runmat-builtins/"))
    .map(({ text }) => text)
    .join("\n");
  if (/\bstruct\s+ResolveContext\b/.test(builtins)) {
    fail("runmat-builtins retains the pre-R05 ResolveContext declaration");
  }
  if (/\benum\s+LiteralValue\b/.test(builtins)) {
    fail("runmat-builtins retains the pre-R05 LiteralValue declaration");
  }

  const mirDataflowPath = "crates/runmat-mir/src/analysis/dataflow.rs";
  const mirDataflow = fs.readFileSync(path.join(repo, mirDataflowPath), "utf8");
  for (const authority of ["runmat_builtins::Type", "runmat_builtins::shape_rules"] ) {
    if (mirDataflow.includes(authority)) fail(`${mirDataflowPath} uses legacy inference authority ${authority}`);
  }
  for (const inference of [
    "infer_binary", "infer_call", "infer_cell_aggregate", "infer_index",
    "infer_index_mutation", "infer_literal", "infer_member_read", "infer_member_write",
    "infer_range", "infer_struct", "infer_tensor_aggregate", "infer_unary",
  ]) {
    if (!mirDataflow.includes(`${inference}(`)) {
      fail(`${mirDataflowPath} does not consume canonical R05 rule ${inference}`);
    }
  }
}

if (manifest.stage === "catalog-separated") {
  const builtinsManifest = fs.readFileSync(path.join(repo, "crates/runmat-builtins/Cargo.toml"), "utf8");
  if (/^runmat-value\s*=/m.test(builtinsManifest)) fail("runmat-builtins depends on runmat-value after catalog separation");
}

if (process.exitCode) process.exit(process.exitCode);
console.log(`RM-1064 value cutover manifest is current (${manifest.stage}; ${manifest.declarations.length} declarations; ${manifest.non_value_relocations.length} non-value relocations; ${manifest.representative_pilots.length} pilots)`);
