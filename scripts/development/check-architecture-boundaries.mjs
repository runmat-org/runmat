#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
let failed = false;

function fail(message) {
  console.error(`architecture boundary check failed: ${message}`);
  failed = true;
}

function read(relativePath) {
  return fs.readFileSync(path.join(repo, relativePath), "utf8");
}

function rustSources(relativeDirectory) {
  const root = path.join(repo, relativeDirectory);
  const sources = [];
  const visit = (directory) => {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const absolute = path.join(directory, entry.name);
      if (entry.isDirectory()) visit(absolute);
      else if (entry.isFile() && entry.name.endsWith(".rs")) {
        sources.push({
          path: path.relative(repo, absolute).split(path.sep).join("/"),
          text: fs.readFileSync(absolute, "utf8"),
        });
      }
    }
  };
  visit(root);
  return sources.sort((left, right) => left.path.localeCompare(right.path));
}

function hasDependency(manifest, dependency) {
  const escaped = dependency.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`^${escaped}\\s*=`, "m").test(manifest);
}

function forbidDependencies(crateName, dependencies) {
  const manifest = read(`crates/${crateName}/Cargo.toml`);
  for (const dependency of dependencies) {
    if (hasDependency(manifest, dependency)) {
      fail(`${crateName} must not depend on ${dependency}`);
    }
  }
}

const upwardValueDependencies = [
  "runmat-builtins",
  "runmat-runtime",
  "runmat-hir",
  "runmat-mir",
  "runmat-vm",
  "runmat-core",
  "runmat-accelerate",
  "runmat-filesystem",
  "runmat-process-host",
];
forbidDependencies("runmat-value", upwardValueDependencies);
forbidDependencies("runmat-runtime", ["runmat-hir"]);
forbidDependencies("runmat-hir", ["runmat-runtime", "runmat-value"]);

const typesManifest = read("crates/runmat-types/Cargo.toml");
for (const dependency of typesManifest.matchAll(/^(runmat-[\w-]+)\s*=/gm)) {
  fail(`runmat-types must remain dependency-neutral; found ${dependency[1]}`);
}

const allRust = rustSources("crates");
const valueDeclarations = allRust
  .filter(({ text }) => /(?:^|\n)\s*pub(?:\([^)]*\))?\s+enum\s+Value(?:\s|\{|<)/.test(text))
  .map(({ path: sourcePath }) => sourcePath);
if (valueDeclarations.length !== 1 || !valueDeclarations[0].startsWith("crates/runmat-value/")) {
  fail(`Value must be declared exactly once by runmat-value; found ${valueDeclarations.join(", ") || "none"}`);
}

for (const { path: sourcePath, text } of allRust) {
  if (sourcePath.startsWith("crates/runmat-runtime/") && /\brunmat_hir\b/.test(text)) {
    fail(`${sourcePath} imports frontend HIR vocabulary`);
  }
  if (sourcePath.startsWith("crates/runmat-hir/") && /\brunmat_runtime\b/.test(text)) {
    fail(`${sourcePath} imports runtime session state`);
  }
  if (/\brunmat_builtins::(?:Value|Tensor|NumericScalar|IntValue)\b/.test(text)) {
    fail(`${sourcePath} addresses live value data through runmat-builtins`);
  }
}

const builtinsSources = allRust
  .filter(({ path: sourcePath }) => sourcePath.startsWith("crates/runmat-builtins/"))
  .map(({ text }) => text)
  .join("\n");
if (/\bpub\s+use\b[^;]*\b(?:Value|Tensor|NumericScalar|IntValue)\b/.test(builtinsSources)) {
  fail("runmat-builtins re-exports live value data");
}
for (const symbol of [
  "ClassDef",
  "PropertyDef",
  "MethodDef",
  "CLASS_REGISTRY",
  "STATIC_VALUES",
  "ENUMERATION_REGISTRY",
]) {
  if (new RegExp(`\\b${symbol}\\b`).test(builtinsSources)) {
    fail(`runmat-builtins contains runtime class/session authority ${symbol}`);
  }
}

if (failed) process.exit(1);
console.log("crate architecture boundaries are valid");
