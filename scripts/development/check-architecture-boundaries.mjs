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

function enforceMigratedBuiltinFamily({
  name,
  roots,
  compositionFiles,
  obsoletePaths,
  testLineCeiling = 320,
  leafLineCeiling = 192,
  byteCeiling = 24 * 1024,
}) {
  for (const sourcePath of obsoletePaths) {
    if (fs.existsSync(path.join(repo, sourcePath))) {
      fail(`${sourcePath} is obsolete ${name} migration debt and must not return`);
    }
  }

  const composition = new Set(compositionFiles);
  for (const rootPath of roots) {
    for (const { path: sourcePath, text } of rustSources(rootPath)) {
      const lines = text.split("\n").length;
      const isTest = sourcePath.includes("/tests/") || sourcePath.endsWith("/tests.rs");
      const lineCeiling = composition.has(sourcePath) ? 64 : isTest ? testLineCeiling : leafLineCeiling;
      if (lines > lineCeiling || Buffer.byteLength(text, "utf8") > byteCeiling) {
        fail(
          `${sourcePath} exceeds its ${name} role boundary ` +
          `(found ${lines} lines; maximum ${lineCeiling} and ${byteCeiling} bytes)`
        );
      }
      if (
        rootPath.includes("runmat-runtime") &&
        /\b(?:BuiltinDescriptor|BuiltinIntegerCapabilityDescriptor)\s*=/.test(text)
      ) {
        fail(`${sourcePath} duplicates catalog-owned ${name} metadata`);
      }
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
forbidDependencies("runmat-mex", [
  "runmat-builtins",
  "runmat-hir",
  "runmat-mir",
  "runmat-vm",
  "runmat-core",
  "runmat-jit",
  "runmat-aot",
  "runmat-native-codegen",
]);

const extensionAbiManifest = read("crates/runmat-extension-abi/Cargo.toml");
if (/^\s*[^#\s][\w-]*\s*=\s*/m.test(extensionAbiManifest.split("[dependencies]")[1] ?? "")) {
  fail("runmat-extension-abi must remain dependency-free");
}

const typesManifest = read("crates/runmat-types/Cargo.toml");
for (const dependency of typesManifest.matchAll(/^(runmat-[\w-]+)\s*=/gm)) {
  fail(`runmat-types must remain dependency-neutral; found ${dependency[1]}`);
}

for (const header of ["matrix.h", "mex.h"]) {
  const text = read(`crates/runmat-mex/include/${header}`);
  if (/runmat_extension|runmat_value|runmat_runtime/i.test(text)) {
    fail(`${header} conflates the MATLAB compatibility API with an internal or RunMat extension ABI`);
  }
}

const allRust = rustSources("crates");
const builtinSourceEntries = fs.readdirSync(
  path.join(repo, "crates/runmat-runtime/src/builtins"),
  { withFileTypes: true },
);
if (builtinSourceEntries.some((entry) => entry.name.startsWith(".runmat-wasm-registry-"))) {
  fail("temporary wasm registry generation directories must not remain in the runtime source tree");
}
const valueDeclarations = allRust
  .filter(({ text }) => /(?:^|\n)\s*pub(?:\([^)]*\))?\s+enum\s+Value(?:\s|\{|<)/.test(text))
  .map(({ path: sourcePath }) => sourcePath);
if (valueDeclarations.length !== 1 || !valueDeclarations[0].startsWith("crates/runmat-value/")) {
  fail(`Value must be declared exactly once by runmat-value; found ${valueDeclarations.join(", ") || "none"}`);
}

const integerClassDeclarations = allRust
  .filter(({ text }) => /(?:^|\n)\s*pub(?:\([^)]*\))?\s+enum\s+IntegerClass(?:\s|\{|<)/.test(text)
    || /(?:^|\n)\s*enum\s+IntegerClass(?:\s|\{|<)/.test(text))
  .map(({ path: sourcePath }) => sourcePath);
if (integerClassDeclarations.length !== 1 || !integerClassDeclarations[0].startsWith("crates/runmat-types/")) {
  fail(
    `IntegerClass must be declared exactly once by runmat-types; found ` +
    `${integerClassDeclarations.join(", ") || "none"}`
  );
}

for (const { path: sourcePath, text } of allRust) {
  if (
    sourcePath.startsWith("crates/runmat-extension-abi/") &&
    /\brunmat_(?:value|types|builtins|runtime|hir|mir|vm|core|jit|aot)\b/.test(text)
  ) {
    fail(`${sourcePath} leaks an internal RunMat crate across the public extension ABI`);
  }
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

const catalogModule = read("crates/runmat-builtins/src/catalog/mod.rs");
if (/\bpub\s+mod\s+entries\s*;/.test(catalogModule)) {
  fail("runmat-builtins must keep its physical catalog entry tree private");
}
for (const { path: sourcePath, text } of allRust) {
  if (
    !sourcePath.startsWith("crates/runmat-builtins/") &&
    /runmat_builtins::catalog::entries\b/.test(text)
  ) {
    fail(`${sourcePath} depends on the private physical catalog entry tree`);
  }
}
if (fs.existsSync(path.join(repo, "crates/runmat-builtins/src/catalog/definitions"))) {
  fail("the obsolete parallel catalog definitions tree must not return");
}

const catalogEntriesRootPath = "crates/runmat-builtins/src/catalog/entries/mod.rs";
const catalogEntriesRoot = read(catalogEntriesRootPath);
const catalogEntriesRootLines = catalogEntriesRoot.split("\n").length;
if (catalogEntriesRootLines > 80) {
  fail(`${catalogEntriesRootPath} must remain a domain-only composition root (found ${catalogEntriesRootLines} lines; maximum 80)`);
}
if (/::(?:ENTRIES|ENTRY_GROUPS)\b/.test(catalogEntriesRoot)) {
  fail(`${catalogEntriesRootPath} must register domains, not family or identity entry slices`);
}
const declaredCatalogDomains = [...catalogEntriesRoot.matchAll(/^(?:pub\(in\s+crate::catalog\)\s+)?mod\s+([a-z][a-z0-9_]*)\s*;/gm)]
  .map((match) => match[1])
  .sort();
const registeredCatalogDomains = [...catalogEntriesRoot.matchAll(/^\s+([a-z][a-z0-9_]*)::extend_entries\(entries\);$/gm)]
  .map((match) => match[1])
  .sort();
if (declaredCatalogDomains.join("\n") !== registeredCatalogDomains.join("\n")) {
  fail(
    `${catalogEntriesRootPath} must register every declared domain exactly once; ` +
    `declared=${declaredCatalogDomains.join(",")}, registered=${registeredCatalogDomains.join(",")}`
  );
}

const inferenceRootPath = "crates/runmat-builtins/src/catalog/inference.rs";
const inferenceRootLines = read(inferenceRootPath).split("\n").length;
if (inferenceRootLines > 48) {
  fail(`${inferenceRootPath} must remain a composition root (found ${inferenceRootLines} lines; maximum 48)`);
}
// These predate domain/family inference packages. The set is shrink-only: a
// migrated family removes its old root leaf, while new semantics must enter
// through a bounded domain package instead of expanding inference.rs.
const legacyRootInferenceLeaves = new Set([
  "acceleration_semantics",
  "aggregate_semantics",
  "introspection_semantics",
  "math_degree_trigonometric",
  "math_exponential",
  "math_hyperbolic",
  "math_inverse",
  "math_logarithms",
  "math_reduction",
  "math_roots",
  "math_rounding",
  "math_trigonometric",
  "metadata_predicate",
  "numeric_classification",
  "numeric_component",
  "parallel_semantics",
  "scalar_logical_reduction",
  "unary_logical_scalar",
]);
const inferenceInfrastructureModules = new Set(["routing", "support"]);
for (const match of read(inferenceRootPath).matchAll(/^mod\s+([a-z][a-z0-9_]*)\s*;/gm)) {
  const moduleName = match[1];
  const moduleDirectory = path.join(
    repo,
    "crates/runmat-builtins/src/catalog/inference",
    moduleName
  );
  const isCompositionModule =
    fs.existsSync(moduleDirectory) && fs.statSync(moduleDirectory).isDirectory();
  if (
    !isCompositionModule &&
    !inferenceInfrastructureModules.has(moduleName) &&
    !legacyRootInferenceLeaves.has(moduleName)
  ) {
    fail(
      `${inferenceRootPath} declares new root semantic leaf ${moduleName}; ` +
      "route it through a bounded domain/family inference package"
    );
  }
}

const inferenceSupportBoundaries = new Map([
  ["crates/runmat-builtins/src/catalog/inference/support/mod.rs", 128],
  ["crates/runmat-builtins/src/catalog/inference/support/facts.rs", 64],
]);
for (const [sourcePath, ceiling] of inferenceSupportBoundaries) {
  const lines = read(sourcePath).split("\n").length;
  if (lines > ceiling) {
    fail(
      `${sourcePath} exceeds its inference-infrastructure boundary ` +
      `(found ${lines} lines; maximum ${ceiling})`
    );
  }
}
for (const sourcePath of [
  "crates/runmat-builtins/src/catalog/inference/support.rs",
  "crates/runmat-builtins/src/catalog/inference/math_fact_transforms.rs",
]) {
  if (fs.existsSync(path.join(repo, sourcePath))) {
    fail(`${sourcePath} is obsolete flat inference infrastructure and must not return`);
  }
}
const legacyInferenceLeafCeilings = new Map([
  ["crates/runmat-builtins/src/catalog/inference/math_inverse.rs", 510],
  ["crates/runmat-builtins/src/catalog/inference/math_logarithms.rs", 481],
  ["crates/runmat-builtins/src/catalog/inference/math_rounding.rs", 269],
  ["crates/runmat-builtins/src/catalog/inference/stats_random/two_parameter.rs", 366],
]);
for (const { path: sourcePath, text } of rustSources("crates/runmat-builtins/src/catalog/inference")) {
  const lines = text.split("\n").length;
  const isTestModule = sourcePath.endsWith("/tests.rs");
  const legacyCeiling = legacyInferenceLeafCeilings.get(sourcePath);
  const leafCeiling = legacyCeiling ?? (isTestModule ? 600 : 256);
  if (lines > leafCeiling) {
    const policy = legacyCeiling === undefined
      ? "bounded inference-family size"
      : "shrink-only legacy inference-family ceiling";
    fail(`${sourcePath} exceeds its ${policy} (found ${lines} lines; maximum ${leafCeiling})`);
  }
  if (/entry\.identity\.name\s*(?:==|!=)|match\s+entry\.identity\.name|matches!\s*\(\s*entry\.identity\.name/.test(text)) {
    fail(`${sourcePath} dispatches inference semantics from a builtin name; add a closed typed rule and route it to its domain module`);
  }
  if (
    sourcePath.includes("/inference/routing/") &&
    lines > 128
  ) {
    fail(`${sourcePath} must remain a thin typed router (found ${lines} lines; maximum 128)`);
  }
  if (
    sourcePath.endsWith("/mod.rs") &&
    lines > 256
  ) {
    fail(`${sourcePath} must remain a bounded module router (found ${lines} lines; maximum 256)`);
  }
}

const inferenceCompositionRoot = read(inferenceRootPath);
for (const forbidden of [
  "BuiltinInferenceRule::",
  "BuiltinDistributedPolicy::",
  "ValueKindFact::Distributed",
]) {
  if (inferenceCompositionRoot.includes(forbidden)) {
    fail(`${inferenceRootPath} owns domain policy (${forbidden}); route it through a typed inference module`);
  }
}

const angleConversionBoundaries = new Map([
  ["crates/runmat-builtins/src/catalog/inference/math/angle_conversion/mod.rs", 32],
  ["crates/runmat-builtins/src/catalog/inference/math/angle_conversion/unary.rs", 128],
  ["crates/runmat-builtins/src/catalog/inference/math/angle_conversion/tests.rs", 128],
  ["crates/runmat-builtins/src/catalog/entries/math/trigonometry/angle_conversion/mod.rs", 32],
  ["crates/runmat-builtins/src/catalog/entries/math/trigonometry/angle_conversion/deg2rad/mod.rs", 192],
  ["crates/runmat-builtins/src/catalog/entries/math/trigonometry/angle_conversion/deg2rad/documentation.rs", 192],
  ["crates/runmat-builtins/src/catalog/entries/math/trigonometry/angle_conversion/rad2deg/mod.rs", 192],
  ["crates/runmat-builtins/src/catalog/entries/math/trigonometry/angle_conversion/rad2deg/documentation.rs", 192],
  ["crates/runmat-runtime/src/builtins/math/trigonometry/angle_conversion/mod.rs", 32],
  ["crates/runmat-runtime/src/builtins/math/trigonometry/angle_conversion/execute.rs", 224],
  ["crates/runmat-runtime/src/builtins/math/trigonometry/angle_conversion/deg2rad.rs", 320],
  ["crates/runmat-runtime/src/builtins/math/trigonometry/angle_conversion/rad2deg.rs", 320],
]);
for (const [sourcePath, ceiling] of angleConversionBoundaries) {
  const lines = read(sourcePath).split("\n").length;
  if (lines > ceiling) {
    fail(
      `${sourcePath} exceeds its angle-conversion boundary ` +
      `(found ${lines} lines; maximum ${ceiling})`
    );
  }
}
for (const sourcePath of [
  "docs/builtins/reference/deg2rad.json",
  "docs/builtins/reference/rad2deg.json",
  "crates/runmat-runtime/src/builtins/builtins-json/deg2rad.json",
  "crates/runmat-runtime/src/builtins/builtins-json/rad2deg.json",
  "crates/runmat-runtime/src/builtins/math/trigonometry/deg2rad.rs",
  "crates/runmat-runtime/src/builtins/math/trigonometry/rad2deg.rs",
]) {
  if (fs.existsSync(path.join(repo, sourcePath))) {
    fail(`${sourcePath} is obsolete deg2rad documentation debt and must not return`);
  }
}
const angleConversionExecution = read(
  "crates/runmat-runtime/src/builtins/math/trigonometry/angle_conversion/execute.rs"
);
for (const identity of ["deg2rad", "rad2deg"]) {
  if (angleConversionExecution.includes(`\"${identity}\"`)) {
    fail(`angle-conversion execution derives behavior from the ${identity} identity`);
  }
}
for (const { path: sourcePath, text } of rustSources("crates/runmat-runtime/src/builtins/math/trigonometry/angle_conversion")) {
  if (/\b(?:BuiltinDescriptor|BuiltinIntegerCapabilityDescriptor)\s*=/.test(text)) {
    fail(`${sourcePath} duplicates catalog-owned angle-conversion metadata`);
  }
}

for (const rootPath of [
  "crates/runmat-builtins/src/catalog/entries/array/combinatorics",
  "crates/runmat-builtins/src/catalog/inference/array/combinatorics",
  "crates/runmat-runtime/src/builtins/array/combinatorics",
]) {
  for (const { path: sourcePath, text } of rustSources(rootPath)) {
    const lines = text.split("\n").length;
    const isTest = sourcePath.includes("/tests/") || sourcePath.endsWith("/tests.rs");
    const isDocumentation = sourcePath.endsWith("/documentation.rs");
    const isCompositionRoot = sourcePath.endsWith("/mod.rs") && !isTest;
    const ownsIdentity = text.includes("_CATALOG_ENTRY") || text.includes("#[runtime_builtin(");
    let ceiling = 256;
    if (isTest) ceiling = 512;
    else if (isDocumentation) ceiling = 256;
    else if (isCompositionRoot) ceiling = ownsIdentity ? 192 : 64;
    if (lines > ceiling) {
      fail(
        `${sourcePath} exceeds its deterministic-combinatorics role boundary ` +
        `(found ${lines} lines; maximum ${ceiling})`
      );
    }
    if (
      rootPath.includes("runmat-runtime") &&
      /\b(?:BuiltinDescriptor|BuiltinIntegerCapabilityDescriptor)\s*=/.test(text)
    ) {
      fail(`${sourcePath} duplicates catalog-owned combinatorics metadata`);
    }
  }
}
for (const sourcePath of [
  "docs/builtins/reference/combinations.json",
  "docs/builtins/reference/nchoosek.json",
  "docs/builtins/reference/perms.json",
  "crates/runmat-runtime/src/builtins/builtins-json/combinations.json",
  "crates/runmat-runtime/src/builtins/builtins-json/nchoosek.json",
  "crates/runmat-runtime/src/builtins/builtins-json/perms.json",
  "crates/runmat-runtime/src/builtins/array/creation/nchoosek.rs",
  "crates/runmat-runtime/src/builtins/array/creation/perms.rs",
]) {
  if (fs.existsSync(path.join(repo, sourcePath))) {
    fail(`${sourcePath} is obsolete deterministic-combinatorics debt and must not return`);
  }
}

for (const rootPath of [
  "crates/runmat-builtins/src/catalog/entries/array/binning",
  "crates/runmat-builtins/src/catalog/inference/array/binning",
  "crates/runmat-runtime/src/builtins/array/binning",
]) {
  for (const { path: sourcePath, text } of rustSources(rootPath)) {
    const lines = text.split("\n").length;
    const isTest = sourcePath.includes("/tests/") || sourcePath.endsWith("/tests.rs");
    const isDocumentation = sourcePath.endsWith("/documentation.rs");
    const relativePath = sourcePath.slice(rootPath.length + 1);
    const isCompositionRoot = relativePath === "mod.rs";
    const ownsIdentity = text.includes("_CATALOG_ENTRY") || text.includes("#[runtime_builtin(");
    let ceiling = 256;
    if (isTest) ceiling = 256;
    else if (isDocumentation) ceiling = 256;
    else if (isCompositionRoot) ceiling = ownsIdentity ? 192 : 64;
    if (lines > ceiling) {
      fail(
        `${sourcePath} exceeds its numeric-binning role boundary ` +
        `(found ${lines} lines; maximum ${ceiling})`
      );
    }
    if (
      rootPath.includes("runmat-runtime") &&
      /\b(?:BuiltinDescriptor|BuiltinIntegerCapabilityDescriptor)\s*=/.test(text)
    ) {
      fail(`${sourcePath} duplicates catalog-owned numeric-binning metadata`);
    }
  }
}
for (const sourcePath of [
  "docs/builtins/reference/discretize.json",
  "crates/runmat-runtime/src/builtins/builtins-json/discretize.json",
]) {
  if (fs.existsSync(path.join(repo, sourcePath))) {
    fail(`${sourcePath} is obsolete discretize documentation debt and must not return`);
  }
}
for (const [sourcePath, ceiling] of [
  ["crates/runmat-builtins/src/catalog/inference.rs", 64],
  ["crates/runmat-builtins/src/catalog/inference/array/mod.rs", 64],
]) {
  const lines = read(sourcePath).split("\n").length;
  if (lines > ceiling) {
    fail(
      `${sourcePath} exceeds its typed inference-router boundary ` +
      `(found ${lines} lines; maximum ${ceiling}); route policy to its owning domain module`
    );
  }
}
for (const [sourcePath, ceiling] of [
  ["crates/runmat-builtins/src/catalog/contract.rs", 96],
  ["crates/runmat-builtins/src/catalog/contract/acceleration.rs", 192],
  ["crates/runmat-builtins/src/catalog/contract/aggregate.rs", 192],
  ["crates/runmat-builtins/src/catalog/contract/array.rs", 192],
  ["crates/runmat-builtins/src/catalog/contract/introspection.rs", 192],
  ["crates/runmat-builtins/src/catalog/contract/io.rs", 192],
  ["crates/runmat-builtins/src/catalog/contract/logical.rs", 192],
  ["crates/runmat-builtins/src/catalog/contract/math.rs", 192],
  ["crates/runmat-builtins/src/catalog/contract/parallel.rs", 192],
  ["crates/runmat-builtins/src/catalog/contract/stats.rs", 192],
]) {
  const lines = read(sourcePath).split("\n").length;
  if (lines > ceiling) {
    fail(
      `${sourcePath} exceeds its typed contract-vocabulary boundary ` +
      `(found ${lines} lines; maximum ${ceiling}); split rules by domain rather than growing the catalog root`
    );
  }
}
if (/pub enum (?:Io|Logical|Math|Array|Parallel|Stats)InferenceRule/.test(
  read("crates/runmat-builtins/src/catalog/contract.rs"),
)) {
  fail("catalog/contract.rs must compose domain-owned rule vocabularies rather than accumulating domain policy");
}
if (fs.existsSync(path.join(repo, "crates/runmat-runtime/src/builtins/array/grouping/legacy.rs"))) {
  fail("array/grouping/legacy.rs must not return after the final identity migration");
}
for (const sourcePath of [
  "docs/builtins/reference/accumarray.json",
  "crates/runmat-runtime/src/builtins/builtins-json/accumarray.json",
  "docs/builtins/reference/grp2idx.json",
  "crates/runmat-runtime/src/builtins/builtins-json/grp2idx.json",
  "docs/builtins/reference/findgroups.json",
  "crates/runmat-runtime/src/builtins/builtins-json/findgroups.json",
  "docs/builtins/reference/groupcounts.json",
  "crates/runmat-runtime/src/builtins/builtins-json/groupcounts.json",
  "docs/builtins/reference/splitapply.json",
  "crates/runmat-runtime/src/builtins/builtins-json/splitapply.json",
]) {
  if (fs.existsSync(path.join(repo, sourcePath))) {
    fail(`${sourcePath} is obsolete catalog-migration documentation debt and must not return`);
  }
}

const groupingCompositionRoots = new Set([
  "crates/runmat-builtins/src/catalog/entries/array/grouping/mod.rs",
  "crates/runmat-builtins/src/catalog/inference/array/grouping/mod.rs",
  "crates/runmat-runtime/src/builtins/array/grouping/mod.rs",
  "crates/runmat-runtime/src/builtins/array/grouping/keys/mod.rs",
  "crates/runmat-runtime/src/builtins/array/grouping/variables/mod.rs",
]);
const accumulationCompositionRoots = new Set([
  "crates/runmat-builtins/src/catalog/entries/array/accumulation/mod.rs",
  "crates/runmat-builtins/src/catalog/entries/array/accumulation/accumarray/mod.rs",
  "crates/runmat-builtins/src/catalog/inference/array/accumulation/mod.rs",
  "crates/runmat-runtime/src/builtins/array/accumulation/mod.rs",
  "crates/runmat-runtime/src/builtins/array/accumulation/accumarray/mod.rs",
]);
for (const rootPath of [
  "crates/runmat-builtins/src/catalog/entries/array/grouping",
  "crates/runmat-builtins/src/catalog/inference/array/grouping",
  "crates/runmat-runtime/src/builtins/array/grouping",
]) {
  for (const { path: sourcePath, text } of rustSources(rootPath)) {
    if (sourcePath.endsWith("/legacy.rs")) continue;
    const lines = text.split("\n").length;
    const isDocumentation = /\/(?:documentation|examples|faqs)\.rs$/.test(sourcePath);
    const isTest = sourcePath.includes("/tests/") || sourcePath.endsWith("/tests.rs");
    const ceiling = groupingCompositionRoots.has(sourcePath)
      ? 64
      : isDocumentation
        ? 500
        : isTest
          ? 400
          : 256;
    if (lines > ceiling || Buffer.byteLength(text, "utf8") > 24 * 1024) {
      fail(
        `${sourcePath} exceeds its grouping role boundary ` +
        `(found ${lines} lines; maximum ${ceiling} and 24 KiB)`
      );
    }
    if (
      rootPath.includes("runmat-runtime") &&
      /\b(?:BuiltinDescriptor|BuiltinIntegerCapabilityDescriptor)\s*=/.test(text)
    ) {
      fail(`${sourcePath} duplicates catalog-owned grouping metadata`);
    }
  }
}
for (const rootPath of [
  "crates/runmat-builtins/src/catalog/entries/array/accumulation",
  "crates/runmat-builtins/src/catalog/inference/array/accumulation",
  "crates/runmat-runtime/src/builtins/array/accumulation",
]) {
  for (const { path: sourcePath, text } of rustSources(rootPath)) {
    const lines = text.split("\n").length;
    const isTest = sourcePath.includes("/tests/") || sourcePath.endsWith("/tests.rs");
    const ceiling = accumulationCompositionRoots.has(sourcePath) ? 80 : isTest ? 400 : 224;
    if (lines > ceiling || Buffer.byteLength(text, "utf8") > 24 * 1024) {
      fail(
        `${sourcePath} exceeds the accumulation module ceiling ` +
        `(found ${lines} lines; maximum ${ceiling} and 24 KiB)`
      );
    }
    if (
      rootPath.includes("runmat-runtime") &&
      /\b(?:BuiltinDescriptor|BuiltinIntegerCapabilityDescriptor)\s*=/.test(text)
    ) {
      fail(`${sourcePath} duplicates catalog-owned accumulation metadata`);
    }
  }
}

const logicalReductionBoundaries = new Map([
  ["crates/runmat-runtime/src/builtins/math/reduction/logical/mod.rs", 128],
  ["crates/runmat-runtime/src/builtins/math/reduction/logical/arguments.rs", 192],
  ["crates/runmat-runtime/src/builtins/math/reduction/logical/shape.rs", 128],
  ["crates/runmat-runtime/src/builtins/math/reduction/logical/host.rs", 400],
  ["crates/runmat-runtime/src/builtins/math/reduction/logical/gpu.rs", 256],
  ["crates/runmat-runtime/src/builtins/math/reduction/all.rs", 160],
  ["crates/runmat-runtime/src/builtins/math/reduction/any.rs", 160],
  ["crates/runmat-runtime/src/builtins/math/reduction/all/tests.rs", 600],
  ["crates/runmat-runtime/src/builtins/math/reduction/any/tests.rs", 600],
]);
for (const [sourcePath, ceiling] of logicalReductionBoundaries) {
  const lines = read(sourcePath).split("\n").length;
  if (lines > ceiling) {
    fail(
      `${sourcePath} exceeds its logical-reduction domain boundary ` +
      `(found ${lines} lines; maximum ${ceiling})`
    );
  }
}

const relationalComparisonBoundaries = new Map([
  ["crates/runmat-builtins/src/catalog/inference/logical/mod.rs", 128],
  ["crates/runmat-builtins/src/catalog/inference/logical/relational.rs", 160],
  ["crates/runmat-builtins/src/catalog/entries/logical/relational/mod.rs", 64],
  ["crates/runmat-builtins/src/catalog/entries/logical/relational/support.rs", 192],
  ["crates/runmat-builtins/src/catalog/entries/logical/relational/ordering_documentation.rs", 160],
  ["crates/runmat-runtime/src/builtins/logical/rel/comparison/mod.rs", 224],
  ["crates/runmat-runtime/src/builtins/logical/rel/comparison/errors.rs", 80],
  ["crates/runmat-runtime/src/builtins/logical/rel/comparison/identity.rs", 80],
  ["crates/runmat-runtime/src/builtins/logical/rel/comparison/operands.rs", 256],
  ["crates/runmat-runtime/src/builtins/logical/rel/integer_comparison/mod.rs", 480],
  ["crates/runmat-runtime/src/builtins/logical/rel/integer_comparison/exact.rs", 768],
  ["crates/runmat-runtime/src/builtins/logical/rel/integer_comparison/gpu.rs", 320],
]);
for (const [sourcePath, ceiling] of relationalComparisonBoundaries) {
  const lines = read(sourcePath).split("\n").length;
  if (lines > ceiling) {
    fail(
      `${sourcePath} exceeds its relational-comparison domain boundary ` +
      `(found ${lines} lines; maximum ${ceiling})`
    );
  }
}
for (const identity of ["eq", "ne", "lt", "le", "gt", "ge"]) {
  const modulePath = `crates/runmat-runtime/src/builtins/logical/rel/${identity}/mod.rs`;
  const moduleLines = read(modulePath).split("\n").length;
  if (moduleLines > 128) {
    fail(`${modulePath} must remain a thin registration and specification leaf (found ${moduleLines} lines; maximum 128)`);
  }
  const testsPath = `crates/runmat-runtime/src/builtins/logical/rel/${identity}/tests.rs`;
  const testLines = read(testsPath).split("\n").length;
  if (testLines > 600) {
    fail(`${testsPath} exceeds its identity-owned test boundary (found ${testLines} lines; maximum 600)`);
  }
}
const relationalRuntimeSources = [
  ...rustSources("crates/runmat-runtime/src/builtins/logical/rel/comparison"),
  ...rustSources("crates/runmat-runtime/src/builtins/logical/rel/integer_comparison"),
];
for (const { path: sourcePath, text } of relationalRuntimeSources) {
  if (/match\s+[^\n{]*\.name\s*\(\s*\)|\.name\s*\(\s*\)\s*(?:==|!=)/.test(text)) {
    fail(`${sourcePath} selects relational semantics from a builtin name; dispatch through RelationalOperator`);
  }
}

const logicalElementwiseBoundaries = new Map([
  ["crates/runmat-builtins/src/catalog/inference/logical/mod.rs", 128],
  ["crates/runmat-builtins/src/catalog/inference/logical/elementwise/mod.rs", 64],
  ["crates/runmat-builtins/src/catalog/inference/logical/elementwise/binary.rs", 128],
  ["crates/runmat-builtins/src/catalog/inference/logical/elementwise/unary.rs", 128],
  ["crates/runmat-builtins/src/catalog/inference/logical/elementwise/output.rs", 128],
  ["crates/runmat-builtins/src/catalog/inference/logical/elementwise/tests.rs", 192],
  ["crates/runmat-builtins/src/catalog/entries/logical/operators/mod.rs", 64],
  ["crates/runmat-builtins/src/catalog/entries/logical/operators/support.rs", 160],
  ["crates/runmat-builtins/src/catalog/entries/logical/operators/tests/mod.rs", 192],
  ["crates/runmat-runtime/src/builtins/logical/bit/truth/mod.rs", 64],
  ["crates/runmat-runtime/src/builtins/logical/bit/truth/binary.rs", 160],
  ["crates/runmat-runtime/src/builtins/logical/bit/truth/unary.rs", 128],
  ["crates/runmat-runtime/src/builtins/logical/bit/truth/contract.rs", 128],
  ["crates/runmat-runtime/src/builtins/logical/bit/truth/evaluate.rs", 128],
  ["crates/runmat-runtime/src/builtins/logical/bit/truth/operand.rs", 192],
  ["crates/runmat-runtime/src/builtins/logical/bit/truth/provider.rs", 160],
  ["crates/runmat-runtime/src/builtins/logical/bit/truth/provider/errors.rs", 64],
  ["crates/runmat-runtime/src/builtins/logical/bit/truth/provider/output.rs", 96],
  ["crates/runmat-runtime/src/builtins/logical/bit/truth/provider/tests.rs", 224],
]);
for (const [sourcePath, ceiling] of logicalElementwiseBoundaries) {
  const lines = read(sourcePath).split("\n").length;
  if (lines > ceiling) {
    fail(
      `${sourcePath} exceeds its logical-elementwise domain boundary ` +
      `(found ${lines} lines; maximum ${ceiling})`
    );
  }
}
for (const identity of ["and", "or", "xor", "not"]) {
  const catalogPath = `crates/runmat-builtins/src/catalog/entries/logical/operators/${identity}/mod.rs`;
  const catalogLines = read(catalogPath).split("\n").length;
  if (catalogLines > 128) {
    fail(`${catalogPath} exceeds its identity-owned catalog boundary (found ${catalogLines} lines; maximum 128)`);
  }
  const documentationPath = `crates/runmat-builtins/src/catalog/entries/logical/operators/${identity}/documentation.rs`;
  const documentationLines = read(documentationPath).split("\n").length;
  if (documentationLines > 128) {
    fail(`${documentationPath} exceeds its identity-owned documentation boundary (found ${documentationLines} lines; maximum 128)`);
  }
  const runtimePath = `crates/runmat-runtime/src/builtins/logical/bit/${identity}.rs`;
  const runtimeLines = read(runtimePath).split("\n").length;
  if (runtimeLines > 128) {
    fail(`${runtimePath} must remain a thin registration and specification leaf (found ${runtimeLines} lines; maximum 128)`);
  }
  const runtimeTestsPath = `crates/runmat-runtime/src/builtins/logical/bit/${identity}/tests.rs`;
  const runtimeTestLines = read(runtimeTestsPath).split("\n").length;
  if (runtimeTestLines > 480) {
    fail(`${runtimeTestsPath} exceeds its identity-owned test boundary (found ${runtimeTestLines} lines; maximum 480)`);
  }
}
const logicalElementwiseRuntimeSources = rustSources(
  "crates/runmat-runtime/src/builtins/logical/bit/truth"
);
for (const { path: sourcePath, text } of logicalElementwiseRuntimeSources) {
  if (/match\s+[^\n{]*\.name\s*\(\s*\)|\.name\s*\(\s*\)\s*(?:==|!=)/.test(text)) {
    fail(`${sourcePath} selects logical semantics from a builtin name; dispatch through the typed logical operator`);
  }
}

const tabularBinaryBoundaries = new Map([
  ["crates/runmat-runtime/src/builtins/common/binary.rs", 160],
  ["crates/runmat-runtime/src/builtins/table/binary/mod.rs", 64],
  ["crates/runmat-runtime/src/builtins/table/binary/plan.rs", 160],
  ["crates/runmat-runtime/src/builtins/table/binary/rows.rs", 192],
  ["crates/runmat-runtime/src/builtins/table/binary/output.rs", 64],
  ["crates/runmat-runtime/src/builtins/table/binary/tests.rs", 160],
]);
for (const [sourcePath, ceiling] of tabularBinaryBoundaries) {
  const lines = read(sourcePath).split("\n").length;
  if (lines > ceiling) {
    fail(
      `${sourcePath} exceeds its tabular-binary domain boundary ` +
      `(found ${lines} lines; maximum ${ceiling})`
    );
  }
}

const bitwiseBoundaries = new Map([
  ["crates/runmat-builtins/src/catalog/inference/math/bitwise/mod.rs", 64],
  ["crates/runmat-builtins/src/catalog/inference/math/bitwise/binary.rs", 160],
  ["crates/runmat-builtins/src/catalog/inference/math/bitwise/complement.rs", 96],
  ["crates/runmat-builtins/src/catalog/inference/math/bitwise/position.rs", 128],
  ["crates/runmat-builtins/src/catalog/inference/math/bitwise/shift.rs", 128],
  ["crates/runmat-builtins/src/catalog/inference/math/bitwise/swapbytes.rs", 96],
  ["crates/runmat-builtins/src/catalog/inference/math/bitwise/tests.rs", 192],
  ["crates/runmat-builtins/src/catalog/entries/math/bitwise/mod.rs", 64],
  ["crates/runmat-builtins/src/catalog/entries/math/bitwise/support.rs", 128],
  ["crates/runmat-builtins/src/catalog/entries/math/bitwise/binary/mod.rs", 32],
  ["crates/runmat-builtins/src/catalog/entries/math/bitwise/binary/support.rs", 192],
  ["crates/runmat-builtins/src/catalog/entries/math/bitwise/swapbytes/mod.rs", 160],
  ["crates/runmat-builtins/src/catalog/entries/math/bitwise/swapbytes/documentation.rs", 192],
  ["crates/runmat-builtins/src/catalog/inference/math/integer_division.rs", 128],
  ["crates/runmat-builtins/src/catalog/inference/math/integer_division_tests.rs", 128],
  ["crates/runmat-builtins/src/catalog/entries/math/integer_division/mod.rs", 32],
  ["crates/runmat-builtins/src/catalog/entries/math/integer_division/idivide/mod.rs", 192],
  ["crates/runmat-builtins/src/catalog/entries/math/integer_division/idivide/documentation.rs", 160],
  ["crates/runmat-runtime/src/builtins/common/integer_value.rs", 128],
  ["crates/runmat-runtime/src/builtins/common/resident_output.rs", 128],
  ["crates/runmat-runtime/src/builtins/math/bitwise/mod.rs", 64],
  ["crates/runmat-runtime/src/builtins/math/bitwise/binary/mod.rs", 64],
  ["crates/runmat-runtime/src/builtins/math/bitwise/binary/tests.rs", 160],
  ["crates/runmat-runtime/src/builtins/math/bitwise/engine/mod.rs", 96],
  ["crates/runmat-runtime/src/builtins/math/bitwise/engine/arguments.rs", 160],
  ["crates/runmat-runtime/src/builtins/math/bitwise/engine/binary.rs", 96],
  ["crates/runmat-runtime/src/builtins/math/bitwise/engine/error.rs", 32],
  ["crates/runmat-runtime/src/builtins/math/bitwise/engine/operand.rs", 288],
  ["crates/runmat-runtime/src/builtins/math/bitwise/engine/output.rs", 224],
  ["crates/runmat-runtime/src/builtins/math/bitwise/engine/position.rs", 384],
  ["crates/runmat-runtime/src/builtins/math/bitwise/engine/resident.rs", 128],
  ["crates/runmat-runtime/src/builtins/math/bitwise/engine/shift.rs", 160],
  ["crates/runmat-runtime/src/builtins/math/bitwise/engine/sparse.rs", 224],
  ["crates/runmat-runtime/src/builtins/math/bitwise/swapbytes.rs", 128],
  ["crates/runmat-runtime/src/builtins/math/bitwise/swapbytes/tests.rs", 160],
  ["crates/runmat-runtime/src/builtins/math/integer_division/mod.rs", 64],
  ["crates/runmat-runtime/src/builtins/math/integer_division/engine/mod.rs", 128],
  ["crates/runmat-runtime/src/builtins/math/integer_division/engine/operand.rs", 160],
  ["crates/runmat-runtime/src/builtins/math/integer_division/engine/rounding.rs", 128],
  ["crates/runmat-runtime/src/builtins/math/integer_division/tests/mod.rs", 32],
  ["crates/runmat-runtime/src/builtins/math/integer_division/tests/rounding.rs", 64],
  ["crates/runmat-runtime/src/builtins/math/integer_division/tests/semantics.rs", 256],
]);
for (const identity of ["bitand", "bitor", "bitxor"]) {
  bitwiseBoundaries.set(
    `crates/runmat-builtins/src/catalog/entries/math/bitwise/binary/${identity}.rs`,
    192
  );
  bitwiseBoundaries.set(
    `crates/runmat-runtime/src/builtins/math/bitwise/binary/${identity}.rs`,
    64
  );
}
for (const identity of ["bitcmp", "bitget", "bitset", "bitshift"]) {
  const identityPath = identity === "bitshift"
    ? `crates/runmat-builtins/src/catalog/entries/math/bitwise/${identity}/mod.rs`
    : `crates/runmat-builtins/src/catalog/entries/math/bitwise/${identity}.rs`;
  bitwiseBoundaries.set(
    identityPath,
    256
  );
  bitwiseBoundaries.set(
    `crates/runmat-runtime/src/builtins/math/bitwise/${identity}.rs`,
    64
  );
}
bitwiseBoundaries.set(
  "crates/runmat-builtins/src/catalog/entries/math/bitwise/bitshift/documentation.rs",
  160
);
for (const { path: sourcePath } of rustSources("crates/runmat-runtime/src/builtins/math/bitwise/engine/tests")) {
  bitwiseBoundaries.set(sourcePath, 384);
}
for (const [sourcePath, ceiling] of bitwiseBoundaries) {
  const lines = read(sourcePath).split("\n").length;
  if (lines > ceiling) {
    fail(
      `${sourcePath} exceeds its bitwise or integer-division domain boundary ` +
      `(found ${lines} lines; maximum ${ceiling})`
    );
  }
}
for (const sourcePath of [
  "crates/runmat-runtime/src/builtins/logical/bit/integer.rs",
  "crates/runmat-runtime/src/builtins/logical/bit/integer_tests.rs",
]) {
  if (fs.existsSync(path.join(repo, sourcePath))) {
    fail(`${sourcePath} is obsolete mixed-domain bitwise-integer debt and must not return`);
  }
}
const bitwiseRuntimeSources = rustSources("crates/runmat-runtime/src/builtins/math/bitwise");
for (const { path: sourcePath, text } of bitwiseRuntimeSources) {
  if (/\bidivide\b|\bIDIVIDE_/.test(text)) {
    fail(`${sourcePath} crosses the bitwise boundary into integer-division policy`);
  }
  if (/match\s+[^\n{]*\.name\s*\(\s*\)|\.name\s*\(\s*\)\s*(?:==|!=)/.test(text)) {
    fail(`${sourcePath} selects bitwise semantics from a builtin name; dispatch through a typed operation`);
  }
}

const errorFunctionBoundaries = new Map([
  ["crates/runmat-builtins/src/catalog/inference/math/error_functions/mod.rs", 32],
  ["crates/runmat-builtins/src/catalog/inference/math/error_functions/real_unary.rs", 224],
  ["crates/runmat-builtins/src/catalog/entries/math/elementwise/error_functions/mod.rs", 32],
  ["crates/runmat-runtime/src/builtins/math/elementwise/error_functions/mod.rs", 32],
  ["crates/runmat-runtime/src/builtins/math/resident_real_unary.rs", 160],
]);
for (const identity of ["erf", "erfcinv"]) {
  errorFunctionBoundaries.set(
    `crates/runmat-builtins/src/catalog/entries/math/elementwise/error_functions/${identity}/mod.rs`,
    128
  );
  errorFunctionBoundaries.set(
    `crates/runmat-builtins/src/catalog/entries/math/elementwise/error_functions/${identity}/documentation.rs`,
    128
  );
  errorFunctionBoundaries.set(
    `crates/runmat-runtime/src/builtins/math/elementwise/error_functions/${identity}.rs`,
    512
  );
}
for (const [sourcePath, ceiling] of errorFunctionBoundaries) {
  const lines = read(sourcePath).split("\n").length;
  if (lines > ceiling) {
    fail(
      `${sourcePath} exceeds its error-function family boundary ` +
      `(found ${lines} lines; maximum ${ceiling})`
    );
  }
}
for (const sourcePath of [
  "crates/runmat-runtime/src/builtins/math/elementwise/erf.rs",
  "crates/runmat-runtime/src/builtins/math/elementwise/erfcinv.rs",
]) {
  if (fs.existsSync(path.join(repo, sourcePath))) {
    fail(`${sourcePath} is an obsolete flat error-function identity and must not return`);
  }
}
for (const { path: sourcePath, text } of rustSources("crates/runmat-runtime/src/builtins/math/elementwise/error_functions")) {
  if (/\b(?:BuiltinDescriptor|BuiltinIntegerCapabilityDescriptor)\s*=/.test(text)) {
    fail(`${sourcePath} duplicates catalog-owned error-function metadata`);
  }
  if (/\.to_string\(\)\.contains\(\s*"unary_(?:erf|erfcinv) not supported"/.test(text)) {
    fail(`${sourcePath} interprets provider capability from error text; use the typed provider result`);
  }
}

enforceMigratedBuiltinFamily({
  name: "powers-of-two family",
  roots: [
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/powers_of_two",
    "crates/runmat-builtins/src/catalog/inference/math/powers_of_two",
    "crates/runmat-runtime/src/builtins/math/elementwise/powers_of_two",
  ],
  compositionFiles: [
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/powers_of_two/mod.rs",
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/powers_of_two/nextpow2/mod.rs",
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/powers_of_two/pow2/mod.rs",
    "crates/runmat-builtins/src/catalog/inference/math/powers_of_two/mod.rs",
    "crates/runmat-builtins/src/catalog/inference/math/powers_of_two/next_exponent/mod.rs",
    "crates/runmat-builtins/src/catalog/inference/math/powers_of_two/power/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/powers_of_two/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/powers_of_two/nextpow2/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/powers_of_two/pow2/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/powers_of_two/pow2/binary/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/powers_of_two/pow2/provider/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/powers_of_two/pow2/tests/mod.rs",
  ],
  obsoletePaths: [
    "crates/runmat-runtime/src/builtins/math/elementwise/nextpow2.rs",
    "docs/builtins/reference/nextpow2.json",
    "crates/runmat-runtime/src/builtins/builtins-json/nextpow2.json",
    "crates/runmat-runtime/src/builtins/math/elementwise/pow2.rs",
    "docs/builtins/reference/pow2.json",
    "crates/runmat-runtime/src/builtins/builtins-json/pow2.json",
    "crates/runmat-builtins/src/catalog/inference/math/powers_of_two/next_exponent.rs",
    "crates/runmat-builtins/src/catalog/inference/math/powers_of_two/tests.rs",
  ],
});

enforceMigratedBuiltinFamily({
  name: "root family",
  roots: [
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/roots",
    "crates/runmat-builtins/src/catalog/inference/math/roots",
    "crates/runmat-runtime/src/builtins/math/elementwise/roots",
  ],
  compositionFiles: [
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/roots/mod.rs",
    "crates/runmat-builtins/src/catalog/inference/math/roots/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/roots/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/roots/sqrt/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/roots/realsqrt/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/roots/sqrt/tests/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/roots/realsqrt/tests/mod.rs",
  ],
  obsoletePaths: [
    "crates/runmat-runtime/src/builtins/math/elementwise/sqrt.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/realsqrt.rs",
    "crates/runmat-builtins/src/catalog/inference/math_roots.rs",
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/roots/documentation.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/logarithm_common.rs",
    "docs/builtins/reference/sqrt.json",
    "docs/builtins/reference/realsqrt.json",
    "crates/runmat-runtime/src/builtins/builtins-json/sqrt.json",
    "crates/runmat-runtime/src/builtins/builtins-json/realsqrt.json",
  ],
  testLineCeiling: 256,
});

enforceMigratedBuiltinFamily({
  name: "exponential family",
  roots: [
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/exponentials",
    "crates/runmat-builtins/src/catalog/inference/math/exponentials",
    "crates/runmat-runtime/src/builtins/math/elementwise/exponentials",
  ],
  compositionFiles: [
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/exponentials/mod.rs",
    "crates/runmat-builtins/src/catalog/inference/math/exponentials/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/exponentials/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/exponentials/exp/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/exponentials/expm1/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/exponentials/exp/tests/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/exponentials/expm1/tests/mod.rs",
  ],
  obsoletePaths: [
    "crates/runmat-builtins/src/catalog/inference/math_exponential.rs",
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/exponentials/exp.rs",
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/exponentials/expm1.rs",
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/exponentials/documentation/mod.rs",
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/exponentials/documentation/exp.rs",
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/exponentials/documentation/expm1.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/exp.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/expm1.rs",
    "docs/builtins/reference/exp.json",
    "docs/builtins/reference/expm1.json",
    "crates/runmat-runtime/src/builtins/builtins-json/exp.json",
    "crates/runmat-runtime/src/builtins/builtins-json/expm1.json",
  ],
  leafLineCeiling: 208,
  testLineCeiling: 256,
});

enforceMigratedBuiltinFamily({
  name: "logarithm family",
  roots: [
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/logarithms",
    "crates/runmat-builtins/src/catalog/inference/math/logarithms",
    "crates/runmat-runtime/src/builtins/math/elementwise/logarithms",
  ],
  compositionFiles: [
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/logarithms/mod.rs",
    "crates/runmat-builtins/src/catalog/inference/math/logarithms/mod.rs",
    "crates/runmat-builtins/src/catalog/inference/math/logarithms/unary/mod.rs",
    "crates/runmat-builtins/src/catalog/inference/math/logarithms/dissection/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/logarithms/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/logarithms/log/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/logarithms/log10/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/logarithms/log1p/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/logarithms/log2/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/logarithms/log/tests/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/logarithms/log10/tests/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/logarithms/log1p/tests/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/logarithms/log2/tests/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/logarithms/log2/dissection/mod.rs",
  ],
  obsoletePaths: [
    "crates/runmat-builtins/src/catalog/inference/math_logarithms.rs",
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/logarithms/natural_and_common.rs",
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/logarithms/log1p.rs",
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/logarithms/log2.rs",
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/logarithms/documentation",
    "crates/runmat-runtime/src/builtins/math/elementwise/log.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/log10.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/log1p.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/log2.rs",
    "docs/builtins/reference/log.json",
    "docs/builtins/reference/log10.json",
    "docs/builtins/reference/log1p.json",
    "docs/builtins/reference/log2.json",
    "crates/runmat-runtime/src/builtins/builtins-json/log.json",
    "crates/runmat-runtime/src/builtins/builtins-json/log10.json",
    "crates/runmat-runtime/src/builtins/builtins-json/log1p.json",
    "crates/runmat-runtime/src/builtins/builtins-json/log2.json",
  ],
  leafLineCeiling: 224,
  testLineCeiling: 256,
});

enforceMigratedBuiltinFamily({
  name: "magnitude/phase/sign family",
  roots: [
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/magnitude_phase_sign",
    "crates/runmat-builtins/src/catalog/inference/math/magnitude_phase_sign",
    "crates/runmat-runtime/src/builtins/math/elementwise/magnitude_phase_sign",
  ],
  compositionFiles: [
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/magnitude_phase_sign/mod.rs",
    "crates/runmat-builtins/src/catalog/inference/math/magnitude_phase_sign/mod.rs",
    "crates/runmat-builtins/src/catalog/inference/math/magnitude_phase_sign/tests/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/magnitude_phase_sign/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/magnitude_phase_sign/abs/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/magnitude_phase_sign/angle/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/magnitude_phase_sign/sign/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/magnitude_phase_sign/abs/tests/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/magnitude_phase_sign/angle/tests/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/magnitude_phase_sign/sign/tests/mod.rs",
  ],
  obsoletePaths: [
    "crates/runmat-builtins/src/catalog/inference/numeric_abs.rs",
    "crates/runmat-builtins/src/catalog/inference/math_components.rs",
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/magnitude_phase_sign/abs.rs",
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/magnitude_phase_sign/angle.rs",
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/magnitude_phase_sign/sign.rs",
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/magnitude_phase_sign/documentation",
    "crates/runmat-runtime/src/builtins/math/elementwise/abs.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/angle.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/sign.rs",
    "docs/builtins/reference/abs.json",
    "docs/builtins/reference/angle.json",
    "docs/builtins/reference/sign.json",
    "crates/runmat-runtime/src/builtins/builtins-json/abs.json",
    "crates/runmat-runtime/src/builtins/builtins-json/angle.json",
    "crates/runmat-runtime/src/builtins/builtins-json/sign.json",
  ],
  leafLineCeiling: 192,
  testLineCeiling: 224,
});

enforceMigratedBuiltinFamily({
  name: "floating-conversion family",
  roots: [
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/floating_conversions",
    "crates/runmat-builtins/src/catalog/inference/math/numeric_conversion",
    "crates/runmat-runtime/src/builtins/math/elementwise/floating_conversions",
  ],
  compositionFiles: [
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/floating_conversions/mod.rs",
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/floating_conversions/double/mod.rs",
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/floating_conversions/single/mod.rs",
    "crates/runmat-builtins/src/catalog/inference/math/numeric_conversion/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/floating_conversions/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/floating_conversions/double/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/floating_conversions/double/tests.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/floating_conversions/single/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/floating_conversions/single/tests.rs",
  ],
  obsoletePaths: [
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/floating_conversions.rs",
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/floating_conversions/definition.rs",
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/floating_conversions/documentation.rs",
    "crates/runmat-builtins/src/catalog/inference/numeric_conversion.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/double.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/single.rs",
    "docs/builtins/reference/double.json",
    "docs/builtins/reference/single.json",
    "crates/runmat-runtime/src/builtins/builtins-json/double.json",
    "crates/runmat-runtime/src/builtins/builtins-json/single.json",
  ],
  leafLineCeiling: 224,
  testLineCeiling: 272,
});

enforceMigratedBuiltinFamily({
  name: "integer-conversion family",
  roots: [
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/numeric_conversions",
    "crates/runmat-builtins/src/catalog/inference/math/numeric_conversion",
    "crates/runmat-runtime/src/builtins/common/integer_conversion",
    "crates/runmat-runtime/src/builtins/math/elementwise/integer_conversions",
  ],
  compositionFiles: [
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/numeric_conversions/mod.rs",
    "crates/runmat-builtins/src/catalog/inference/math/numeric_conversion/mod.rs",
    "crates/runmat-runtime/src/builtins/common/integer_conversion/mod.rs",
    "crates/runmat-runtime/src/builtins/common/integer_conversion/tests.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/integer_conversions/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/integer_conversions/tests.rs",
  ],
  obsoletePaths: [
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/numeric_conversions.rs",
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/numeric_conversions/documentation.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/int32.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/integer_cast.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/integer_cast_builtins.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/uint8.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/uint16.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/uint32.rs",
  ],
  leafLineCeiling: 224,
  testLineCeiling: 224,
});

enforceMigratedBuiltinFamily({
  name: "complex-component family",
  roots: [
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/complex_components",
    "crates/runmat-builtins/src/catalog/inference/math/complex_components",
    "crates/runmat-runtime/src/builtins/math/elementwise/complex_components",
  ],
  compositionFiles: [
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/complex_components/mod.rs",
    "crates/runmat-builtins/src/catalog/inference/math/complex_components/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/complex_components/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/complex_components/conj/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/complex_components/conj/tests/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/complex_components/imag/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/complex_components/imag/tests/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/complex_components/projection/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/complex_components/real/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/complex_components/real/tests/mod.rs",
  ],
  obsoletePaths: [
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/complex_components/contracts.rs",
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/complex_components/documentation",
    "crates/runmat-builtins/src/catalog/inference/numeric_component.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/conj.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/imag.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/real.rs",
    "docs/builtins/reference/conj.json",
    "docs/builtins/reference/imag.json",
    "docs/builtins/reference/real.json",
    "crates/runmat-runtime/src/builtins/builtins-json/conj.json",
    "crates/runmat-runtime/src/builtins/builtins-json/imag.json",
    "crates/runmat-runtime/src/builtins/builtins-json/real.json",
  ],
  leafLineCeiling: 224,
  testLineCeiling: 256,
});

enforceMigratedBuiltinFamily({
  name: "complex-construction family",
  roots: [
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/complex_construction",
    "crates/runmat-builtins/src/catalog/inference/math/complex_construction",
    "crates/runmat-runtime/src/builtins/math/elementwise/complex_construction",
  ],
  compositionFiles: [
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/complex_construction/mod.rs",
    "crates/runmat-builtins/src/catalog/inference/math/complex_construction/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/complex_construction/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/complex_construction/complex/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/complex_construction/complex/tests/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/complex_construction/complex/tests/host/mod.rs",
  ],
  obsoletePaths: [
    "crates/runmat-runtime/src/builtins/math/elementwise/complex.rs",
    "docs/builtins/reference/complex.json",
    "crates/runmat-runtime/src/builtins/builtins-json/complex.json",
  ],
  leafLineCeiling: 224,
  testLineCeiling: 320,
});

enforceMigratedBuiltinFamily({
  name: "typed binary inference families",
  roots: [
    "crates/runmat-builtins/src/catalog/inference/math/hypotenuse",
    "crates/runmat-builtins/src/catalog/inference/math/remainder",
    "crates/runmat-builtins/src/catalog/inference/math/two_argument_angle",
  ],
  compositionFiles: [
    "crates/runmat-builtins/src/catalog/inference/math/hypotenuse/mod.rs",
    "crates/runmat-builtins/src/catalog/inference/math/remainder/mod.rs",
    "crates/runmat-builtins/src/catalog/inference/math/two_argument_angle/mod.rs",
  ],
  obsoletePaths: [
    "crates/runmat-builtins/src/catalog/inference/math_binary.rs",
    "crates/runmat-builtins/src/catalog/inference/math_binary/tests.rs",
  ],
  leafLineCeiling: 192,
  testLineCeiling: 192,
});

enforceMigratedBuiltinFamily({
  name: "hypotenuse runtime family",
  roots: [
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/hypot",
    "crates/runmat-runtime/src/builtins/math/elementwise/hypot",
  ],
  compositionFiles: [
    "crates/runmat-runtime/src/builtins/math/elementwise/hypot/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/hypot/host/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/hypot/tests/mod.rs",
  ],
  obsoletePaths: [
    "crates/runmat-runtime/src/builtins/math/elementwise/hypot.rs",
  ],
  leafLineCeiling: 224,
  testLineCeiling: 224,
});

enforceMigratedBuiltinFamily({
  name: "numeric-limit family",
  roots: [
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/numeric_limits",
    "crates/runmat-builtins/src/catalog/inference/math/numeric_limits",
    "crates/runmat-runtime/src/builtins/math/elementwise/numeric_limits",
  ],
  compositionFiles: [
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/numeric_limits/mod.rs",
    "crates/runmat-builtins/src/catalog/inference/math/numeric_limits/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/numeric_limits/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/numeric_limits/tests/mod.rs",
  ],
  obsoletePaths: [
    "crates/runmat-builtins/src/catalog/inference/numeric_limit.rs",
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/numeric_limits/documentation/mod.rs",
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/numeric_limits/documentation/integer.rs",
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/numeric_limits/documentation/floating.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/numeric_limits.rs",
    "docs/builtins/reference/intmin.json",
    "docs/builtins/reference/intmax.json",
    "docs/builtins/reference/realmin.json",
    "docs/builtins/reference/realmax.json",
    "docs/builtins/reference/flintmax.json",
    "crates/runmat-runtime/src/builtins/builtins-json/intmin.json",
    "crates/runmat-runtime/src/builtins/builtins-json/intmax.json",
    "crates/runmat-runtime/src/builtins/builtins-json/realmin.json",
    "crates/runmat-runtime/src/builtins/builtins-json/realmax.json",
    "crates/runmat-runtime/src/builtins/builtins-json/flintmax.json",
  ],
  leafLineCeiling: 192,
  testLineCeiling: 256,
});

enforceMigratedBuiltinFamily({
  name: "heaviside identity",
  roots: [
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/heaviside",
    "crates/runmat-builtins/src/catalog/inference/math/heaviside",
    "crates/runmat-runtime/src/builtins/math/elementwise/heaviside",
  ],
  compositionFiles: [
    "crates/runmat-runtime/src/builtins/math/elementwise/heaviside/mod.rs",
    "crates/runmat-runtime/src/builtins/math/elementwise/heaviside/tests/mod.rs",
  ],
  obsoletePaths: [
    "crates/runmat-runtime/src/builtins/math/elementwise/heaviside.rs",
    "docs/builtins/reference/heaviside.json",
    "crates/runmat-runtime/src/builtins/builtins-json/heaviside.json",
  ],
  leafLineCeiling: 192,
  testLineCeiling: 224,
});

enforceMigratedBuiltinFamily({
  name: "clear-console identity",
  roots: [
    "crates/runmat-builtins/src/catalog/entries/io/console/clc",
  ],
  compositionFiles: [
    "crates/runmat-builtins/src/catalog/entries/io/console/mod.rs",
  ],
  obsoletePaths: [
    "docs/builtins/reference/clc.json",
    "crates/runmat-runtime/src/builtins/builtins-json/clc.json",
  ],
  leafLineCeiling: 192,
  testLineCeiling: 192,
});

enforceMigratedBuiltinFamily({
  name: "current-directory identity",
  roots: [
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/pwd",
    "crates/runmat-runtime/src/builtins/io/repl_fs/pwd",
  ],
  compositionFiles: [
    "crates/runmat-runtime/src/builtins/io/repl_fs/pwd/mod.rs",
  ],
  obsoletePaths: [
    "crates/runmat-runtime/src/builtins/io/repl_fs/pwd.rs",
    "docs/builtins/reference/pwd.json",
    "crates/runmat-runtime/src/builtins/builtins-json/pwd.json",
  ],
  leafLineCeiling: 192,
  testLineCeiling: 192,
});

enforceMigratedBuiltinFamily({
  name: "working-directory mutation identity",
  roots: [
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/cd",
    "crates/runmat-runtime/src/builtins/io/repl_fs/cd",
  ],
  compositionFiles: [
    "crates/runmat-runtime/src/builtins/io/repl_fs/cd/mod.rs",
  ],
  obsoletePaths: [
    "crates/runmat-runtime/src/builtins/io/repl_fs/cd.rs",
    "docs/builtins/reference/cd.json",
    "crates/runmat-runtime/src/builtins/builtins-json/cd.json",
  ],
  leafLineCeiling: 192,
  testLineCeiling: 192,
});

enforceMigratedBuiltinFamily({
  name: "search-path query and replacement identity",
  roots: [
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/path",
    "crates/runmat-runtime/src/builtins/io/repl_fs/path",
  ],
  compositionFiles: [
    "crates/runmat-runtime/src/builtins/io/repl_fs/path/mod.rs",
  ],
  obsoletePaths: [
    "crates/runmat-runtime/src/builtins/io/repl_fs/path.rs",
    "docs/builtins/reference/path.json",
    "crates/runmat-runtime/src/builtins/builtins-json/path.json",
  ],
  leafLineCeiling: 192,
  testLineCeiling: 192,
});

for (const [sourcePath, ceiling] of new Map([
  ["crates/runmat-builtins/src/catalog/inference/routing/io/mod.rs", 32],
  ["crates/runmat-builtins/src/catalog/entries/io/mod.rs", 32],
  ["crates/runmat-builtins/src/catalog/entries/io/console/mod.rs", 32],
  ["crates/runmat-builtins/src/catalog/entries/io/repl_fs/mod.rs", 32],
  ["crates/runmat-builtins/src/catalog/entries/io/repl_fs/exports.rs", 32],
  ["crates/runmat-builtins/src/catalog/entries/io/repl_fs/inference/mod.rs", 32],
  ["crates/runmat-builtins/src/catalog/entries/io/repl_fs/inference/directory.rs", 32],
  ["crates/runmat-builtins/src/catalog/entries/io/repl_fs/inference/file.rs", 32],
  ["crates/runmat-builtins/src/catalog/entries/io/repl_fs/inference/path.rs", 32],
  ["crates/runmat-builtins/src/catalog/entries/io/repl_fs/inference/working_directory.rs", 32],
  ["crates/runmat-builtins/src/catalog/entries/io/repl_fs/installation_path/mod.rs", 32],
  ["crates/runmat-builtins/src/catalog/entries/io/repl_fs/installation_path/inference.rs", 32],
  ["crates/runmat-builtins/src/catalog/entries/io/repl_fs/text_input.rs", 32],
  ["crates/runmat-builtins/src/catalog/entries/io/repl_fs/registry.rs", 32],
  ["crates/runmat-builtins/src/catalog/entries/io/repl_fs/search_path/mod.rs", 32],
])) {
  const lines = read(sourcePath).split("\n").length;
  if (lines > ceiling) {
    fail(`${sourcePath} exceeds its typed IO catalog composition boundary (found ${lines} lines; maximum ${ceiling})`);
  }
}
if (fs.existsSync(path.join(repo, "crates/runmat-builtins/src/catalog/inference/routing/io.rs"))) {
  fail("flat IO inference routing must not replace the domain hierarchy");
}
const ioRouting = rustSources("crates/runmat-builtins/src/catalog/inference/routing/io")
  .map(({ text }) => text)
  .join("\n");
if (/IoInferenceRule::(?:ClearConsole|ChangeDirectory|CurrentDirectory|SearchPath)\b/.test(ioRouting)) {
  fail("IO inference routing must use typed domain subrules instead of a growing flat builtin list");
}
if (rustSources("crates/runmat-builtins/src/catalog/inference/io").length > 0) {
  fail("IO inference belongs beside its catalog entries; the secondary inference tree must not return");
}
for (const { path: sourcePath, text } of rustSources(
  "crates/runmat-builtins/src/catalog/entries/io/repl_fs/inference",
)) {
  if (/"[^"\n]*"/.test(text)) {
    fail(`${sourcePath} is a typed routing layer and must not select policy from string literals`);
  }
  if (/\b(?:if|for|while|loop)\b/.test(text)) {
    fail(`${sourcePath} is a typed routing layer and must not acquire conditional domain policy`);
  }
  const matchCount = (text.match(/\bmatch\b/g) ?? []).length;
  if (matchCount !== 1) {
    fail(`${sourcePath} must contain exactly one exhaustive typed dispatch match (found ${matchCount})`);
  }
}
for (const obsolete of [
  "crates/runmat-builtins/src/catalog/inference/routing/io/console.rs",
  "crates/runmat-builtins/src/catalog/inference/routing/io/repl_fs.rs",
]) {
  if (fs.existsSync(path.join(repo, obsolete))) {
    fail(`${obsolete} duplicates the domain-owned IO inference router`);
  }
}

enforceMigratedBuiltinFamily({
  name: "installation-root identity",
  roots: [
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/installation_path",
    "crates/runmat-runtime/src/builtins/io/repl_fs/installation_path",
  ],
  compositionFiles: [
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/installation_path/mod.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/installation_path/mod.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/installation_path/matlabroot/mod.rs",
  ],
  obsoletePaths: [
    "docs/builtins/reference/matlabroot.json",
    "crates/runmat-runtime/src/builtins/builtins-json/matlabroot.json",
  ],
  leafLineCeiling: 192,
  testLineCeiling: 192,
});
const installationSources = [
  ...rustSources("crates/runmat-builtins/src/catalog/entries/io/repl_fs/installation_path"),
  ...rustSources("crates/runmat-runtime/src/builtins/io/repl_fs/installation_path"),
].map(({ text }) => text).join("\n");
if (/compat::MATLABROOT_DESCRIPTOR|type_resolvers::string_type/.test(installationSources)) {
  fail("matlabroot must use its identity-local catalog and inference authority");
}
const replCompatibility = read("crates/runmat-runtime/src/builtins/io/repl_fs/compat.rs");
if (/name\s*=\s*"matlabroot"|\bMATLABROOT_DESCRIPTOR\b|\bmatlabroot_builtin\b/.test(replCompatibility)) {
  fail("the REPL-filesystem compatibility bucket must not regain matlabroot ownership");
}

const cdRuntimePath = "crates/runmat-runtime/src/builtins/io/repl_fs/cd/mod.rs";
if (/\b(?:category|summary|keywords|descriptor|type_resolver|accel)\s*=/.test(read(cdRuntimePath))) {
  fail(`${cdRuntimePath} duplicates catalog-owned metadata or inference`);
}
if (/"(?:cd|chdir)"\s*(?:\||=>)/.test(read("crates/runmat-builtins/src/semantics.rs"))) {
  fail("legacy name-selected semantics must not reclaim cd authority");
}
if (/\bpub fn cd_type\b/.test(read("crates/runmat-runtime/src/builtins/io/type_resolvers.rs"))) {
  fail("legacy runtime type resolver must not reclaim cd inference authority");
}
if (/\bgather_if_needed_async\b/.test(rustSources("crates/runmat-runtime/src/builtins/io/repl_fs/cd").map(({ text }) => text).join("\n"))) {
  fail("cd must reject non-text values instead of gathering accelerator buffers");
}

const pathRuntimePath = "crates/runmat-runtime/src/builtins/io/repl_fs/path/mod.rs";
if (/\b(?:category|summary|keywords|descriptor|type_resolver|accel)\s*=/.test(read(pathRuntimePath))) {
  fail(`${pathRuntimePath} duplicates catalog-owned metadata or inference`);
}
if (/"path"\s*(?:\||=>)/.test(read("crates/runmat-builtins/src/semantics.rs"))) {
  fail("legacy name-selected semantics must not reclaim path authority");
}
if (/\bpub fn path_type\b/.test(read("crates/runmat-runtime/src/builtins/io/type_resolvers.rs"))) {
  fail("legacy runtime type resolver must not reclaim path inference authority");
}

for (const identity of ["addpath", "rmpath"]) {
  enforceMigratedBuiltinFamily({
    name: `${identity} search-path mutation identity`,
    roots: [
      `crates/runmat-builtins/src/catalog/entries/io/repl_fs/${identity}`,
      "crates/runmat-builtins/src/catalog/entries/io/repl_fs/search_path",
      `crates/runmat-runtime/src/builtins/io/repl_fs/${identity}`,
    ],
    compositionFiles: [
      `crates/runmat-builtins/src/catalog/entries/io/repl_fs/${identity}/mod.rs`,
      `crates/runmat-runtime/src/builtins/io/repl_fs/${identity}/mod.rs`,
    ],
    obsoletePaths: [
      `crates/runmat-runtime/src/builtins/io/repl_fs/${identity}.rs`,
      `docs/builtins/reference/${identity}.json`,
      `crates/runmat-runtime/src/builtins/builtins-json/${identity}.json`,
    ],
    leafLineCeiling: 192,
    testLineCeiling: 192,
  });
}
enforceMigratedBuiltinFamily({
  name: "recursive search-path generation identity",
  roots: [
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/genpath",
    "crates/runmat-runtime/src/builtins/io/repl_fs/genpath",
  ],
  compositionFiles: [
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/genpath/mod.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/genpath/mod.rs",
  ],
  obsoletePaths: [
    "crates/runmat-runtime/src/builtins/io/repl_fs/genpath.rs",
    "docs/builtins/reference/genpath.json",
    "crates/runmat-runtime/src/builtins/builtins-json/genpath.json",
  ],
  leafLineCeiling: 192,
  testLineCeiling: 192,
});
enforceMigratedBuiltinFamily({
  name: "search-path persistence identity",
  roots: [
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/savepath",
    "crates/runmat-runtime/src/builtins/io/repl_fs/savepath",
  ],
  compositionFiles: [
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/savepath/mod.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/savepath/mod.rs",
  ],
  obsoletePaths: [
    "crates/runmat-runtime/src/builtins/io/repl_fs/savepath.rs",
    "docs/builtins/reference/savepath.json",
    "crates/runmat-runtime/src/builtins/builtins-json/savepath.json",
  ],
  leafLineCeiling: 192,
  testLineCeiling: 192,
});
enforceMigratedBuiltinFamily({
  name: "process-environment family",
  roots: [
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/environment",
    "crates/runmat-runtime/src/builtins/io/repl_fs/environment",
  ],
  compositionFiles: [
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/environment/mod.rs",
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/environment/inference.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/environment/mod.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/environment/getenv/mod.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/environment/setenv/mod.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/environment/isenv/mod.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/environment/unsetenv/mod.rs",
  ],
  obsoletePaths: [
    "crates/runmat-runtime/src/builtins/io/repl_fs/getenv.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/setenv.rs",
    "docs/builtins/reference/getenv.json",
    "docs/builtins/reference/setenv.json",
    "docs/builtins/reference/isenv.json",
    "docs/builtins/reference/unsetenv.json",
    "crates/runmat-runtime/src/builtins/builtins-json/getenv.json",
    "crates/runmat-runtime/src/builtins/builtins-json/setenv.json",
    "crates/runmat-runtime/src/builtins/builtins-json/isenv.json",
    "crates/runmat-runtime/src/builtins/builtins-json/unsetenv.json",
  ],
  leafLineCeiling: 192,
  testLineCeiling: 192,
});
enforceMigratedBuiltinFamily({
  name: "temporary-path family",
  roots: [
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/temporary_path",
    "crates/runmat-runtime/src/builtins/io/repl_fs/temporary_path",
  ],
  compositionFiles: [
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/temporary_path/mod.rs",
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/temporary_path/inference.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/temporary_path/mod.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/temporary_path/tempdir/mod.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/temporary_path/tempname/mod.rs",
  ],
  obsoletePaths: [
    "crates/runmat-runtime/src/builtins/io/repl_fs/tempdir.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/tempname.rs",
    "docs/builtins/reference/tempdir.json",
    "docs/builtins/reference/tempname.json",
    "crates/runmat-runtime/src/builtins/builtins-json/tempdir.json",
    "crates/runmat-runtime/src/builtins/builtins-json/tempname.json",
  ],
  leafLineCeiling: 192,
  testLineCeiling: 192,
});
enforceMigratedBuiltinFamily({
  name: "lexical-path family",
  roots: [
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/path_syntax",
    "crates/runmat-runtime/src/builtins/io/repl_fs/path_syntax",
  ],
  compositionFiles: [
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/path_syntax/mod.rs",
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/path_syntax/inference.rs",
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/path_syntax/fullfile/mod.rs",
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/path_syntax/fileparts/mod.rs",
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/path_syntax/filesep/mod.rs",
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/path_syntax/pathsep/mod.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/path_syntax/mod.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/path_syntax/fullfile/mod.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/path_syntax/fileparts/mod.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/path_syntax/filesep/mod.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/path_syntax/pathsep/mod.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/path_syntax/lexical/mod.rs",
  ],
  obsoletePaths: [
    "crates/runmat-runtime/src/builtins/io/repl_fs/fullfile.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/fileparts.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/filesep.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/pathsep.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/path_syntax/lexical.rs",
    "docs/builtins/reference/fullfile.json",
    "docs/builtins/reference/fileparts.json",
    "docs/builtins/reference/filesep.json",
    "docs/builtins/reference/pathsep.json",
    "crates/runmat-runtime/src/builtins/builtins-json/fullfile.json",
    "crates/runmat-runtime/src/builtins/builtins-json/fileparts.json",
    "crates/runmat-runtime/src/builtins/builtins-json/filesep.json",
    "crates/runmat-runtime/src/builtins/builtins-json/pathsep.json",
  ],
  leafLineCeiling: 192,
  testLineCeiling: 192,
});
enforceMigratedBuiltinFamily({
  name: "directory-lifecycle family",
  roots: [
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/directory_lifecycle",
    "crates/runmat-runtime/src/builtins/io/repl_fs/directory_lifecycle",
  ],
  compositionFiles: [
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/directory_lifecycle/mod.rs",
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/directory_lifecycle/inference.rs",
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/directory_lifecycle/mkdir/mod.rs",
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/directory_lifecycle/rmdir/mod.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/directory_lifecycle/mod.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/directory_lifecycle/mkdir/mod.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/directory_lifecycle/rmdir/mod.rs",
  ],
  obsoletePaths: [
    "crates/runmat-runtime/src/builtins/io/repl_fs/mkdir.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/rmdir.rs",
    "docs/builtins/reference/mkdir.json",
    "docs/builtins/reference/rmdir.json",
    "crates/runmat-runtime/src/builtins/builtins-json/mkdir.json",
    "crates/runmat-runtime/src/builtins/builtins-json/rmdir.json",
  ],
  leafLineCeiling: 192,
  testLineCeiling: 192,
});
enforceMigratedBuiltinFamily({
  name: "file-transfer family",
  roots: [
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/file_transfer",
    "crates/runmat-runtime/src/builtins/io/repl_fs/file_transfer",
  ],
  compositionFiles: [
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/file_transfer/mod.rs",
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/file_transfer/inference.rs",
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/file_transfer/copyfile/mod.rs",
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/file_transfer/movefile/mod.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/file_transfer/mod.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/file_transfer/copyfile/mod.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/file_transfer/movefile/mod.rs",
  ],
  obsoletePaths: [
    "crates/runmat-runtime/src/builtins/io/repl_fs/copyfile.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/movefile.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/file_transfer/copyfile/legacy.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/file_transfer/movefile/legacy.rs",
    "docs/builtins/reference/copyfile.json",
    "docs/builtins/reference/movefile.json",
    "crates/runmat-runtime/src/builtins/builtins-json/copyfile.json",
    "crates/runmat-runtime/src/builtins/builtins-json/movefile.json",
  ],
  leafLineCeiling: 192,
  testLineCeiling: 192,
});
enforceMigratedBuiltinFamily({
  name: "path-predicate family",
  roots: [
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/path_predicate",
    "crates/runmat-runtime/src/builtins/io/repl_fs/path_predicate",
  ],
  compositionFiles: [
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/path_predicate/mod.rs",
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/path_predicate/inference.rs",
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/path_predicate/isfile/mod.rs",
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/path_predicate/isfolder/mod.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/path_predicate/mod.rs",
  ],
  obsoletePaths: [
    "docs/builtins/reference/isfile.json",
    "docs/builtins/reference/isfolder.json",
    "crates/runmat-runtime/src/builtins/builtins-json/isfile.json",
    "crates/runmat-runtime/src/builtins/builtins-json/isfolder.json",
  ],
  leafLineCeiling: 192,
  testLineCeiling: 192,
});
enforceMigratedBuiltinFamily({
  name: "directory-listing family",
  roots: [
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/directory_listing",
    "crates/runmat-runtime/src/builtins/io/repl_fs/directory_listing",
  ],
  compositionFiles: [
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/directory_listing/mod.rs",
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/directory_listing/inference.rs",
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/directory_listing/dir/mod.rs",
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/directory_listing/ls/mod.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/directory_listing/mod.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/directory_listing/dir/mod.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/directory_listing/ls/mod.rs",
  ],
  obsoletePaths: [
    "crates/runmat-runtime/src/builtins/io/repl_fs/dir.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/ls.rs",
    "docs/builtins/reference/dir.json",
    "docs/builtins/reference/ls.json",
    "crates/runmat-runtime/src/builtins/builtins-json/dir.json",
    "crates/runmat-runtime/src/builtins/builtins-json/ls.json",
  ],
  leafLineCeiling: 192,
  testLineCeiling: 192,
});
enforceMigratedBuiltinFamily({
  name: "source-inventory family",
  roots: [
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/source_inventory",
    "crates/runmat-runtime/src/builtins/io/repl_fs/source_inventory",
  ],
  compositionFiles: [
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/source_inventory/mod.rs",
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/source_inventory/inference.rs",
    "crates/runmat-builtins/src/catalog/entries/io/repl_fs/source_inventory/what/mod.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/source_inventory/mod.rs",
    "crates/runmat-runtime/src/builtins/io/repl_fs/source_inventory/what/mod.rs",
  ],
  obsoletePaths: [
    "docs/builtins/reference/what.json",
    "crates/runmat-runtime/src/builtins/builtins-json/what.json",
  ],
  leafLineCeiling: 192,
  testLineCeiling: 192,
});
for (const sourcePath of [
  "crates/runmat-builtins/src/catalog/inference.rs",
  "crates/runmat-builtins/src/catalog/entries/io/mod.rs",
  "crates/runmat-builtins/src/catalog/entries/io/repl_fs/inference/mod.rs",
  "crates/runmat-builtins/src/catalog/entries/io/repl_fs/registry.rs",
]) {
  if (/"what"/.test(read(sourcePath))) {
    fail(`${sourcePath} must route source-inventory inference by typed rules, not builtin names`);
  }
}
if (/(?:WHAT_(?:DESCRIPTOR|INTEGER_AUDIT)|name\s*=\s*"what"|what_builtin)/.test(
  read("crates/runmat-runtime/src/builtins/io/repl_fs/compat.rs"),
)) {
  fail("the REPL-filesystem compatibility bucket must not reclaim what ownership");
}
for (const sourcePath of [
  "crates/runmat-builtins/src/catalog/inference.rs",
  "crates/runmat-builtins/src/catalog/entries/io/mod.rs",
  "crates/runmat-builtins/src/catalog/entries/io/repl_fs/inference/mod.rs",
  "crates/runmat-builtins/src/catalog/entries/io/repl_fs/registry.rs",
]) {
  if (/"(?:dir|ls)"/.test(read(sourcePath))) {
    fail(`${sourcePath} must route directory-listing inference by typed rules, not builtin names`);
  }
}
if (/\bpub fn (?:dir|ls)_type\b/.test(
  read("crates/runmat-runtime/src/builtins/io/type_resolvers.rs"),
)) {
  fail("legacy dir/ls type resolvers must not return after catalog cutover");
}
if (/"(?:dir|ls)"\s*(?:\||=>)/.test(
  read("crates/runmat-builtins/src/semantics.rs"),
)) {
  fail("legacy name-selected semantics must not reclaim directory-listing authority");
}
for (const { path: sourcePath, text } of rustSources(
  "crates/runmat-runtime/src/builtins/io/repl_fs/directory_listing",
)) {
  if (/glob::glob\b/.test(text)) {
    fail(`${sourcePath} must enumerate through runmat-filesystem, not the host glob walker`);
  }
}
for (const sourcePath of [
  "crates/runmat-builtins/src/catalog/inference.rs",
  "crates/runmat-builtins/src/catalog/entries/io/mod.rs",
  "crates/runmat-builtins/src/catalog/entries/io/repl_fs/inference/mod.rs",
  "crates/runmat-builtins/src/catalog/entries/io/repl_fs/registry.rs",
  "crates/runmat-builtins/src/catalog/entries/io/repl_fs/path_predicate/inference.rs",
]) {
  if (/"(?:isfile|isfolder)"/.test(read(sourcePath))) {
    fail(`${sourcePath} must route path-predicate inference by typed rules, not builtin names`);
  }
}
const replFsCompatibility = read("crates/runmat-runtime/src/builtins/io/repl_fs/compat.rs");
if (/(?:ISFILE|ISFOLDER)_(?:DESCRIPTOR|INTEGER_AUDIT)|name\s*=\s*"(?:isfile|isfolder)"|path_predicate_builtin/.test(replFsCompatibility)) {
  fail("the REPL-filesystem compatibility bucket must not reclaim path-predicate ownership");
}
if (/"(?:isfile|isfolder)"\s*(?:\||=>)/.test(
  read("crates/runmat-builtins/src/semantics.rs"),
)) {
  fail("legacy name-selected semantics must not reclaim path-predicate authority");
}
for (const sourcePath of [
  "crates/runmat-builtins/src/catalog/inference.rs",
  "crates/runmat-builtins/src/catalog/entries/io/mod.rs",
  "crates/runmat-builtins/src/catalog/entries/io/repl_fs/inference/mod.rs",
  "crates/runmat-builtins/src/catalog/entries/io/repl_fs/registry.rs",
  "crates/runmat-builtins/src/catalog/entries/io/repl_fs/file_transfer/mod.rs",
  "crates/runmat-builtins/src/catalog/entries/io/repl_fs/file_transfer/inference.rs",
]) {
  if (/"(?:copyfile|movefile)"/.test(read(sourcePath))) {
    fail(`${sourcePath} must route file-transfer inference by typed rules, not builtin names`);
  }
}
if (/\bpub fn (?:copyfile|movefile)_type\b/.test(
  read("crates/runmat-runtime/src/builtins/io/type_resolvers.rs"),
)) {
  fail("legacy runtime type resolvers must not reclaim file-transfer inference authority");
}
if (/"(?:copyfile|movefile)"\s*(?:\||=>)/.test(
  read("crates/runmat-builtins/src/semantics.rs"),
)) {
  fail("legacy name-selected semantics must not reclaim file-transfer authority");
}
for (const sourcePath of [
  "crates/runmat-builtins/src/catalog/inference.rs",
  "crates/runmat-builtins/src/catalog/entries/io/mod.rs",
  "crates/runmat-builtins/src/catalog/entries/io/repl_fs/inference/mod.rs",
  "crates/runmat-builtins/src/catalog/entries/io/repl_fs/registry.rs",
  "crates/runmat-builtins/src/catalog/entries/io/repl_fs/directory_lifecycle/mod.rs",
  "crates/runmat-builtins/src/catalog/entries/io/repl_fs/directory_lifecycle/inference.rs",
]) {
  if (/"(?:mkdir|rmdir)"/.test(read(sourcePath))) {
    fail(`${sourcePath} must route directory lifecycle inference by typed rules, not builtin names`);
  }
}
if (/\bpub fn (?:mkdir|rmdir)_type\b/.test(
  read("crates/runmat-runtime/src/builtins/io/type_resolvers.rs"),
)) {
  fail("legacy runtime type resolvers must not reclaim directory-lifecycle inference authority");
}
if (/"(?:mkdir|rmdir)"\s*(?:\||=>)/.test(
  read("crates/runmat-builtins/src/semantics.rs"),
)) {
  fail("legacy name-selected semantics must not reclaim directory-lifecycle authority");
}
for (const [sourcePath, ceiling] of [
  ["crates/runmat-runtime/src/warnings/mod.rs", 32],
  ["crates/runmat-runtime/src/warnings/emission.rs", 128],
  ["crates/runmat-runtime/src/warnings/policy.rs", 256],
  ["crates/runmat-runtime/src/warnings/tests.rs", 128],
]) {
  const lines = read(sourcePath).split("\n").length;
  if (lines > ceiling) {
    fail(`${sourcePath} exceeds its warning-service role boundary (found ${lines} lines; maximum ${ceiling})`);
  }
}
const warningBuiltin = read("crates/runmat-runtime/src/builtins/diagnostics/warning.rs");
if (/\b(?:struct WarningPolicy|struct WarningManager|static WARNING_MANAGER)\b/.test(warningBuiltin)) {
  fail("the warning builtin must not own cross-cutting warning policy state");
}
for (const sourcePath of [
  "crates/runmat-builtins/src/catalog/inference.rs",
  "crates/runmat-builtins/src/catalog/entries/io/mod.rs",
  "crates/runmat-builtins/src/catalog/entries/io/repl_fs/inference/mod.rs",
  "crates/runmat-builtins/src/catalog/entries/io/repl_fs/registry.rs",
  "crates/runmat-builtins/src/catalog/entries/io/repl_fs/path_syntax/mod.rs",
  "crates/runmat-builtins/src/catalog/entries/io/repl_fs/path_syntax/inference.rs",
]) {
  const source = read(sourcePath);
  if (/"(?:fullfile|fileparts|filesep|pathsep)"/.test(source)) {
    fail(`${sourcePath} must dispatch lexical-path behavior through typed rules and entries`);
  }
}
if (/\bpub fn (?:fullfile|fileparts|filesep|pathsep)_type\b/.test(
  read("crates/runmat-runtime/src/builtins/io/type_resolvers.rs"),
)) {
  fail("legacy runtime type resolvers must not reclaim lexical-path inference authority");
}
if (/"(?:fullfile|fileparts|filesep|pathsep)"\s*(?:\||=>)/.test(
  read("crates/runmat-builtins/src/semantics.rs"),
)) {
  fail("legacy name-selected semantics must not reclaim lexical-path authority");
}
for (const sourcePath of [
  "crates/runmat-builtins/src/catalog/inference.rs",
  "crates/runmat-builtins/src/catalog/entries/io/mod.rs",
  "crates/runmat-builtins/src/catalog/entries/io/repl_fs/inference/mod.rs",
  "crates/runmat-builtins/src/catalog/entries/io/repl_fs/registry.rs",
  "crates/runmat-builtins/src/catalog/entries/io/repl_fs/temporary_path/mod.rs",
  "crates/runmat-builtins/src/catalog/entries/io/repl_fs/temporary_path/inference.rs",
]) {
  const source = read(sourcePath);
  if (/"(?:tempdir|tempname)"/.test(source)) {
    fail(`${sourcePath} must dispatch temporary-path behavior through typed rules and entries`);
  }
}
if (/\bpub fn (?:tempdir|tempname)_type\b/.test(
  read("crates/runmat-runtime/src/builtins/io/type_resolvers.rs"),
)) {
  fail("legacy runtime type resolvers must not reclaim temporary-path inference authority");
}
if (/"(?:tempdir|tempname)"\s*(?:\||=>)/.test(
  read("crates/runmat-builtins/src/semantics.rs"),
)) {
  fail("legacy name-selected semantics must not reclaim temporary-path authority");
}
for (const sourcePath of [
  "crates/runmat-builtins/src/catalog/entries/io/repl_fs/environment/mod.rs",
  "crates/runmat-builtins/src/catalog/entries/io/repl_fs/environment/inference.rs",
  "crates/runmat-builtins/src/catalog/entries/io/repl_fs/inference/mod.rs",
  "crates/runmat-builtins/src/catalog/entries/io/repl_fs/registry.rs",
]) {
  const source = read(sourcePath);
  if (/"(?:getenv|setenv|isenv|unsetenv)"/.test(source)) {
    fail(`${sourcePath} must dispatch process-environment behavior through typed rules and entries`);
  }
}
for (const { path: sourcePath, text } of rustSources(
  "crates/runmat-runtime/src/builtins/io/repl_fs/environment",
)) {
  if (/\bstd::env::/.test(text)) {
    fail(`${sourcePath} bypasses the native/browser process-environment service`);
  }
}
const legacyIoResolvers = read("crates/runmat-runtime/src/builtins/io/type_resolvers.rs");
if (/\bpub fn (?:addpath|rmpath|genpath|savepath|getenv|setenv|isenv|unsetenv)_type\b/.test(legacyIoResolvers)) {
  fail("legacy runtime type resolvers must not reclaim migrated search-path or environment inference authority");
}
if (/"(?:addpath|rmpath|genpath|savepath|getenv|setenv|isenv|unsetenv)"\s*(?:\||=>)/.test(read("crates/runmat-builtins/src/semantics.rs"))) {
  fail("legacy name-selected semantics must not reclaim migrated search-path or environment authority");
}

if (/\b(?:category|summary|keywords|descriptor|type_resolver)\s*=/.test(
  read("crates/runmat-runtime/src/builtins/io/repl_fs/pwd/mod.rs"),
)) {
  fail("pwd runtime binding duplicates catalog-owned metadata or inference");
}
if (/"pwd"\s*(?:\||=>)/.test(read("crates/runmat-builtins/src/semantics.rs"))) {
  fail("legacy name-selected semantics must not reclaim pwd authority");
}
if (/\bpub fn pwd_type\b/.test(
  read("crates/runmat-runtime/src/builtins/io/type_resolvers.rs"),
)) {
  fail("legacy runtime type resolver must not reclaim pwd inference authority");
}

const clcRuntimePath = "crates/runmat-runtime/src/builtins/io/clc.rs";
const clcRuntime = read(clcRuntimePath);
if (clcRuntime.split("\n").length > 96) {
  fail(`${clcRuntimePath} exceeds its binding-and-control boundary`);
}
if (/\b(?:BuiltinDescriptor|BuiltinContractDeclaration)\s*=/.test(clcRuntime)) {
  fail(`${clcRuntimePath} duplicates catalog-owned clc metadata`);
}
if (/"clc"\s*(?:\||=>)/.test(read("crates/runmat-builtins/src/semantics.rs"))) {
  fail("legacy name-selected semantics must not reclaim clc authority");
}

enforceMigratedBuiltinFamily({
  name: "provider restoration service",
  roots: [
    "crates/runmat-runtime/src/builtins/common/provider_restore",
  ],
  compositionFiles: [
    "crates/runmat-runtime/src/builtins/common/provider_restore/mod.rs",
  ],
  obsoletePaths: [
    "crates/runmat-runtime/src/builtins/common/provider_restore.rs",
  ],
  leafLineCeiling: 224,
});

for (const [sourcePath, ceiling] of new Map([
  ["crates/runmat-builtins/src/catalog/inference/math/binary_containers.rs", 160],
  ["crates/runmat-builtins/src/catalog/inference/math/test_support.rs", 64],
])) {
  const lines = read(sourcePath).split("\n").length;
  if (lines > ceiling) {
    fail(`${sourcePath} exceeds its typed binary inference support boundary (found ${lines} lines; maximum ${ceiling})`);
  }
}

const gammaFunctionBoundaries = new Map([
  ["crates/runmat-builtins/src/catalog/inference/math/gamma_functions/mod.rs", 40],
  ["crates/runmat-builtins/src/catalog/inference/math/gamma_functions/common.rs", 64],
  ["crates/runmat-builtins/src/catalog/inference/math/gamma_functions/gamma.rs", 96],
  ["crates/runmat-builtins/src/catalog/inference/math/gamma_functions/gammaln.rs", 112],
  ["crates/runmat-builtins/src/catalog/inference/math/gamma_functions/tests.rs", 192],
  ["crates/runmat-builtins/src/catalog/entries/math/elementwise/gamma_functions/mod.rs", 32],
  ["crates/runmat-runtime/src/builtins/math/elementwise/gamma_functions/mod.rs", 32],
  ["crates/runmat-runtime/src/builtins/math/elementwise/gamma_functions/gamma/mod.rs", 320],
  ["crates/runmat-runtime/src/builtins/math/elementwise/gamma_functions/gamma/tests.rs", 320],
  ["crates/runmat-runtime/src/builtins/math/elementwise/gamma_functions/gammaln/mod.rs", 384],
  ["crates/runmat-runtime/src/builtins/math/elementwise/gamma_functions/gammaln/tests.rs", 640],
]);
for (const identity of ["gamma", "gammaln"]) {
  gammaFunctionBoundaries.set(
    `crates/runmat-builtins/src/catalog/entries/math/elementwise/gamma_functions/${identity}/mod.rs`,
    192
  );
  gammaFunctionBoundaries.set(
    `crates/runmat-builtins/src/catalog/entries/math/elementwise/gamma_functions/${identity}/documentation.rs`,
    160
  );
}
for (const [sourcePath, ceiling] of gammaFunctionBoundaries) {
  const lines = read(sourcePath).split("\n").length;
  if (lines > ceiling) {
    fail(
      `${sourcePath} exceeds its gamma-function family boundary ` +
      `(found ${lines} lines; maximum ${ceiling})`
    );
  }
}
for (const sourcePath of [
  "crates/runmat-builtins/src/catalog/inference/math_special.rs",
  "crates/runmat-builtins/src/catalog/entries/math/elementwise/gamma",
  "crates/runmat-builtins/src/catalog/entries/math/elementwise/gammaln",
  "crates/runmat-runtime/src/builtins/math/elementwise/gamma.rs",
  "crates/runmat-runtime/src/builtins/math/elementwise/gammaln.rs",
  "crates/runmat-runtime/src/builtins/math/elementwise/gammaln",
]) {
  if (fs.existsSync(path.join(repo, sourcePath))) {
    fail(`${sourcePath} is obsolete flat gamma-function debt and must not return`);
  }
}
for (const { path: sourcePath, text } of rustSources("crates/runmat-runtime/src/builtins/math/elementwise/gamma_functions")) {
  if (/\b(?:BuiltinDescriptor|BuiltinIntegerCapabilityDescriptor)\s*=/.test(text)) {
    fail(`${sourcePath} duplicates catalog-owned gamma-function metadata`);
  }
  if (/\.to_string\(\)\.contains\(\s*"unary_(?:gamma|gammaln) not supported"/.test(text)) {
    fail(`${sourcePath} interprets provider capability from error text; use the typed provider result`);
  }
}

const distributedInferenceBoundaries = new Map([
  ["crates/runmat-builtins/src/catalog/inference/distributed/mod.rs", 64],
  ["crates/runmat-builtins/src/catalog/inference/distributed/admission.rs", 96],
  ["crates/runmat-builtins/src/catalog/inference/distributed/mapping.rs", 96],
]);
for (const [sourcePath, ceiling] of distributedInferenceBoundaries) {
  const lines = read(sourcePath).split("\n").length;
  if (lines > ceiling) {
    fail(
      `${sourcePath} exceeds its distributed inference boundary ` +
      `(found ${lines} lines; maximum ${ceiling})`
    );
  }
}
if (fs.existsSync(path.join(repo, "crates/runmat-builtins/src/catalog/inference/distributed_semantics.rs"))) {
  fail("the monolithic distributed_semantics.rs inference module must not return");
}

const discreteNumberTheoryRoots = [
  "crates/runmat-builtins/src/catalog/entries/math/discrete",
  "crates/runmat-builtins/src/catalog/inference/math/discrete",
  "crates/runmat-runtime/src/builtins/math/discrete",
];
for (const sourceRoot of discreteNumberTheoryRoots) {
  for (const { path: sourcePath, text } of rustSources(sourceRoot)) {
    const lines = text.split("\n").length;
    const relativePath = sourcePath.slice(sourceRoot.length + 1);
    const isCompositionRoot = relativePath === "mod.rs";
    const isDocumentation = sourcePath.endsWith("/documentation.rs");
    const isTest = sourcePath.endsWith("/tests.rs") || sourcePath.endsWith("_tests.rs");
    const ceiling = isCompositionRoot ? 64 : isDocumentation ? 320 : isTest ? 512 : 256;
    if (lines > ceiling) {
      fail(
        sourcePath + " exceeds its role-based discrete number-theory boundary " +
        "(found " + lines + " lines; maximum " + ceiling + ")"
      );
    }
  }
}
for (const sourcePath of [
  "crates/runmat-runtime/src/builtins/math/elementwise/factorial.rs",
  "crates/runmat-runtime/src/builtins/math/elementwise/factorial",
  "crates/runmat-runtime/src/builtins/math/discrete/factor.rs",
  "crates/runmat-runtime/src/builtins/math/discrete/lcm.rs",
  "crates/runmat-runtime/src/builtins/math/discrete/isprime.rs",
  "crates/runmat-runtime/src/builtins/math/discrete/integer_number_theory.rs",
  "crates/runmat-runtime/src/builtins/math/discrete/primes.rs",
  "docs/builtins/reference/factorial.json",
  "docs/builtins/reference/factor.json",
  "docs/builtins/reference/isprime.json",
  "docs/builtins/reference/lcm.json",
  "docs/builtins/reference/primes.json",
  "crates/runmat-runtime/src/builtins/builtins-json/factorial.json",
  "crates/runmat-runtime/src/builtins/builtins-json/factor.json",
  "crates/runmat-runtime/src/builtins/builtins-json/isprime.json",
  "crates/runmat-runtime/src/builtins/builtins-json/lcm.json",
  "crates/runmat-runtime/src/builtins/builtins-json/primes.json",
]) {
  if (fs.existsSync(path.join(repo, sourcePath))) {
    fail(`${sourcePath} is obsolete discrete number-theory identity or documentation debt and must not return`);
  }
}
for (const sourceRoot of ["crates/runmat-runtime/src/builtins/math/discrete"]) {
  for (const { path: sourcePath, text } of rustSources(sourceRoot)) {
    if (/\b(?:BuiltinDescriptor|BuiltinIntegerCapabilityDescriptor)\s*=/.test(text)) {
      fail(`${sourcePath} duplicates catalog-owned discrete number-theory metadata`);
    }
  }
}
for (const { path: sourcePath, text } of rustSources("crates/runmat-runtime/src/builtins/math/discrete/factorial")) {
  if (/\.to_string\(\)\.contains\(|\.message\(\)\.contains\([^)]*(?:unsupported|factorial)/.test(text)) {
    fail(`${sourcePath} selects factorial behavior from error text; use typed runtime policy`);
  }
}

const logicalPredicateRoots = [
  "crates/runmat-builtins/src/catalog/entries/logical/tests",
  "crates/runmat-runtime/src/builtins/logical/tests",
];
for (const sourceRoot of logicalPredicateRoots) {
  for (const { path: sourcePath, text } of rustSources(sourceRoot)) {
    const lines = text.split("\n").length;
    const relativePath = sourcePath.slice(sourceRoot.length + 1);
    const isCompositionRoot = relativePath === "mod.rs";
    const isDocumentation = sourcePath.endsWith("/documentation.rs");
    const isTest = sourcePath.endsWith("/tests.rs");
    const isLegacyClassificationEngine = relativePath === "classification.rs";
    const ceiling = isCompositionRoot
      ? 64
      : isDocumentation
        ? 320
        : isTest
          ? 512
          : isLegacyClassificationEngine
            ? 448
            : 256;
    if (lines > ceiling) {
      fail(
        sourcePath + " exceeds its role-based logical-predicate boundary " +
        "(found " + lines + " lines; maximum " + ceiling + ")"
      );
    }
  }
}
for (const sourcePath of [
  "crates/runmat-runtime/src/builtins/cells/core/iscell.rs",
  "crates/runmat-runtime/src/builtins/cells/core/iscellstr.rs",
  "docs/builtins/reference/iscell.json",
  "docs/builtins/reference/iscellstr.json",
  "crates/runmat-runtime/src/builtins/builtins-json/iscell.json",
  "crates/runmat-runtime/src/builtins/builtins-json/iscellstr.json",
]) {
  if (fs.existsSync(path.join(repo, sourcePath))) {
    fail(`${sourcePath} is obsolete logical-predicate identity or documentation debt and must not return`);
  }
}
for (const { path: sourcePath, text } of rustSources("crates/runmat-runtime/src/builtins/logical/tests")) {
  if (/\b(?:BuiltinDescriptor|BuiltinIntegerCapabilityDescriptor|BuiltinIntegerAuditDescriptor)\s*=/.test(text)) {
    fail(`${sourcePath} duplicates catalog-owned logical-predicate metadata`);
  }
}

enforceMigratedBuiltinFamily({
  name: "binary-arithmetic catalog",
  roots: ["crates/runmat-builtins/src/catalog/entries/math/elementwise/binary_arithmetic"],
  compositionFiles: [
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/binary_arithmetic/mod.rs",
    "crates/runmat-builtins/src/catalog/entries/math/elementwise/binary_arithmetic/inference/mod.rs",
  ],
  obsoletePaths: [],
});
const binaryArithmeticRuntimeRoot =
  "crates/runmat-runtime/src/builtins/math/elementwise/binary_arithmetic";
const binaryArithmeticRuntimeComposition = read(`${binaryArithmeticRuntimeRoot}/mod.rs`);
if (binaryArithmeticRuntimeComposition.split("\n").length > 64) {
  fail("binary-arithmetic runtime composition must remain a bounded module router");
}
if (fs.existsSync(path.join(repo, `${binaryArithmeticRuntimeRoot}/output_prototype.rs`))) {
  fail("binary-arithmetic output-prototype service must remain split by domain role");
}
const binaryArithmeticProviderSupport = read(
  `${binaryArithmeticRuntimeRoot}/provider_support.rs`
);
if (
  binaryArithmeticProviderSupport.split("\n").length > 160 ||
  Buffer.byteLength(binaryArithmeticProviderSupport, "utf8") > 16 * 1024
) {
  fail("binary-arithmetic provider support exceeds its bounded family role");
}
for (const { path: sourcePath, text } of rustSources(binaryArithmeticRuntimeRoot)) {
  if (/\b(?:BuiltinDescriptor|BuiltinIntegerCapabilityDescriptor)\s*=/.test(text)) {
    fail(`${sourcePath} duplicates catalog-owned binary-arithmetic metadata`);
  }
  if (/match\s+[^\n{]*(?:builtin_)?name|(?:builtin_)?name\s*(?:==|!=)/.test(text)) {
    fail(`${sourcePath} selects binary-arithmetic semantics from an identity string`);
  }
}
for (const identity of ["minus", "plus", "times"]) {
  const obsoleteFlatPath = `crates/runmat-runtime/src/builtins/math/elementwise/${identity}.rs`;
  if (fs.existsSync(path.join(repo, obsoleteFlatPath))) {
    fail(`${obsoleteFlatPath} is obsolete flat binary-arithmetic layout debt`);
  }
  const identityPrototypePath = `${binaryArithmeticRuntimeRoot}/${identity}/output_prototype.rs`;
  if (fs.existsSync(path.join(repo, identityPrototypePath))) {
    fail(`${identityPrototypePath} duplicates the family-owned output-prototype service`);
  }
  const identityPrototypeRoot = `${binaryArithmeticRuntimeRoot}/${identity}/output_prototype`;
  if (fs.existsSync(path.join(repo, identityPrototypeRoot))) {
    fail(`${identityPrototypeRoot} duplicates the family-owned output-prototype service`);
  }
}
for (const { path: sourcePath, text } of rustSources(binaryArithmeticRuntimeRoot)) {
  if (/\b(?:broadcast_reps|extract_scalar_f64|gpu_scalar_value)\b/.test(text)) {
    fail(`${sourcePath} restores an identity-local provider helper; use provider_support`);
  }
}
const binaryArithmeticRuntimeBoundaries = new Map([
  [`${binaryArithmeticRuntimeRoot}/mod.rs`, 64],
  [`${binaryArithmeticRuntimeRoot}/output_prototype/mod.rs`, 96],
  [`${binaryArithmeticRuntimeRoot}/output_prototype/analysis.rs`, 128],
  [`${binaryArithmeticRuntimeRoot}/output_prototype/conversion.rs`, 128],
  [`${binaryArithmeticRuntimeRoot}/output_prototype/parse.rs`, 64],
  [`${binaryArithmeticRuntimeRoot}/output_prototype/placement.rs`, 160],
  [`${binaryArithmeticRuntimeRoot}/provider_support.rs`, 160],
  [`${binaryArithmeticRuntimeRoot}/minus/mod.rs`, 160],
  [`${binaryArithmeticRuntimeRoot}/minus/host.rs`, 384],
  [`${binaryArithmeticRuntimeRoot}/minus/provider.rs`, 192],
  [`${binaryArithmeticRuntimeRoot}/minus/tests/mod.rs`, 64],
  [`${binaryArithmeticRuntimeRoot}/minus/tests/host.rs`, 448],
  [`${binaryArithmeticRuntimeRoot}/minus/tests/provider.rs`, 320],
  [`${binaryArithmeticRuntimeRoot}/minus/tests/wgpu.rs`, 128],
  [`${binaryArithmeticRuntimeRoot}/plus/mod.rs`, 160],
  [`${binaryArithmeticRuntimeRoot}/plus/host.rs`, 384],
  [`${binaryArithmeticRuntimeRoot}/plus/provider.rs`, 256],
  [`${binaryArithmeticRuntimeRoot}/plus/tests/mod.rs`, 64],
  [`${binaryArithmeticRuntimeRoot}/plus/tests/host.rs`, 448],
  [`${binaryArithmeticRuntimeRoot}/plus/tests/provider.rs`, 256],
  [`${binaryArithmeticRuntimeRoot}/plus/tests/wgpu.rs`, 256],
  [`${binaryArithmeticRuntimeRoot}/times/mod.rs`, 160],
  [`${binaryArithmeticRuntimeRoot}/times/host.rs`, 384],
  [`${binaryArithmeticRuntimeRoot}/times/provider.rs`, 192],
  [`${binaryArithmeticRuntimeRoot}/times/tests/mod.rs`, 64],
  [`${binaryArithmeticRuntimeRoot}/times/tests/host.rs`, 448],
  [`${binaryArithmeticRuntimeRoot}/times/tests/provider.rs`, 320],
  [`${binaryArithmeticRuntimeRoot}/times/tests/wgpu.rs`, 128],
]);
for (const [sourcePath, ceiling] of binaryArithmeticRuntimeBoundaries) {
  const text = read(sourcePath);
  const lines = text.split("\n").length;
  if (lines > ceiling || Buffer.byteLength(text, "utf8") > 24 * 1024) {
    fail(
      `${sourcePath} exceeds its binary-arithmetic role boundary ` +
      `(found ${lines} lines; maximum ${ceiling} and 24576 bytes)`
    );
  }
}

const legacyCatalogTestsPath = "crates/runmat-builtins/src/catalog/tests.rs";
const legacyCatalogTestLines = read(legacyCatalogTestsPath).split("\n").length;
if (legacyCatalogTestLines > 4561) {
  fail(`${legacyCatalogTestsPath} is a legacy centralized test boundary and must only shrink (found ${legacyCatalogTestLines} lines; ceiling 4561); put new tests beside their owning catalog, inference, or validation module`);
}

if (failed) process.exit(1);
console.log("crate architecture boundaries are valid");
