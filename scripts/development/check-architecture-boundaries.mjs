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
forbidDependencies("runmat-analysis/core", [
  "runmat-lexer",
  "runmat-parser",
  "runmat-analysis-language",
  "runmat-builtins",
  "runmat-value",
  "runmat-runtime",
  "runmat-lsp",
  "runmat-filesystem",
]);
forbidDependencies("runmat-analysis/language", [
  "runmat-builtins",
  "runmat-value",
  "runmat-runtime",
  "runmat-lsp",
  "runmat-filesystem",
  "runmat-execution",
  "runmat-analysis-execution",
]);

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
  if (
    sourcePath.startsWith("crates/runmat-analysis/language/") &&
    /\brunmat_(?:builtins|value|runtime|lsp|filesystem|execution|analysis_execution)\b/.test(text)
  ) {
    fail(`${sourcePath} crosses the source-to-domain compiler dependency boundary`);
  }
  if (
    sourcePath.startsWith("crates/runmat-analysis/core/") &&
    /\brunmat_(?:lexer|parser|analysis_language|builtins|value|runtime|lsp|filesystem)\b/.test(text)
  ) {
    fail(`${sourcePath} crosses the canonical analysis-core dependency boundary`);
  }
  if (
    (sourcePath.startsWith("crates/runmat-analysis/fea/") ||
      sourcePath.startsWith("crates/runmat-analysis/execution/")) &&
    /\brunmat_(?:lexer|parser|analysis_language)\b/.test(text)
  ) {
    fail(`${sourcePath} imports FEA source-language machinery below the canonical contract boundary`);
  }
  if (
    (sourcePath.startsWith("crates/runmat-lexer/src/fea/") ||
      sourcePath.startsWith("crates/runmat-parser/src/fea/")) &&
    /\brunmat_(?:analysis|geometry|runtime|builtins|value|execution)\b/.test(text)
  ) {
    fail(`${sourcePath} imports domain or runtime behavior into the FEA syntax frontend`);
  }
}

const newFeaBoundarySources = allRust.filter(
  ({ path: sourcePath }) =>
    sourcePath.startsWith("crates/runmat-analysis/language/") ||
    sourcePath.startsWith("crates/runmat-types/src/operation/") ||
    sourcePath.startsWith("crates/runmat-lexer/src/fea/") ||
    sourcePath.startsWith("crates/runmat-parser/src/fea/"),
);
for (const { path: sourcePath, text } of newFeaBoundarySources) {
  if (/#\s*!?\s*\[\s*allow\b/.test(text)) {
    fail(`${sourcePath} adds an allow suppression to the FEA language boundary`);
  }
  if (/\b[A-Za-z_][A-Za-z0-9_]*[Vv]2\b/.test(text)) {
    fail(`${sourcePath} adds a version-suffixed FEA language identifier`);
  }
  if (text.split("\n").length > 600) {
    fail(`${sourcePath} exceeds the 600-line FEA module concentration limit`);
  }
}

const feaOwnedSources = allRust.filter(
  ({ path: sourcePath }) =>
    sourcePath.startsWith("crates/runmat-analysis/core/") ||
    sourcePath.startsWith("crates/runmat-analysis/language/") ||
    sourcePath.startsWith("crates/runmat-analysis/fea/") ||
    sourcePath.startsWith("crates/runmat-runtime/src/analysis/"),
);
for (const { path: sourcePath, text } of feaOwnedSources) {
  if (/#\s*!?\s*\[\s*allow\b/.test(text)) {
    fail(`${sourcePath} contains an allow suppression in the FEA system`);
  }
  if (/\b[A-Za-z_][A-Za-z0-9_]*[Vv]2\b/.test(text)) {
    fail(`${sourcePath} contains a version-suffixed identifier in the FEA system`);
  }
}

// Existing numerical modules predate the current domain split. This is a debt ratchet, not an
// exemption: unlisted production modules may not exceed 600 lines, listed modules may not grow,
// and a listing becomes an error as soon as its module is split below the limit.
const moduleLineLimit = 600;
const moduleSizeDebt = JSON.parse(read("scripts/development/fea-module-size-debt.json"));
const productionFeaSources = feaOwnedSources.filter(
  ({ path: sourcePath }) =>
    !sourcePath.endsWith("tests.rs") && !sourcePath.includes("/tests/"),
);
const productionFeaPaths = new Set(productionFeaSources.map(({ path: sourcePath }) => sourcePath));
for (const [sourcePath, ceiling] of Object.entries(moduleSizeDebt)) {
  if (!productionFeaPaths.has(sourcePath)) {
    fail(`module-size debt entry is missing or no longer a production FEA source: ${sourcePath}`);
  }
  if (!Number.isInteger(ceiling) || ceiling <= moduleLineLimit) {
    fail(`module-size debt ceiling for ${sourcePath} must be an integer above ${moduleLineLimit}`);
  }
}
for (const { path: sourcePath, text } of productionFeaSources) {
  const lineCount = text.endsWith("\n") ? text.split("\n").length - 1 : text.split("\n").length;
  const ceiling = moduleSizeDebt[sourcePath];
  if (lineCount > moduleLineLimit && ceiling === undefined) {
    fail(`${sourcePath} has ${lineCount} lines and adds untracked module-size debt`);
  } else if (ceiling !== undefined && lineCount > ceiling) {
    fail(`${sourcePath} grew from its ${ceiling}-line debt ceiling to ${lineCount} lines`);
  } else if (ceiling !== undefined && lineCount < ceiling && lineCount > moduleLineLimit) {
    fail(`${sourcePath} shrank to ${lineCount} lines; lower its ${ceiling}-line debt ceiling`);
  } else if (ceiling !== undefined && lineCount <= moduleLineLimit) {
    fail(`${sourcePath} is now ${lineCount} lines; remove its stale module-size debt entry`);
  }
}

const runtimeAnalysisSource = read("crates/runmat-runtime/src/analysis/mod.rs");
const runtimeAcousticKernelDeclarations = [
  /\bfn\s+solve_acoustic_/,
  /\bfn\s+recover_acoustic_/,
  /\bfn\s+acoustic_(?:axis_derivative|boundary_summary|domain_topology|drive_frequency|helmholtz|known_answer|material_summary|node_count|residual_norm|source_vector|sweep_frequencies)/,
  /\bstruct\s+Acoustic(?:BoundarySummary|DomainSystem|DomainTopology|GraphEdge|KnownAnswerMetrics|MaterialSummary)\b/,
];
if (runtimeAcousticKernelDeclarations.some((pattern) => pattern.test(runtimeAnalysisSource))) {
  fail("runmat-runtime must adapt acoustic operations without owning acoustic numerical kernels");
}
const runtimeCfdKernelDeclarations = [
  /\bfn\s+(?:solve|recover)_cfd_/,
  /\bfn\s+cfd_(?:assembly|boundary|known_answer|residual|reynolds|velocity_pressure)/,
  /\bstruct\s+Cfd(?:BoundarySummary|DomainTopology|KnownAnswerMetrics|VelocityPressureSolution)\b/,
];
if (runtimeCfdKernelDeclarations.some((pattern) => pattern.test(runtimeAnalysisSource))) {
  fail("runmat-runtime must adapt CFD operations without owning CFD numerical kernels");
}
const runtimeChtKernelDeclarations = [
  /\bfn\s+(?:build|solve)_cht_/,
  /\bfn\s+cht_(?:advection|interface|known_answer)/,
  /\bstruct\s+Cht(?:ConjugateInterfaceSolution|InterfaceClosure|KnownAnswerMetrics)\b/,
];
if (runtimeChtKernelDeclarations.some((pattern) => pattern.test(runtimeAnalysisSource))) {
  fail("runmat-runtime must adapt CHT operations without owning CHT numerical kernels");
}
const runtimeFsiKernelDeclarations = [
  /\bfn\s+(?:build|solve)_fsi_/,
  /\bfn\s+fsi_(?:known_answer|structural_compliance)/,
  /\bstruct\s+Fsi(?:InterfaceClosure|KnownAnswerMetrics|PartitionedInterfaceStep|StructuralInterfaceResponse)\b/,
];
if (runtimeFsiKernelDeclarations.some((pattern) => pattern.test(runtimeAnalysisSource))) {
  fail("runmat-runtime must adapt FSI operations without owning FSI numerical kernels");
}
if (
  allRust.some(
    ({ path: sourcePath, text }) =>
      sourcePath.startsWith("crates/runmat-runtime/") &&
      /\b(?:struct|enum)\s+(?:SolverMeshFlowTopology|FlowTopologyProjection)\b/.test(text),
  )
) {
  fail("runmat-runtime must consume, not declare, canonical flow-topology projections");
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
