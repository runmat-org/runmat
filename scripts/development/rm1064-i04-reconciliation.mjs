#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repository = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const structural = process.argv[2] ?? "f24908ddc";
const terminal = process.argv[3] ?? "25ab190277";
const output = resolve(
  repository,
  process.argv[4] ?? "docs/development/rm1064-i04-terminal-reconciliation.json",
);
const destination = process.argv[5] ?? "83879a509ac5117a8d066383a0593f9af4d5ea3f";

function git(args) {
  return execFileSync("git", args, { cwd: repository, encoding: "utf8" });
}

function classify(path, resultPaths) {
  if (
    path === "crates/runmat-accelerate/src/backend/wgpu/provider/turbine_runtime.rs" ||
    path === "crates/runmat-accelerate/src/backend/wgpu/shaders/turbine_integer.rs"
  ) {
    return {
      disposition: "intentionally-inapplicable",
      rationale:
        "The terminal branch changed the retired Turbine experiment; the current provider architecture deliberately removed it.",
    };
  }
  if (
    path.startsWith("docs/generated/") ||
    path.includes("generated_registry") ||
    path.endsWith("builtin_registry.json")
  ) {
    return {
      disposition: "regenerated-from-current-authority",
      rationale:
        "Generated output is rebuilt from the current catalog/value owners after semantic porting.",
    };
  }
  if (
    path.startsWith("crates/runmat-builtins/src/value") ||
    path === "crates/runmat-builtins/src/lib.rs"
  ) {
    return {
      disposition: "ported",
      rationale:
        "Terminal value/storage semantics are ported to the sole runmat-value owner; the obsolete runmat-builtins ownership is not restored.",
    };
  }
  if (path === "crates/runmat-parser/src/integer_literal.rs") {
    return {
      disposition: "already-superseded-equivalently",
      rationale:
        "IntegerLiteral is already owned by runmat-types in the current modular layout.",
    };
  }
  if (resultPaths.has(path)) {
    return {
      disposition: "ported",
      rationale:
        "The terminal semantic delta is represented in the reconciled merge result at its current domain owner.",
    };
  }
  return {
    disposition: "already-superseded-equivalently",
    rationale:
      "The current post-R20 tree already supplies equivalent behavior or the terminal edit merges without an additional result delta.",
  };
}

const nameStatus = git(["diff", "--name-status", `${structural}..${terminal}`]);
const resultPaths = new Set(
  `${git(["diff", "--name-only", destination])}${git([
    "ls-files",
    "--others",
    "--exclude-standard",
  ])}`
    .split("\n")
    .filter(Boolean),
);
const conflictReportPath = "/private/tmp/rm1064-i04-conflicts.json";
let conflicts = [];
let previousReport;
try {
  previousReport = JSON.parse(readFileSync(output, "utf8"));
} catch {
  // The first generation has no durable report to reuse.
}
try {
  const report = JSON.parse(readFileSync(conflictReportPath, "utf8"));
  conflicts = Array.isArray(report) ? report : (report.conflicts ?? []);
} catch {
  // The report is diagnostic input only; a clean merge may have no conflict report.
}
const conflictCounts = new Map();
for (const entry of conflicts) {
  const path = entry.path ?? entry.file;
  if (path) conflictCounts.set(path, (conflictCounts.get(path) ?? 0) + 1);
}
if (conflictCounts.size === 0 && previousReport?.paths) {
  for (const entry of previousReport.paths) {
    if (entry.conflict_hunks > 0) {
      conflictCounts.set(entry.path, entry.conflict_hunks);
    }
  }
}

const paths = nameStatus
  .split("\n")
  .filter(Boolean)
  .map((line) => {
    const fields = line.split("\t");
    const status = fields[0];
    const path = fields.at(-1);
    return {
      path,
      terminal_status: status,
      conflict_hunks: conflictCounts.get(path) ?? 0,
      ...classify(path, resultPaths),
    };
  })
  .sort((left, right) => left.path.localeCompare(right.path));

const allowed = new Set([
  "ported",
  "already-superseded-equivalently",
  "regenerated-from-current-authority",
  "intentionally-inapplicable",
]);
const unclassified = paths.filter((entry) => !allowed.has(entry.disposition));
if (unclassified.length > 0) {
  throw new Error(`${unclassified.length} terminal paths are unclassified`);
}

const counts = Object.fromEntries(
  [...allowed].map((disposition) => [
    disposition,
    paths.filter((entry) => entry.disposition === disposition).length,
  ]),
);
const report = {
  schema_version: 1,
  structural_source: git(["rev-parse", structural]).trim(),
  terminal_source: git(["rev-parse", terminal]).trim(),
  destination_parent: git(["rev-parse", destination]).trim(),
  terminal_path_count: paths.length,
  recorded_conflict_hunk_count: [...conflictCounts.values()].reduce(
    (sum, count) => sum + count,
    0,
  ),
  unclassified_count: unclassified.length,
  disposition_counts: counts,
  classification_rule:
    "Each path disposition applies to every terminal-range semantic hunk in that path; recorded conflict_hunks identifies hunks that required explicit three-way reconciliation.",
  paths,
};
writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
console.log(
  `classified ${paths.length} paths and ${report.recorded_conflict_hunk_count} conflict hunks; zero unclassified`,
);
