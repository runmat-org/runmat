#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export function diagnosticInventory(jsonl) {
  const counts = new Map();
  for (const line of jsonl.split("\n")) {
    if (!line.trim()) continue;
    const record = JSON.parse(line);
    if (record.reason !== "compiler-message") continue;
    const diagnostic = record.message;
    if (!["error", "warning"].includes(diagnostic.level)) continue;
    const primary = diagnostic.spans.filter(span => span.is_primary)
      .map(span => ({ ...span, file_name: span.file_name.replaceAll("\\", "/") }));
    if (!diagnostic.code?.code?.startsWith("clippy::") || !primary.length
        || primary.some(span => !span.file_name.startsWith("crates/runmat-analysis/fea/"))) {
      throw new Error(`Unreviewed compiler diagnostic: ${diagnostic.message}`);
    }
    const identity = JSON.stringify([
      diagnostic.code.code, diagnostic.message,
      primary.map(span => [span.file_name, span.text.map(line => line.text.trim()).join("\n")]),
    ]);
    counts.set(identity, (counts.get(identity) ?? 0) + 1);
  }
  return counts;
}

export function matchesDebt(actual, expected) {
  return actual.size === expected.size
    && [...actual].every(([identity, count]) => expected.get(identity) === count);
}

function main() {
  const [log, exitCode] = process.argv.slice(2);
  if (!log || !["0", "101"].includes(exitCode)) throw new Error("Expected a Cargo JSONL log and exit code 0 or 101");
  const ledger = JSON.parse(fs.readFileSync(new URL("../../qualification/quality/fea-lint-debt-0.7.json", import.meta.url)));
  const actual = diagnosticInventory(fs.readFileSync(log, "utf8"));
  if (exitCode === "0" && actual.size === 0) {
    console.log("FEA strict Clippy passes without a debt exception.");
    return;
  }
  if (ledger.schema_version !== 1 || ledger.issue !== "RM-1287"
      || !matchesDebt(actual, new Map(ledger.diagnostics))) {
    throw new Error("FEA Clippy diagnostics differ from the reviewed debt; review the retained census");
  }
  console.log(`Reviewed FEA Clippy debt conserved: ${ledger.distinct_sites} sites, ${[...actual.values()].reduce((a, b) => a + b, 0)} emissions. See RM-1287.`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; }
}
