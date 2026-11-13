import { REPORT_SCHEMA } from "./sharding.mjs";

const DIGEST = /^sha256:[a-f0-9]{64}$/;
const LANE = "harness-defined-browser-and-native";

export function validateMachineReport(report) {
  exact(report, ["schemaVersion", "metadata", "summary", "results"], "shard report");
  if (report.schemaVersion !== REPORT_SCHEMA) throw new Error("Unsupported shard report schema");
  exact(report.metadata, ["index", "count", "source", "artifact", "lane", "inventory"], "shard metadata");
  const { index, count } = report.metadata;
  if (!Number.isInteger(index) || !Number.isInteger(count) || index < 0 || count < 1 || index >= count) {
    throw new Error("Invalid shard index/count metadata");
  }
  commonMetadata(report.metadata);
  validateInventory(report.metadata.inventory, false);
  validateResults(report);
  return report;
}

export function validateCombinedMachineReport(report) {
  exact(report, ["schemaVersion", "metadata", "summary", "results"], "combined report");
  if (report.schemaVersion !== REPORT_SCHEMA) throw new Error("Unsupported combined report schema");
  exact(report.metadata, ["source", "artifact", "lane", "inventory", "shards", "constituentArtifacts"], "combined metadata");
  commonMetadata(report.metadata);
  validateInventory(report.metadata.inventory, true);
  const shards = report.metadata.shards;
  const artifacts = report.metadata.constituentArtifacts;
  if (!Array.isArray(shards) || !shards.length || shards.some((entry, index) => entry !== index)) {
    throw new Error("Combined report has invalid shard provenance");
  }
  if (!Array.isArray(artifacts) || artifacts.length !== shards.length
      || artifacts.some((entry) => typeof entry !== "string" || !entry.trim())
      || new Set(artifacts).size !== artifacts.length) {
    throw new Error("Combined report has invalid constituent artifact provenance");
  }
  validateResults(report);
  const keys = report.results.map((entry) => entry.key);
  if (JSON.stringify(keys) !== JSON.stringify(report.metadata.inventory.keys) || new Set(keys).size !== keys.length) {
    throw new Error("Combined report results do not match its inventory");
  }
  return report;
}

function commonMetadata(metadata) {
  if (typeof metadata.source !== "string" || !metadata.source.trim()
      || typeof metadata.artifact !== "string" || !metadata.artifact.trim() || metadata.lane !== LANE) {
    throw new Error("Invalid report source/artifact/lane metadata");
  }
}

function validateInventory(inventory, combined) {
  exact(inventory, ["digest", "count", "keys", "range"], "report inventory");
  exact(inventory.range, ["startInclusive", "endExclusive"], "report inventory range");
  const { count, keys, range } = inventory;
  if (!DIGEST.test(inventory.digest) || !Number.isInteger(count) || count < 0
      || !Array.isArray(keys) || keys.length !== count
      || keys.some((key) => typeof key !== "string" || !key.length) || new Set(keys).size !== keys.length
      || !Number.isInteger(range.startInclusive) || !Number.isInteger(range.endExclusive)
      || range.startInclusive < 0 || range.endExclusive < range.startInclusive || range.endExclusive > count) {
    throw new Error("Invalid report inventory metadata");
  }
  if (combined && (range.startInclusive !== 0 || range.endExclusive !== count)) {
    throw new Error("Combined report has an invalid full inventory range");
  }
}

function validateResults(report) {
  if (!Array.isArray(report.results)) throw new Error("Report results must be an array");
  for (const result of report.results) {
    exact(result, ["key", "builtin", "exampleIndex", "harness", "matches", "expected", "actual", "image", "imageError"], "report result");
    if (typeof result.key !== "string" || !result.key || typeof result.builtin !== "string" || !result.builtin
        || !Number.isInteger(result.exampleIndex) || result.exampleIndex < 0 || typeof result.harness !== "string" || !result.harness
        || typeof result.matches !== "boolean" || typeof result.expected !== "string" || typeof result.actual !== "string"
        || !nullableString(result.image) || !nullableString(result.imageError)) throw new Error("Invalid report result record");
  }
  exact(report.summary, ["total", "passed", "failed"], "report summary");
  const failed = report.results.filter((result) => !result.matches).length;
  if (report.summary.total !== report.results.length || report.summary.passed !== report.results.length - failed
      || report.summary.failed !== failed) throw new Error("Inconsistent report summary");
}

function exact(value, keys, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be an object`);
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (JSON.stringify(actual) !== JSON.stringify(expected)) throw new Error(`${label} fields must be exactly ${keys.join(", ")}`);
}
function nullableString(value) { return value === null || typeof value === "string"; }
