import { compareCodePoint, sorted } from "./constants.mjs";
import { validateCombinedMachineReport, validateMachineReport } from "../../runtime/builtin-example-verifier/report-schema.mjs";

const SAFE_IDENTITY = /^[A-Za-z][A-Za-z0-9_.]*$/;
const DIGEST = /^sha256:[a-f0-9]{64}$/;
const LANES = new Set(["browser", "native"]);

export function parseVerificationManifest(value) {
  requireKind(value, "runmat-builtin-migration-verification-manifest");
  exact(value, ["schema_version", "kind", "batch", "factory_audit", "example_reports", "expectations"], "verification manifest");
  const batch = requireObject(value.batch, "batch");
  exact(batch, ["artifact", "source", "identities", "factory_inventory_digest", "example_inventory_digest", "combined_example_artifact"], "verification batch");
  const identities = uniqueStrings(batch.identities, "batch identities", false, SAFE_IDENTITY).map(lower);
  const expectations = requireArray(value.expectations, "expectations").map(parseExpectation);
  const expectedIdentities = sorted(expectations.map((entry) => entry.identity));
  if (JSON.stringify(sorted(identities)) !== JSON.stringify(expectedIdentities)) {
    throw new Error("Expectations must cover each batch identity exactly once");
  }
  const exampleKeys = expectations.flatMap((entry) => entry.example_keys);
  if (new Set(exampleKeys).size !== exampleKeys.length) throw new Error("Example keys must be unique across expectations");
  const reports = requireArray(value.example_reports, "example reports").map((entry) => parseReference(entry, "example report"));
  if (new Set(reports.map((entry) => entry.artifact)).size !== reports.length) throw new Error("Example report artifacts must be unique");
  reports.sort((a, b) => compareCodePoint(a.artifact, b.artifact));
  const source = nonempty(batch.source, "batch source");
  if (source === "unspecified") throw new Error("batch source must be an exact source identity, not unspecified");
  return {
    schema_version: 1,
    kind: value.kind,
    batch: {
      artifact: nonempty(batch.artifact, "batch artifact"),
      source,
      identities: sorted(identities),
      factory_inventory_digest: digest(batch.factory_inventory_digest, "factory inventory digest"),
      example_inventory_digest: digest(batch.example_inventory_digest, "example inventory digest"),
      combined_example_artifact: nonempty(batch.combined_example_artifact, "combined example artifact"),
    },
    factory_audit: parseReference(value.factory_audit, "factory audit"),
    example_reports: reports,
    expectations: expectations.sort((a, b) => compareCodePoint(a.identity, b.identity)),
  };
}

export function validateFactoryAudit(value) {
  if (!value || value.schema_version !== 2 || value.kind !== "runmat-builtin-migration-audit") {
    throw new Error("Expected schema_version 2 and kind runmat-builtin-migration-audit");
  }
  exact(value, ["schema_version", "kind", "authority", "metadata", "requested_identities", "global_diagnostics", "summary", "result", "identities"], "factory audit");
  if (value.authority !== "development-verification-only") throw new Error("Invalid factory audit authority");
  const metadata = requireObject(value.metadata, "factory audit metadata");
  exact(metadata, ["source", "artifact", "inventory"], "factory audit metadata");
  const inventory = requireObject(metadata.inventory, "factory audit inventory");
  exact(inventory, ["schema_version", "kind", "digest", "identities"], "factory audit inventory");
  if (inventory.schema_version !== 1 || inventory.kind !== "runmat-builtin-migration-inventory") throw new Error("Unsupported factory inventory schema");
  uniqueStrings(inventory.identities, "factory inventory identities", false);
  const requested = sorted(uniqueStrings(value.requested_identities, "factory requested identities", false, SAFE_IDENTITY).map(lower));
  if (!Array.isArray(value.identities) || !["pass", "fail"].includes(value.result)
      || !Array.isArray(value.global_diagnostics)) throw new Error("Invalid factory audit result");
  const resultIdentities = value.identities.map((entry) => {
    exact(entry, ["identity", "result", "failures", "evidence"], "factory identity result");
    if (!entry || typeof entry.identity !== "string" || !SAFE_IDENTITY.test(entry.identity)
        || !["pass", "fail"].includes(entry.result) || !Array.isArray(entry.failures)) {
      throw new Error("Invalid factory identity audit record");
    }
    return entry.identity.toLowerCase();
  });
  if (new Set(resultIdentities).size !== resultIdentities.length
      || JSON.stringify(sorted(resultIdentities)) !== JSON.stringify(requested)) {
    throw new Error("Factory identity results must exactly match requested identities");
  }
  exact(value.summary, ["identities", "passed", "failed", "global_errors"], "factory audit summary");
  const failed = value.identities.filter((entry) => entry.result === "fail").length;
  if (value.summary.identities !== value.identities.length || value.summary.passed !== value.identities.length - failed
      || value.summary.failed !== failed || value.summary.global_errors !== value.global_diagnostics.length) {
    throw new Error("Inconsistent factory audit summary");
  }
  return {
    value,
    source: nonempty(metadata.source, "factory audit source"),
    artifact: nonempty(metadata.artifact, "factory audit artifact"),
    inventory_digest: digest(inventory.digest, "factory inventory digest"),
  };
}

export function classifyExampleReport(value) {
  if (!value || value.schemaVersion !== "runmat.builtin-example-report.v2") throw new Error("Unsupported example report schema");
  const kind = Number.isInteger(value.metadata?.index) ? "shard" : Array.isArray(value.metadata?.shards) ? "combined" : "unknown";
  if (kind === "shard") validateMachineReport(value);
  else if (kind === "combined") validateCombinedMachineReport(value);
  else throw new Error("Example report is neither a shard nor a combined artifact");
  const metadata = value.metadata;
  const inventory = metadata.inventory;
  return {
    value,
    kind,
    source: nonempty(metadata.source, "example report source"),
    artifact: nonempty(metadata.artifact, "example report artifact"),
    inventory_digest: inventory.digest,
  };
}

function parseExpectation(value) {
  const entry = requireObject(value, "expectation");
  exact(entry, ["identity", "example_keys", "required_lanes"], "expectation");
  const identity = nonempty(entry.identity, "expectation identity").toLowerCase();
  if (!SAFE_IDENTITY.test(identity)) throw new Error(`Unsafe expectation identity ${identity}`);
  return {
    identity,
    example_keys: sorted(uniqueStrings(entry.example_keys, `${identity} example keys`, true)),
    required_lanes: sorted(uniqueStrings(entry.required_lanes, `${identity} required lanes`, true).map((lane) => {
      if (!LANES.has(lane)) throw new Error(`Unsupported required lane ${lane}`);
      return lane;
    })),
  };
}

function parseReference(value, label) {
  const reference = requireObject(value, label);
  exact(reference, ["path", "artifact"], label);
  return { path: nonempty(reference.path, `${label} path`), artifact: nonempty(reference.artifact, `${label} artifact`) };
}

function requireKind(value, kind) {
  if (!value || value.schema_version !== 1 || value.kind !== kind) throw new Error(`Expected schema_version 1 and kind ${kind}`);
}
function requireObject(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be an object`);
  return value;
}
function requireArray(value, label) { if (!Array.isArray(value) || !value.length) throw new Error(`${label} must be a nonempty array`); return value; }
function nonempty(value, label) { if (typeof value !== "string" || !value.trim()) throw new Error(`${label} must be a nonempty string`); return value.trim(); }
function digest(value, label) { const result = nonempty(value, label); if (!DIGEST.test(result)) throw new Error(`${label} must be sha256 hex`); return result; }
function uniqueStrings(value, label, allowEmpty = false, pattern = null) {
  if (!Array.isArray(value) || (!allowEmpty && !value.length)) throw new Error(`${label} must be ${allowEmpty ? "an" : "a nonempty"} array`);
  const entries = value.map((entry) => nonempty(entry, label));
  if (pattern && entries.some((entry) => !pattern.test(entry))) throw new Error(`${label} contains an unsafe value`);
  if (new Set(entries.map(lower)).size !== entries.length) throw new Error(`${label} must be unique`);
  return entries;
}
function lower(value) { return value.toLowerCase(); }
function exact(value, keys, label) {
  requireObject(value, label);
  const actual = sorted(Object.keys(value));
  const expected = sorted(keys);
  if (JSON.stringify(actual) !== JSON.stringify(expected)) throw new Error(`${label} fields must be exactly ${keys.join(", ")}`);
}
