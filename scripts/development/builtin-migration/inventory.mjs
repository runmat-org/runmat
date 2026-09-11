import { CATALOG_ROOT, RUNTIME_ROOT, SHADOW_ROOT, SIDECAR_ROOT, WASM_REGISTRY } from "./constants.mjs";
import { compareCodePoint } from "./constants.mjs";
import { emptyDispositionInput, normalizeDisposition, validateDispositionInput, validateReviewedRelationships } from "./dispositions.mjs";
import { finalizeRecords, inventorySummary } from "./finalize.mjs";
import { createRecords } from "./records.mjs";
import { attachWasmEvidence, scanSurfaces } from "./surfaces.mjs";
import { evidenceDigest } from "./evidence.mjs";
import { sourceSnapshot } from "./snapshot.mjs";
import { authorityFor, parseCompiledInventory } from "./compiled-inventory.mjs";
import { migrationFinding } from "./compiled-runtime-schema.mjs";
import { migrationFindingsDigest } from "./migration-findings.mjs";
import { array, digest, exact, kind } from "./schema.mjs";

export function buildInventory(repository, dispositionInput = emptyDispositionInput(), options = {}) {
  if (!options.compiledInventory) throw new Error("inventory requires a compiled migration inventory; pass --compiled-inventory");
  const compiled = parseCompiledInventory(options.compiledInventory);
  validateDispositionInput(dispositionInput);
  const { records, record } = createRecords();
  const diagnostics = [];
  scanSurfaces(repository, record, diagnostics);
  for (const identity of compiled.identities) record(identity);
  attachWasmEvidence(repository, records);
  for (const [identity, input] of Object.entries(dispositionInput.identities)) record(identity).input = normalizeDisposition(input);
  validateReviewedRelationships(records, diagnostics);
  const identities = finalizeRecords(repository, records).map((entry) => ({
    ...entry,
    semantic_authority: authorityFor(compiled, entry.identity),
    lexical_observations: { authority: "discovery_only", ownership: entry.ownership, registrations: entry.registrations, dependencies: entry.dependencies, provider: entry.provider },
  }));
  const generatedFrom = [
    "Cargo.toml",
    CATALOG_ROOT, RUNTIME_ROOT, SIDECAR_ROOT, SHADOW_ROOT, WASM_REGISTRY,
    "scripts/development/builtin-migration", "scripts/development/builtin-migration-factory.mjs",
    "scripts/development/check-architecture-boundaries.mjs",
  ];
  const source = sourceSnapshot(repository, generatedFrom, options.revision ?? null);
  const scannedPaths = [...new Set(identities.flatMap((entry) => [
    ...entry.source_metrics.files.map((file) => file.path), ...entry.tests.paths,
    ...entry.documentation.sources.map((document) => document.path),
  ]))].sort(compareCodePoint);
  const sourceEntries = new Map(source.files.map((entry) => [entry.path, entry]));
  const uncovered = scannedPaths.filter((entry) => !sourceEntries.has(entry));
  if (uncovered.length) throw new Error(`source snapshot does not cover scanned files: ${uncovered.join(", ")}`);
  const scannedSourceCoverage = { paths: scannedPaths, digest: evidenceDigest(scannedPaths.map((entry) => sourceEntries.get(entry))) };
  const payload = {
    schema_version: 2,
    kind: "runmat-builtin-migration-inventory",
    authority: "development-evidence-only",
    generated_from: generatedFrom,
    source,
    compiled_inventory: { schema_version: options.compiledInventory.schema_version, kind: options.compiledInventory.kind, digest: compiled.digest, build: compiled.snapshot.build },
    migration_findings: compiled.findings,
    migration_findings_digest: migrationFindingsDigest(compiled.findings),
    scanned_source_coverage: scannedSourceCoverage,
    dispositions_digest: evidenceDigest(dispositionInput),
    summary: inventorySummary(identities),
    diagnostics: diagnostics.sort(compareDiagnostic),
    identities,
  };
  return { ...payload, digest: evidenceDigest(payload) };
}

export function parseInventoryEvidence(value) {
  kind(value, 2, "runmat-builtin-migration-inventory", "migration inventory");
  exact(value, ["schema_version", "kind", "authority", "generated_from", "source", "compiled_inventory", "migration_findings", "migration_findings_digest", "scanned_source_coverage", "dispositions_digest", "summary", "diagnostics", "identities", "digest"], "migration inventory");
  if (value.authority !== "development-evidence-only") throw new Error("migration inventory has invalid authority");
  digest(value.digest, "migration inventory digest"); digest(value.source?.digest, "migration inventory source digest"); digest(value.compiled_inventory?.digest, "compiled inventory evidence digest"); digest(value.migration_findings_digest, "migration findings digest"); digest(value.scanned_source_coverage?.digest, "scanned source coverage digest"); digest(value.dispositions_digest, "migration dispositions digest");
  array(value.migration_findings, "migration inventory findings", { empty: true }).forEach(migrationFinding);
  if (migrationFindingsDigest(value.migration_findings) !== value.migration_findings_digest) throw new Error("migration finding digest mismatch");
  array(value.identities, "migration inventory identities", { empty: true }); array(value.diagnostics, "migration inventory diagnostics", { empty: true });
  const { digest: ignored, ...payload } = value;
  if (evidenceDigest(payload) !== value.digest) throw new Error("migration inventory artifact digest mismatch");
  return value;
}

function compareDiagnostic(left, right) {
  return compareCodePoint(`${left.code}:${left.path}:${left.detail}`, `${right.code}:${right.path}:${right.detail}`);
}

export { buildDispositionSeed, emptyDispositionInput, validateDispositionInput } from "./dispositions.mjs";
