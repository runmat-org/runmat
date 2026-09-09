import { CATALOG_ROOT, RUNTIME_ROOT, SHADOW_ROOT, SIDECAR_ROOT, WASM_REGISTRY } from "./constants.mjs";
import { compareCodePoint } from "./constants.mjs";
import { emptyDispositionInput, normalizeDisposition, validateDispositionInput, validateReviewedRelationships } from "./dispositions.mjs";
import { finalizeRecords, inventorySummary } from "./finalize.mjs";
import { createRecords } from "./records.mjs";
import { attachWasmEvidence, scanSurfaces } from "./surfaces.mjs";

export function buildInventory(repository, dispositionInput = emptyDispositionInput()) {
  validateDispositionInput(dispositionInput);
  const { records, record } = createRecords();
  const diagnostics = [];
  scanSurfaces(repository, record, diagnostics);
  attachWasmEvidence(repository, records);
  for (const [identity, input] of Object.entries(dispositionInput.identities)) record(identity).input = normalizeDisposition(input);
  validateReviewedRelationships(records, diagnostics);
  const identities = finalizeRecords(repository, records);
  return {
    schema_version: 1,
    kind: "runmat-builtin-migration-inventory",
    authority: "development-evidence-only",
    generated_from: [CATALOG_ROOT, RUNTIME_ROOT, SIDECAR_ROOT, SHADOW_ROOT, WASM_REGISTRY],
    summary: inventorySummary(identities),
    diagnostics: diagnostics.sort(compareDiagnostic),
    identities,
  };
}

function compareDiagnostic(left, right) {
  return compareCodePoint(`${left.code}:${left.path}:${left.detail}`, `${right.code}:${right.path}:${right.detail}`);
}

export { buildDispositionSeed, emptyDispositionInput, validateDispositionInput } from "./dispositions.mjs";
