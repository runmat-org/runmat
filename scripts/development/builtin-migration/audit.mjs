import { compareCodePoint, sorted } from "./constants.mjs";

export function auditInventory(inventory, requestedIdentities) {
  const byIdentity = new Map(inventory.identities.map((entry) => [entry.identity, entry]));
  const identities = sorted(new Set(requestedIdentities.map((entry) => entry.toLowerCase())));
  const results = identities.map((identity) => auditIdentity(identity, byIdentity));
  const globalErrors = inventory.diagnostics.filter((entry) => entry.severity === "error");
  return {
    schema_version: 1,
    kind: "runmat-builtin-migration-audit",
    authority: "development-verification-only",
    requested_identities: identities,
    global_diagnostics: globalErrors,
    summary: {
      identities: results.length,
      passed: results.filter((entry) => entry.result === "pass").length,
      failed: results.filter((entry) => entry.result === "fail").length,
      global_errors: globalErrors.length,
    },
    result: results.every((entry) => entry.result === "pass") && globalErrors.length === 0 ? "pass" : "fail",
    identities: results,
  };
}

export function parseBatch(value) {
  if (!value || value.schema_version !== 1 || value.kind !== "runmat-builtin-migration-batch" || !Array.isArray(value.identities)) {
    throw new Error("Batch input must use schema_version 1, kind runmat-builtin-migration-batch, and an identities array");
  }
  if (!value.identities.length || value.identities.some((entry) => typeof entry !== "string" || !/^[A-Za-z][A-Za-z0-9_.]*$/.test(entry))) {
    throw new Error("Batch identities must be a nonempty array of safe identity strings");
  }
  if (new Set(value.identities.map((entry) => entry.toLowerCase())).size !== value.identities.length) throw new Error("Batch identities must be unique ignoring case");
  return value.identities;
}

function auditIdentity(identity, byIdentity) {
  const row = byIdentity.get(identity);
  if (!row) return result(identity, [{ code: "identity-not-in-inventory", detail: identity }]);
  const failures = [];
  for (const field of row.unresolved) failures.push({ code: "unresolved-field", detail: field });
  duplicateBindings(row, failures);
  if (row.disposition.kind === "canonical") auditCanonical(row, failures);
  else if (row.disposition.kind === "alias") auditAlias(row, byIdentity, failures);
  else if (row.disposition.kind === "internal") auditInternal(row, failures);
  else failures.push({ code: "unresolved-disposition", detail: row.disposition.reason });
  return result(identity, failures, row);
}

function auditCanonical(row, failures) {
  exact(failures, "catalog-authority-count", row.ownership.catalog.length, 1);
  minimum(failures, "runtime-binding-count", row.registrations.runtime.length, 1);
  minimum(failures, "catalog-documentation-count", row.ownership.catalog_documentation.length, 1);
  exact(failures, "legacy-sidecar-count", row.ownership.sidecars.length, 0);
  exact(failures, "runtime-shadow-count", row.ownership.runtime_documentation_shadows.length, 0);
  exact(failures, "legacy-resolver-count", row.dependencies.legacy_resolver_paths.length, 0);
  minimum(failures, "test-evidence-count", row.tests.paths.length, 1);
  minimum(failures, "typed-example-count", row.examples.discovered_count, 1);
  for (const binding of row.registrations.native_link.runtime_binding_inputs) {
    if (!binding.builtin_path) failures.push({ code: "missing-native-link-input", detail: `${binding.path}:${binding.function ?? "?"}#${binding.binding_variant}` });
  }
}

function auditAlias(row, byIdentity, failures) {
  const target = byIdentity.get(row.disposition.canonical);
  if (!target) failures.push({ code: "dangling-alias", detail: row.disposition.canonical });
  else if (target.disposition.kind !== "canonical") failures.push({ code: "alias-target-not-canonical", detail: `${target.identity}:${target.disposition.kind}` });
  exact(failures, "alias-copied-documentation", row.ownership.catalog_documentation.length + row.ownership.sidecars.length + row.ownership.runtime_documentation_shadows.length, 0);
}

function auditInternal(row, failures) {
  minimum(failures, "internal-runtime-binding-count", row.registrations.runtime.length, 1);
  exact(failures, "internal-public-documentation-count", row.ownership.catalog_documentation.length + row.ownership.sidecars.length, 0);
  exact(failures, "internal-catalog-authority-count", row.ownership.catalog.length, 0);
}

function duplicateBindings(row, failures) {
  const counts = new Map();
  for (const binding of row.registrations.runtime) {
    const key = `${binding.binding_variant}:${binding.function ?? "?"}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  for (const [key, count] of counts) if (count > 1) failures.push({ code: "duplicate-runtime-binding", detail: `${key}:${count}` });
}

function exact(failures, code, actual, expected) { if (actual !== expected) failures.push({ code, expected, actual }); }
function minimum(failures, code, actual, expectedMinimum) { if (actual < expectedMinimum) failures.push({ code, expected_minimum: expectedMinimum, actual }); }
function result(identity, failures, row = null) {
  return { identity, result: failures.length ? "fail" : "pass", failures: failures.sort((a, b) => compareCodePoint(`${a.code}:${a.detail ?? ""}`, `${b.code}:${b.detail ?? ""}`)), evidence: row ? { disposition: row.disposition, ownership: row.ownership, registrations: row.registrations, unresolved: row.unresolved } : null };
}
