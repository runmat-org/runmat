import fs from "node:fs";
import path from "node:path";
import { compareCodePoint, sorted } from "./constants.mjs";
import { evidenceDigest } from "./evidence.mjs";
import { parseGateResult } from "./gate-result.mjs";
import { requiredGateNames } from "./gate-requirements.mjs";
import { validateLeaseDiff } from "./lease.mjs";
import { parsePrepareResult } from "./prepare.mjs";
import { SAFE_IDENTITY, exact, kind, stableId, uniqueStrings } from "./schema.mjs";
import { parseCompletedSourceDisposition } from "./source-fields.mjs";

export function auditMigration(repository, inventory, control, lease, batchValue, evidence) {
  stableId(evidence.artifact_id, "audit artifact id");
  const requested = parseBatch(batchValue);
  const bundle = lease.bundle;
  const failures = [];
  if (JSON.stringify(requested) !== JSON.stringify(sorted(bundle.identities))) failures.push(issue("partial-bundle", "audit batch must cover the complete reviewed bundle"));
  try { validateLeaseDiff(bundle, evidence.changed_paths ?? []); } catch (error) { failures.push(issue("lease-violation", error.message)); }
  const expected = {
    source_revision: inventory.source.revision, source_digest: inventory.source.digest,
    inventory_digest: inventory.digest, control_manifest_digest: control.digest, bundle_id: bundle.id,
    storage_policy: control.value.storage_policy,
    source_files: inventory.source.files,
    repository, gate_plans: bundle.gate_plans, compiled_build: inventory.compiled_inventory.build,
  };
  const gates = parseGates(evidence.gate_results ?? [], expected, requested, failures);
  const prepares = indexPrepare(evidence.prepare_results ?? [], expected, lease, requested, failures);
  const dispositions = indexDispositions(evidence.source_dispositions ?? [], prepares, requested, failures);
  const inventoryByIdentity = new Map(inventory.identities.map((entry) => [entry.identity, entry]));
  const identities = requested.map((id) => auditIdentity(repository, id, inventoryByIdentity.get(id), control.identities.get(id), gates, prepares, dispositions));
  failures.push(...inventory.diagnostics.filter((entry) => entry.severity === "error").map((entry) => issue("inventory-error", `${entry.code}:${entry.path ?? ""}`)));
  const passed = identities.filter((entry) => entry.result === "pass").length;
  return {
    schema_version: 3, kind: "runmat-builtin-migration-audit", authority: "development-verification-evidence-only",
    artifact_id: evidence.artifact_id, source: inventory.source, inventory_digest: inventory.digest,
    control_manifest_digest: control.digest, bundle_id: bundle.id, lease_id: lease.value.lease_id,
    requested_identities: requested, evidence: {
      gate_artifacts: [...gates.values()].map((entry) => entry.artifact_id).sort(compareCodePoint),
      prepare_digests: [...prepares.values()].map(evidenceDigest).sort(compareCodePoint),
      source_disposition_digests: [...dispositions.values()].map(evidenceDigest).sort(compareCodePoint),
    },
    summary: { identities: identities.length, passed, failed: identities.length - passed, global_failures: failures.length },
    result: passed === identities.length && failures.length === 0 ? "pass" : "fail", global_failures: failures, identities,
  };
}

export function parseBatch(value) {
  kind(value, 1, "runmat-builtin-migration-batch", "migration batch");
  exact(value, ["schema_version", "kind", "identities"], "migration batch");
  return uniqueStrings(value.identities, "batch identities", { pattern: SAFE_IDENTITY, lower: true }).sort(compareCodePoint);
}

function parseGates(values, expected, requested, failures) {
  const result = new Map();
  for (const raw of values) {
    try {
      const gate = parseGateResult(raw, expected);
      if (result.has(gate.gate)) throw new Error(`duplicate gate ${gate.gate}`);
      if (JSON.stringify(gate.identities) !== JSON.stringify(requested)) throw new Error(`${gate.gate}: identities do not exactly match the batch`);
      if (gate.result !== "pass") throw new Error(`${gate.gate}: result is ${gate.result}`);
      result.set(gate.gate, gate);
    } catch (error) { failures.push(issue("invalid-gate-evidence", error.message)); }
  }
  return result;
}

function indexPrepare(values, expected, lease, requested, failures) {
  const result = new Map();
  for (const raw of values) {
    try {
      const parsed = parsePrepareResult(raw, { bundle_id: expected.bundle_id, control_manifest_digest: expected.control_manifest_digest, lease_id: lease.value.lease_id });
      if (!requested.includes(parsed.identity)) throw new Error(`${parsed.identity}: prepare result is outside the batch`);
      if (parsed.source.revision !== expected.source_revision || parsed.source.digest !== expected.source_digest || parsed.inventory_digest !== expected.inventory_digest) throw new Error(`${parsed.identity}: stale prepare result`);
      if (result.has(parsed.identity)) throw new Error(`${parsed.identity}: duplicate prepare result`);
      result.set(parsed.identity, parsed);
    } catch (error) { failures.push(issue("invalid-prepare-evidence", error.message)); }
  }
  return result;
}

function indexDispositions(values, prepares, requested, failures) {
  const result = new Map();
  for (const raw of values) {
    try {
      const id = String(raw?.identity ?? "").toLowerCase();
      if (!requested.includes(id)) throw new Error(`${id || "unknown"}: source disposition is outside the batch`);
      const prepared = prepares.get(id);
      if (!prepared) throw new Error(`${id}: source disposition has no matching prepare result`);
      const parsed = parseCompletedSourceDisposition(raw, id, prepared.checklist_baseline_digest);
      if (evidenceDigest(raw) === prepared.checklist_digest) throw new Error(`${id}: unchanged pending prepare checklist is not completed evidence`);
      if (result.has(id)) throw new Error(`${id}: duplicate source disposition`);
      result.set(id, parsed);
    } catch (error) { failures.push(issue("invalid-source-disposition", error.message)); }
  }
  return result;
}

function auditIdentity(repository, id, observed, controlled, gates, prepares, dispositions) {
  const failures = [];
  if (!observed) failures.push(issue("identity-not-in-inventory", id));
  if (!controlled) failures.push(issue("identity-not-in-control", id));
  if (!observed || !controlled) return identityResult(id, failures);
  if (observed.unresolved.length) failures.push(issue("unresolved-inventory", observed.unresolved.join(",")));
  if (observed.disposition.kind !== controlled.disposition.kind) failures.push(issue("disposition-mismatch", `${observed.disposition.kind}:${controlled.disposition.kind}`));
  if (!observed.spellings.includes(controlled.public_spelling)) failures.push(issue("public-spelling-not-observed", controlled.public_spelling));
  if (!prepares.has(id)) failures.push(issue("prepare-result-missing", id));
  const legacySources = observed.ownership.sidecars.length + observed.ownership.runtime_documentation_shadows.length;
  if (legacySources && !dispositions.has(id)) failures.push(issue("source-field-disposition-missing", id));
  if (dispositions.has(id)) verifyDestinations(id, dispositions.get(id), gates, failures);
  for (const gate of requiredGateNames(controlled)) if (!gates.has(gate)) failures.push(issue("required-gate-missing", gate));
  for (const removal of controlled.expected_removals) verifyRemoval(repository, removal, gates, failures);
  if (controlled.disposition.kind === "canonical") {
    const authority = observed.semantic_authority;
    if (authority.catalog_entries.length !== controlled.expected_authorities.catalog_entry_count) failures.push(issue("catalog-authority-count", `${authority.catalog_entries.length}:${controlled.expected_authorities.catalog_entry_count}`));
    const actualBindings = authority.implementation_provenance.filter((entry) => entry.authority === "canonical_binding").map((entry) => `${entry.source_file}:${entry.function}:${entry.binding_variant}`).sort(compareCodePoint);
    const expectedBindings = controlled.expected_authorities.runtime_bindings.map((entry) => `${entry.path}:${entry.function}:${entry.variant}`).sort(compareCodePoint);
    if (JSON.stringify(actualBindings) !== JSON.stringify(expectedBindings)) failures.push(issue("runtime-binding-set-mismatch", `${actualBindings.join("|")} != ${expectedBindings.join("|")}`));
    if (observed.ownership.sidecars.length) failures.push(issue("legacy-sidecar-present", observed.ownership.sidecars.join(",")));
    if (observed.ownership.runtime_documentation_shadows.length) failures.push(issue("runtime-shadow-present", observed.ownership.runtime_documentation_shadows.join(",")));
    if (observed.dependencies.legacy_resolver_paths.length) failures.push(issue("legacy-resolver-present", observed.dependencies.legacy_resolver_paths.join(",")));
  } else if (controlled.disposition.kind === "alias") {
    const target = controlled.disposition.target.toLowerCase();
    if (!observed.disposition.canonical || observed.disposition.canonical.toLowerCase() !== target) failures.push(issue("alias-target-mismatch", target));
    if (observed.ownership.catalog_documentation.length + legacySources) failures.push(issue("alias-copied-documentation", id));
  } else if (observed.semantic_authority.catalog_entries.length || observed.ownership.catalog_documentation.length + observed.ownership.sidecars.length) {
    failures.push(issue("internal-public-authority-present", id));
  }
  return identityResult(id, failures);
}

function verifyRemoval(repository, removal, gates, failures) {
  if (removal.kind === "file") {
    if (fs.existsSync(path.join(repository, removal.path))) failures.push(issue("expected-file-removal-present", removal.path));
  } else {
    const gate = gates.get("source-removal");
    const suffix = `${removal.path}:${removal.locator.kind}:${removal.locator.name}`;
    if (!gate) failures.push(issue("source-removal-gate-missing", suffix));
    else {
      const baseline = gate.checks.find((entry) => entry.id === `baseline:${suffix}` && entry.evidence_digest === removal.baseline_digest);
      const absent = gate.checks.find((entry) => entry.id === `absent:${suffix}` && entry.result === "pass");
      if (!baseline || !absent) failures.push(issue("source-removal-proof-incomplete", suffix));
    }
  }
}

function verifyDestinations(identity, disposition, gates, failures) {
  const documentation = gates.get("documentation-cutover");
  for (const source of disposition.sources) for (const leaf of source.leaves) {
    if (leaf.disposition === "removed") continue;
    if (leaf.destination.catalog_identity.toLowerCase() !== identity) { failures.push(issue("destination-identity-mismatch", leaf.destination.catalog_identity)); continue; }
    const id = `destination:${identity}:${source.path}:${leaf.pointer}:${leaf.destination.kind}:${leaf.destination.pointer}`;
    if (!documentation?.checks.some((entry) => entry.id === id && entry.result === "pass" && entry.evidence_digest === leaf.destination.value_digest)) failures.push(issue("destination-proof-missing", id));
  }
}

function identityResult(identity, failures) { return { identity, result: failures.length ? "fail" : "pass", failures: failures.sort((a, b) => compareCodePoint(`${a.code}:${a.detail}`, `${b.code}:${b.detail}`)) }; }
function issue(code, detail) { return { code, detail }; }
