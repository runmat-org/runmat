import fs from "node:fs";
import path from "node:path";

import { compareCodePoint } from "./constants.mjs";
import { evidenceDigest } from "./evidence.mjs";

export function buildInventoryDeltaProof(repository, baseline, current, control) {
  const bundle = control.bundles.get(control.active_bundle_id);
  if (!bundle) throw new Error(`${control.active_bundle_id}: inventory delta bundle is absent`);
  const failures = [];
  if (current.compiled_inventory.build.operating_system !== baseline.compiled_inventory.build.operating_system
      || current.compiled_inventory.build.architecture !== baseline.compiled_inventory.build.architecture) {
    failures.push("compiled target differs from the frozen baseline");
  }
  const baselineRows = new Map(baseline.identities.map((entry) => [entry.identity, entry]));
  const currentRows = new Map(current.identities.map((entry) => [entry.identity, entry]));
  if (JSON.stringify([...baselineRows.keys()].sort(compareCodePoint)) !== JSON.stringify([...currentRows.keys()].sort(compareCodePoint))) {
    failures.push("identity set differs from the frozen baseline");
  }
  failures.push(...changedPathFailures(baseline, current, bundle));
  failures.push(...current.diagnostics.filter((entry) => entry.severity === "error").map((entry) => `inventory diagnostic ${entry.code}:${entry.path ?? ""}`));
  failures.push(...findingFailures(baseline, current, control, bundle));

  const identities = bundle.identities.map((identity) => {
    const identityFailures = auditIdentity(repository, identity, baselineRows.get(identity), currentRows.get(identity), control.identities.get(identity));
    return {
      identity,
      baseline_row_digest: evidenceDigest(baselineRows.get(identity)),
      current_row_digest: currentRows.has(identity) ? evidenceDigest(currentRows.get(identity)) : null,
      result: identityFailures.length === 0 ? "pass" : "fail",
      failures: identityFailures,
    };
  });
  for (const [identity, baselineRow] of baselineRows) {
    if (bundle.identities.includes(identity)) continue;
    const currentRow = currentRows.get(identity);
    if (!currentRow || evidenceDigest(baselineRow.semantic_authority) !== evidenceDigest(currentRow.semantic_authority)) {
      failures.push(`${identity}: compiled authority changed outside the reviewed bundle`);
    }
  }
  const payload = {
    schema_version: 1,
    kind: "runmat-builtin-inventory-delta-proof",
    authority: "machine-derived-migration-evidence",
    baseline_source_revision: baseline.source.revision,
    subject_source_revision: current.source.revision,
    baseline_inventory_digest: baseline.digest,
    subject_inventory_digest: current.digest,
    subject_source_digest: current.source.digest,
    subject_compiled_inventory_digest: current.compiled_inventory.digest,
    control_manifest_digest: control.digest,
    bundle_id: bundle.id,
    changed_paths: changedPaths(baseline.source.files, current.source.files),
    failures: failures.sort(compareCodePoint),
    identities,
    result: failures.length === 0 && identities.every((entry) => entry.result === "pass") ? "pass" : "fail",
  };
  return { ...payload, digest: evidenceDigest(payload) };
}

export function inventoryDeltaChecks(proof) {
  return proof.identities.map((entry) => ({
    id: `inventory-delta:${entry.identity}`,
    result: proof.result === "pass" && entry.result === "pass" ? "pass" : "fail",
    evidence_digest: evidenceDigest(entry),
  }));
}

function changedPathFailures(baseline, current, bundle) {
  const scopes = [...bundle.authored_write_set, ...bundle.integration_outputs.map((entry) => ({ kind: "file", path: entry.path }))];
  return changedPaths(baseline.source.files, current.source.files)
    .filter((sourcePath) => !scopes.some((scope) => withinScope(sourcePath, scope)))
    .map((sourcePath) => `${sourcePath}: source changed outside the reviewed bundle scopes`);
}

function changedPaths(baselineFiles, currentFiles) {
  const baseline = new Map(baselineFiles.map((entry) => [entry.path, `${entry.mode}:${entry.content_digest}`]));
  const current = new Map(currentFiles.map((entry) => [entry.path, `${entry.mode}:${entry.content_digest}`]));
  return [...new Set([...baseline.keys(), ...current.keys()])]
    .filter((sourcePath) => baseline.get(sourcePath) !== current.get(sourcePath))
    .sort(compareCodePoint);
}

function withinScope(sourcePath, scope) {
  return scope.kind === "file" ? sourcePath === scope.path : sourcePath === scope.path || sourcePath.startsWith(`${scope.path}/`);
}

function findingFailures(baseline, current, control, bundle) {
  const currentDigests = new Set(current.migration_findings.map(evidenceDigest));
  const reviewed = new Map(control.migrationFindings.map((entry) => [entry.finding_digest, entry]));
  const failures = [];
  for (const finding of baseline.migration_findings) {
    const digest = evidenceDigest(finding);
    const disposition = reviewed.get(digest);
    if (!disposition) {
      failures.push(`${digest}: baseline finding has no reviewed disposition`);
      continue;
    }
    const mustResolve = disposition.disposition === "bundle-work" && disposition.bundle_id === bundle.id;
    if (mustResolve && currentDigests.has(digest)) failures.push(`${digest}: bundle-owned migration finding remains unresolved`);
    if (!mustResolve && !currentDigests.has(digest)) failures.push(`${digest}: migration finding changed outside this bundle`);
  }
  const baselineDigests = new Set(baseline.migration_findings.map(evidenceDigest));
  for (const digest of currentDigests) if (!baselineDigests.has(digest)) failures.push(`${digest}: current inventory introduced an unreviewed migration finding`);
  return failures;
}

function auditIdentity(repository, identity, baseline, current, controlled) {
  const failures = [];
  if (!baseline) failures.push("baseline inventory row is absent");
  if (!current) failures.push("current inventory row is absent");
  if (!controlled) failures.push("reviewed control row is absent");
  if (!baseline || !current || !controlled) return failures;
  if (current.unresolved.length) failures.push(`current inventory remains unresolved: ${current.unresolved.join(",")}`);
  const authority = current.semantic_authority;
  if (controlled.disposition.kind === "canonical") {
    if (authority.catalog_entries.length !== controlled.expected_authorities.catalog_entry_count) failures.push("catalog authority count differs from review");
    if (authority.constants.length !== controlled.expected_authorities.catalog_constant_count) failures.push("catalog constant authority count differs from review");
    const actualConstants = authority.runtime_constants.map((entry) => entry.name).sort(compareCodePoint);
    const expectedConstants = [...controlled.expected_authorities.runtime_constants].sort(compareCodePoint);
    if (JSON.stringify(actualConstants) !== JSON.stringify(expectedConstants)) failures.push("runtime constant set differs from review");
    if (authority.legacy_functions.length) failures.push("legacy function authority remains");
    if (authority.legacy_documentation.length) failures.push("legacy documentation authority remains");
    if (current.ownership.sidecars.length) failures.push("legacy documentation sidecar remains");
    if (current.ownership.runtime_documentation_shadows.length) failures.push("runtime documentation shadow remains");
    if (current.dependencies.legacy_resolver_paths.length) failures.push("legacy resolver ownership remains");
    const actualBindings = authority.implementation_provenance
      .filter((entry) => entry.authority === "canonical_binding")
      .map((entry) => `${entry.source_file}:${entry.function}:${entry.binding_variant}`)
      .sort(compareCodePoint);
    const expectedBindings = controlled.expected_authorities.runtime_bindings
      .map((entry) => `${entry.path}:${entry.function}:${entry.variant}`)
      .sort(compareCodePoint);
    if (JSON.stringify(actualBindings) !== JSON.stringify(expectedBindings)) failures.push("canonical runtime binding provenance differs from review");
  } else if (controlled.disposition.kind === "alias") {
    if (current.disposition.kind !== "alias" || current.disposition.canonical !== controlled.disposition.target.toLowerCase()) failures.push("alias target differs from review");
    if (authority.catalog_entries.length || current.ownership.catalog_documentation.length || current.ownership.sidecars.length) failures.push("alias retains copied public authority");
  } else if (authority.catalog_entries.length || authority.legacy_functions.length || current.ownership.catalog_documentation.length || current.ownership.sidecars.length) {
    failures.push("internal identity retains public authority");
  }
  for (const removal of controlled.expected_removals) {
    if (fs.existsSync(path.join(repository, removal.path))) failures.push(`${removal.path}: reviewed file removal remains present`);
  }
  return failures.sort(compareCodePoint);
}
