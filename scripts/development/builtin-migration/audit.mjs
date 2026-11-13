import { subjectAuthorityPathFailures } from "./authority-paths.mjs";
import { compareCodePoint, sorted } from "./constants.mjs";
import { assertControlBaseline, assertControlSubject } from "./control.mjs";
import { evidenceDigest } from "./evidence.mjs";
import { parseGateResult } from "./gate-result.mjs";
import { requiredGateNames } from "./gate-requirements.mjs";
import { captureMigrationPhases } from "./integration-phases.mjs";
import { finalIdentityAuthorityFailures } from "./inventory-delta.mjs";
import { assertLeaseBaseInventory } from "./lease.mjs";
import { parsePrepareResult } from "./prepare.mjs";
import { SAFE_IDENTITY, exact, kind, stableId, uniqueStrings } from "./schema.mjs";
import { parseCompletedSourceDisposition } from "./source-fields.mjs";

export function auditMigration(
  repository, controlBaseline, leaseBase, subject, control, lease, batchValue, evidence,
  clock = Date.now,
) {
  assertControlBaseline(control, controlBaseline);
  assertLeaseBaseInventory(lease, control, leaseBase);
  assertControlSubject(control, subject);
  stableId(evidence.artifact_id, "audit artifact id");
  const requested = parseBatch(batchValue);
  const bundle = lease.bundle;
  const failures = [];
  const phases = captureMigrationPhases(
    repository, lease, control, subject, evidence.authored_revision, clock,
  );
  if (JSON.stringify(requested) !== JSON.stringify(sorted(bundle.identities))) failures.push(issue("partial-bundle", "audit batch must cover the complete reviewed bundle"));
  failures.push(...subjectAuthorityPathFailures(control, subject, [bundle.id])
    .map((failure) => issue("authority-path", failure)));
  const prepares = indexPrepare(evidence.prepare_results ?? [], leaseBase, control, lease, requested, failures);
  const dispositions = indexDispositions(evidence.source_dispositions ?? [], prepares, requested, failures);
  const expected = {
    source_revision: subject.source.revision, source_digest: subject.source.digest,
    control_baseline_source_revision: controlBaseline.source.revision,
    control_baseline_inventory_digest: controlBaseline.digest,
    lease_base_inventory_digest: leaseBase.digest,
    subject_inventory_digest: subject.digest,
    control_manifest_digest: control.digest, bundle_id: bundle.id,
    lease_id: lease.value.lease_id,
    lease_digest: lease.value.digest,
    queue_phase: lease.value.queue_phase,
    storage_policy: control.value.storage_policy,
    source_files: subject.source.files,
    repository, gate_plans: bundle.gate_plans, compiled_build: subject.compiled_inventory.build,
    execution_targets: control.executionTargets,
    subject_compiled_inventory_digest: subject.compiled_inventory.digest,
    documentation_source_dispositions: requested.flatMap((identity) => {
      const prepared = prepares.get(identity);
      const disposition = dispositions.get(identity);
      return prepared && disposition
        ? [{ baseline_digest: prepared.checklist_baseline_digest, value: disposition }]
        : [];
    }),
  };
  const gates = parseGates(evidence.gate_results ?? [], expected, requested, failures);
  const inventoryByIdentity = new Map(subject.identities.map((entry) => [entry.identity, entry]));
  const identities = requested.map((id) => auditIdentity(repository, id, inventoryByIdentity.get(id), control.identities.get(id), gates, prepares, dispositions));
  failures.push(...subject.diagnostics.filter((entry) => entry.severity === "error").map((entry) => issue("inventory-error", `${entry.code}:${entry.path ?? ""}`)));
  const passed = identities.filter((entry) => entry.result === "pass").length;
  return {
    schema_version: 9, kind: "runmat-builtin-migration-audit", authority: "development-verification-evidence-only",
    artifact_id: evidence.artifact_id, source: subject.source,
    control_baseline_inventory_digest: controlBaseline.digest,
    lease_base_inventory_digest: leaseBase.digest,
    subject_inventory_digest: subject.digest,
    control_manifest_digest: control.digest, bundle_id: bundle.id, lease_id: lease.value.lease_id,
    lease_digest: lease.value.digest,
    queue_phase: lease.value.queue_phase,
    accepted_seals: lease.value.accepted_seals,
    accepted_seal_set_digest: lease.value.accepted_seal_set_digest,
    barrier_seals: lease.value.barrier_seals,
    barrier_seal_set_digest: lease.value.barrier_seal_set_digest,
    phases,
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

function indexPrepare(values, baseline, control, lease, requested, failures) {
  const result = new Map();
  for (const raw of values) {
    try {
      const parsed = parsePrepareResult(raw, { bundle_id: lease.bundle.id, control_manifest_digest: control.digest, lease_id: lease.value.lease_id, lease_digest: lease.value.digest });
      if (!requested.includes(parsed.identity)) throw new Error(`${parsed.identity}: prepare result is outside the batch`);
      if (parsed.source.revision !== baseline.source.revision || parsed.source.digest !== baseline.source.digest || parsed.inventory_digest !== baseline.digest) throw new Error(`${parsed.identity}: stale prepare result`);
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
  const expectedDisposition = controlled.public_identity.kind === "primary"
    ? "canonical" : controlled.public_identity.kind;
  if (observed.disposition.kind !== expectedDisposition) failures.push(issue("disposition-mismatch", `${observed.disposition.kind}:${expectedDisposition}`));
  const expectedSpelling = controlled.public_identity.kind === "primary"
    ? controlled.public_identity.primary_spelling.spelling
    : controlled.public_identity.kind === "alias"
      ? controlled.public_identity.alias_spelling.spelling : null;
  if (expectedSpelling !== null && !observed.spellings.includes(expectedSpelling)) failures.push(issue("public-spelling-not-observed", expectedSpelling));
  if (!prepares.has(id)) failures.push(issue("prepare-result-missing", id));
  const legacySources = observed.ownership.sidecars.length + observed.ownership.runtime_documentation_shadows.length;
  if (legacySources && !dispositions.has(id)) failures.push(issue("source-field-disposition-missing", id));
  if (dispositions.has(id)) verifyDestinations(id, dispositions.get(id), gates, failures);
  for (const gate of requiredGateNames(controlled)) if (!gates.has(gate)) failures.push(issue("required-gate-missing", gate));
  failures.push(...finalIdentityAuthorityFailures(id, observed, controlled)
    .map((failure) => issue("final-authority", failure)));
  return identityResult(id, failures);
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
