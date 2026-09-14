import { compareCodePoint } from "./constants.mjs";
import { assertValidatedControl } from "./control.mjs";
import { parseMigrationPhases } from "./integration-phases.mjs";
import { parseSourceSnapshot } from "./snapshot.mjs";
import { requiredGateNames } from "./gate-requirements.mjs";
import {
  validateAcceptedSealSet, validateBarrierSealSet,
} from "./seal-set-schema.mjs";
import { array, digest, enumValue, exact, integer, kind, nonempty, repositoryPath, sourceRevision, stableId, uniqueStrings, SAFE_IDENTITY } from "./schema.mjs";

export function parseVerificationManifest(value, control) {
  assertValidatedControl(control);
  kind(value, 8, "runmat-builtin-migration-verification-manifest", "verification manifest");
  exact(value, ["schema_version", "kind", "authority", "batch", "audit", "gate_results", "expectations"], "verification manifest");
  if (value.authority !== "reviewed-verification-request") throw new Error("verification manifest has invalid authority");
  exact(value.batch, ["artifact_id", "source_revision", "source_digest", "control_baseline_inventory_digest", "lease_base_inventory_digest", "subject_inventory_digest", "control_manifest_digest", "bundle_id", "lease_id", "lease_digest", "queue_phase", "accepted_seals", "accepted_seal_set_digest", "barrier_seals", "barrier_seal_set_digest", "identities", "phases"], "verification batch");
  if (value.batch.control_manifest_digest !== control.digest) {
    throw new Error("verification manifest belongs to another control manifest");
  }
  const bundleId = stableId(value.batch.bundle_id, "verification bundle id");
  const bundle = control.bundles.get(bundleId);
  if (!bundle) throw new Error(`${bundleId}: verification manifest references unknown bundle`);
  const queuePhase = enumValue(
    value.batch.queue_phase, ["pilot", "production"], "verification queue phase",
  );
  const identities = uniqueStrings(value.batch.identities, "verification identities", { pattern: SAFE_IDENTITY, lower: true }).sort(compareCodePoint);
  const accepted = validateAcceptedSealSet(
    value.batch.control_manifest_digest, value.batch.accepted_seals,
    value.batch.accepted_seal_set_digest, "verification accepted seals",
  );
  const barriers = validateBarrierSealSet({
    controlManifestDigest: control.digest,
    bundleId,
    queuePhase,
    seals: value.batch.barrier_seals,
    observedDigest: value.batch.barrier_seal_set_digest,
    label: "verification barrier seals",
  });
  const batch = {
    artifact_id: stableId(value.batch.artifact_id, "verification artifact id"),
    source_revision: sourceRevision(value.batch.source_revision, "verification source revision"),
    source_digest: digest(value.batch.source_digest, "verification source digest"),
    control_baseline_inventory_digest: digest(value.batch.control_baseline_inventory_digest, "verification control baseline inventory digest"),
    lease_base_inventory_digest: digest(value.batch.lease_base_inventory_digest, "verification lease base inventory digest"),
    subject_inventory_digest: digest(value.batch.subject_inventory_digest, "verification subject inventory digest"),
    control_manifest_digest: digest(value.batch.control_manifest_digest, "verification control digest"),
    bundle_id: bundleId,
    lease_id: stableId(value.batch.lease_id, "verification lease id"),
    lease_digest: digest(value.batch.lease_digest, "verification lease digest"),
    queue_phase: queuePhase, identities,
    accepted_seals: accepted.seals,
    accepted_seal_set_digest: digest(value.batch.accepted_seal_set_digest, "verification accepted seal-set digest"),
    barrier_seals: barriers.seals,
    barrier_seal_set_digest: digest(value.batch.barrier_seal_set_digest, "verification barrier seal-set digest"),
    phases: parseMigrationPhases(value.batch.phases),
  };
  if (batch.source_revision !== batch.phases.integrated_revision) {
    throw new Error("verification source revision differs from its integrated phase revision");
  }
  const audit = parseReference(value.audit, "audit reference");
  const gates = array(value.gate_results, "gate references").map((entry) => parseReference(entry, "gate reference"));
  if (!gates.length || new Set(gates.map((entry) => entry.artifact_id)).size !== gates.length) throw new Error("gate references must be nonempty and unique");
  const expectations = array(value.expectations, "verification expectations").map(parseExpectation).sort((a, b) => compareCodePoint(a.identity, b.identity));
  if (JSON.stringify(expectations.map((entry) => entry.identity)) !== JSON.stringify(identities)) throw new Error("expectations must cover each batch identity exactly once");
  const expectedIdentities = [...bundle.identities].sort(compareCodePoint);
  if (JSON.stringify(identities) !== JSON.stringify(expectedIdentities)) {
    throw new Error("verification manifest identities differ from the reviewed bundle");
  }
  for (const expectation of expectations) {
    const required = requiredGateNames(control.identities.get(expectation.identity));
    if (JSON.stringify(expectation.required_gates) !== JSON.stringify(required)) {
      throw new Error(`${expectation.identity}: verification expectations differ from reviewed control`);
    }
  }
  return { ...value, batch, audit, gate_results: gates.sort((a, b) => compareCodePoint(a.artifact_id, b.artifact_id)), expectations };
}

export function validateAudit(value, batch) {
  kind(value, 9, "runmat-builtin-migration-audit", "migration audit");
  exact(value, ["schema_version", "kind", "authority", "artifact_id", "source", "control_baseline_inventory_digest", "lease_base_inventory_digest", "subject_inventory_digest", "control_manifest_digest", "bundle_id", "lease_id", "lease_digest", "queue_phase", "accepted_seals", "accepted_seal_set_digest", "barrier_seals", "barrier_seal_set_digest", "phases", "requested_identities", "evidence", "summary", "result", "global_failures", "identities"], "migration audit");
  if (value.authority !== "development-verification-evidence-only") throw new Error("migration audit has invalid authority");
  stableId(value.artifact_id, "migration audit artifact id");
  digest(value.control_baseline_inventory_digest, "migration audit control baseline inventory digest");
  digest(value.lease_base_inventory_digest, "migration audit lease base inventory digest");
  digest(value.subject_inventory_digest, "migration audit subject inventory digest");
  digest(value.control_manifest_digest, "migration audit control digest");
  stableId(value.bundle_id, "migration audit bundle id");
  stableId(value.lease_id, "migration audit lease id");
  digest(value.lease_digest, "migration audit lease digest");
  const queuePhase = enumValue(value.queue_phase, ["pilot", "production"], "migration audit queue phase");
  validateAcceptedSealSet(
    value.control_manifest_digest, value.accepted_seals, value.accepted_seal_set_digest,
    "migration audit accepted seals",
  );
  validateBarrierSealSet({
    controlManifestDigest: value.control_manifest_digest,
    bundleId: value.bundle_id,
    queuePhase,
    seals: value.barrier_seals,
    observedDigest: value.barrier_seal_set_digest,
    label: "migration audit barrier seals",
  });
  const phases = parseMigrationPhases(value.phases);
  parseSourceSnapshot(value.source, "migration audit source", { emptyFiles: true });
  exact(value.evidence, ["gate_artifacts", "prepare_digests", "source_disposition_digests"], "migration audit evidence");
  uniqueStrings(value.evidence.gate_artifacts, "migration audit gate artifacts", { empty: true });
  for (const field of ["prepare_digests", "source_disposition_digests"]) array(value.evidence[field], `migration audit ${field}`, { empty: true }).forEach((entry) => digest(entry, `migration audit ${field}`));
  exact(value.summary, ["identities", "passed", "failed", "global_failures"], "migration audit summary");
  for (const field of ["identities", "passed", "failed", "global_failures"]) integer(value.summary[field], `migration audit summary ${field}`);
  enumValue(value.result, ["pass", "fail"], "migration audit result");
  array(value.global_failures, "migration audit global failures", { empty: true }).forEach((entry) => { exact(entry, ["code", "detail"], "migration audit failure"); nonempty(entry.code, "migration audit failure code"); nonempty(entry.detail, "migration audit failure detail"); });
  array(value.identities, "migration audit identity results").forEach((entry) => {
    exact(entry, ["identity", "result", "failures"], "migration audit identity result");
    if (!SAFE_IDENTITY.test(nonempty(entry.identity, "migration audit identity"))) throw new Error("migration audit identity is unsafe");
    if (!["pass", "fail"].includes(entry.result)) throw new Error("migration audit identity result is invalid");
    array(entry.failures, "migration audit identity failures", { empty: true }).forEach((failure) => { exact(failure, ["code", "detail"], "migration audit identity failure"); nonempty(failure.code, "migration audit identity failure code"); nonempty(failure.detail, "migration audit identity failure detail"); });
    if ((entry.result === "pass") !== (entry.failures.length === 0)) throw new Error(`${entry.identity}: audit result conflicts with its failures`);
  });
  const requested = uniqueStrings(value.requested_identities, "migration audit requested identities", { pattern: SAFE_IDENTITY, lower: true }).sort(compareCodePoint);
  if (JSON.stringify(requested) !== JSON.stringify(value.requested_identities)) throw new Error("migration audit requested identities must be normalized and canonically ordered");
  const resultIds = value.identities.map((entry) => entry.identity).sort(compareCodePoint);
  if (new Set(resultIds).size !== resultIds.length || JSON.stringify(resultIds) !== JSON.stringify(value.identities.map((entry) => entry.identity))) throw new Error("migration audit identity results must be unique and canonically ordered");
  if (JSON.stringify(requested) !== JSON.stringify(resultIds)) throw new Error("migration audit result identities do not exactly match its request");
  const failed = value.identities.filter((entry) => entry.result === "fail").length;
  const expectedResult = failed === 0 && value.global_failures.length === 0 ? "pass" : "fail";
  if (value.summary.identities !== value.identities.length || value.summary.passed !== value.identities.length - failed || value.summary.failed !== failed || value.summary.global_failures !== value.global_failures.length || value.result !== expectedResult) throw new Error("migration audit result or summary is inconsistent");
  if (value.artifact_id !== batch.audit_artifact_id || value.source.revision !== batch.source_revision || value.source.digest !== batch.source_digest || value.control_baseline_inventory_digest !== batch.control_baseline_inventory_digest || value.lease_base_inventory_digest !== batch.lease_base_inventory_digest || value.subject_inventory_digest !== batch.subject_inventory_digest || value.control_manifest_digest !== batch.control_manifest_digest || value.bundle_id !== batch.bundle_id || value.lease_id !== batch.lease_id || value.lease_digest !== batch.lease_digest || value.queue_phase !== batch.queue_phase || JSON.stringify(value.accepted_seals) !== JSON.stringify(batch.accepted_seals) || value.accepted_seal_set_digest !== batch.accepted_seal_set_digest || JSON.stringify(value.barrier_seals) !== JSON.stringify(batch.barrier_seals) || value.barrier_seal_set_digest !== batch.barrier_seal_set_digest || JSON.stringify(phases) !== JSON.stringify(batch.phases)) throw new Error("migration audit provenance is stale or mismatched");
  if (JSON.stringify(value.requested_identities) !== JSON.stringify(batch.identities)) throw new Error("migration audit identities differ from verification batch");
  if (value.result !== "pass" || value.global_failures.length || value.identities.some((entry) => entry.result !== "pass")) throw new Error("migration audit is not passing");
  return value;
}

function parseExpectation(value) {
  exact(value, ["identity", "required_gates"], "verification expectation");
  const identity = nonempty(value.identity, "expectation identity").toLowerCase();
  if (!SAFE_IDENTITY.test(identity)) throw new Error(`unsafe expectation identity ${identity}`);
  return { identity, required_gates: uniqueStrings(value.required_gates, `${identity} required gates`).sort(compareCodePoint) };
}

function parseReference(value, label) {
  exact(value, ["path", "artifact_id", "digest"], label);
  return { path: repositoryPath(value.path, `${label} path`), artifact_id: stableId(value.artifact_id, `${label} artifact id`), digest: digest(value.digest, `${label} digest`) };
}
