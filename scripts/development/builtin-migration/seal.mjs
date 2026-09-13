import { evidenceDigest } from "./evidence.mjs";
import { assertValidatedControl } from "./control.mjs";
import { parseGateResult } from "./gate-result.mjs";
import { parseMigrationPhases, validateSealedMigrationPhases } from "./integration-phases.mjs";
import { assertActiveLease } from "./lease.mjs";
import {
  requiredPilotBarrierBundleIds, requiredProductionBarrierBundleIds,
} from "./queue-barriers.mjs";
import { validateQueueSeal } from "./queue-seal.mjs";
import {
  parseSealReferences, validateAcceptedSealSet, validateBarrierSealSet,
} from "./seal-set-schema.mjs";
import { parseVerificationResult } from "./verification-result.mjs";
import { array, digest, enumValue, exact, kind, repositoryPath, sourceRevision, stableId, uniqueStrings, SAFE_IDENTITY } from "./schema.mjs";

export function sealBundle(
  manifest, verification, gateValues, barrierValues = [], control = null,
  repository = null, lease = null, clock = Date.now,
) {
  assertValidatedControl(control);
  const parsed = parseSealManifest(manifest);
  const activeLease = assertActiveLease(lease, control, clock);
  const failures = [];
  if (control.digest !== parsed.control_manifest_digest || !control.bundles.has(parsed.bundle_id)) throw new Error("seal requires its exact parsed control manifest");
  if (activeLease.value.digest !== parsed.lease_digest
    || activeLease.value.lease_id !== parsed.lease_id
    || activeLease.bundle.id !== parsed.bundle_id) {
    throw new Error("seal requires the exact active authored lease");
  }
  if (parsed.queue_phase !== activeLease.value.queue_phase) {
    throw new Error("seal queue phase differs from its exact active authored lease");
  }
  if (parsed.control_baseline_inventory_digest !== control.baseline.inventory_digest) throw new Error("seal control baseline inventory differs from the reviewed control baseline");
  if (!repository) throw new Error("seal requires the trusted repository path for phase and gate-plan verification");
  const phases = validateSealedMigrationPhases(repository, control, parsed.bundle_id, parsed.phases);
  if (parsed.source_revision !== phases.integrated_revision) {
    throw new Error("seal source revision differs from its integrated phase revision");
  }
  const requiredBarriers = parsed.queue_phase === "pilot"
    ? requiredPilotBarrierBundleIds(control, parsed.bundle_id)
    : requiredProductionBarrierBundleIds(control, parsed.bundle_id);
  const declaredBarriers = parsed.barrier_seals.map((entry) => entry.bundle_id).sort();
  if (JSON.stringify(requiredBarriers) !== JSON.stringify(declaredBarriers)) throw new Error("seal barrier references do not exactly match the control DAG and cohort barriers");
  validateBarrierSealSet({
    controlManifestDigest: control.digest,
    bundleId: parsed.bundle_id,
    queuePhase: parsed.queue_phase,
    seals: parsed.barrier_seals,
    observedDigest: parsed.barrier_seal_set_digest,
    label: "seal barrier seals",
  });
  validateAcceptedSealSet(
    control.digest, parsed.accepted_seals, parsed.accepted_seal_set_digest,
    "seal accepted seals",
  );
  const acceptedByBundle = new Map(parsed.accepted_seals.map((entry) => [entry.bundle_id, entry]));
  for (const barrier of parsed.barrier_seals) {
    if (JSON.stringify(acceptedByBundle.get(barrier.bundle_id)) !== JSON.stringify(barrier)) {
      throw new Error("seal barrier set is not an exact subset of its accepted seal set");
    }
  }
  const verified = reconcileVerification(parsed, verification, control, failures);
  const expected = {
    source_revision: parsed.source_revision, source_digest: parsed.source_digest,
    control_baseline_source_revision: control.baseline.revision,
    control_baseline_inventory_digest: parsed.control_baseline_inventory_digest,
    lease_base_inventory_digest: parsed.lease_base_inventory_digest,
    subject_inventory_digest: parsed.subject_inventory_digest,
    control_manifest_digest: parsed.control_manifest_digest, bundle_id: parsed.bundle_id,
    lease_id: parsed.lease_id,
    lease_digest: parsed.lease_digest,
    queue_phase: parsed.queue_phase,
    gate_plans: control.bundles.get(parsed.bundle_id).gate_plans,
    execution_targets: control.executionTargets,
    repository,
  };
  const gates = new Map();
  for (const reference of parsed.integration_gates) {
    const loaded = gateValues.find((entry) => entry.reference.artifact_id === reference.artifact_id);
    if (!loaded || evidenceDigest(loaded.value) !== reference.digest) { failures.push(issue("integration-gate-reference-invalid", reference.artifact_id)); continue; }
    try {
      const gate = parseGateResult(loaded.value, expected);
      if (gate.artifact_id !== reference.artifact_id || gate.result !== "pass") throw new Error(`${gate.gate}: evidence is not passing`);
      if (JSON.stringify(gate.identities) !== JSON.stringify(parsed.identities)) throw new Error(`${gate.gate}: identities differ from bundle`);
      if (gates.has(gate.gate)) throw new Error(`duplicate integration gate ${gate.gate}`);
      gates.set(gate.gate, gate);
    } catch (error) { failures.push(issue("integration-gate-invalid", error.message)); }
  }
  for (const required of ["deterministic-products", "inventory-delta"]) if (!gates.has(required)) failures.push(issue("integration-gate-missing", required));
  for (const reference of parsed.accepted_seals) {
    const loaded = barrierValues.find((entry) => entry.reference.artifact_id === reference.artifact_id);
    try {
      if (!loaded || JSON.stringify(loaded.reference) !== JSON.stringify(reference)) {
        throw new Error("loaded reference differs from accepted seal reference");
      }
      validateQueueSeal(loaded.value, reference, control);
    } catch (error) { failures.push(issue("barrier-seal-invalid", error.message)); }
  }
  return {
    schema_version: 6, kind: "runmat-builtin-migration-seal-result", authority: "development-integration-evidence-only",
    seal_id: parsed.seal_id, bundle_id: parsed.bundle_id, identities: parsed.identities,
    lease_id: parsed.lease_id, lease_digest: parsed.lease_digest, phases,
    queue_phase: parsed.queue_phase,
    source_revision: parsed.source_revision, source_digest: parsed.source_digest,
    control_baseline_inventory_digest: parsed.control_baseline_inventory_digest,
    lease_base_inventory_digest: parsed.lease_base_inventory_digest,
    subject_inventory_digest: parsed.subject_inventory_digest,
    control_manifest_digest: parsed.control_manifest_digest, verification_digest: parsed.verification.digest,
    accepted_seals: parsed.accepted_seals,
    accepted_seal_set_digest: parsed.accepted_seal_set_digest,
    barrier_seals: parsed.barrier_seals,
    barrier_seal_set_digest: parsed.barrier_seal_set_digest,
    ordinary_gate_proof: verified?.ordinaryGateProof.value ?? null,
    integration_gate_results: [...gates.values()].map((entry) => ({
      gate: entry.gate,
      path: parsed.integration_gates.find((reference) =>
        reference.artifact_id === entry.artifact_id).path,
      artifact_id: entry.artifact_id,
      digest: parsed.integration_gates.find((reference) =>
        reference.artifact_id === entry.artifact_id).digest,
    })).sort((left, right) => left.gate.localeCompare(right.gate)),
    result: failures.length ? "fail" : "pass", failures,
  };
}

export function parseSealManifest(value) {
  kind(value, 6, "runmat-builtin-migration-seal-manifest", "seal manifest");
  exact(value, ["schema_version", "kind", "authority", "seal_id", "bundle_id", "lease_id", "lease_digest", "queue_phase", "identities", "source_revision", "source_digest", "control_baseline_inventory_digest", "lease_base_inventory_digest", "subject_inventory_digest", "control_manifest_digest", "accepted_seals", "accepted_seal_set_digest", "barrier_seals", "barrier_seal_set_digest", "phases", "verification", "integration_gates", "review"], "seal manifest");
  if (value.authority !== "reviewed-integration-request") throw new Error("seal manifest has invalid authority");
  const result = {
    ...value, seal_id: stableId(value.seal_id, "seal id"), bundle_id: stableId(value.bundle_id, "seal bundle id"),
    lease_id: stableId(value.lease_id, "seal lease id"),
    lease_digest: digest(value.lease_digest, "seal lease digest"),
    queue_phase: enumValue(value.queue_phase, ["pilot", "production"], "seal queue phase"),
    identities: uniqueStrings(value.identities, "seal identities", { pattern: SAFE_IDENTITY, lower: true }).sort(),
    source_revision: sourceRevision(value.source_revision, "seal source revision"), source_digest: digest(value.source_digest, "seal source digest"),
    control_baseline_inventory_digest: digest(value.control_baseline_inventory_digest, "seal control baseline inventory digest"),
    lease_base_inventory_digest: digest(value.lease_base_inventory_digest, "seal lease base inventory digest"),
    subject_inventory_digest: digest(value.subject_inventory_digest, "seal subject inventory digest"),
    control_manifest_digest: digest(value.control_manifest_digest, "seal control digest"),
    accepted_seals: parseSealReferences(value.accepted_seals, "seal accepted seals"),
    accepted_seal_set_digest: digest(value.accepted_seal_set_digest, "seal accepted seal-set digest"),
    barrier_seals: parseSealReferences(value.barrier_seals, "seal barrier seals"),
    barrier_seal_set_digest: digest(value.barrier_seal_set_digest, "seal barrier seal-set digest"),
    phases: parseMigrationPhases(value.phases),
    verification: reference(value.verification, "seal verification"),
    integration_gates: array(value.integration_gates, "seal integration gates").map((entry) => reference(entry, "seal integration gate")),
  };
  exact(value.review, ["status", "evidence"], "seal review");
  if (value.review.status !== "reviewed") throw new Error("seal request must be reviewed");
  uniqueStrings(value.review.evidence, "seal review evidence");
  return result;
}

function reconcileVerification(manifest, loaded, control, failures) {
  if (!loaded || evidenceDigest(loaded) !== manifest.verification.digest) {
    failures.push(issue("verification-reference-invalid", manifest.verification.artifact_id));
    return null;
  }
  try {
    const parsed = parseVerificationResult(loaded, control, { requirePassing: true });
    if (loaded.artifact_id !== manifest.verification.artifact_id || loaded.bundle_id !== manifest.bundle_id || loaded.lease_id !== manifest.lease_id || loaded.lease_digest !== manifest.lease_digest || loaded.queue_phase !== manifest.queue_phase || JSON.stringify(loaded.identities) !== JSON.stringify(manifest.identities) || JSON.stringify(loaded.phases) !== JSON.stringify(manifest.phases) || loaded.source_revision !== manifest.source_revision || loaded.source_digest !== manifest.source_digest || loaded.control_baseline_inventory_digest !== manifest.control_baseline_inventory_digest || loaded.lease_base_inventory_digest !== manifest.lease_base_inventory_digest || loaded.subject_inventory_digest !== manifest.subject_inventory_digest || loaded.control_manifest_digest !== manifest.control_manifest_digest || JSON.stringify(loaded.accepted_seals) !== JSON.stringify(manifest.accepted_seals) || loaded.accepted_seal_set_digest !== manifest.accepted_seal_set_digest || JSON.stringify(loaded.barrier_seals) !== JSON.stringify(manifest.barrier_seals) || loaded.barrier_seal_set_digest !== manifest.barrier_seal_set_digest) {
      throw new Error("verification provenance differs from seal request");
    }
    return parsed;
  } catch (error) {
    failures.push(issue("verification-not-passing", error.message));
    return null;
  }
}

function reference(value, label) { exact(value, ["path", "artifact_id", "digest"], label); return { path: repositoryPath(value.path, `${label} path`), artifact_id: stableId(value.artifact_id, `${label} artifact id`), digest: digest(value.digest, `${label} digest`) }; }
function issue(code, detail) { return { code, detail }; }
