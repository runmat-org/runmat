import { compareCodePoint } from "./constants.mjs";
import { assertValidatedControl } from "./control.mjs";
import { deepImmutable } from "./immutable.mjs";
import { evidenceDigest } from "./evidence.mjs";
import { parseMigrationPhases } from "./migration-phase-schema.mjs";
import {
  assertCompleteOrdinaryGateProof, parseOrdinaryGateProof,
} from "./ordinary-gate-proof.mjs";
import {
  requiredPilotBarrierBundleIds, requiredProductionBarrierBundleIds,
} from "./queue-barriers.mjs";
import { validateAcceptedSealSet, validateBarrierSealSet } from "./seal-set-schema.mjs";
import {
  SAFE_IDENTITY, array, digest, enumValue, exact, repositoryPath, sourceRevision,
  stableId, uniqueStrings,
} from "./schema.mjs";

const VALIDATED_QUEUE_SEALS = new WeakSet();

export function validateQueueSeal(value, reference, control) {
  assertValidatedControl(control);
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`queue seal ${reference.artifact_id} is missing`);
  exact(value, [
    "schema_version", "kind", "authority", "seal_id", "bundle_id", "identities",
    "lease_id", "lease_digest", "queue_phase", "phases", "source_revision", "source_digest",
    "control_baseline_inventory_digest", "lease_base_inventory_digest",
    "subject_inventory_digest", "control_manifest_digest", "verification_digest",
    "accepted_seals", "accepted_seal_set_digest", "barrier_seals",
    "barrier_seal_set_digest", "ordinary_gate_proof", "integration_gate_results",
    "result", "failures",
  ], `queue seal ${reference.artifact_id}`);
  if (evidenceDigest(value) !== reference.digest) throw new Error(`queue seal ${reference.artifact_id} digest mismatch`);
  if (value.schema_version !== 6 || value.kind !== "runmat-builtin-migration-seal-result"
    || value.authority !== "development-integration-evidence-only" || value.result !== "pass") {
    throw new Error(`queue seal ${reference.artifact_id} is not a passing seal result`);
  }
  const sealId = stableId(value.seal_id, `queue seal ${reference.artifact_id} seal id`);
  const bundleId = stableId(value.bundle_id, `queue seal ${reference.artifact_id} bundle id`);
  const leaseId = stableId(value.lease_id, `queue seal ${reference.artifact_id} lease id`);
  const leaseDigest = digest(value.lease_digest, `queue seal ${reference.artifact_id} lease digest`);
  const queuePhase = enumValue(
    value.queue_phase, ["pilot", "production"],
    `queue seal ${reference.artifact_id} queue phase`,
  );
  if (sealId !== reference.artifact_id) throw new Error(`queue seal ${reference.artifact_id} artifact id mismatch`);
  if (bundleId !== reference.bundle_id) throw new Error(`queue seal ${reference.artifact_id} bundle mismatch`);
  if (value.control_manifest_digest !== control.digest) throw new Error(`queue seal ${reference.artifact_id} belongs to another control manifest`);
  if (value.control_baseline_inventory_digest !== control.baseline.inventory_digest) throw new Error(`queue seal ${reference.artifact_id} has a stale control baseline inventory`);
  const phases = parseMigrationPhases(value.phases);
  const source = sourceRevision(value.source_revision, `queue seal ${reference.artifact_id} source revision`);
  if (source !== phases.integrated_revision) throw new Error(`queue seal ${reference.artifact_id} source revision differs from its integrated phase`);
  for (const [field, label] of [
    ["source_digest", "source"],
    ["control_baseline_inventory_digest", "control baseline inventory"],
    ["lease_base_inventory_digest", "lease base inventory"],
    ["subject_inventory_digest", "subject inventory"],
    ["control_manifest_digest", "control manifest"],
    ["verification_digest", "verification"],
  ]) digest(value[field], `queue seal ${reference.artifact_id} ${label} digest`);
  const accepted = validateAcceptedSealSet(
    control.digest, value.accepted_seals, value.accepted_seal_set_digest,
    `queue seal ${reference.artifact_id} accepted`,
  ).seals;
  const barriers = validateBarrierSealSet({
    controlManifestDigest: control.digest,
    bundleId,
    queuePhase,
    seals: value.barrier_seals,
    observedDigest: value.barrier_seal_set_digest,
    label: `queue seal ${reference.artifact_id} barrier`,
  }).seals;
  const expectedBarriers = queuePhase === "pilot"
    ? requiredPilotBarrierBundleIds(control, bundleId)
    : requiredProductionBarrierBundleIds(control, bundleId);
  if (JSON.stringify(barriers.map((entry) => entry.bundle_id)) !== JSON.stringify(expectedBarriers)) {
    throw new Error(`queue seal ${reference.artifact_id} barrier coverage mismatch`);
  }
  const ordinaryGateProof = assertCompleteOrdinaryGateProof(parseOrdinaryGateProof(
    value.ordinary_gate_proof, control, bundleId, queuePhase,
  ));
  const integrationGateResults = parseIntegrationGateResults(
    value.integration_gate_results, reference.artifact_id,
  );
  if (array(value.failures, `queue seal ${reference.artifact_id} failures`, { empty: true }).length !== 0) {
    throw new Error(`queue seal ${reference.artifact_id} contains failures`);
  }
  const identities = [...control.bundles.get(reference.bundle_id).identities].sort(compareCodePoint);
  const observedIdentities = uniqueStrings(
    value.identities, `queue seal ${reference.artifact_id} identities`,
    { pattern: SAFE_IDENTITY, lower: true },
  );
  if (JSON.stringify(observedIdentities) !== JSON.stringify(identities)) {
    throw new Error(`queue seal ${reference.artifact_id} identity coverage mismatch`);
  }
  const result = deepImmutable({
    value,
    reference,
    controlDigest: control.digest,
    sealId,
    bundleId,
    leaseId,
    leaseDigest,
    integratedRevision: phases.integrated_revision,
    sourceDigest: value.source_digest,
    subjectInventoryDigest: value.subject_inventory_digest,
    queuePhase,
    ordinaryGateProof,
    acceptedSeals: accepted,
    barrierSeals: barriers,
    integrationGateResults,
  });
  VALIDATED_QUEUE_SEALS.add(result);
  return result;
}

export function assertValidatedQueueSeal(value, control) {
  assertValidatedControl(control);
  if (!VALIDATED_QUEUE_SEALS.has(value)) {
    throw new Error("operation requires the exact validated queue seal");
  }
  if (value.controlDigest !== control.digest) {
    throw new Error("queue seal was validated for another control manifest");
  }
  return value;
}

function parseIntegrationGateResults(value, sealId) {
  const rows = array(value, `queue seal ${sealId} integration gate results`).map((entry) => {
    exact(entry, ["gate", "path", "artifact_id", "digest"], `queue seal ${sealId} integration gate result`);
    return {
      gate: stableId(entry.gate, `queue seal ${sealId} integration gate name`),
      path: repositoryPath(entry.path, `queue seal ${sealId} integration gate path`),
      artifact_id: stableId(entry.artifact_id, `queue seal ${sealId} integration gate artifact id`),
      digest: digest(entry.digest, `queue seal ${sealId} integration gate digest`),
    };
  });
  const names = rows.map((entry) => entry.gate);
  if (JSON.stringify(names) !== JSON.stringify(["deterministic-products", "inventory-delta"])) {
    throw new Error(`queue seal ${sealId} integration gates must exactly cover deterministic-products and inventory-delta`);
  }
  return rows;
}
