import { compareCodePoint } from "./constants.mjs";
import { evidenceDigest } from "./evidence.mjs";
import { parseMigrationPhases } from "./migration-phase-schema.mjs";
import { requiredBarrierBundleIds } from "./queue-barriers.mjs";
import { validateAcceptedSealSet, validateBarrierSealSet } from "./seal-set-schema.mjs";
import {
  SAFE_IDENTITY, array, digest, exact, sourceRevision, stableId, uniqueStrings,
} from "./schema.mjs";

export function validateQueueSeal(value, reference, control) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`queue seal ${reference.artifact_id} is missing`);
  exact(value, [
    "schema_version", "kind", "authority", "seal_id", "bundle_id", "identities",
    "lease_id", "lease_digest", "phases", "source_revision", "source_digest",
    "control_baseline_inventory_digest", "lease_base_inventory_digest",
    "subject_inventory_digest", "control_manifest_digest", "verification_digest",
    "accepted_seals", "accepted_seal_set_digest", "barrier_seals",
    "barrier_seal_set_digest", "integration_gate_artifacts", "result", "failures",
  ], `queue seal ${reference.artifact_id}`);
  if (evidenceDigest(value) !== reference.digest) throw new Error(`queue seal ${reference.artifact_id} digest mismatch`);
  if (value.schema_version !== 5 || value.kind !== "runmat-builtin-migration-seal-result"
    || value.authority !== "development-integration-evidence-only" || value.result !== "pass") {
    throw new Error(`queue seal ${reference.artifact_id} is not a passing seal result`);
  }
  const sealId = stableId(value.seal_id, `queue seal ${reference.artifact_id} seal id`);
  const bundleId = stableId(value.bundle_id, `queue seal ${reference.artifact_id} bundle id`);
  stableId(value.lease_id, `queue seal ${reference.artifact_id} lease id`);
  digest(value.lease_digest, `queue seal ${reference.artifact_id} lease digest`);
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
  const barriers = validateBarrierSealSet(
    control.digest, bundleId, value.barrier_seals, value.barrier_seal_set_digest,
    `queue seal ${reference.artifact_id} barrier`,
  ).seals;
  const expectedBarriers = requiredBarrierBundleIds(control, bundleId);
  if (JSON.stringify(barriers.map((entry) => entry.bundle_id)) !== JSON.stringify(expectedBarriers)) {
    throw new Error(`queue seal ${reference.artifact_id} barrier coverage mismatch`);
  }
  const integrationArtifacts = uniqueStrings(
    value.integration_gate_artifacts, `queue seal ${reference.artifact_id} integration gate artifacts`,
  );
  if (JSON.stringify(integrationArtifacts) !== JSON.stringify([...integrationArtifacts].sort(compareCodePoint))) {
    throw new Error(`queue seal ${reference.artifact_id} integration gate artifacts must be canonically ordered`);
  }
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
  return {
    integratedRevision: phases.integrated_revision,
    sourceDigest: value.source_digest,
    acceptedSeals: accepted,
    barrierSeals: barriers,
  };
}
