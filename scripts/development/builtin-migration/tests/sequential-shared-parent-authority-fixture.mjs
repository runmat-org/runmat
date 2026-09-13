import fs from "node:fs";
import path from "node:path";

import { compareCodePoint } from "../constants.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { requiredGateNames } from "../gate-requirements.mjs";
import { buildOrdinaryGateProof } from "../ordinary-gate-proof.mjs";
import { validateQueueState } from "../queue.mjs";
import { validateQueueCheckpoint } from "../queue-checkpoint.mjs";

export function acceptFirstBundle({ fixture, subjectInventory, phases }) {
  return acceptBundle({
    fixture,
    subjectInventory,
    phases,
    leaseRecord: fixture.firstLease,
    predecessorState: fixture.queueState,
    predecessorStatePath: "test-artifacts/queue-state-0.json",
    predecessorCheckpoint: fixture.queueCheckpoint,
    predecessorCheckpointPath: "test-artifacts/queue-checkpoint-0.json",
    checkpointHistory: new Map([[
      fixture.queueCheckpoint.digest, fixture.queueCheckpointValue,
    ]]),
    sealId: "seal-alpha",
    sealPath: "test-artifacts/seal-alpha.json",
  });
}

export function acceptBundle({
  fixture, subjectInventory, phases, leaseRecord,
  predecessorState, predecessorStatePath,
  predecessorCheckpoint, predecessorCheckpointPath, checkpointHistory,
  sealId, sealPath,
}) {
  const artifactRoot = path.join(path.dirname(fixture.repository), "sequential-authority");
  fs.mkdirSync(artifactRoot, { recursive: true });
  const lease = leaseRecord.lease;
  const bundleId = lease.value.bundle_id;
  const seal = passingSeal({
    fixture, subjectInventory, phases, lease, bundleId, sealId,
  });
  const physicalSealPath = path.join(artifactRoot, `${sealId}.json`);
  writeJson(physicalSealPath, seal);
  const reference = {
    path: sealPath,
    artifact_id: seal.seal_id,
    digest: evidenceDigest(seal),
    bundle_id: seal.bundle_id,
  };
  const references = [...predecessorState.acceptedSeals, reference]
    .sort((left, right) => compareCodePoint(left.bundle_id, right.bundle_id));
  const queueStatePayload = {
    schema_version: 4,
    kind: "runmat-builtin-migration-queue-state",
    authority: "reviewed-monotonic-scheduling-input",
    control_manifest_digest: fixture.control.digest,
    phase: "pilot",
    pilot_transition: null,
    predecessor: {
      state_path: predecessorStatePath,
      state_digest: predecessorState.stateDigest,
      checkpoint_path: predecessorCheckpointPath,
      checkpoint_digest: predecessorCheckpoint.digest,
    },
    bundles: {
      ...structuredClone(predecessorState.value.bundles),
      [bundleId]: { artifact: seal.seal_id, state: "verified" },
    },
    seals: references,
  };
  const queueStateValue = {
    ...queueStatePayload, digest: evidenceDigest(queueStatePayload),
  };
  const queueStatePath = path.join(artifactRoot, `${sealId}-queue-state.json`);
  writeJson(queueStatePath, queueStateValue);
  const queueState = validateQueueState(
    readJson(queueStatePath), fixture.control,
    (sealReference) => sealReference.bundle_id === bundleId
      ? readJson(physicalSealPath)
      : predecessorState.sealedBundles.find(
        (entry) => entry.reference.bundle_id === sealReference.bundle_id,
      )?.seal.value,
    () => predecessorState,
  );
  const acceptedDigest = evidenceDigest({
    schema_version: 1,
    kind: "runmat-builtin-migration-accepted-seal-set",
    authority: "derived-from-validated-seals",
    control_manifest_digest: fixture.control.digest,
    seals: references,
  });
  const checkpointPayload = {
    schema_version: 2,
    kind: "runmat-builtin-migration-queue-checkpoint",
    authority: "reviewer-authored-current-queue-checkpoint",
    control_manifest_digest: fixture.control.digest,
    queue_state_digest: queueState.stateDigest,
    predecessor_checkpoint_digest: predecessorCheckpoint.digest,
    phase: "pilot",
    head_event: { kind: "seal", seal: reference },
    source_revision: subjectInventory.source.revision,
    source_digest: subjectInventory.source.digest,
    inventory_digest: subjectInventory.digest,
    accepted_seal_set_digest: acceptedDigest,
    review: { status: "reviewed", evidence: ["serialized successor checkpoint review"] },
  };
  const checkpointValue = {
    ...checkpointPayload, digest: evidenceDigest(checkpointPayload),
  };
  const checkpointPath = path.join(artifactRoot, `${sealId}-queue-checkpoint.json`);
  writeJson(checkpointPath, checkpointValue);
  const queueCheckpoint = validateQueueCheckpoint(
    readJson(checkpointPath), checkpointValue.digest, queueState, fixture.control,
    (predecessor) => checkpointHistory.get(predecessor.checkpoint_digest),
  );
  return {
    seal, reference, queueState, queueStateValue, queueCheckpoint, checkpointValue,
  };
}

function passingSeal({ fixture, subjectInventory, phases, lease, bundleId, sealId }) {
  const gates = new Map();
  for (const identity of fixture.control.bundles.get(bundleId).identities) {
    for (const gate of requiredGateNames(fixture.control.identities.get(identity))) {
      gates.set(gate, { gate, reference: gateReference(gate) });
    }
  }
  const ordinaryGateProof = buildOrdinaryGateProof(
    fixture.control, bundleId, "pilot", gates,
  );
  return {
    schema_version: 6,
    kind: "runmat-builtin-migration-seal-result",
    authority: "development-integration-evidence-only",
    seal_id: sealId,
    bundle_id: bundleId,
    identities: [...fixture.control.bundles.get(bundleId).identities],
    lease_id: lease.value.lease_id,
    lease_digest: lease.value.digest,
    queue_phase: "pilot",
    phases: structuredClone(phases),
    source_revision: subjectInventory.source.revision,
    source_digest: subjectInventory.source.digest,
    control_baseline_inventory_digest: fixture.inventory.digest,
    lease_base_inventory_digest: lease.value.lease_base_inventory.inventory_digest,
    subject_inventory_digest: subjectInventory.digest,
    control_manifest_digest: fixture.control.digest,
    verification_digest: `sha256:${"a".repeat(64)}`,
    accepted_seals: lease.value.accepted_seals,
    accepted_seal_set_digest: lease.value.accepted_seal_set_digest,
    barrier_seals: lease.value.barrier_seals,
    barrier_seal_set_digest: lease.value.barrier_seal_set_digest,
    ordinary_gate_proof: ordinaryGateProof,
    integration_gate_results: ["deterministic-products", "inventory-delta"]
      .map((gate) => ({ gate, ...gateReference(gate) })),
    result: "pass",
    failures: [],
  };
}

function gateReference(gate) {
  return {
    path: `test-artifacts/gates/${gate}.json`, artifact_id: `gate-${gate}`,
    digest: `sha256:${Buffer.from(gate).toString("hex").padEnd(64, "0").slice(0, 64)}`,
  };
}

function readJson(target) {
  return JSON.parse(fs.readFileSync(target, "utf8"));
}

function writeJson(target, value) {
  fs.writeFileSync(target, `${JSON.stringify(value, null, 2)}\n`);
}
