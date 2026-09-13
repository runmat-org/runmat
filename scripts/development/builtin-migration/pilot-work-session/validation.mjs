import { assertValidatedControl } from "../control.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { bindPilotLeaseAuthority } from "../lease-authority.mjs";
import { assertLoadedQueueAuthority } from "../queue-authority/index.mjs";
import { assertValidatedQueueSeal } from "../queue-seal.mjs";
import { pilotWorkSessionId } from "./paths.mjs";

export function validateStartBindings({
  value, session, control, queueAuthority, leaseAuthority,
}) {
  const validatedControl = assertValidatedControl(control);
  const queue = assertLoadedQueueAuthority(queueAuthority, { session, control: validatedControl });
  const bound = bindPilotLeaseAuthority({
    leaseAuthority, queueAuthority: queue, session, control: validatedControl,
  });
  const lease = bound.leaseAuthority;
  const leaseValue = lease.lease.value;
  if (value.control_manifest_digest !== validatedControl.digest
    || value.pilot_policy_digest !== validatedControl.pilotPolicyDigest
    || value.pilot_id !== validatedControl.pilotPolicy.pilotId) {
    throw new Error("pilot work-session start belongs to another control or pilot policy");
  }
  assertQueueBinding(value.initial_queue, queue, "pilot work-session start initial queue");
  assertArtifactBinding(value.initial_lease, lease, "pilot work-session start initial lease");
  if (value.bundle_id !== leaseValue.bundle_id
    || value.session_id !== pilotWorkSessionId(value.pilot_id, value.bundle_id)) {
    throw new Error("pilot work-session bundle or session identity differs from its authority");
  }
  assertSameSource(value.source, queue.checkpoint.value, "pilot work-session start");
  assertTimeWithinLease(value.started_at, leaseValue, "pilot work-session start");
  return { queue, lease, bound };
}

export function validateCompletionBindings({
  value, session, control, start, finalLeaseAuthority,
  preIntegrationQueue, successorQueue,
}) {
  const validatedControl = assertValidatedControl(control);
  const preIntegration = assertLoadedQueueAuthority(
    preIntegrationQueue, { session, control: validatedControl },
  );
  const finalBound = bindPilotLeaseAuthority({
    leaseAuthority: finalLeaseAuthority,
    queueAuthority: preIntegration,
    session,
    control: validatedControl,
  });
  const finalLease = finalBound.leaseAuthority;
  const finalLeaseValue = finalLease.lease.value;
  const queue = assertLoadedQueueAuthority(
    successorQueue, { session, control: validatedControl },
  );
  if (value.control_manifest_digest !== validatedControl.digest
    || value.pilot_policy_digest !== validatedControl.pilotPolicyDigest
    || value.pilot_id !== validatedControl.pilotPolicy.pilotId) {
    throw new Error("pilot work-session completion belongs to another control or pilot policy");
  }
  assertArtifactBinding(value.start, start, "pilot work-session completion start");
  assertArtifactBinding(
    value.final_lease, finalLease, "pilot work-session completion final lease",
  );
  assertQueueBinding(
    value.pre_integration_queue, preIntegration,
    "pilot work-session completion pre-integration queue",
  );
  assertQueueBinding(value.successor_queue, queue, "pilot work-session completion");
  if (value.pilot_id !== start.pilotId || value.bundle_id !== start.bundleId
    || value.session_id !== start.sessionId
    || value.session_id !== pilotWorkSessionId(value.pilot_id, value.bundle_id)) {
    throw new Error("pilot work-session completion identity differs from its start");
  }
  if (finalLeaseValue.bundle_id !== start.bundleId) {
    throw new Error("pilot work-session final lease belongs to another bundle");
  }
  if (finalLeaseValue.owner !== start.initialLeaseAuthority.lease.value.owner) {
    throw new Error("pilot work-session final lease belongs to another reviewed owner");
  }
  if (!queueIsDescendantOf(preIntegration, start.initialQueue)) {
    throw new Error("pilot work-session final pre-integration queue does not descend from its start queue");
  }
  if (preIntegration.state.sealedBundleIds.includes(start.bundleId)) {
    throw new Error("pilot work-session bundle was already sealed before final integration");
  }
  const predecessor = queue.state.predecessor;
  if (queue.state.value.phase !== "pilot" || queue.checkpoint.value.phase !== "pilot"
    || predecessor === null
    || predecessor.state_path !== preIntegration.observations.state.path
    || predecessor.state_digest !== preIntegration.state.stateDigest
    || predecessor.checkpoint_path !== preIntegration.observations.checkpoint.path
    || predecessor.checkpoint_digest !== preIntegration.checkpoint.digest) {
    throw new Error("pilot work-session completion requires the direct successor of its final pre-integration queue");
  }
  const sealed = directSuccessorSeal(preIntegration, queue);
  assertValidatedQueueSeal(sealed.seal, validatedControl);
  if (JSON.stringify(value.seal) !== JSON.stringify(sealed.reference)
    || sealed.reference.bundle_id !== start.bundleId
    || sealed.lease_id !== finalLeaseValue.lease_id
    || sealed.lease_digest !== finalLeaseValue.digest) {
    throw new Error("pilot work-session completion seal differs from its exact lease or bundle");
  }
  assertSealLeaseProjection(sealed.seal.value, finalLeaseValue);
  const head = queue.checkpoint.value.head_event;
  if (head?.kind !== "seal" || JSON.stringify(head.seal) !== JSON.stringify(sealed.reference)) {
    throw new Error("pilot work-session completion seal is not the exact queue head");
  }
  assertSameSource(value.source, queue.checkpoint.value, "pilot work-session completion");
  if (sealed.integrated_revision !== value.source.revision
    || sealed.source_digest !== value.source.source_digest
    || sealed.subject_inventory_digest !== value.source.inventory_digest) {
    throw new Error("pilot work-session completion source differs from its exact seal");
  }
  if (!sealed.ordinary_gate_proof) {
    throw new Error("pilot work-session completion requires complete ordinary-gate proof");
  }
  if (sealed.seal.integrationGateResults.length !== 2
    || JSON.stringify(sealed.seal.integrationGateResults.map((entry) => entry.gate))
      !== JSON.stringify(["deterministic-products", "inventory-delta"])) {
    throw new Error("pilot work-session completion requires the exact integration gate-result identities");
  }
  if (Date.parse(value.ended_at) < Date.parse(start.startedAt)) {
    throw new Error("pilot work-session end precedes its machine-observed start");
  }
  assertTimeWithinLease(
    value.ended_at, finalLeaseValue, "pilot work-session completion",
  );
  return { queue, preIntegration, finalLease, sealed };
}

function assertSealLeaseProjection(seal, lease) {
  const expectedIntegrationOutputs = lease.forbidden_integration_outputs;
  const projection = [
    [seal.bundle_id, lease.bundle_id, "bundle"],
    [seal.control_manifest_digest, lease.control_manifest_digest, "control manifest"],
    [seal.lease_id, lease.lease_id, "lease id"],
    [seal.lease_digest, lease.digest, "lease digest"],
    [seal.queue_phase, lease.queue_phase, "queue phase"],
    [seal.phases.lease_base_revision, lease.base_revision, "lease base revision"],
    [
      seal.lease_base_inventory_digest,
      lease.lease_base_inventory.inventory_digest,
      "lease base inventory digest",
    ],
    [
      seal.phases.authored_write_set_digest,
      evidenceDigest(lease.authored_write_set),
      "authored write-set digest",
    ],
    [
      seal.phases.integration_outputs_digest,
      evidenceDigest(expectedIntegrationOutputs),
      "integration-output digest",
    ],
    [JSON.stringify(seal.accepted_seals), JSON.stringify(lease.accepted_seals), "accepted seals"],
    [seal.accepted_seal_set_digest, lease.accepted_seal_set_digest, "accepted seal-set digest"],
    [JSON.stringify(seal.barrier_seals), JSON.stringify(lease.barrier_seals), "barrier seals"],
    [seal.barrier_seal_set_digest, lease.barrier_seal_set_digest, "barrier seal-set digest"],
    [
      JSON.stringify(seal.phases.reviewed_authored_write_set),
      JSON.stringify(lease.authored_write_set),
      "reviewed authored write set",
    ],
    [
      JSON.stringify(seal.phases.reviewed_integration_outputs),
      JSON.stringify(expectedIntegrationOutputs),
      "reviewed integration outputs",
    ],
  ];
  const mismatch = projection.find(([observed, expected]) => observed !== expected);
  if (mismatch) {
    throw new Error(
      `pilot work-session completion seal ${mismatch[2]} differs from its exact final lease`,
    );
  }
}

export function directSuccessorSeal(preIntegrationQueue, successorQueue) {
  const prior = new Set(
    preIntegrationQueue.state.acceptedSeals.map((entry) => entry.bundle_id),
  );
  const appended = successorQueue.state.sealedBundles.filter(
    (entry) => !prior.has(entry.reference.bundle_id),
  );
  if (appended.length !== 1) {
    throw new Error("pilot work-session completion requires exactly one appended queue seal");
  }
  return appended[0];
}

function queueIsDescendantOf(descendant, ancestor) {
  const expected = {
    statePath: ancestor.observations.state.path,
    stateDigest: ancestor.state.stateDigest,
    checkpointPath: ancestor.observations.checkpoint.path,
    checkpointDigest: ancestor.checkpoint.digest,
  };
  let state = descendant.state;
  let binding = {
    statePath: descendant.observations.state.path,
    stateDigest: descendant.state.stateDigest,
    checkpointPath: descendant.observations.checkpoint.path,
    checkpointDigest: descendant.checkpoint.digest,
  };
  for (;;) {
    if (JSON.stringify(binding) === JSON.stringify(expected)) return true;
    if (state.predecessor === null) return false;
    binding = {
      statePath: state.predecessor.state_path,
      stateDigest: state.predecessor.state_digest,
      checkpointPath: state.predecessor.checkpoint_path,
      checkpointDigest: state.predecessor.checkpoint_digest,
    };
    state = state.predecessorState;
  }
}

export function assertQueueBinding(binding, queue, label) {
  for (const role of ["state", "checkpoint"]) {
    const observation = queue.observations[role];
    if (binding[role].path !== observation.path
      || binding[role].semantic_digest !== observation.semanticDigest
      || binding[role].content_digest !== observation.contentDigest) {
      throw new Error(`${label} ${role} observation mismatch`);
    }
  }
}

export function assertArtifactBinding(binding, authority, label) {
  if (binding.path !== authority.reference.path
    || binding.semantic_digest !== authority.reference.digest
    || binding.content_digest !== authority.artifact.contentDigest) {
    throw new Error(`${label} observation mismatch`);
  }
}

function assertSameSource(binding, checkpoint, label) {
  if (binding.revision !== checkpoint.source_revision
    || binding.source_digest !== checkpoint.source_digest
    || binding.inventory_digest !== checkpoint.inventory_digest) {
    throw new Error(`${label} source or inventory identity mismatch`);
  }
}

function assertTimeWithinLease(timestamp, lease, label) {
  const observed = Date.parse(timestamp);
  if (observed < Date.parse(lease.issued_at) || observed >= Date.parse(lease.expires_at)) {
    throw new Error(`${label} timestamp is outside its active lease interval`);
  }
}
