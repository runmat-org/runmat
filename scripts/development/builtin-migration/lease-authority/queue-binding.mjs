import { revalidateObservedArtifacts } from "../authority-loading/index.mjs";
import { assertValidatedControl } from "../control.mjs";
import { assertActiveLease } from "../lease.mjs";
import { acceptedSealSet, barrierSealSet } from "../queue.mjs";
import { assertLoadedQueueAuthority } from "../queue-authority/index.mjs";
import {
  assertLoadedLeaseAuthority, revalidateLeaseAuthority,
} from "./loader.mjs";

const BOUND_PILOT_LEASES = new WeakMap();
const BINDINGS_BY_LEASE = new WeakMap();

export function bindPilotLeaseAuthority({
  leaseAuthority: leaseValue, queueAuthority: queueValue, session, control: controlValue,
}) {
  const control = assertValidatedControl(controlValue);
  const leaseAuthority = assertLoadedLeaseAuthority(leaseValue, session, control);
  const queueAuthority = assertLoadedQueueAuthority(queueValue, { session, control });
  const lease = leaseAuthority.lease.value;
  const state = queueAuthority.state;
  const checkpoint = queueAuthority.checkpoint.value;
  if (lease.queue_phase !== "pilot" || state.value.phase !== "pilot"
    || checkpoint.phase !== "pilot") {
    throw new Error("loaded pilot lease and queue authority must all be in pilot phase");
  }
  if (!control.pilotPolicy.waveByBundle.has(lease.bundle_id)) {
    throw new Error(`${lease.bundle_id}: loaded lease bundle is not in the reviewed pilot`);
  }
  if (state.sealedBundleIds.includes(lease.bundle_id)) {
    throw new Error(`${lease.bundle_id}: loaded lease target is already sealed`);
  }
  if (lease.queue_checkpoint_digest !== queueAuthority.checkpoint.digest) {
    throw new Error("loaded lease differs from the exact queue checkpoint");
  }
  const accepted = acceptedSealSet(state, control);
  const barriers = barrierSealSet(state, control, lease.bundle_id);
  if (lease.accepted_seal_set_digest !== accepted.value.digest
    || JSON.stringify(lease.accepted_seals) !== JSON.stringify(accepted.value.seals)) {
    throw new Error("loaded lease accepted seals differ from the exact queue authority");
  }
  if (lease.barrier_seal_set_digest !== barriers.value.digest
    || JSON.stringify(lease.barrier_seals) !== JSON.stringify(barriers.value.seals)) {
    throw new Error("loaded lease barriers differ from the exact queue authority");
  }
  if (lease.base_revision !== checkpoint.source_revision
    || lease.lease_base_inventory.source_digest !== checkpoint.source_digest
    || lease.lease_base_inventory.inventory_digest !== checkpoint.inventory_digest) {
    throw new Error("loaded lease base differs from the exact queue authority");
  }
  revalidateObservedArtifacts(session);
  const existing = BINDINGS_BY_LEASE.get(leaseAuthority)?.get(queueAuthority);
  if (existing) {
    return existing;
  }
  const result = Object.freeze({ leaseAuthority, queueAuthority, control });
  BOUND_PILOT_LEASES.set(result, { session, control });
  let bindings = BINDINGS_BY_LEASE.get(leaseAuthority);
  if (!bindings) {
    bindings = new WeakMap();
    BINDINGS_BY_LEASE.set(leaseAuthority, bindings);
  }
  bindings.set(queueAuthority, result);
  return result;
}

export function assertBoundPilotLeaseAuthority(value, { session, control: controlValue }) {
  const control = assertValidatedControl(controlValue);
  const record = BOUND_PILOT_LEASES.get(value);
  if (!record || record.session !== session || record.control !== control) {
    throw new Error("operation requires the exact bound pilot lease authority");
  }
  assertLoadedLeaseAuthority(value.leaseAuthority, session, control);
  assertLoadedQueueAuthority(value.queueAuthority, { session, control });
  return value;
}

export function assertActivePilotLeaseAuthority(value, context) {
  const bound = assertBoundPilotLeaseAuthority(value, context);
  revalidateLeaseAuthority(bound.leaseAuthority, context.session, context.control);
  assertActiveLease(bound.leaseAuthority.lease, context.control);
  return bound;
}
