import { assertValidatedControl } from "../control.mjs";
import { deepImmutable } from "../immutable.mjs";
import { assertActiveLease } from "../lease.mjs";
import { acceptedSealSet, assertValidatedQueueState } from "../queue.mjs";
import { assertValidatedQueueCheckpoint } from "../queue-checkpoint.mjs";
import { applyModuleCompositionTransitions } from "./projection.mjs";

export function deriveEffectiveModuleComposition({
  control, queueState, queueCheckpoint, lease, clock = Date.now,
}) {
  assertValidatedControl(control);
  const state = assertValidatedQueueState(queueState, control);
  const checkpoint = assertValidatedQueueCheckpoint(queueCheckpoint, control, state);
  const active = assertActiveLease(lease, control, clock);
  const accepted = acceptedSealSet(state, control);
  if (active.value.queue_checkpoint_digest !== checkpoint.digest
    || active.value.accepted_seal_set_digest !== accepted.value.digest
    || JSON.stringify(active.value.accepted_seals) !== JSON.stringify(accepted.value.seals)) {
    throw new Error("active lease differs from the validated current queue checkpoint");
  }
  if (state.sealedBundleIds.includes(active.bundle.id)) {
    throw new Error(`${active.bundle.id}: active composition bundle is already sealed`);
  }
  const baseline = control.moduleComposition.baseline;
  if (baseline === null) return null;
  const transitions = state.acceptedSeals.map((reference) =>
    control.moduleComposition.transitions.get(reference.bundle_id)).filter(Boolean);
  transitions.push(requiredTransition(control, active.bundle.id, "active"));
  return deepImmutable(applyModuleCompositionTransitions(baseline, transitions));
}

function requiredTransition(control, bundleId, role) {
  const transition = control.moduleComposition.transitions.get(bundleId);
  if (!transition) throw new Error(`${bundleId}: ${role} bundle has no reviewed composition transition`);
  return transition;
}
