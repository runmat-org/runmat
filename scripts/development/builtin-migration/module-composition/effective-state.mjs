import { assertValidatedControl } from "../control.mjs";
import { compareCodePoint } from "../constants.mjs";
import { deepImmutable } from "../immutable.mjs";
import { assertActiveLease } from "../lease.mjs";
import { acceptedSealSet, assertValidatedQueueState } from "../queue.mjs";
import { assertValidatedQueueCheckpoint } from "../queue-checkpoint.mjs";
import { applyModuleCompositionTransitions } from "./projection.mjs";

export function deriveEffectiveModuleComposition({
  control, queueState, queueCheckpoint, lease, clock = Date.now,
}) {
  return deriveModuleCompositionMaterializationState({
    control, queueState, queueCheckpoint, lease, clock,
  })?.effective ?? null;
}

export function deriveModuleCompositionMaterializationState({
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
  const acceptedTransitions = state.acceptedSeals.flatMap((reference) => {
    if (!control.moduleComposition.transitions.has(reference.bundle_id)) {
      throw new Error(`${reference.bundle_id}: accepted bundle has no reviewed composition disposition`);
    }
    const transition = control.moduleComposition.transitions.get(reference.bundle_id);
    return transition === null ? [] : [transition];
  });
  const prior = applyModuleCompositionTransitions(baseline, acceptedTransitions);
  const activeTransition = transitionForActiveBundle(control, active.bundle);
  const effective = activeTransition === null
    ? prior
    : applyModuleCompositionTransitions(prior, [activeTransition]);
  const priorPresentProductIds = new Set([...control.integrationProducts.values()]
    .filter((product) => product.verification.kind === "rust_module_composition"
      && product.baseline_digest !== null)
    .map((product) => product.product_id));
  for (const transition of acceptedTransitions) {
    for (const change of transition.changes) priorPresentProductIds.add(change.product_id);
  }
  return deepImmutable({
    prior,
    effective,
    priorPresentProductIds: [...priorPresentProductIds].sort(compareCodePoint),
  });
}

function transitionForActiveBundle(control, bundle) {
  if (!control.moduleComposition.transitions.has(bundle.id)) {
    throw new Error(`${bundle.id}: active bundle has no reviewed composition disposition`);
  }
  const transition = control.moduleComposition.transitions.get(bundle.id);
  const ownsCompositionProduct = bundle.integration_outputs.some((output) => {
    const product = control.integrationProducts.get(output.product_id);
    if (!product) {
      throw new Error(`${bundle.id}: active bundle references unknown integration product ${output.product_id}`);
    }
    return product.verification.kind === "rust_module_composition";
  });
  if (!ownsCompositionProduct) {
    if (transition !== null) {
      throw new Error(`${bundle.id}: active bundle without composition products has a transition`);
    }
    return null;
  }
  if (transition === null) {
    throw new Error(`${bundle.id}: active bundle has no reviewed composition transition`);
  }
  return transition;
}
