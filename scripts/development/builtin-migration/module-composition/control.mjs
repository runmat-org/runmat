import { compareCodePoint } from "../constants.mjs";
import { pathAllowed } from "../path-scope.mjs";
import { bindModuleCompositionProjection } from "./binding.mjs";
import {
  applyModuleCompositionTransitions,
  parseModuleCompositionTransitionShape,
} from "./projection.mjs";

export function validateModuleCompositionControl(
  baselineValue, integrationProducts, bundles, { deferEffectiveSequence = false } = {},
) {
  const products = [...integrationProducts.values()];
  const baseline = bindModuleCompositionProjection(products, baselineValue);
  const transitions = new Map();
  const childChanges = new Map();
  const productStateOwners = new Map();
  for (const [bundleId, bundle] of bundles) {
    const productIds = bundle.integration_product_refs.filter((productId) => {
      const product = integrationProducts.get(productId);
      if (product?.verification.kind !== "rust_module_composition") return false;
      if (product.lifecycle.kind === "reviewed-baseline-only") {
        throw new Error(`${bundleId}: reviewed-baseline-only composition product ${productId} cannot be bundle referenced`);
      }
      return true;
    });
    const value = bundle.module_composition_transition;
    if (productIds.length === 0) {
      if (value !== null) {
        throw new Error(`${bundleId}: bundle without composition products cannot declare a transition`);
      }
      transitions.set(bundleId, null);
      continue;
    }
    if (baseline === null || value === null) {
      throw new Error(`${bundleId}: reviewed composition products require a bundle transition`);
    }
    const transition = parseModuleCompositionTransitionShape(value, baseline);
    if (transition.transition_id !== bundleId) {
      throw new Error(`${bundleId}: module composition transition id must equal its bundle id`);
    }
    const changedProducts = transition.product_states.map((entry) => entry.product_id)
      .sort(compareCodePoint);
    if (JSON.stringify(changedProducts) !== JSON.stringify(productIds)) {
      throw new Error(`${bundleId}: composition transition must exactly cover its reviewed composition products`);
    }
    const referencedProductPaths = new Set(
      productIds.map((productId) => integrationProducts.get(productId).path),
    );
    for (const change of transition.changes) {
      const key = `${change.product_id}\0${(change.after ?? change.before).module}`;
      const currentChange = {
        bundle_id: bundleId,
        semantic: !isParentMetadataOnlyReplacement(change),
      };
      const priorChanges = childChanges.get(key) ?? [];
      for (const priorChange of priorChanges) {
        const prerequisiteOrdered = dependsOn(bundles, bundleId, priorChange.bundle_id)
          || dependsOn(bundles, priorChange.bundle_id, bundleId);
        if (!prerequisiteOrdered) {
          throw new Error(
            `${bundleId}: composition child is also changed by ${priorChange.bundle_id} and is not prerequisite-ordered`,
          );
        }
        if (currentChange.semantic && priorChange.semantic) {
          throw new Error(
            `${bundleId}: composition child is also semantically changed by ${priorChange.bundle_id}`,
          );
        }
      }
      priorChanges.push(currentChange);
      childChanges.set(key, priorChanges);
      const parentMetadataOnly = isParentMetadataOnlyReplacement(change);
      for (const child of [change.before, change.after].filter(Boolean)) {
        if (!parentMetadataOnly
            && !pathAllowed(bundle.authored_write_set, child.source_path)
            && !referencedProductPaths.has(child.source_path)) {
          throw new Error(`${bundleId}: composition child ${child.source_path} is outside its authored scope`);
        }
      }
    }
    for (const state of transition.product_states.filter((entry) =>
      entry.before_state !== entry.after_state)) {
      const key = `${state.product_id}\0@state`;
      const priorBundle = productStateOwners.get(key);
      if (priorBundle) throw new Error(`${bundleId}: composition product state is also changed by ${priorBundle}`);
      productStateOwners.set(key, bundleId);
    }
    transitions.set(bundleId, transition);
  }
  const effective = deferEffectiveSequence
    ? null
    : validateEffectiveSequence(baseline, transitions, bundles);
  return {
    baseline,
    effective,
    transitions: new Map([...transitions].sort(([left], [right]) => compareCodePoint(left, right))),
  };
}

// Declaration and aggregation order live in the parent module's composition
// metadata. Reordering an otherwise byte-identical child does not authorize or
// require writing that child's source file; the transition's reviewed parent
// product reference is the relevant authority boundary.
function isParentMetadataOnlyReplacement(change) {
  if (change.operation !== "replace" || change.before === null || change.after === null
      || change.before.source_path !== change.after.source_path) return false;
  const withoutParentOrder = (child) => ({
    ...child,
    declaration_order: 0,
    aggregation_sources: child.aggregation_sources.map((source) => ({ ...source, order: 0 })),
  });
  return JSON.stringify(withoutParentOrder(change.before))
    === JSON.stringify(withoutParentOrder(change.after));
}

function validateEffectiveSequence(baseline, transitions, bundles) {
  const order = prerequisiteOrder(bundles);
  validateSharedProductOrdering(transitions, bundles);
  let effective = baseline;
  for (const bundleId of order) {
    const transition = transitions.get(bundleId);
    if (transition !== null) {
      effective = applyModuleCompositionTransitions(effective, [transition]);
    }
  }
  return effective;
}

function validateSharedProductOrdering(transitions, bundles) {
  const productBundles = new Map();
  for (const [bundleId, transition] of transitions) {
    if (transition === null) continue;
    for (const state of transition.product_states) {
      const owners = productBundles.get(state.product_id) ?? [];
      owners.push({
        bundle_id: bundleId,
        before_state: state.before_state,
        after_state: state.after_state,
      });
      productBundles.set(state.product_id, owners);
    }
  }
  for (const [productId, owners] of productBundles) {
    for (let leftIndex = 0; leftIndex < owners.length; leftIndex += 1) {
      for (let rightIndex = leftIndex + 1; rightIndex < owners.length; rightIndex += 1) {
        const left = owners[leftIndex];
        const right = owners[rightIndex];
        const stableSharedState = left.before_state === left.after_state
          && right.before_state === right.after_state
          && left.before_state === right.before_state;
        const leftDependsOnRight = dependsOn(bundles, left.bundle_id, right.bundle_id);
        const rightDependsOnLeft = dependsOn(bundles, right.bundle_id, left.bundle_id);
        if (!stableSharedState && !leftDependsOnRight && !rightDependsOnLeft) {
          throw new Error(
            `${productId}: composition transitions ${left.bundle_id} and ${right.bundle_id} must be prerequisite-ordered; neither bundle reaches the other through reviewed prerequisites`,
          );
        }
      }
    }
  }
}

function dependsOn(bundles, bundleId, expectedPrerequisite, visited = new Set()) {
  if (visited.has(bundleId)) return false;
  visited.add(bundleId);
  const bundle = bundles.get(bundleId);
  for (const prerequisite of bundle?.prerequisites ?? []) {
    if (prerequisite.bundle_id === expectedPrerequisite
      || dependsOn(bundles, prerequisite.bundle_id, expectedPrerequisite, visited)) {
      return true;
    }
  }
  return false;
}

function prerequisiteOrder(bundles) {
  const remaining = new Map([...bundles].map(([bundleId, bundle]) => [
    bundleId,
    new Set((bundle.prerequisites ?? []).map((entry) => entry.bundle_id)),
  ]));
  const result = [];
  while (remaining.size > 0) {
    const ready = [...remaining]
      .filter(([, prerequisites]) => [...prerequisites]
        .every((bundleId) => !remaining.has(bundleId)))
      .map(([bundleId]) => bundleId)
      .sort(compareCodePoint);
    if (ready.length === 0) {
      throw new Error("module composition prerequisites contain a cycle");
    }
    for (const bundleId of ready) {
      remaining.delete(bundleId);
      result.push(bundleId);
    }
  }
  return result;
}
