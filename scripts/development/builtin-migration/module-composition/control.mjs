import { compareCodePoint } from "../constants.mjs";
import { pathAllowed } from "../path-scope.mjs";
import { bindModuleCompositionProjection } from "./binding.mjs";
import { parseModuleCompositionTransition } from "./projection.mjs";

export function validateModuleCompositionControl(
  baselineValue, integrationProducts, bundles,
) {
  const products = [...integrationProducts.values()];
  const baseline = bindModuleCompositionProjection(products, baselineValue);
  const transitions = new Map();
  const changedKeys = new Map();
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
    const transition = parseModuleCompositionTransition(value, baseline);
    if (transition.transition_id !== bundleId) {
      throw new Error(`${bundleId}: module composition transition id must equal its bundle id`);
    }
    const changedProducts = transition.product_states.map((entry) => entry.product_id)
      .sort(compareCodePoint);
    if (JSON.stringify(changedProducts) !== JSON.stringify(productIds)) {
      throw new Error(`${bundleId}: composition transition must exactly cover its reviewed composition products`);
    }
    for (const change of transition.changes) {
      const key = `${change.product_id}\0${(change.after ?? change.before).module}`;
      const priorBundle = changedKeys.get(key);
      if (priorBundle) {
        throw new Error(`${bundleId}: composition child is also changed by ${priorBundle}`);
      }
      changedKeys.set(key, bundleId);
      for (const child of [change.before, change.after].filter(Boolean)) {
        if (!pathAllowed(bundle.authored_write_set, child.source_path)) {
          throw new Error(`${bundleId}: composition child ${child.source_path} is outside its authored scope`);
        }
      }
    }
    for (const state of transition.product_states.filter((entry) =>
      entry.before_state !== entry.after_state)) {
      const key = `${state.product_id}\0@state`;
      const priorBundle = changedKeys.get(key);
      if (priorBundle) throw new Error(`${bundleId}: composition product state is also changed by ${priorBundle}`);
      changedKeys.set(key, bundleId);
    }
    transitions.set(bundleId, transition);
  }
  return {
    baseline,
    transitions: new Map([...transitions].sort(([left], [right]) => compareCodePoint(left, right))),
  };
}
