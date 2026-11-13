import { compareCodePoint } from "../constants.mjs";
import { materializationChildKey } from "./repository-state.mjs";

export function deriveAuthorizedPendingChildKeys(products, priorById) {
  const activatedPaths = new Set(products
    .filter((product) => product.state === "present"
      && priorById.get(product.product_id)?.state === "absent")
    .map((product) => product.path));
  const keys = new Set();
  for (const product of products) {
    const priorChildren = new Set((priorById.get(product.product_id)?.children ?? [])
      .map((child) => child.source_path));
    for (const child of product.children) {
      if ((activatedPaths.has(child.source_path) || child.companion_section != null)
        && !priorChildren.has(child.source_path)) {
        keys.add(materializationChildKey(product.product_id, child.source_path));
      }
    }
  }
  return keys;
}

export function selectMaterializationProducts(control, bundle, requestedProductIds = null) {
  if (requestedProductIds === null) {
    const implicit = new Set([...control.integrationProducts.values()]
      .filter((product) => product.verification.kind === "rust_module_composition"
        && product.lifecycle.kind === "reviewed-baseline-only")
      .map((product) => product.product_id));
    for (const output of bundle.integration_outputs) {
      const product = control.integrationProducts.get(output.product_id);
      if (!product) throw new Error(`${bundle.id}: active bundle references unknown integration product ${output.product_id}`);
      if (product.verification.kind === "rust_module_composition") {
        implicit.add(product.product_id);
      }
    }
    const compositionProductIds = [...implicit].sort(compareCodePoint);
    return {
      selectedProductIds: compositionProductIds,
      compositionProductIds,
    };
  }
  const integrationProductIds = [];
  for (const output of bundle.integration_outputs) {
    const product = control.integrationProducts.get(output.product_id);
    if (!product) throw new Error(`${bundle.id}: active bundle references unknown integration product ${output.product_id}`);
    integrationProductIds.push(product.product_id);
  }
  const expected = [...integrationProductIds].sort(compareCodePoint);
  const requested = [...requestedProductIds];
  if (new Set(requested).size !== requested.length) {
    throw new Error("materialization product selection contains duplicates");
  }
  if (JSON.stringify(requested) !== JSON.stringify([...requested].sort(compareCodePoint))) {
    throw new Error("materialization product selection must use canonical product-id order");
  }
  if (JSON.stringify(requested) !== JSON.stringify(expected)) {
    throw new Error(`${bundle.id}: materialization product selection must exactly match the active lease integration products`);
  }
  const compositionProductIds = requested.filter((id) =>
    control.integrationProducts.get(id).verification.kind === "rust_module_composition");
  return {
    selectedProductIds: requested,
    compositionProductIds,
  };
}
