import { inspectMaterializationParentStates } from "./repository-state.mjs";

export function inspectCanonicalMaterializationTargets(
  repository, products, expectedDigests,
) {
  const present = new Set(products.filter((product) => product.state === "present")
    .map((product) => product.product_id));
  if (!(expectedDigests instanceof Map) || expectedDigests.size !== present.size
    || [...expectedDigests.keys()].some((productId) => !present.has(productId))) {
    throw new Error("canonical composition digest set differs from expected present products");
  }
  return inspectMaterializationParentStates(repository, products).map((observed) => {
    const expectedState = present.has(observed.product_id) ? "present" : "absent";
    if (observed.state !== expectedState) {
      throw new Error(`${observed.product_id}: parent presence differs from the authorized prior projection`);
    }
    if (expectedState === "present") {
      const expected = expectedDigests.get(observed.product_id);
      if (!/^sha256:[a-f0-9]{64}$/.test(expected ?? "")
        || observed.content_digest !== expected) {
        throw new Error(`${observed.product_id}: parent bytes differ from the canonical prior projection`);
      }
    }
    return observed;
  });
}
