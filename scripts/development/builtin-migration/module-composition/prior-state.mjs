import crypto from "node:crypto";

import { inspectMaterializationParentStates } from "./repository-state.mjs";

export function inspectCanonicalMaterializationTargets(
  repository, products, expectedPresentProductIds, expectedContents,
) {
  const present = new Set(expectedPresentProductIds);
  const productIds = new Set(products.map((product) => product.product_id));
  if (present.size !== expectedPresentProductIds.length
    || [...present].some((productId) => !productIds.has(productId))) {
    throw new Error("canonical composition presence does not exactly cover selected products");
  }
  if (!(expectedContents instanceof Map) || expectedContents.size !== present.size
    || [...expectedContents.keys()].some((productId) => !present.has(productId))) {
    throw new Error("canonical composition render set differs from expected present products");
  }
  return inspectMaterializationParentStates(repository, products).map((observed) => {
    const expectedState = present.has(observed.product_id) ? "present" : "absent";
    if (observed.state !== expectedState) {
      throw new Error(`${observed.product_id}: parent presence differs from the authorized prior projection`);
    }
    if (expectedState === "present") {
      const expected = expectedContents.get(observed.product_id);
      if (typeof expected !== "string"
        || observed.content_digest !== digestText(expected)) {
        throw new Error(`${observed.product_id}: parent bytes differ from the canonical prior projection`);
      }
    }
    return observed;
  });
}

function digestText(value) {
  return `sha256:${crypto.createHash("sha256").update(value).digest("hex")}`;
}
