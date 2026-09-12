import { moduleCompositionProductRegistry } from "./registry.mjs";

export function validateReviewedModuleCompositionAuthority(products, projection) {
  const definitions = moduleCompositionProductRegistry();
  const reviewed = [...products.values()]
    .filter((entry) => entry.verification.kind === "rust_module_composition");
  assertExactIds(reviewed, definitions, "reviewed module composition products");
  for (let index = 0; index < definitions.length; index += 1) {
    const expected = definitions[index];
    const observed = reviewed[index];
    assertEqual(observed.path, expected.path, expected.product_id, "path");
    assertEqual(
      observed.verification.crate_role, expected.crate_role,
      expected.product_id, "crate role",
    );
    assertEqual(
      observed.verification.module_path, expected.module_path,
      expected.product_id, "module path",
    );
  }

  return validateFixedModuleCompositionProjection(projection);
}

export function validateFixedModuleCompositionProjection(projection) {
  const definitions = moduleCompositionProductRegistry();
  if (projection === null) throw new Error("fixed module composition products require their reviewed projection");
  assertExactIds(projection.products, definitions, "module composition projection");
  for (let index = 0; index < definitions.length; index += 1) {
    const expected = definitions[index];
    const observed = projection.products[index];
    assertEqual(observed.path, expected.path, expected.product_id, "projection path");
    assertEqual(observed.crate_role, expected.crate_role, expected.product_id, "projection crate role");
    assertEqual(observed.module_path, expected.module_path, expected.product_id, "projection module path");
    if (!same(observed.aggregations, expected.aggregations)) {
      throw new Error(`${expected.product_id}: projection aggregation roles do not match the fixed registry`);
    }
    const expectedExports = observed.children.length ? expected.aggregation_exports : [];
    if (!same(observed.aggregation_exports, expectedExports)) {
      throw new Error(`${expected.product_id}: projection aggregation exports do not match the fixed registry`);
    }
  }
  return projection;
}

function assertExactIds(observed, expected, label) {
  const observedIds = observed.map((entry) => entry.product_id);
  const expectedIds = expected.map((entry) => entry.product_id);
  if (same(observedIds, expectedIds)) return;
  const observedSet = new Set(observedIds);
  const expectedSet = new Set(expectedIds);
  const missing = expectedIds.filter((id) => !observedSet.has(id));
  const unexpected = observedIds.filter((id) => !expectedSet.has(id));
  const details = [
    missing.length ? `missing ${missing.join(", ")}` : null,
    unexpected.length ? `unexpected ${unexpected.join(", ")}` : null,
  ].filter(Boolean).join("; ");
  throw new Error(`${label} must exactly cover the fixed registry${details ? `: ${details}` : ""}`);
}

function assertEqual(observed, expected, productId, field) {
  if (observed !== expected) {
    throw new Error(`${productId}: ${field} does not match the fixed module composition registry`);
  }
}

function same(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}
