import { parseModuleCompositionProjection } from "./schema.mjs";

export function bindModuleCompositionProjection(products, value) {
  const definitions = products.filter((entry) => entry.verification.kind === "rust_module_composition");
  if (definitions.length === 0) {
    if (value !== null && value !== undefined) throw new Error("unused module composition projection is not admitted");
    return null;
  }
  if (value === null || value === undefined) throw new Error("reviewed module composition products require their exact projection");
  const projection = parseModuleCompositionProjection(value);
  const expected = definitions.map((entry) => ({
    product_id: entry.product_id,
    path: entry.path,
    crate_role: entry.verification.crate_role,
    module_path: entry.verification.module_path,
  }));
  const observed = projection.products.map((entry) => ({
    product_id: entry.product_id,
    path: entry.path,
    crate_role: entry.crate_role,
    module_path: entry.module_path,
  }));
  if (JSON.stringify(observed) !== JSON.stringify(expected)) {
    throw new Error("module composition projection does not exactly cover the reviewed composition products");
  }
  return projection;
}
