import { compareCodePoint } from "../constants.mjs";

export function validateSharedTargetPackageAuthority(topology, integrationProducts) {
  const productsByPath = new Map([...integrationProducts.values()]
    .filter((product) => product.verification.kind === "rust_module_composition")
    .map((product) => [product.path, product]));
  for (const [targetPackage, bundleIds] of sharedTargetPackages(topology)) {
    for (const expected of targetPackageProducts(targetPackage)) {
      const product = productsByPath.get(expected.path);
      if (!product) {
        throw new Error(
          `${targetPackage}: shared target package used by bundles ${bundleIds.join(", ")} requires integration product authority at ${expected.path}`,
        );
      }
      if (product.verification.crate_role !== expected.crate_role
        || product.verification.module_path !== expected.module_path) {
        throw new Error(
          `${product.product_id}: shared target-package authority does not match ${expected.crate_role} parent ${expected.module_path}`,
        );
      }
    }
  }
}

function sharedTargetPackages(topology) {
  const owners = new Map();
  for (const [bundleId, bundle] of topology.bundles) {
    for (const target of bundle.composition.target_packages) {
      const key = `${target.domain}/${target.family}`;
      const bundles = owners.get(key) ?? new Set();
      bundles.add(bundleId);
      owners.set(key, bundles);
    }
  }
  return [...owners]
    .filter(([, bundles]) => bundles.size > 1)
    .map(([key, bundles]) => [key, [...bundles].sort(compareCodePoint)])
    .sort(([left], [right]) => compareCodePoint(left, right));
}

function targetPackageProducts(targetPackage) {
  const moduleSuffix = targetPackage.replaceAll("/", "::");
  return [
    {
      crate_role: "catalog",
      path: `crates/runmat-builtins/src/catalog/entries/${targetPackage}/mod.rs`,
      module_path: `crate::catalog::entries::${moduleSuffix}`,
    },
    {
      crate_role: "runtime",
      path: `crates/runmat-runtime/src/builtins/${targetPackage}/mod.rs`,
      module_path: `crate::builtins::${moduleSuffix}`,
    },
  ];
}
