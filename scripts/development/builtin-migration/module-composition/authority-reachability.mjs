import path from "node:path";

export function validateAuthorityReachability(moduleComposition, identities) {
  if (!moduleComposition?.effective) {
    throw new Error("authority reachability requires the complete effective module composition");
  }
  for (const [identity, control] of identities) {
    for (const authorityPath of authorityPaths(control)) {
      validateAuthorityPath(identity, authorityPath, moduleComposition.effective.products);
    }
  }
}

function validateAuthorityPath(identity, authorityPath, products) {
  const product = nearestParentProduct(authorityPath, products);
  if (!product) {
    throw new Error(`${identity}: authority ${authorityPath} has no module-composition parent`);
  }
  const childPath = immediateChildPath(product.path, authorityPath);
  if (!product.children.some((child) => child.source_path === childPath)) {
    throw new Error(
      `${identity}: authority ${authorityPath} is unreachable because ${product.product_id} does not declare ${childPath}`,
    );
  }
}

function nearestParentProduct(authorityPath, products) {
  return products
    .filter((product) => isWithin(path.posix.dirname(product.path), authorityPath))
    .sort((left, right) => right.path.length - left.path.length)[0] ?? null;
}

function immediateChildPath(parentPath, authorityPath) {
  const parent = path.posix.dirname(parentPath);
  const relative = authorityPath.slice(parent.length + 1);
  const [first, ...rest] = relative.split("/");
  return rest.length === 0 ? authorityPath : `${parent}/${first}/mod.rs`;
}

function isWithin(parent, candidate) {
  return candidate !== parent && candidate.startsWith(`${parent}/`);
}

function authorityPaths(control) {
  const expected = control.expected_authorities;
  return [...new Set([
    expected.catalog_package,
    expected.catalog_alias_package,
    expected.catalog_constant_package,
    control.implementation.callable.kind === "owned"
      ? control.implementation.callable.owner_path : null,
    control.implementation.constant.kind === "owned"
      ? control.implementation.constant.owner_path : null,
  ].filter(Boolean))];
}
