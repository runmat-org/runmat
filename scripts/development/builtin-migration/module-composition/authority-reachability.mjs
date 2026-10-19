import path from "node:path";
import { compareCodePoint } from "../constants.mjs";

export function validateAuthorityReachability(moduleComposition, identities) {
  if (!moduleComposition?.effective) {
    throw new Error("authority reachability requires the complete effective module composition");
  }
  validateSharedParentAuthority(moduleComposition.effective.products, identities);
  for (const [identity, control] of identities) {
    for (const authorityPath of authorityPaths(control)) {
      validateAuthorityPath(identity, authorityPath, moduleComposition.effective.products);
    }
  }
}

function validateSharedParentAuthority(products, identities) {
  const productsByPath = new Map(products.map((product) => [product.path, product]));
  const owners = new Map();
  for (const [, control] of identities) {
    for (const authorityPath of authorityPaths(control)) {
      const product = parentProductChain(authorityPath, products).at(-1);
      const parentPath = product?.children.find((child) => childOwns(child, authorityPath))
        ?.source_path;
      if (!parentPath) continue;
      const bundles = owners.get(parentPath) ?? new Set();
      bundles.add(control.bundle_id);
      owners.set(parentPath, bundles);
    }
  }
  for (const [parentPath, bundles] of owners) {
    if (bundles.size < 2 || productsByPath.has(parentPath)) continue;
    throw new Error(
      `${parentPath}: shared authority parent used by bundles ${[...bundles].sort(compareCodePoint).join(", ")} is absent from integration product authority`,
    );
  }
}

function validateAuthorityPath(identity, authorityPath, products) {
  const chain = parentProductChain(authorityPath, products);
  if (chain.length === 0) {
    throw new Error(`${identity}: authority ${authorityPath} has no module-composition parent`);
  }
  for (const [index, product] of chain.entries()) {
    if (product.state !== "present") {
      throw new Error(
        `${identity}: authority ${authorityPath} is unreachable because ${product.product_id} is absent`,
      );
    }
    const nextProduct = chain[index + 1] ?? null;
    if (nextProduct === null && product.path === authorityPath) continue;
    const child = nextProduct === null
      ? product.children.find((entry) => childOwns(entry, authorityPath))
      : product.children.find((entry) => entry.source_path === nextProduct.path);
    if (!child) {
      const expected = nextProduct?.path ?? authorityPath;
      throw new Error(
        `${identity}: authority ${authorityPath} is unreachable because ${product.product_id} does not declare a child owning ${expected}`,
      );
    }
  }
}

function parentProductChain(authorityPath, products) {
  return products
    .filter((product) => authorityPath === product.path
      || isWithin(productNamespace(product.path), authorityPath))
    .sort((left, right) => pathDepth(productNamespace(left.path))
      - pathDepth(productNamespace(right.path)));
}

function pathDepth(value) { return value.split("/").length; }

function childOwns(child, authorityPath) {
  if (child.source_kind === "directory") {
    const namespace = path.posix.dirname(child.source_path);
    return authorityPath === child.source_path || isWithin(namespace, authorityPath);
  }
  const namespace = child.source_path.slice(0, -".rs".length);
  return authorityPath === child.source_path || isWithin(namespace, authorityPath);
}

function productNamespace(productPath) {
  return productPath.endsWith("/mod.rs")
    ? path.posix.dirname(productPath)
    : productPath.slice(0, -".rs".length);
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
