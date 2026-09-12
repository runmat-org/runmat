import { compareCodePoint } from "./constants.mjs";
import { pathAllowed } from "./control-graph.mjs";
import { implementationOwners } from "./identity-authority.mjs";

export function validateAuthorityDependencyPolicy(bundles, identities, integrationProducts) {
  for (const [identity, control] of identities) {
    const bundle = bundles.get(control.bundle_id);
    if (!bundle) throw new Error(`${identity}: authority policy references an unknown bundle`);
    const prerequisites = new Set(bundle.prerequisites.map((entry) => entry.bundle_id));
    const catalogPackage = control.expected_authorities.catalog_package;
    if (catalogPackage !== null && !pathAllowed(bundle.authored_write_set, catalogPackage)) {
      throw new Error(`${identity}: catalog package is outside its bundle's authored scopes`);
    }
    for (const dependency of control.shared_dependencies) {
      if (dependency.ownership === "bundle") {
        if (dependency.owner_id !== bundle.id) {
          throw new Error(`${identity}: bundle-owned dependency must name ${bundle.id} as owner`);
        }
        requirePathInScopes(dependency, bundle.authored_write_set, identity, bundle.id);
      } else if (dependency.ownership === "prerequisite") {
        if (!prerequisites.has(dependency.owner_id)) {
          throw new Error(`${identity}: dependency owner ${dependency.owner_id} is not a direct prerequisite`);
        }
        const owner = bundles.get(dependency.owner_id);
        requirePathInScopes(dependency, owner.authored_write_set, identity, owner.id);
      } else {
        const product = integrationProducts.get(dependency.owner_id);
        if (!product) {
          throw new Error(`${identity}: integration dependency owner ${dependency.owner_id} is not reviewed`);
        }
        if (dependency.path !== product.path) {
          throw new Error(`${identity}: integration dependency path differs from ${product.product_id}`);
        }
        if (!bundle.integration_product_refs.includes(product.product_id)) {
          throw new Error(`${identity}: integration dependency ${product.product_id} is not referenced by its bundle`);
        }
      }
    }
    for (const ownerPath of implementationOwners(control.implementation)) {
      if (!pathAllowed(bundle.authored_write_set, ownerPath)
        && !control.shared_dependencies.some((dependency) => dependency.path === ownerPath)) {
        throw new Error(`${identity}: implementation owner ${ownerPath} is neither bundle-owned nor a reviewed dependency`);
      }
    }
  }
  for (const bundle of bundles.values()) {
    for (const removal of bundle.expected_removals) {
      if (!pathAllowed(bundle.authored_write_set, removal.path)) {
        throw new Error(`${bundle.id}: removal ${removal.path} is outside its authored scopes`);
      }
    }
  }
}

export function subjectAuthorityPathFailures(control, inventory, bundleIds) {
  const sourcePaths = new Set(inventory.source.files.map((entry) => entry.path));
  const rows = new Map(inventory.identities.map((entry) => [entry.identity, entry]));
  const selectedBundles = selectedBundleSet(control, bundleIds);
  const failures = [];
  for (const bundleId of selectedBundles) {
    const bundle = control.bundles.get(bundleId);
    for (const removal of bundle.expected_removals) {
      if (sourcePaths.has(removal.path)) failures.push(`${bundle.id}: expected removal remains at ${removal.path}`);
    }
    for (const product of bundle.integration_outputs) {
      if (!sourcePaths.has(product.path)) failures.push(`${bundle.id}: integration product is absent at ${product.path}`);
    }
  }
  for (const [identity, expected] of control.identities) {
    if (!selectedBundles.has(expected.bundle_id)) continue;
    const row = rows.get(identity);
    if (!row) continue;
    for (const ownerPath of implementationOwners(expected.implementation)) {
      requireExistingPath(ownerPath, sourcePaths, failures, identity, "implementation owner");
    }
    requireExistingPath(
      expected.expected_authorities.catalog_package,
      sourcePaths,
      failures,
      identity,
      "catalog package",
    );
    for (const dependency of expected.shared_dependencies) {
      requireExistingPath(dependency.path, sourcePaths, failures, identity, "shared dependency");
    }
    const expectedCatalog = expected.expected_authorities.catalog_package;
    const observedCatalog = [...new Set((row.semantic_authority.catalog_provenance ?? [])
      .map((entry) => entry.provenance?.source_file)
      .filter(Boolean))].sort(compareCodePoint);
    const requiredCatalog = expectedCatalog === null ? [] : [expectedCatalog];
    if (expected.expected_authorities.catalog_entry_count > 0
      && JSON.stringify(observedCatalog) !== JSON.stringify(requiredCatalog)) {
      failures.push(`${identity}: catalog provenance does not resolve to its reviewed package`);
    }
    const observedBindings = (row.semantic_authority.implementation_provenance ?? [])
      .map(observedCallableBindingKey)
      .sort(compareCodePoint);
    const expectedBindings = expected.implementation.callable.kind === "owned"
      ? expected.implementation.callable.bindings.map((entry) =>
        reviewedCallableBindingKey(expected.implementation.callable.owner_path, entry))
        .sort(compareCodePoint)
      : [];
    if (JSON.stringify(observedBindings) !== JSON.stringify(expectedBindings)) {
      failures.push(`${identity}: implementation provenance does not resolve to its reviewed bindings`);
    }
    const observedConstants = (row.semantic_authority.runtime_constants ?? [])
      .map((entry) => `${entry.source_file}\0${entry.name}\0${entry.builtin_path}`)
      .sort(compareCodePoint);
    const expectedConstants = expected.implementation.constant.kind === "owned"
      ? expected.implementation.constant.bindings.map((entry) =>
        `${expected.implementation.constant.owner_path}\0${entry.constant}\0${entry.builtin_path}`)
        .sort(compareCodePoint)
      : [];
    if (JSON.stringify(observedConstants) !== JSON.stringify(expectedConstants)) {
      failures.push(`${identity}: constant provenance does not resolve to its reviewed bindings`);
    }
  }
  return failures.sort(compareCodePoint);
}

function selectedBundleSet(control, bundleIds) {
  const selected = bundleIds === undefined
    ? [...control.bundles.keys()]
    : [...bundleIds];
  const result = new Set();
  for (const bundleId of selected) {
    if (!control.bundles.has(bundleId)) throw new Error(`authority validation references unknown bundle ${bundleId}`);
    result.add(bundleId);
  }
  return result;
}

function requirePathInScopes(dependency, scopes, identity, owner) {
  if (!pathAllowed(scopes, dependency.path)) {
    throw new Error(`${identity}: dependency ${dependency.path} is outside owner ${owner}'s authored scopes`);
  }
}

function requireExistingPath(value, sourcePaths, failures, identity, label) {
  if (value !== null && !sourcePaths.has(value)) failures.push(`${identity}: ${label} is absent at ${value}`);
}

function observedCallableBindingKey(entry) {
  const kind = entry.authority === "canonical_binding" ? "canonical_binding" : "legacy_function";
  return `${kind}\0${entry.source_file}\0${entry.function}\0${entry.binding_variant ?? ""}\0${entry.builtin_path}`;
}

function reviewedCallableBindingKey(ownerPath, entry) {
  return `${entry.kind}\0${ownerPath}\0${entry.function}\0${entry.variant ?? ""}\0${entry.builtin_path}`;
}
