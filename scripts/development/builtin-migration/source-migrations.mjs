import path from "node:path";

import { compareCodePoint } from "./constants.mjs";
import { implementationOwners } from "./identity-authority.mjs";
import {
  compositionReexportKey,
  parseCompositionReexport,
} from "./module-composition/schema.mjs";
import { pathAllowed } from "./path-scope.mjs";
import { parseUniqueReviewedEvidence } from "./reviewed-evidence.mjs";
import {
  array, digest, enumValue, exact, nonempty, repositoryPath, SAFE_IDENTITY, stableId,
  uniqueStrings,
} from "./schema.mjs";

export const SOURCE_MIGRATION_STRATEGIES = Object.freeze([
  "module-decomposition",
  "module-support-reparent",
]);

export function parseSourceMigrations(value, bundleId, identities, inventory, baselineEvidence) {
  const allowedIdentities = new Set(identities);
  const sourceFiles = new Map(inventory.source.files.map((entry) => [entry.path, entry.content_digest]));
  const foldedSourcePaths = new Set(inventory.source.files.map((entry) => entry.path.toLowerCase()));
  return parseSourceMigrationEvidence(value, `${bundleId} source migrations`).map((entry) => {
    const { strategy, source_path: sourcePath, source_baseline_digest: sourceDigest } = entry;
    if (sourceFiles.get(sourcePath) !== sourceDigest) {
      throw new Error(`${bundleId}: source migration ${sourcePath} differs from the exact baseline source`);
    }
    for (const destination of entry.destination_paths) {
      if (foldedSourcePaths.has(destination.toLowerCase())) {
        throw new Error(`${bundleId}: source migration destination exists at the baseline: ${destination}`);
      }
    }
    const affected = entry.identity_destinations.map(({ identity }) => identity);
    if (affected.some((identity) => !allowedIdentities.has(identity))) {
      throw new Error(`${bundleId}: source migration identities must belong to the bundle`);
    }
    if (strategy === "module-support-reparent" && affected.length !== 0) {
      throw new Error(`${bundleId}: support reparent cannot claim identity destinations`);
    }
    if (strategy === "module-decomposition" && affected.length === 0) {
      throw new Error(`${bundleId}: module decomposition requires identity destinations`);
    }
    const evidencedIdentities = [...new Set(baselineEvidence
      .filter((evidence) => evidence.path === sourcePath
        && evidence.source_snapshot === "present"
        && evidence.content_digest === sourceDigest)
      .flatMap((evidence) => evidence.affected_identities))].sort(compareCodePoint);
    if (JSON.stringify(affected) !== JSON.stringify(evidencedIdentities)) {
      throw new Error(`${bundleId}: source migration ${sourcePath} identities differ from complete baseline evidence`);
    }
    return entry;
  });
}

export function parseSourceMigrationEvidence(value, label = "source migrations") {
  const migrations = array(value, label, { empty: true }).map((entry) => {
    exact(entry, [
      "strategy", "source_path", "source_baseline_digest", "promoted_target",
      "destination_paths", "support_destinations", "identity_destinations", "reason", "review",
    ], `${label} entry`);
    const strategy = enumValue(entry.strategy, SOURCE_MIGRATION_STRATEGIES, `${label} strategy`);
    const sourcePath = repositoryPath(entry.source_path, `${label} source`);
    const sourceDigest = digest(entry.source_baseline_digest, `${label} baseline digest`);
    const promotedTarget = parsePromotedTarget(entry.promoted_target, `${label} promoted target`);
    const destinations = array(entry.destination_paths, `${label} destinations`)
      .map((destination) => repositoryPath(destination, `${label} destination`));
    requireCanonicalUnique(destinations, `${label} destinations`);
    requireCaseFoldUnique([sourcePath, ...destinations], `${label} endpoints`);
    const destinationSet = new Set(destinations);
    const supportDestinations = parseSupportDestinations(
      entry.support_destinations, destinationSet, label,
    );
    if (supportDestinations.length === 0) {
      throw new Error(`${label}: source migration requires a typed support destination`);
    }
    const identityDestinations = parseIdentityDestinations(
      entry.identity_destinations, destinationSet, label,
    );
    validateDestinationPartition(
      destinations, promotedTarget, supportDestinations, identityDestinations, label,
    );
    if (nonempty(entry.reason, `${label} reason`) !== entry.reason) {
      throw new Error(`${label} reason must not contain surrounding whitespace`);
    }
    parseUniqueReviewedEvidence(entry.review, `${label} review`);
    return {
      strategy,
      source_path: sourcePath,
      source_baseline_digest: sourceDigest,
      promoted_target: promotedTarget,
      destination_paths: destinations,
      support_destinations: supportDestinations,
      identity_destinations: identityDestinations,
      reason: entry.reason,
      review: entry.review,
    };
  });
  requireCanonicalUnique(migrations.map(migrationKey), label);
  return migrations;
}

export function validateSourceMigrationControl(
  bundles, identities, integrationProducts, moduleComposition,
) {
  const endpoints = new Map();
  const integrationPaths = new Set([...integrationProducts.values()]
    .map((entry) => entry.path.toLowerCase()));
  const effectivePaths = new Set((moduleComposition.effective?.products ?? [])
    .map((entry) => entry.path.toLowerCase()));
  const effectiveById = new Map((moduleComposition.effective?.products ?? [])
    .map((product) => [product.product_id, product]));
  const effectiveByPath = new Map((moduleComposition.effective?.products ?? [])
    .map((product) => [product.path, product]));
  for (const bundle of bundles.values()) {
    const removals = new Set(bundle.expected_removals.map((entry) => entry.path));
    for (const migration of bundle.source_migrations) {
      if (removals.has(migration.source_path)) {
        throw new Error(`${bundle.id}: source migration source duplicates expected removal ${migration.source_path}`);
      }
      for (const endpoint of [migration.source_path, ...migration.destination_paths]) {
        if (!pathAllowed(bundle.authored_write_set, endpoint)) {
          throw new Error(`${bundle.id}: source migration endpoint is outside authored scopes: ${endpoint}`);
        }
        if (effectivePaths.has(endpoint.toLowerCase())) {
          throw new Error(`${bundle.id}: source migration endpoint is an effective composition product: ${endpoint}`);
        }
        if (integrationPaths.has(endpoint.toLowerCase())) {
          throw new Error(`${bundle.id}: source migration endpoint is integration-owned: ${endpoint}`);
        }
        const endpointKey = endpoint.toLowerCase();
        const prior = endpoints.get(endpointKey);
        if (prior) {
          throw new Error(`${bundle.id}: source migration endpoint ${endpoint} is already owned by ${prior}`);
        }
        endpoints.set(endpointKey, bundle.id);
      }
      const replacement = exactReplacement(bundle, migration.source_path);
      const promotedPath = replacement.after.source_path;
      if (migration.promoted_target.path !== promotedPath) {
        throw new Error(`${bundle.id}: source migration promoted target differs from its exact composition replacement`);
      }
      const promotedDirectory = path.posix.dirname(promotedPath);
      for (const destination of migration.destination_paths) {
        if (!destination.startsWith(`${promotedDirectory}/`)) {
          throw new Error(`${bundle.id}: source migration destination is not beneath promoted directory ${promotedDirectory}`);
        }
      }
      const promotedProduct = resolvePromotedProduct(
        migration.promoted_target, integrationProducts, effectiveById, effectiveByPath, bundle.id,
      );
      if (promotedProduct) {
        validateRegisteredTargetTransition(migration, bundle, promotedProduct);
      }
      validateSupportDestinations(migration, promotedProduct, bundle.id);
      validateIdentityDestinations(migration, identities, promotedProduct, bundle.id);
    }
  }
}

function validateRegisteredTargetTransition(migration, bundle, promotedProduct) {
  const productId = promotedProduct.product_id;
  if (!bundle.integration_product_refs.includes(productId)) {
    throw new Error(`${bundle.id}: registered promoted target is not referenced by its source-migration bundle`);
  }
  const transition = bundle.module_composition_transition;
  const states = transition?.product_states?.filter((state) => state.product_id === productId) ?? [];
  if (states.length !== 1 || states[0].after_state !== "present") {
    throw new Error(`${bundle.id}: registered promoted target lacks one exact present product-state transition`);
  }
  const changes = transition.changes.filter((change) => (
    change.product_id === productId && change.after !== null
  ));
  const changedChildren = new Map(changes.map((change) => [change.after.source_path, change.after]));
  for (const support of migration.support_destinations) {
    const child = changedChildren.get(support.destination_path);
    if (!child || child.role !== "support"
      || JSON.stringify(child.reexports) !== JSON.stringify(support.reexports)) {
      throw new Error(`${bundle.id}: registered migrated support is not introduced by its exact bundle transition`);
    }
  }
  const supportPaths = new Set(migration.support_destinations
    .map(({ destination_path: destination }) => destination));
  for (const destination of migration.identity_destinations
    .flatMap(({ destination_paths: paths }) => paths)
    .filter((destination) => !supportPaths.has(destination))) {
    if (changedChildren.get(destination)?.role !== "identity") {
      throw new Error(`${bundle.id}: registered migrated identity is not introduced by its exact bundle transition`);
    }
  }
}

function parsePromotedTarget(value, label) {
  if (value?.kind === "registered") {
    exact(value, ["kind", "path", "product_id"], label);
    return {
      kind: "registered",
      path: repositoryPath(value.path, `${label} path`),
      product_id: stableId(value.product_id, `${label} product id`),
    };
  }
  exact(value, ["kind", "path"], label);
  if (value.kind !== "authored") throw new Error(`${label} kind must be authored or registered`);
  return { kind: "authored", path: repositoryPath(value.path, `${label} path`) };
}

function parseSupportDestinations(value, destinations, label) {
  const rows = array(value, `${label} support destinations`, { empty: true }).map((entry) => {
    exact(entry, ["destination_path", "reexports"], `${label} support destination`);
    const destinationPath = repositoryPath(
      entry.destination_path, `${label} support destination path`,
    );
    if (!destinations.has(destinationPath)) {
      throw new Error(`${label}: support destination is absent from destination paths: ${destinationPath}`);
    }
    const reexports = array(entry.reexports, `${label} support destination reexports`, { empty: true })
      .map((reexport) => parseCompositionReexport(
        reexport, null, `${label} support destination`,
      ));
    requireCanonicalUnique(
      reexports.map(compositionReexportKey), `${label} support destination reexports`,
    );
    return { destination_path: destinationPath, reexports };
  });
  requireCanonicalUnique(rows.map(({ destination_path: destination }) => destination),
    `${label} support destinations`);
  return rows;
}

function parseIdentityDestinations(value, destinations, label) {
  const rows = array(value, `${label} identity destinations`, { empty: true }).map((entry) => {
    exact(entry, ["identity", "destination_paths"], `${label} identity destination`);
    const normalized = uniqueStrings([entry.identity], `${label} identity destination identity`, {
      pattern: SAFE_IDENTITY, lower: true,
    })[0];
    if (normalized !== entry.identity) {
      throw new Error(`${label} identity destinations must use exact lowercase identity spelling`);
    }
    const paths = array(entry.destination_paths, `${label} ${normalized} destination paths`)
      .map((destination) => repositoryPath(destination, `${label} ${normalized} destination`));
    requireCanonicalUnique(paths, `${label} ${normalized} destination paths`);
    for (const destination of paths) if (!destinations.has(destination)) {
      throw new Error(`${label}: ${normalized} destination is absent from destination paths: ${destination}`);
    }
    return { identity: normalized, destination_paths: paths };
  });
  requireCanonicalUnique(rows.map(({ identity }) => identity), `${label} identity destinations`);
  return rows;
}

function validateDestinationPartition(
  destinations, promotedTarget, supportDestinations, identityDestinations, label,
) {
  const expected = [...new Set([
    ...(promotedTarget.kind === "authored" ? [promotedTarget.path] : []),
    ...supportDestinations.map(({ destination_path: destination }) => destination),
    ...identityDestinations.flatMap(({ destination_paths: paths }) => paths),
  ])].sort(compareCodePoint);
  if (JSON.stringify(destinations) !== JSON.stringify(expected)) {
    throw new Error(`${label}: destination paths are not exactly partitioned by promoted, support, and identity ownership`);
  }
}

function resolvePromotedProduct(target, integrationProducts, effectiveById, effectiveByPath, bundleId) {
  const effectiveAtPath = effectiveByPath.get(target.path);
  if (target.kind === "authored") {
    if (effectiveAtPath) {
      throw new Error(`${bundleId}: authored promoted target is registered as ${effectiveAtPath.product_id}`);
    }
    return null;
  }
  const registered = integrationProducts.get(target.product_id);
  if (!registered || registered.path !== target.path
    || registered.verification?.kind !== "rust_module_composition") {
    throw new Error(`${bundleId}: registered promoted target is absent from exact integration-product authority`);
  }
  const effective = effectiveById.get(target.product_id);
  if (!effective || effective.path !== target.path || effectiveAtPath !== effective) {
    throw new Error(`${bundleId}: registered promoted target is absent from effective composition`);
  }
  return effective;
}

function validateSupportDestinations(migration, promotedProduct, bundleId) {
  if (!promotedProduct) return;
  const supportChildren = new Map(promotedProduct.children
    .filter((child) => child.role === "support")
    .map((child) => [child.source_path, child]));
  const expectedSupportPaths = migration.destination_paths
    .filter((destination) => supportChildren.has(destination));
  const reviewedSupportPaths = migration.support_destinations
    .map(({ destination_path: destination }) => destination);
  if (JSON.stringify(reviewedSupportPaths) !== JSON.stringify(expectedSupportPaths)) {
    throw new Error(`${bundleId}: registered support destinations do not exactly cover migrated support children`);
  }
  for (const support of migration.support_destinations) {
    const child = supportChildren.get(support.destination_path);
    if (!child) {
      throw new Error(`${bundleId}: registered support destination is not an exact support child: ${support.destination_path}`);
    }
    if (JSON.stringify(support.reexports) !== JSON.stringify(child.reexports)) {
      throw new Error(`${bundleId}: support destination reexports differ from its exact registered support child: ${support.destination_path}`);
    }
  }
}

function validateIdentityDestinations(migration, identities, promotedProduct, bundleId) {
  const identityChildren = new Set((promotedProduct?.children ?? [])
    .filter((child) => child.role === "identity")
    .map((child) => child.source_path));
  const supportChildren = new Set(migration.support_destinations
    .map(({ destination_path: destination }) => destination));
  for (const binding of migration.identity_destinations) {
    const identity = identities.get(binding.identity);
    if (!identity || identity.bundle_id !== bundleId) {
      throw new Error(`${bundleId}: source migration identity destination lacks exact bundle identity authority: ${binding.identity}`);
    }
    if (binding.destination_paths.some((destination) => supportChildren.has(destination))
      && identity.public_identity?.kind !== "internal") {
      throw new Error(`${bundleId}: only an internal identity may remain owned by migrated support`);
    }
    const owners = [...new Set(implementationOwners(identity.implementation))].sort(compareCodePoint);
    if (JSON.stringify(binding.destination_paths) !== JSON.stringify(owners)) {
      throw new Error(`${bundleId}: ${binding.identity} migration destinations differ from final owned callable/constant paths`);
    }
    if (promotedProduct && binding.destination_paths.some((destination) =>
      !identityChildren.has(destination) && !supportChildren.has(destination))) {
      throw new Error(`${bundleId}: ${binding.identity} destination is not an exact registered identity or support child`);
    }
  }
}

function exactReplacement(bundle, sourcePath) {
  const matches = (bundle.module_composition_transition?.changes ?? []).filter((change) =>
    change.operation === "replace"
    && change.before?.source_kind === "file"
    && change.before.source_path === sourcePath
    && change.after?.source_kind === "directory");
  if (matches.length !== 1) {
    throw new Error(`${bundle.id}: source migration ${sourcePath} requires one exact same-bundle file-to-directory composition replacement`);
  }
  return matches[0];
}

function migrationKey(entry) {
  return `${entry.source_path}\0${entry.strategy}\0${entry.destination_paths.join("\0")}`;
}

function requireCanonicalUnique(values, label) {
  if (new Set(values).size !== values.length
    || JSON.stringify(values) !== JSON.stringify([...values].sort(compareCodePoint))) {
    throw new Error(`${label} must be unique and canonically ordered`);
  }
}

function requireCaseFoldUnique(values, label) {
  const folded = values.map((value) => value.toLowerCase());
  if (new Set(folded).size !== folded.length) {
    throw new Error(`${label} must not collide case-insensitively`);
  }
}
