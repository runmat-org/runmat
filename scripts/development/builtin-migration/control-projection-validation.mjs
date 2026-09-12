import { compareCodePoint } from "./constants.mjs";
import { validateAuthorityDependencyPolicy } from "./authority-paths.mjs";
import { validateBundleGraph } from "./control-graph.mjs";
import { requiredGatePlanNames } from "./gate-requirements.mjs";
import {
  observedIdentityForms, primarySpelling, validateIdentityAuthorityGraph,
} from "./identity-authority.mjs";
import {
  parseIntegrationProductRegistry, validateGeneratedRegistryCoverage,
  validateIntegrationProductCoverage,
} from "./integration-products.mjs";
import { parseFindingDispositions } from "./migration-findings.mjs";
import { materializeTopologyControl } from "./topology/control-projection.mjs";
import { assertValidatedTopologyView } from "./topology/freeze.mjs";
import { parseTargetPolicy } from "./target-policy.mjs";
import {
  parseExceptionManifestPolicy, parseIdentityControlPolicy,
  parseOperationalBundleControlPolicy, validateStorageTargetCoverage,
} from "./control-authoring/policy-schema.mjs";

export function validateControlProjection({
  inventory,
  topology,
  baselineContext,
  bundleControls: bundleControlValues,
  identityControls: identityControlValues,
  integrationProducts: integrationProductValues,
  migrationFindings,
  exceptionManifest,
  targetPolicy: targetPolicyValue,
  storagePolicy,
}) {
  assertValidatedTopologyView(topology);
  validateBaselineContext(baselineContext, inventory, topology);
  const integrationProducts = parseIntegrationProductRegistry(integrationProductValues, inventory);
  const bundleControls = parseBundleControls(bundleControlValues, inventory, topology);
  const identityControls = parseIdentityControls(identityControlValues);
  validateIntegrationProductCoverage(bundleControls, integrationProducts);
  const materialized = materializeTopologyControl(topology, {
    topology_digest: topology.digest,
    bundleControls,
    identityControls,
    integrationProducts,
  });
  const bundles = materialized.bundles;
  validateBundleGraph(bundles, materialized.identities);
  validateGeneratedRegistryCoverage(
    inventory,
    bundles,
    materialized.identities,
    integrationProducts,
  );
  validateAuthorityDependencyPolicy(bundles, materialized.identities, integrationProducts);
  validateIdentityAuthorityGraph(materialized.identities);
  validateGateCoverage(bundles, materialized.identities);
  validateInventoryAuthority(materialized.identities, inventory);
  parseFindingDispositions(migrationFindings, bundles, inventory.migration_findings);
  parseExceptionManifestPolicy(exceptionManifest, bundles);
  const targetPolicy = parseTargetPolicy(targetPolicyValue);
  const parsedExecutionTargets = targetPolicy.migrationExecutionTargets;
  validateStorageTargetCoverage(storagePolicy, parsedExecutionTargets);
  return {
    bundles,
    identities: materialized.identities,
    integrationProducts,
    targetPolicy,
    executionTargets: parsedExecutionTargets,
  };
}

function parseBundleControls(value, inventory, topology) {
  requireRecord(value, "control bundle policies");
  return new Map(Object.keys(value).sort(compareCodePoint).map((id) => {
    const identities = topology.bundles.get(id)?.identities ?? [];
    parseOperationalBundleControlPolicy(value[id], id, inventory, identities);
    return [id, value[id]];
  }));
}

function parseIdentityControls(value) {
  requireRecord(value, "control identity policies");
  return new Map(Object.keys(value).sort(compareCodePoint).map((id) => [id, parseIdentityControlPolicy(value[id], id)]));
}

function validateBaselineContext(value, inventory, topology) {
  if (topology.baseline.inventory_digest !== inventory.digest
    || topology.baseline.revision !== inventory.source.revision
    || value.source_digest !== inventory.source.digest
    || value.dispositions_digest !== inventory.dispositions_digest
    || value.migration_findings_digest !== inventory.migration_findings_digest
    || value.compiled_target?.operating_system !== inventory.compiled_inventory.build.operating_system
    || value.compiled_target?.architecture !== inventory.compiled_inventory.build.architecture) {
    throw new Error("control projection baseline differs from the exact inventory and reviewed topology");
  }
}

function validateGateCoverage(bundles, identities) {
  for (const bundle of bundles.values()) {
    const available = new Set(bundle.gate_plans.map((plan) => plan.gate));
    const required = requiredGatePlanNames(
      bundle.identities.map((id) => identities.get(id)),
    );
    for (const gate of required) {
      if (!available.has(gate)) throw new Error(`${bundle.id}: required gate ${gate} has no reviewed gate plan`);
    }
  }
}

function validateInventoryAuthority(identities, inventory) {
  const rows = new Map(inventory.identities.map((entry) => [entry.identity, entry]));
  if (JSON.stringify([...identities.keys()].sort(compareCodePoint)) !== JSON.stringify([...rows.keys()].sort(compareCodePoint))) {
    throw new Error("control identities do not exactly cover the baseline inventory");
  }
  for (const [id, control] of identities) {
    const row = rows.get(id);
    if (row.classification_input.review.status !== "reviewed") throw new Error(`${id}: baseline identity disposition is not reviewed`);
    const spelling = primarySpelling(control.public_identity);
    if (spelling !== null && !row.spellings.includes(spelling)) throw new Error(`${id}: public identity spelling differs from the reviewed inventory`);
    if (JSON.stringify(control.forms) !== JSON.stringify(observedIdentityForms(row))) {
      throw new Error(`${id}: reviewed callable and constant forms differ from compiled observations`);
    }
    if ((row.classification_input.domain !== null && row.classification_input.domain !== control.domain)
      || (row.classification_input.family !== null && row.classification_input.family !== control.family)) {
      throw new Error(`${id}: domain or family differs from its reviewed disposition override`);
    }
    if (row.disposition.kind === "canonical" && control.public_identity.kind !== "primary") {
      throw new Error(`${id}: public identity differs from the reviewed inventory`);
    }
    if (row.disposition.kind === "alias" && (control.public_identity.kind !== "alias"
      || row.disposition.canonical !== control.public_identity.canonical_identity.toLowerCase())) {
      throw new Error(`${id}: alias target differs from the reviewed inventory`);
    }
    if (row.disposition.kind === "internal" && (control.public_identity.kind !== "internal"
      || row.disposition.reason !== control.public_identity.reason)) {
      throw new Error(`${id}: internal identity differs from the reviewed inventory`);
    }
  }
}

function requireRecord(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be an object`);
}
