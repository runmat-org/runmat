import { compareCodePoint } from "./constants.mjs";
import { validateBundleGraph } from "./control-graph.mjs";
import { requiredGateNames } from "./gate-requirements.mjs";
import { parseFindingDispositions } from "./migration-findings.mjs";
import { materializeTopologyControl } from "./topology/control-projection.mjs";
import { assertValidatedTopologyView } from "./topology/freeze.mjs";
import {
  parseExceptionManifestPolicy, parseIdentityControlPolicy,
  parseOperationalBundleControlPolicy, parseExecutionTargets,
  validateStorageTargetCoverage,
} from "./control-authoring/policy-schema.mjs";

export function validateControlProjection({
  inventory,
  topology,
  baselineContext,
  bundleControls: bundleControlValues,
  identityControls: identityControlValues,
  migrationFindings,
  exceptionManifest,
  executionTargets,
  storagePolicy,
}) {
  assertValidatedTopologyView(topology);
  validateBaselineContext(baselineContext, inventory, topology);
  const bundleControls = parseBundleControls(bundleControlValues, inventory);
  const identityControls = parseIdentityControls(identityControlValues);
  const materialized = materializeTopologyControl(topology, {
    topology_digest: topology.digest,
    bundleControls,
    identityControls,
  });
  const bundles = normalizeBundles(materialized.bundles);
  validateBundleGraph(bundles, materialized.identities);
  validateIdentityGraph(materialized.identities);
  validateGateCoverage(bundles, materialized.identities);
  validateInventoryAuthority(materialized.identities, inventory);
  parseFindingDispositions(migrationFindings, bundles, inventory.migration_findings);
  parseExceptionManifestPolicy(exceptionManifest, bundles);
  const parsedExecutionTargets = parseExecutionTargets(executionTargets);
  validateStorageTargetCoverage(storagePolicy, parsedExecutionTargets);
  return { bundles, identities: materialized.identities, executionTargets: parsedExecutionTargets };
}

function parseBundleControls(value, inventory) {
  requireRecord(value, "control bundle policies");
  return new Map(Object.keys(value).sort(compareCodePoint).map((id) => {
    parseOperationalBundleControlPolicy(value[id], id, inventory);
    return [id, value[id]];
  }));
}

function parseIdentityControls(value) {
  requireRecord(value, "control identity policies");
  return new Map(Object.keys(value).sort(compareCodePoint).map((id) => [id, parseIdentityControlPolicy(value[id], id)]));
}

function normalizeBundles(values) {
  return new Map([...values].map(([id, bundle]) => [id, {
    ...bundle,
    integration_outputs: bundle.integration_outputs.map((entry) => ({ kind: "file", ...entry })),
  }]));
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

function validateIdentityGraph(identities) {
  const spellings = new Map();
  for (const [id, entry] of identities) {
    const folded = entry.public_spelling.toLowerCase();
    if (spellings.has(folded)) throw new Error(`public spellings collide case-insensitively: ${spellings.get(folded)} and ${entry.public_spelling}`);
    spellings.set(folded, entry.public_spelling);
    if (entry.disposition.kind === "alias") {
      const target = identities.get(entry.disposition.target.toLowerCase());
      if (!target) throw new Error(`${id}: alias target is absent from the control manifest`);
      if (target.disposition.kind !== "canonical") throw new Error(`${id}: alias target must be canonical`);
    }
    if (entry.runtime_owner === null && entry.disposition.kind === "canonical"
      && (entry.expected_authorities.runtime_bindings.length > 0
        || entry.expected_authorities.runtime_constants.length === 0)) {
      throw new Error(`${id}: canonical callable identity requires a runtime owner`);
    }
  }
}

function validateGateCoverage(bundles, identities) {
  for (const bundle of bundles.values()) {
    const available = new Set(bundle.gate_plans.map((plan) => plan.gate));
    const required = new Set(bundle.identities.flatMap((id) => requiredGateNames(identities.get(id))));
    for (const gate of required) {
      if (!available.has(gate)) throw new Error(`${bundle.id}: required gate ${gate} has no reviewed gate plan`);
    }
  }
}

function validateInventoryAuthority(identities, inventory) {
  const rows = new Map(inventory.identities.map((entry) => [entry.identity, entry]));
  const sourceFiles = new Map(inventory.source.files.map((entry) => [entry.path, entry.content_digest]));
  if (JSON.stringify([...identities.keys()].sort(compareCodePoint)) !== JSON.stringify([...rows.keys()].sort(compareCodePoint))) {
    throw new Error("control identities do not exactly cover the baseline inventory");
  }
  for (const [id, control] of identities) {
    const row = rows.get(id);
    if (row.classification_input.review.status !== "reviewed") throw new Error(`${id}: baseline identity disposition is not reviewed`);
    if (!row.spellings.includes(control.public_spelling)) throw new Error(`${id}: public spelling differs from the reviewed inventory`);
    if ((row.classification_input.domain !== null && row.classification_input.domain !== control.domain)
      || (row.classification_input.family !== null && row.classification_input.family !== control.family)) {
      throw new Error(`${id}: domain or family differs from its reviewed disposition override`);
    }
    if (row.disposition.kind !== control.disposition.kind) throw new Error(`${id}: disposition differs from the reviewed inventory`);
    if (row.disposition.kind === "alias" && row.disposition.canonical !== control.disposition.target.toLowerCase()) {
      throw new Error(`${id}: alias target differs from the reviewed inventory`);
    }
    if (row.disposition.kind === "internal" && row.disposition.reason !== control.disposition.reason) {
      throw new Error(`${id}: internal reason differs from the reviewed inventory`);
    }
    for (const removal of control.expected_removals) {
      if (sourceFiles.get(removal.path) !== removal.baseline_digest) {
        throw new Error(`${id}: file removal baseline does not match the content-derived source snapshot`);
      }
    }
  }
}

function requireRecord(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be an object`);
}
