import { evidenceDigest } from "./evidence.mjs";
import { deepImmutable } from "./immutable.mjs";
import { validateBundleGraph } from "./control-graph.mjs";
import { validateControlProjection } from "./control-projection-validation.mjs";
import { parseFindingDispositions } from "./migration-findings.mjs";
import { parseGatePlans } from "./gate-plan.mjs";
import { requiredGateNames } from "./gate-requirements.mjs";
import { materializeTopologyControl } from "./topology/control-projection.mjs";
import { assertValidatedTopologyView } from "./topology/freeze.mjs";
import { assertValidatedControlReview } from "./control-authoring/authority.mjs";
import {
  executionTargetKey, parseExceptionManifestPolicy, parseIdentityControlPolicy,
  parseOperationalBundleControlPolicy, parseReviewedEvidence,
} from "./control-authoring/policy-schema.mjs";
import {
  SAFE_IDENTITY, array, digest, enumValue, exact, identity, integer, kind, nonempty,
  object, repositoryPath, sourceRevision, stableId, uniqueStrings,
} from "./schema.mjs";

export { MATURITY_GATES } from "./control-authoring/policy-schema.mjs";

const COHORTS = ["C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07"];
const SEMANTIC_COHORTS = ["prerequisite", "A", "B", "C", "D", "E", "F", "G"];
const VALIDATED_CONTROLS = new WeakSet();

export function validateControlManifestStructure(value, { inventory: current, reviewedTopology } = {}) {
  if (!current || !reviewedTopology) {
    throw new Error("control manifest validation requires the exact inventory and deterministically validated topology");
  }
  assertValidatedTopologyView(reviewedTopology);
  kind(value, 2, "runmat-builtin-migration-control-manifest", "control manifest");
  exact(value, ["schema_version", "kind", "authority", "program", "inputs", "topology_digest", "candidate_digest", "attestation_digest", "baseline_context", "cohorts", "bundle_controls", "identity_controls", "migration_findings", "exception_manifest", "execution_targets", "storage_policy", "review", "digest"], "control manifest");
  if (value.authority !== "reviewed-development-control") throw new Error("control manifest has invalid authority");
  if (value.program !== "RM-1064/C00-C07") throw new Error("control manifest has unexpected program");
  digest(value.topology_digest, "reviewed topology digest");
  parseControlInputs(value.inputs, reviewedTopology);
  digest(value.candidate_digest, "control candidate digest");
  digest(value.attestation_digest, "control attestation digest");
  digest(value.digest, "control manifest digest");
  const projection = validateControlProjection({
    inventory: current,
    topology: reviewedTopology,
    baselineContext: value.baseline_context,
    bundleControls: value.bundle_controls,
    identityControls: value.identity_controls,
    migrationFindings: value.migration_findings,
    exceptionManifest: value.exception_manifest,
    executionTargets: value.execution_targets,
    storagePolicy: value.storage_policy,
  });
  const baseline = parseBaselineContext(value.baseline_context, reviewedTopology);
  const cohorts = parseCohorts(value.cohorts);
  const bundleControls = new Map(Object.entries(object(value.bundle_controls, "control bundle policies")).map(([id, entry]) => [id, parseBundleControl(id, entry, current)]));
  const identityControls = new Map();
  for (const [id, entry] of Object.entries(object(value.identity_controls, "control identity policies"))) {
    const normalized = id.toLowerCase();
    if (identityControls.has(normalized)) throw new Error(`control identity policies collide case-insensitively at ${id}`);
    identityControls.set(normalized, parseIdentityControl(id, entry));
  }
  const materialized = materializeTopologyControl(reviewedTopology, {
    topology_digest: value.topology_digest,
    bundleControls,
    identityControls,
  });
  const bundles = new Map([...materialized.bundles].map(([id, entry]) => [id, parseBundle(id, entry, current)]));
  const identities = new Map([...materialized.identities].map(([id, entry]) => [id, parseIdentity(id, entry, bundles, cohorts)]));
  const publicSpellings = new Map();
  for (const entry of identities.values()) {
    const folded = entry.public_spelling.toLowerCase();
    if (publicSpellings.has(folded)) {
      throw new Error(`public spellings collide case-insensitively: ${publicSpellings.get(folded)} and ${entry.public_spelling}`);
    }
    publicSpellings.set(folded, entry.public_spelling);
  }
  if (!bundles.size || !identities.size) throw new Error("control manifest must contain bundles and identities");
  const migrationFindings = parseFindingDispositions(value.migration_findings, bundles, current?.migration_findings ?? null);
  parseExceptionManifestPolicy(value.exception_manifest, bundles);
  parseReviewedEvidence(value.review, "control manifest review");
  validateBundleGraph(bundles, identities);
  validateIdentityGraph(identities);
  validateGatePlanCoverage(bundles, identities);
  if (current) validateBaseline(baseline, current, identities);
  const { digest: _ignored, ...payload } = value;
  if (evidenceDigest(payload) !== value.digest) throw new Error("control manifest digest mismatch");
  return deepImmutable({
    value,
    inputs: value.inputs,
    topology_digest: value.topology_digest,
    baseline,
    executionTargets: projection.executionTargets,
    cohorts,
    bundles,
    identities,
    migrationFindings,
    digest: value.digest,
  });
}

export function parseControlManifest(value, { inventory, reviewedTopology, reviewedControl } = {}) {
  if (!inventory || !reviewedTopology || !reviewedControl) {
    throw new Error("control manifest parsing requires the exact inventory, deterministically validated topology, and validated control review chain");
  }
  assertValidatedControlReview(reviewedControl);
  if (JSON.stringify(value) !== JSON.stringify(reviewedControl.controlValue)) {
    throw new Error("control manifest differs from the deterministically reviewed control candidate and attestation");
  }
  const parsed = validateControlManifestStructure(value, { inventory, reviewedTopology });
  VALIDATED_CONTROLS.add(parsed);
  return parsed;
}

function parseControlInputs(value, topology) {
  exact(value, ["baseline_inventory_digest", "control_draft_digest", "reviewed_topology_digest", "control_scaffold_digest", "review_set_digest", "global_review_digest", "bundle_review_digests"], "control review inputs");
  for (const field of ["baseline_inventory_digest", "control_draft_digest", "reviewed_topology_digest", "control_scaffold_digest", "review_set_digest", "global_review_digest"]) {
    digest(value[field], `control review input ${field}`);
  }
  if (value.baseline_inventory_digest !== topology.baseline.inventory_digest
    || value.control_draft_digest !== topology.baseline.control_draft_digest
    || value.reviewed_topology_digest !== topology.digest) {
    throw new Error("control review inputs differ from the exact reviewed topology chain");
  }
  const rows = array(value.bundle_review_digests, "control bundle review digests");
  const ids = rows.map((entry) => {
    exact(entry, ["bundle_id", "digest"], "control bundle review digest");
    digest(entry.digest, `${entry.bundle_id} bundle review digest`);
    return stableId(entry.bundle_id, "control bundle review id");
  });
  const expected = [...topology.bundles.keys()].sort();
  if (JSON.stringify(ids) !== JSON.stringify(expected)) {
    throw new Error("control bundle review digests do not exactly cover the reviewed topology");
  }
}

export function assertValidatedControl(value) {
  if (!VALIDATED_CONTROLS.has(value)) throw new Error("operation requires the exact validated control manifest");
  return value;
}

export function assertControlBaseline(control, inventory) {
  assertValidatedControl(control);
  assertInventoryIntegrity(inventory, "control baseline inventory");
  validateBaselineBinding(control.baseline, inventory);
  return inventory;
}

export function assertControlSubject(control, inventory) {
  assertValidatedControl(control);
  assertInventoryIntegrity(inventory, "control subject inventory");
  const build = inventory.compiled_inventory?.build;
  const target = executionTargetKey(build ?? {});
  if (!control.executionTargets.some((entry) => executionTargetKey(entry) === target)) {
    throw new Error("control subject compiled target is not a reviewed execution target");
  }
  const expected = [...control.identities.keys()].sort();
  const observed = inventory.identities.map((entry) => entry.identity).sort();
  if (JSON.stringify(observed) !== JSON.stringify(expected)) {
    throw new Error("control subject identities differ from the reviewed control identity set");
  }
  const rows = new Map(inventory.identities.map((entry) => [entry.identity, entry]));
  for (const [identity, reviewed] of control.identities) {
    validateReviewedClassification(reviewed, rows.get(identity));
  }
  return inventory;
}

function parseBaselineContext(value, topology) {
  exact(value, ["source_digest", "dispositions_digest", "migration_findings_digest", "compiled_target"], "control baseline context");
  exact(value.compiled_target, ["operating_system", "architecture"], "control baseline compiled target");
  return {
    revision: sourceRevision(topology?.baseline?.revision, "reviewed topology baseline revision"),
    source_digest: digest(value.source_digest, "control baseline source digest"),
    inventory_digest: digest(topology?.baseline?.inventory_digest, "reviewed topology baseline inventory digest"),
    dispositions_digest: digest(value.dispositions_digest, "control baseline dispositions digest"),
    migration_findings_digest: digest(value.migration_findings_digest, "control baseline migration findings digest"),
    compiled_target: { operating_system: nonempty(value.compiled_target.operating_system, "control baseline operating system"), architecture: nonempty(value.compiled_target.architecture, "control baseline architecture") },
  };
}

function parseBundleControl(id, value, current) {
  stableId(id, "bundle control id");
  parseOperationalBundleControlPolicy(value, id, current);
  return value;
}

function parseIdentityControl(id, value) {
  return parseIdentityControlPolicy(value, id);
}

function parseCohorts(value) {
  const rows = array(value, "control cohorts");
  if (rows.length !== COHORTS.length) throw new Error("control manifest must define C00 through C07 exactly once");
  const result = new Map();
  for (const row of rows) {
    exact(row, ["id", "semantic", "order"], "control cohort");
    const id = enumValue(row.id, COHORTS, "cohort id");
    if (result.has(id)) throw new Error(`duplicate cohort ${id}`);
    const expected = COHORTS.indexOf(id);
    if (row.order !== expected || row.semantic !== SEMANTIC_COHORTS[expected]) throw new Error(`${id}: cohort order or semantic label is inconsistent`);
    result.set(id, row);
  }
  return result;
}

function parseBundle(id, value, current) {
  stableId(id, "bundle id");
  exact(value, ["id", "identities", "atomic_reason", "prerequisites", "authored_write_set", "integration_outputs", "gate_plans", "owner_role", "complexity", "review"], `${id} bundle`);
  if (value.id !== id) throw new Error(`${id}: bundle key and id differ`);
  const result = {
    ...value,
    identities: uniqueStrings(value.identities, `${id} identities`, { pattern: SAFE_IDENTITY, lower: true }),
    atomic_reason: nonempty(value.atomic_reason, `${id} atomic reason`),
    prerequisites: array(value.prerequisites, `${id} prerequisites`, { empty: true }).map((entry) => parsePrerequisite(entry, id)),
    authored_write_set: array(value.authored_write_set, `${id} authored write set`).map((entry) => parseScope(entry, `${id} authored scope`)),
    integration_outputs: array(value.integration_outputs, `${id} integration outputs`, { empty: true }).map((entry) => parseIntegrationOutput(entry, id)),
    gate_plans: parseGatePlans(value.gate_plans, id, current),
    owner_role: nonempty(value.owner_role, `${id} owner role`),
    complexity: parseComplexity(value.complexity, `${id} complexity`),
  };
  parseReviewedEvidence(value.review, `${id} bundle review`);
  return result;
}

function parseIdentity(id, value, bundles, cohorts) {
  const normalized = identity(id, "control identity").toLowerCase();
  exact(value, ["identity", "public_spelling", "disposition", "cohort", "bundle_id", "domain", "family", "runtime_owner", "shared_dependencies", "complexity", "maturity", "expected_authorities", "expected_removals", "baseline_evidence", "owner", "review"], `${id} identity`);
  if (value.identity !== normalized) throw new Error(`${id}: identity key and normalized identity differ`);
  const bundle = bundles.get(value.bundle_id);
  if (!bundle) throw new Error(`${id}: unknown bundle ${value.bundle_id}`);
  if (!cohorts.has(value.cohort)) throw new Error(`${id}: unknown cohort ${value.cohort}`);
  parseDisposition(value.disposition, normalized);
  if (value.runtime_owner === null) {
    if (value.disposition.kind === "canonical"
      && (value.expected_authorities.runtime_bindings.length > 0
        || value.expected_authorities.runtime_constants.length === 0)) {
      throw new Error(`${id}: canonical callable identity requires a runtime owner`);
    }
  } else repositoryPath(value.runtime_owner, `${id} runtime owner`);
  return value;
}

function parseDisposition(value, id) {
  object(value, `${id} disposition`);
  if (value.kind === "canonical") {
    exact(value, ["kind", "target"], `${id} canonical disposition`);
    if (value.target.toLowerCase() !== id) throw new Error(`${id}: canonical target must be self`);
  } else if (value.kind === "alias") {
    exact(value, ["kind", "target"], `${id} alias disposition`);
    identity(value.target, `${id} alias target`);
    if (value.target.toLowerCase() === id) throw new Error(`${id}: alias target cannot be self`);
  } else if (value.kind === "internal") {
    exact(value, ["kind", "reason", "evidence"], `${id} internal disposition`);
    nonempty(value.reason, `${id} internal reason`);
    uniqueStrings(value.evidence, `${id} internal evidence`);
  } else throw new Error(`${id}: unsupported disposition`);
}

function parsePrerequisite(value, id) {
  exact(value, ["bundle_id", "kind"], `${id} prerequisite`);
  stableId(value.bundle_id, `${id} prerequisite bundle`);
  enumValue(value.kind, ["representation", "infrastructure", "semantic", "cohort"], `${id} prerequisite kind`);
  return value;
}

export function parseScope(value, label) {
  exact(value, ["kind", "path"], label);
  enumValue(value.kind, ["file", "tree"], `${label} kind`);
  repositoryPath(value.path, `${label} path`);
  return value;
}

function parseIntegrationOutput(value, id) {
  exact(value, ["product_id", "path", "producer"], `${id} integration output`);
  nonempty(value.product_id, `${id} integration product id`);
  repositoryPath(value.path, `${id} integration output path`);
  if (value.producer !== "integration") throw new Error(`${id}: integration output producer must be integration`);
  return { kind: "file", ...value };
}

function parseComplexity(value, label) {
  exact(value, ["class", "weight", "basis"], label);
  enumValue(value.class, ["low", "medium", "high", "dynamic"], `${label} class`);
  integer(value.weight, `${label} weight`, 1);
  uniqueStrings(value.basis, `${label} basis`);
  return value;
}

function validateBaseline(baseline, current, identities) {
  assertInventoryIntegrity(current, "control baseline inventory");
  validateBaselineBinding(baseline, current);
  const observed = current.identities.map((entry) => entry.identity).sort();
  const reviewed = [...identities.keys()].sort();
  if (JSON.stringify(observed) !== JSON.stringify(reviewed)) throw new Error("control manifest identities do not exactly cover the baseline inventory");
  const inventoryIdentities = new Map(current.identities.map((entry) => [entry.identity, entry]));
  for (const [id, entry] of identities) {
    validateReviewedClassification(entry, inventoryIdentities.get(id));
  }
  const sourceFiles = new Map(current.source.files.map((entry) => [entry.path, entry.content_digest]));
  for (const entry of identities.values()) for (const removal of entry.expected_removals) {
    const proof = entry.baseline_evidence.find((candidate) => candidate.path === removal.path && candidate.digest === removal.baseline_digest);
    if (proof.locator !== null || sourceFiles.get(removal.path) !== removal.baseline_digest) throw new Error(`${entry.identity}: file removal baseline does not match the content-derived source snapshot`);
  }
}

function validateBaselineBinding(baseline, current) {
  if (current.source?.revision !== baseline.revision) throw new Error("control baseline revision does not match current source revision");
  if (current.source?.digest !== baseline.source_digest) throw new Error("control baseline source digest does not match current source snapshot");
  if (current.digest !== baseline.inventory_digest) throw new Error("control baseline inventory digest does not match current inventory");
  if (current.dispositions_digest !== baseline.dispositions_digest) throw new Error("control baseline dispositions digest does not match reviewed dispositions");
  if (current.migration_findings_digest !== baseline.migration_findings_digest) throw new Error("control baseline migration findings digest does not match compiled findings");
  if (current.compiled_inventory.build.operating_system !== baseline.compiled_target.operating_system || current.compiled_inventory.build.architecture !== baseline.compiled_target.architecture) throw new Error("control baseline compiled target does not match the inventory build");
}

function assertInventoryIntegrity(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be an inventory object`);
  digest(value.digest, `${label} digest`);
  const { digest: _ignored, ...payload } = value;
  if (evidenceDigest(payload) !== value.digest) throw new Error(`${label} artifact digest mismatch`);
}

function validateReviewedClassification(control, inventory) {
  if (inventory.classification_input.review.status !== "reviewed") {
    throw new Error(`${control.identity}: baseline identity disposition is not reviewed`);
  }
  if (!inventory.spellings.includes(control.public_spelling)) {
    throw new Error(`${control.identity}: public spelling differs from the reviewed inventory`);
  }
  const reviewedDomain = inventory.classification_input.domain;
  const reviewedFamily = inventory.classification_input.family;
  if ((reviewedDomain !== null && reviewedDomain !== control.domain)
    || (reviewedFamily !== null && reviewedFamily !== control.family)) {
    throw new Error(`${control.identity}: domain or family differs from its reviewed disposition override`);
  }
  const observed = inventory.disposition;
  if (observed.kind !== control.disposition.kind) {
    throw new Error(`${control.identity}: disposition differs from the reviewed inventory`);
  }
  if (observed.kind === "canonical" && control.disposition.target !== control.identity) {
    throw new Error(`${control.identity}: canonical target differs from the reviewed inventory`);
  }
  if (observed.kind === "alias" && observed.canonical !== control.disposition.target.toLowerCase()) {
    throw new Error(`${control.identity}: alias target differs from the reviewed inventory`);
  }
  if (observed.kind === "internal" && observed.reason !== control.disposition.reason) {
    throw new Error(`${control.identity}: internal reason differs from the reviewed inventory`);
  }
}

function validateIdentityGraph(identities) {
  for (const [id, entry] of identities) {
    if (entry.disposition.kind !== "alias") continue;
    const target = identities.get(entry.disposition.target.toLowerCase());
    if (!target) throw new Error(`${id}: alias target is absent from the control manifest`);
    if (target.disposition.kind !== "canonical") throw new Error(`${id}: alias target must be canonical`);
  }
}

function validateGatePlanCoverage(bundles, identities) {
  for (const bundle of bundles.values()) {
    const required = new Set(bundle.identities.flatMap((id) => requiredGateNames(identities.get(id))));
    for (const gate of required) if (!bundle.gate_plans.has(gate)) throw new Error(`${bundle.id}: required gate ${gate} has no reviewed gate plan`);
  }
}
