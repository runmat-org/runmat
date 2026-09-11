import { evidenceDigest } from "./evidence.mjs";
import { validateBundleGraph } from "./control-graph.mjs";
import { parseFindingDispositions } from "./migration-findings.mjs";
import { parseGatePlans } from "./gate-plan.mjs";
import { requiredGateNames } from "./gate-requirements.mjs";
import {
  SAFE_IDENTITY, array, digest, enumValue, exact, identity, integer, kind, nonempty,
  absolutePath, filesystemIdentity, object, repositoryPath, sourceRevision, stableId, uniqueStrings,
} from "./schema.mjs";

export const MATURITY_GATES = Object.freeze([
  "identity", "disposition", "catalog-contract", "requested-output", "effects-capabilities",
  "inference", "runtime-facts", "runtime-binding", "native-ir-deopt", "placement",
  "provider", "fusion", "link-reachability", "interop-parallel", "documentation",
  "examples", "tests", "wasm",
]);

const COHORTS = ["C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07"];
const SEMANTIC_COHORTS = ["prerequisite", "A", "B", "C", "D", "E", "F", "G"];

export function parseControlManifest(value, current = null) {
  kind(value, 1, "runmat-builtin-migration-control-manifest", "control manifest");
  exact(value, ["schema_version", "kind", "authority", "program", "control_draft_digest", "baseline", "cohorts", "bundles", "identities", "migration_findings", "exception_manifest", "storage_policy", "review"], "control manifest");
  if (value.authority !== "reviewed-development-control") throw new Error("control manifest has invalid authority");
  if (value.program !== "RM-1064/C00-C07") throw new Error("control manifest has unexpected program");
  digest(value.control_draft_digest, "control draft digest");
  const baseline = parseBaseline(value.baseline);
  const cohorts = parseCohorts(value.cohorts);
  const bundles = new Map(Object.entries(object(value.bundles, "control bundles")).map(([id, entry]) => [id, parseBundle(id, entry, current)]));
  const identities = new Map();
  for (const [id, entry] of Object.entries(object(value.identities, "control identities"))) {
    const normalized = id.toLowerCase();
    if (identities.has(normalized)) throw new Error(`control identities collide case-insensitively at ${id}`);
    identities.set(normalized, parseIdentity(id, entry, bundles, cohorts));
  }
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
  parseExceptionManifest(value.exception_manifest);
  parseStoragePolicy(value.storage_policy);
  parseReview(value.review, "control manifest review");
  validateBundleGraph(bundles, identities);
  validateIdentityGraph(identities);
  validateGatePlanCoverage(bundles, identities);
  if (current) validateBaseline(baseline, current, identities);
  return { value, baseline, cohorts, bundles, identities, migrationFindings, digest: evidenceDigest(value) };
}

function parseBaseline(value) {
  exact(value, ["revision", "source_digest", "inventory_digest", "dispositions_digest", "migration_findings_digest", "compiled_target"], "control baseline");
  exact(value.compiled_target, ["operating_system", "architecture"], "control baseline compiled target");
  return {
    revision: sourceRevision(value.revision, "control baseline revision"),
    source_digest: digest(value.source_digest, "control baseline source digest"),
    inventory_digest: digest(value.inventory_digest, "control baseline inventory digest"),
    dispositions_digest: digest(value.dispositions_digest, "control baseline dispositions digest"),
    migration_findings_digest: digest(value.migration_findings_digest, "control baseline migration findings digest"),
    compiled_target: { operating_system: nonempty(value.compiled_target.operating_system, "control baseline operating system"), architecture: nonempty(value.compiled_target.architecture, "control baseline architecture") },
  };
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
  exact(value, ["id", "identities", "domain", "family", "atomic_reason", "prerequisites", "authored_write_set", "integration_outputs", "gate_plans", "owner_role", "complexity", "review"], `${id} bundle`);
  if (value.id !== id) throw new Error(`${id}: bundle key and id differ`);
  const result = {
    ...value,
    identities: uniqueStrings(value.identities, `${id} identities`, { pattern: SAFE_IDENTITY, lower: true }),
    domain: nonempty(value.domain, `${id} domain`),
    family: nonempty(value.family, `${id} family`),
    atomic_reason: nonempty(value.atomic_reason, `${id} atomic reason`),
    prerequisites: array(value.prerequisites, `${id} prerequisites`, { empty: true }).map((entry) => parsePrerequisite(entry, id)),
    authored_write_set: array(value.authored_write_set, `${id} authored write set`).map((entry) => parseScope(entry, `${id} authored scope`)),
    integration_outputs: array(value.integration_outputs, `${id} integration outputs`, { empty: true }).map((entry) => parseIntegrationOutput(entry, id)),
    gate_plans: parseGatePlans(value.gate_plans, id, current),
    owner_role: nonempty(value.owner_role, `${id} owner role`),
    complexity: parseComplexity(value.complexity, `${id} complexity`),
  };
  parseReview(value.review, `${id} bundle review`);
  return result;
}

function parseIdentity(id, value, bundles, cohorts) {
  const normalized = identity(id, "control identity").toLowerCase();
  exact(value, ["identity", "public_spelling", "disposition", "cohort", "bundle_id", "domain", "family", "runtime_owner", "shared_dependencies", "complexity", "maturity", "expected_authorities", "expected_removals", "baseline_evidence", "owner", "review"], `${id} identity`);
  if (value.identity !== normalized) throw new Error(`${id}: identity key and normalized identity differ`);
  const bundle = bundles.get(value.bundle_id);
  if (!bundle) throw new Error(`${id}: unknown bundle ${value.bundle_id}`);
  if (!cohorts.has(value.cohort)) throw new Error(`${id}: unknown cohort ${value.cohort}`);
  if (value.domain !== bundle.domain || value.family !== bundle.family) throw new Error(`${id}: identity domain/family must match its bundle`);
  parseDisposition(value.disposition, normalized);
  parseMaturity(value.maturity, id);
  parseAuthorities(value.expected_authorities, id);
  const removals = array(value.expected_removals, `${id} expected removals`, { empty: true });
  const baselineEvidence = array(value.baseline_evidence, `${id} baseline evidence`, { empty: true });
  removals.forEach((entry) => parseRemoval(entry, id));
  baselineEvidence.forEach((entry) => parseBaselineEvidence(entry, id));
  for (const removal of removals) {
    if (!baselineEvidence.some((entry) => entry.path === removal.path && entry.digest === removal.baseline_digest)) throw new Error(`${id}: expected removal ${removal.path} lacks matching baseline path/digest evidence`);
  }
  array(value.shared_dependencies, `${id} shared dependencies`, { empty: true }).forEach((entry) => parseSharedDependency(entry, id));
  parseComplexity(value.complexity, `${id} complexity`);
  const publicSpelling = identity(value.public_spelling, `${id} public spelling`);
  if (publicSpelling.toLowerCase() !== normalized) throw new Error(`${id}: public spelling must case-fold to the identity key`);
  if (value.runtime_owner === null) {
    if (value.disposition.kind === "canonical") throw new Error(`${id}: canonical identity requires a runtime owner`);
  } else repositoryPath(value.runtime_owner, `${id} runtime owner`);
  nonempty(value.owner, `${id} owner`);
  parseReview(value.review, `${id} review`);
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

function parseMaturity(value, id) {
  exact(value, MATURITY_GATES, `${id} maturity`);
  for (const gate of MATURITY_GATES) {
    const entry = value[gate];
    exact(entry, ["applicability", "reason", "evidence"], `${id} maturity ${gate}`);
    enumValue(entry.applicability, ["required", "not-applicable"], `${id} maturity ${gate} applicability`);
    uniqueStrings(entry.evidence, `${id} maturity ${gate} evidence`, { empty: true });
    if (entry.applicability === "required" && entry.reason !== null) throw new Error(`${id}: required maturity ${gate} cannot have a reason`);
    if (entry.applicability === "not-applicable" && (!String(entry.reason ?? "").trim() || entry.evidence.length === 0)) throw new Error(`${id}: inapplicable maturity ${gate} requires reason and evidence`);
  }
}

function parseAuthorities(value, id) {
  exact(value, ["catalog_package", "catalog_entry_count", "documentation", "runtime_bindings", "native_link", "wasm_registry"], `${id} expected authorities`);
  if (value.catalog_package !== null) repositoryPath(value.catalog_package, `${id} catalog package`);
  integer(value.catalog_entry_count, `${id} catalog entry count`);
  enumValue(value.documentation, ["catalog", "alias", "none"], `${id} documentation authority`);
  array(value.runtime_bindings, `${id} runtime bindings`, { empty: true }).forEach((entry) => {
    exact(entry, ["path", "function", "variant"], `${id} runtime binding`);
    repositoryPath(entry.path, `${id} runtime binding path`);
    nonempty(entry.function, `${id} runtime binding function`);
    nonempty(entry.variant, `${id} runtime binding variant`);
  });
  enumValue(value.native_link, ["required", "not-applicable"], `${id} native link`);
  enumValue(value.wasm_registry, ["required", "not-applicable"], `${id} wasm registry`);
}

function parseRemoval(value, id) {
  exact(value, ["kind", "path", "baseline_digest"], `${id} file removal`);
  if (value.kind !== "file") throw new Error(`${id}: expected removals are complete files; in-file authority changes belong to compiled inventory delta evidence`);
  repositoryPath(value.path, `${id} removal path`);
  digest(value.baseline_digest, `${id} removal baseline digest`);
}

function parseBaselineEvidence(value, id) {
  exact(value, ["kind", "path", "locator", "digest"], `${id} baseline evidence`);
  enumValue(value.kind, ["catalog", "runtime", "sidecar", "runtime-shadow", "resolver", "provider", "fusion", "test", "example"], `${id} evidence kind`);
  repositoryPath(value.path, `${id} evidence path`);
  if (value.locator !== null) throw new Error(`${id}: baseline source-item locators are obsolete; compiled and lexical inventory rows are the typed item authority`);
  digest(value.digest, `${id} evidence digest`);
}

function parseSharedDependency(value, id) {
  exact(value, ["kind", "path", "ownership", "owner_id"], `${id} shared dependency`);
  enumValue(value.kind, ["runtime-owner", "legacy-resolver", "provider-service", "fusion-service", "catalog-composition", "runtime-composition", "registry"], `${id} dependency kind`);
  repositoryPath(value.path, `${id} dependency path`);
  enumValue(value.ownership, ["bundle", "prerequisite", "integration"], `${id} dependency ownership`);
  nonempty(value.owner_id, `${id} dependency owner`);
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

function parseExceptionManifest(value) {
  exact(value, ["entries", "review"], "exception manifest");
  array(value.entries, "exception entries", { empty: true }).forEach((entry) => {
    exact(entry, ["id", "scope", "reason", "expires_after_bundle", "evidence"], "exception entry");
    nonempty(entry.id, "exception id");
    repositoryPath(entry.scope, "exception scope");
    nonempty(entry.reason, "exception reason");
    nonempty(entry.expires_after_bundle, "exception expiry bundle");
    uniqueStrings(entry.evidence, "exception evidence");
  });
  parseReview(value.review, "exception manifest review");
}

function parseStoragePolicy(value) {
  exact(value, ["volume_roles", "targets_must_be_disjoint", "occt_default"], "storage policy");
  exact(value.volume_roles, ["source_worktree", "target_temp"], "storage volume roles");
  const source = parseVolumePolicy(value.volume_roles.source_worktree, "source-worktree");
  const target = parseVolumePolicy(value.volume_roles.target_temp, "target-temp");
  if (source.filesystem_id === target.filesystem_id) throw new Error("source/worktree and target/temp roles must use disjoint filesystem identities");
  if (value.targets_must_be_disjoint !== true || value.occt_default !== "disabled-unless-affected") throw new Error("storage policy must require disjoint targets and scoped OCCT");
}

function parseVolumePolicy(value, role) {
  exact(value, ["role", "mount_path", "filesystem_id", "minimum_free_bytes", "pause_below_bytes", "maximum_observation_age_seconds"], `${role} volume policy`);
  if (value.role !== role) throw new Error(`${role} volume policy has the wrong role`);
  absolutePath(value.mount_path, `${role} mount path`);
  filesystemIdentity(value.filesystem_id, `${role} filesystem id`);
  integer(value.minimum_free_bytes, `${role} minimum free bytes`, 1);
  integer(value.pause_below_bytes, `${role} pause below bytes`, 1);
  integer(value.maximum_observation_age_seconds, `${role} maximum observation age`, 1);
  if (value.pause_below_bytes < value.minimum_free_bytes) throw new Error(`${role} pause threshold cannot be below minimum free bytes`);
  return value;
}

function parseReview(value, label) {
  exact(value, ["status", "evidence"], label);
  if (value.status !== "reviewed") throw new Error(`${label} must be reviewed`);
  uniqueStrings(value.evidence, `${label} evidence`);
}

function validateBaseline(baseline, current, identities) {
  if (current.source?.revision !== baseline.revision) throw new Error("control baseline revision does not match current source revision");
  if (current.source?.digest !== baseline.source_digest) throw new Error("control baseline source digest does not match current source snapshot");
  if (current.digest !== baseline.inventory_digest) throw new Error("control baseline inventory digest does not match current inventory");
  if (current.dispositions_digest !== baseline.dispositions_digest) throw new Error("control baseline dispositions digest does not match reviewed dispositions");
  if (current.migration_findings_digest !== baseline.migration_findings_digest) throw new Error("control baseline migration findings digest does not match compiled findings");
  if (current.compiled_inventory.build.operating_system !== baseline.compiled_target.operating_system || current.compiled_inventory.build.architecture !== baseline.compiled_target.architecture) throw new Error("control baseline compiled target does not match the inventory build");
  const observed = current.identities.map((entry) => entry.identity).sort();
  const reviewed = [...identities.keys()].sort();
  if (JSON.stringify(observed) !== JSON.stringify(reviewed)) throw new Error("control manifest identities do not exactly cover the baseline inventory");
  const sourceFiles = new Map(current.source.files.map((entry) => [entry.path, entry.content_digest]));
  for (const entry of identities.values()) for (const removal of entry.expected_removals) {
    const proof = entry.baseline_evidence.find((candidate) => candidate.path === removal.path && candidate.digest === removal.baseline_digest);
    if (proof.locator !== null || sourceFiles.get(removal.path) !== removal.baseline_digest) throw new Error(`${entry.identity}: file removal baseline does not match the content-derived source snapshot`);
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
