import { compareCodePoint } from "../constants.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { GATE_PRODUCERS } from "../gate-kinds.mjs";
import { GATE_PARSERS, parseGatePlans } from "../gate-plan.mjs";
import {
  SAFE_IDENTITY, absolutePath, array, digest, enumValue, exact, filesystemIdentity,
  identity, integer, nonempty, repositoryPath, stableId, uniqueStrings,
} from "../schema.mjs";

export function assertScaffoldTopologyBinding(scaffold, topology) {
  if (topology.baseline?.control_draft_digest !== scaffold.bindings?.control_draft_digest
    || topology.baseline?.inventory_digest !== scaffold.bindings?.inventory_digest
    || topology.baseline?.revision !== scaffold.bindings?.revision
    || scaffold.bindings?.reviewed_topology_digest !== topology.digest) {
    throw new Error("reviewed topology and control authoring scaffold bindings differ");
  }
}

export function assertEvidenceDigest(value, label) {
  digest(value.digest, `${label} digest`);
  const { digest: _ignored, ...payload } = value;
  if (evidenceDigest(payload) !== value.digest) throw new Error(`${label} digest mismatch`);
}

export const MATURITY_GATES = Object.freeze([
  "identity", "disposition", "catalog-contract", "requested-output", "effects-capabilities",
  "inference", "runtime-facts", "runtime-binding", "native-ir-deopt", "placement",
  "provider", "fusion", "link-reachability", "interop-parallel", "documentation",
  "examples", "tests", "wasm",
]);

export function parseBundleControlPolicy(value, id) {
  return parseBundlePolicy(value, id, (plans) => parseGateReferences(plans, id));
}

export function parseOperationalBundleControlPolicy(value, id, inventory) {
  return parseBundlePolicy(value, id, (plans) => parseGatePlans(plans, id, inventory));
}

function parseBundlePolicy(value, id, parsePlans) {
  exact(value, ["prerequisites", "additional_authored_write_set", "integration_outputs", "gate_plans", "owner_role", "complexity", "review"], `${id} bundle control`);
  array(value.prerequisites, `${id} prerequisites`, { empty: true }).forEach((entry) => {
    exact(entry, ["bundle_id", "kind"], `${id} prerequisite`);
    stableId(entry.bundle_id, `${id} prerequisite bundle`);
    enumValue(entry.kind, ["representation", "infrastructure", "semantic", "cohort"], `${id} prerequisite kind`);
  });
  array(value.additional_authored_write_set, `${id} additional authored write set`, { empty: true }).forEach((entry) => parseScope(entry, `${id} additional authored scope`));
  array(value.integration_outputs, `${id} integration outputs`, { empty: true }).forEach((entry) => {
    exact(entry, ["product_id", "path", "producer"], `${id} integration output`);
    nonempty(entry.product_id, `${id} integration product id`);
    repositoryPath(entry.path, `${id} integration output path`);
    if (entry.producer !== "integration") throw new Error(`${id}: integration output producer must be integration`);
  });
  parsePlans(value.gate_plans);
  nonempty(value.owner_role, `${id} owner role`);
  parseComplexity(value.complexity, `${id} complexity`);
  parseReviewedEvidence(value.review, `${id} bundle control review`);
  return value;
}

export function parseIdentityControlPolicy(value, id) {
  const normalized = identity(id, "identity control id").toLowerCase();
  exact(value, [
    "public_spelling", "runtime_owner", "shared_dependencies", "complexity", "maturity",
    "expected_authorities", "expected_removals", "baseline_evidence", "owner", "review",
  ], `${id} identity control`);
  const spelling = identity(value.public_spelling, `${id} public spelling`);
  if (spelling.toLowerCase() !== normalized) throw new Error(`${id}: public spelling must case-fold to the identity key`);
  if (value.runtime_owner !== null) repositoryPath(value.runtime_owner, `${id} runtime owner`);
  array(value.shared_dependencies, `${id} shared dependencies`, { empty: true }).forEach((entry) => parseSharedDependency(entry, id));
  parseComplexity(value.complexity, `${id} complexity`);
  parseMaturity(value.maturity, id);
  parseAuthorities(value.expected_authorities, id);
  const removals = array(value.expected_removals, `${id} expected removals`, { empty: true });
  const evidence = array(value.baseline_evidence, `${id} baseline evidence`, { empty: true });
  removals.forEach((entry) => parseRemoval(entry, id));
  evidence.forEach((entry) => parseBaselineEvidence(entry, id));
  for (const removal of removals) {
    if (!evidence.some((entry) => entry.path === removal.path && entry.digest === removal.baseline_digest)) {
      throw new Error(`${id}: expected removal ${removal.path} lacks matching baseline path/digest evidence`);
    }
  }
  nonempty(value.owner, `${id} owner`);
  parseReviewedEvidence(value.review, `${id} identity control review`);
  return value;
}

export function parseProgramProfile(value, id) {
  stableId(id, "global program profile id");
  exact(value, ["program", "review"], `${id} global program profile`);
  const program = value.program;
  if (program?.kind === "repository_script") {
    exact(program, ["kind", "path", "content_digest", "approved_executables"], `${id} repository script`);
    repositoryPath(program.path, `${id} repository script path`);
    digest(program.content_digest, `${id} repository script content digest`);
  } else if (program?.kind === "cargo_binary") {
    exact(program, ["kind", "package", "binary", "manifest_path", "manifest_digest", "approved_executables"], `${id} cargo binary`);
    nonempty(program.package, `${id} cargo package`);
    nonempty(program.binary, `${id} cargo binary name`);
    repositoryPath(program.manifest_path, `${id} Cargo manifest path`);
    digest(program.manifest_digest, `${id} Cargo manifest digest`);
  } else throw new Error(`${id}: unsupported global program profile kind`);
  const keys = array(program.approved_executables, `${id} approved executables`).map((entry) => {
    exact(entry, ["operating_system", "architecture", "content_digest"], `${id} approved executable`);
    const key = `${nonempty(entry.operating_system, `${id} executable operating system`)}\0${nonempty(entry.architecture, `${id} executable architecture`)}`;
    digest(entry.content_digest, `${id} executable digest`);
    return key;
  });
  requireCanonicalUnique(keys, `${id} approved executables`);
  parseReviewedEvidence(value.review, `${id} global program profile review`);
  return value;
}

export function parseExceptionManifestPolicy(value, bundles) {
  exact(value, ["entries", "review"], "exception manifest");
  const ids = array(value.entries, "exception entries", { empty: true }).map((entry) => {
    exact(entry, ["id", "scope", "reason", "expires_after_bundle", "evidence"], "exception entry");
    const id = stableId(entry.id, "exception id");
    repositoryPath(entry.scope, "exception scope");
    nonempty(entry.reason, "exception reason");
    stableId(entry.expires_after_bundle, "exception expiry bundle");
    if (!bundles.has(entry.expires_after_bundle)) throw new Error(`${id}: exception expiry references unknown bundle ${entry.expires_after_bundle}`);
    uniqueStrings(entry.evidence, "exception evidence");
    return id;
  });
  requireCanonicalUnique(ids, "exception entries");
  parseReviewedEvidence(value.review, "exception manifest review");
  return value;
}

export function parseStoragePolicy(value) {
  exact(value, ["host_profiles", "targets_must_be_disjoint", "occt_default"], "storage policy");
  if (!value.host_profiles || typeof value.host_profiles !== "object" || Array.isArray(value.host_profiles)) {
    throw new Error("storage host profiles must be an object");
  }
  const ids = Object.keys(value.host_profiles);
  requireCanonicalUnique(ids, "storage host profiles");
  if (ids.length === 0) throw new Error("storage policy requires at least one host profile");
  const selectors = [];
  for (const id of ids) {
    stableId(id, "storage host profile id");
    const profile = value.host_profiles[id];
    exact(profile, ["operating_system", "architecture", "execution_host", "volume_roles"], `${id} storage host profile`);
    selectors.push(`${nonempty(profile.operating_system, `${id} operating system`)}\0${nonempty(profile.architecture, `${id} architecture`)}\0${nonempty(profile.execution_host, `${id} execution host`)}`);
    exact(profile.volume_roles, ["source_worktree", "target_temp"], `${id} storage volume roles`);
    const source = parseVolumePolicy(profile.volume_roles.source_worktree, "source-worktree");
    const target = parseVolumePolicy(profile.volume_roles.target_temp, "target-temp");
    if (source.filesystem_id === target.filesystem_id) throw new Error(`${id}: source/worktree and target/temp roles must use disjoint filesystem identities`);
  }
  if (new Set(selectors).size !== selectors.length) throw new Error("storage host profile selectors must be unique");
  if (value.targets_must_be_disjoint !== true || value.occt_default !== "disabled-unless-affected") throw new Error("storage policy must require disjoint targets and scoped OCCT");
  return value;
}

export function parseReviewedEvidence(value, label) {
  exact(value, ["status", "evidence"], label);
  if (value.status !== "reviewed") throw new Error(`${label} must be reviewed`);
  uniqueStrings(value.evidence, `${label} evidence`);
  return value;
}

function parseGateReferences(value, id) {
  const keys = array(value, `${id} gate plans`, { empty: true }).map((entry) => {
    exact(entry, ["gate", "program_profile_id", "arguments", "working_directory", "parser", "expected_artifact_roles"], `${id} gate plan`);
    const gate = enumValue(entry.gate, Object.keys(GATE_PRODUCERS), `${id} gate`);
    stableId(entry.program_profile_id, `${id} gate program profile`);
    if (entry.working_directory !== "repository") throw new Error(`${id}: gate working directory must be repository`);
    enumValue(entry.parser, GATE_PARSERS, `${id} gate parser`);
    uniqueStrings(entry.arguments, `${id} gate arguments`, { empty: true });
    const roles = uniqueStrings(entry.expected_artifact_roles, `${id} gate artifact roles`, { empty: true });
    if (JSON.stringify(roles) !== JSON.stringify([...roles].sort(compareCodePoint))) throw new Error(`${id}: gate artifact roles must use canonical order`);
    return gate;
  });
  requireCanonicalUnique(keys, `${id} gate plans`);
}

function parseScope(value, label) {
  exact(value, ["kind", "path"], label);
  enumValue(value.kind, ["file", "tree"], `${label} kind`);
  repositoryPath(value.path, `${label} path`);
}

function parseComplexity(value, label) {
  exact(value, ["class", "weight", "basis"], label);
  enumValue(value.class, ["low", "medium", "high", "dynamic"], `${label} class`);
  integer(value.weight, `${label} weight`, 1);
  uniqueStrings(value.basis, `${label} basis`);
}

function parseSharedDependency(value, id) {
  exact(value, ["kind", "path", "ownership", "owner_id"], `${id} shared dependency`);
  enumValue(value.kind, [
    "runtime-owner", "legacy-resolver", "provider-service", "fusion-service",
    "catalog-composition", "runtime-composition", "registry",
  ], `${id} dependency kind`);
  repositoryPath(value.path, `${id} dependency path`);
  enumValue(value.ownership, ["bundle", "prerequisite", "integration"], `${id} dependency ownership`);
  nonempty(value.owner_id, `${id} dependency owner`);
}

function parseMaturity(value, id) {
  exact(value, MATURITY_GATES, `${id} maturity`);
  for (const gate of MATURITY_GATES) {
    const entry = value[gate];
    exact(entry, ["applicability", "reason", "evidence"], `${id} maturity ${gate}`);
    enumValue(entry.applicability, ["required", "not-applicable"], `${id} maturity ${gate} applicability`);
    uniqueStrings(entry.evidence, `${id} maturity ${gate} evidence`, { empty: true });
    if (entry.applicability === "required" && entry.reason !== null) {
      throw new Error(`${id}: required maturity ${gate} cannot have a reason`);
    }
    if (entry.applicability === "not-applicable"
      && (!String(entry.reason ?? "").trim() || entry.evidence.length === 0)) {
      throw new Error(`${id}: inapplicable maturity ${gate} requires reason and evidence`);
    }
  }
}

function parseAuthorities(value, id) {
  exact(value, [
    "catalog_package", "catalog_entry_count", "catalog_constant_count", "documentation",
    "runtime_bindings", "runtime_constants", "native_link", "wasm_registry",
  ], `${id} expected authorities`);
  if (value.catalog_package !== null) repositoryPath(value.catalog_package, `${id} catalog package`);
  integer(value.catalog_entry_count, `${id} catalog entry count`);
  integer(value.catalog_constant_count, `${id} catalog constant count`);
  enumValue(value.documentation, ["catalog", "alias", "none"], `${id} documentation authority`);
  array(value.runtime_bindings, `${id} runtime bindings`, { empty: true }).forEach((entry) => {
    exact(entry, ["path", "function", "variant"], `${id} runtime binding`);
    repositoryPath(entry.path, `${id} runtime binding path`);
    nonempty(entry.function, `${id} runtime binding function`);
    nonempty(entry.variant, `${id} runtime binding variant`);
  });
  const constants = uniqueStrings(value.runtime_constants, `${id} runtime constants`, {
    empty: true,
    pattern: SAFE_IDENTITY,
  });
  if (JSON.stringify(constants) !== JSON.stringify([...constants].sort(compareCodePoint))) {
    throw new Error(`${id}: runtime constants must use canonical order`);
  }
  enumValue(value.native_link, ["required", "not-applicable"], `${id} native link`);
  enumValue(value.wasm_registry, ["required", "not-applicable"], `${id} wasm registry`);
}

function parseRemoval(value, id) {
  exact(value, ["kind", "path", "baseline_digest"], `${id} file removal`);
  if (value.kind !== "file") throw new Error(`${id}: expected removal must be a file`);
  repositoryPath(value.path, `${id} removal path`);
  digest(value.baseline_digest, `${id} removal baseline digest`);
}

function parseBaselineEvidence(value, id) {
  exact(value, ["kind", "path", "locator", "digest"], `${id} baseline evidence`);
  enumValue(value.kind, [
    "catalog", "runtime", "sidecar", "runtime-shadow", "resolver",
    "provider", "fusion", "test", "example",
  ], `${id} evidence kind`);
  repositoryPath(value.path, `${id} evidence path`);
  if (value.locator !== null) throw new Error(`${id}: baseline source-item locators are obsolete; compiled and lexical inventory rows are the typed item authority`);
  digest(value.digest, `${id} evidence digest`);
}

function parseVolumePolicy(value, role) {
  exact(value, [
    "role", "mount_path", "filesystem_id", "minimum_free_bytes", "pause_below_bytes",
    "maximum_observation_age_seconds",
  ], `${role} volume policy`);
  if (value.role !== role) throw new Error(`${role} volume policy has the wrong role`);
  absolutePath(value.mount_path, `${role} mount path`);
  filesystemIdentity(value.filesystem_id, `${role} filesystem id`);
  integer(value.minimum_free_bytes, `${role} minimum free bytes`, 1);
  integer(value.pause_below_bytes, `${role} pause below bytes`, 1);
  integer(value.maximum_observation_age_seconds, `${role} maximum observation age`, 1);
  if (value.pause_below_bytes < value.minimum_free_bytes) {
    throw new Error(`${role} pause threshold cannot be below minimum free bytes`);
  }
  return value;
}

function requireCanonicalUnique(keys, label) {
  if (new Set(keys).size !== keys.length) throw new Error(`${label} must be unique`);
  if (JSON.stringify(keys) !== JSON.stringify([...keys].sort(compareCodePoint))) {
    throw new Error(`${label} must use canonical order`);
  }
}
