import { compareCodePoint } from "../constants.mjs";
import {
  parseBundleRemovals, validateCompleteBundleBaselineEvidence,
} from "../baseline-evidence.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { executionTargetKey, parseExecutionTargets } from "../execution-target.mjs";
import { GATE_PRODUCERS } from "../gate-kinds.mjs";
import { GATE_PARSERS, parseGatePlans, parseGateProgram } from "../gate-plan.mjs";
import { parseIntegrationProductReferences } from "../integration-products.mjs";
import { parseUniqueReviewedEvidence } from "../reviewed-evidence.mjs";
import { parseSourceMigrations } from "../source-migrations.mjs";
import {
  parseIdentityForms, parseImplementationAuthority, parsePublicIdentity,
} from "../identity-authority.mjs";
import {
  absolutePath, array, digest, enumValue, exact, filesystemIdentity,
  identity, integer, nonempty, repositoryPath, stableId, uniqueStrings,
} from "../schema.mjs";

export { executionTargetKey, parseExecutionTargets } from "../execution-target.mjs";

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
  "provider", "fusion", "link-reachability", "foreign", "parallel", "host",
  "documentation", "native-example", "browser-example", "browser-runtime", "tests",
  "wasm-registry",
]);

export function parseBundleControlPolicy(value, id, identities, inventory) {
  return parseBundlePolicy(
    value,
    id,
    identities,
    inventory,
    (plans) => parseGateReferences(plans, id),
  );
}

export function parseOperationalBundleControlPolicy(value, id, inventory, identities) {
  return parseBundlePolicy(
    value,
    id,
    identities,
    inventory,
    (plans) => parseGatePlans(plans, id, inventory),
  );
}

function parseBundlePolicy(value, id, identities, inventory, parsePlans) {
  exact(value, ["prerequisites", "additional_authored_write_set", "integration_product_refs", "module_composition_transition", "source_migrations", "expected_removals", "baseline_evidence", "gate_plans", "owner_role", "complexity", "review"], `${id} bundle control`);
  array(value.prerequisites, `${id} prerequisites`, { empty: true }).forEach((entry) => {
    exact(entry, ["bundle_id", "kind"], `${id} prerequisite`);
    stableId(entry.bundle_id, `${id} prerequisite bundle`);
    enumValue(entry.kind, ["representation", "infrastructure", "semantic", "cohort"], `${id} prerequisite kind`);
  });
  array(value.additional_authored_write_set, `${id} additional authored write set`, { empty: true }).forEach((entry) => parseScope(entry, `${id} additional authored scope`));
  parseIntegrationProductReferences(value.integration_product_refs, id);
  if (value.module_composition_transition !== null
    && (!value.module_composition_transition || typeof value.module_composition_transition !== "object"
      || Array.isArray(value.module_composition_transition))) {
    throw new Error(`${id} module composition transition must be an object or null`);
  }
  const baselineEvidence = validateCompleteBundleBaselineEvidence(
    value.baseline_evidence,
    inventory,
    identities,
    id,
  );
  parseSourceMigrations(value.source_migrations, id, identities, inventory, baselineEvidence);
  parseBundleRemovals(value.expected_removals, id, identities, baselineEvidence);
  parsePlans(value.gate_plans);
  nonempty(value.owner_role, `${id} owner role`);
  parseComplexity(value.complexity, `${id} complexity`);
  parseUniqueReviewedEvidence(value.review, `${id} bundle control review`);
  return value;
}

export function parseIdentityControlPolicy(value, id) {
  const normalized = identity(id, "identity control id").toLowerCase();
  exact(value, [
    "public_identity", "forms", "implementation", "shared_dependencies", "complexity", "maturity",
    "expected_authorities", "owner", "review",
  ], `${id} identity control`);
  parsePublicIdentity(value.public_identity, normalized);
  parseIdentityForms(value.forms, normalized);
  parseImplementationAuthority(value.implementation, normalized);
  const dependencyKeys = array(value.shared_dependencies, `${id} shared dependencies`, { empty: true })
    .map((entry) => {
      parseSharedDependency(entry, id);
      return `${entry.ownership}\0${entry.owner_id}\0${entry.kind}\0${entry.path}`;
    });
  requireCanonicalUnique(dependencyKeys, `${id} shared dependencies`);
  parseComplexity(value.complexity, `${id} complexity`);
  parseMaturity(value.maturity, id);
  parseAuthorities(value.expected_authorities, id);
  validateAuthorityCoherence(value, id);
  nonempty(value.owner, `${id} owner`);
  parseUniqueReviewedEvidence(value.review, `${id} identity control review`);
  return value;
}

export function parseProgramProfile(value, id) {
  stableId(id, "global program profile id");
  exact(value, ["program", "review"], `${id} global program profile`);
  parseGateProgram(value.program, id);
  parseUniqueReviewedEvidence(value.review, `${id} global program profile review`);
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
  parseUniqueReviewedEvidence(value.review, "exception manifest review");
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

export function validateStorageTargetCoverage(storagePolicy, executionTargets) {
  parseStoragePolicy(storagePolicy);
  const profileTargets = [...new Set(Object.values(storagePolicy.host_profiles).map(executionTargetKey))]
    .sort(compareCodePoint);
  const reviewedTargets = parseExecutionTargets(executionTargets).map(executionTargetKey);
  if (JSON.stringify(profileTargets) !== JSON.stringify(reviewedTargets)) {
    throw new Error("storage host profiles must exactly cover every reviewed execution target");
  }
}

function parseGateReferences(value, id) {
  const keys = array(value, `${id} gate plans`, { empty: true }).map((entry) => {
    exact(entry, ["gate", "program_profile_id", "arguments", "working_directory", "parser", "expected_artifact_roles"], `${id} gate plan`);
    const gate = enumValue(entry.gate, Object.keys(GATE_PRODUCERS), `${id} gate`);
    stableId(entry.program_profile_id, `${id} gate program profile`);
    if (entry.working_directory !== "repository") throw new Error(`${id}: gate working directory must be repository`);
    enumValue(entry.parser, GATE_PARSERS, `${id} gate parser`);
    array(entry.arguments, `${id} gate arguments`, { empty: true }).forEach((argument) => {
      nonempty(argument, `${id} gate argument`);
      if (argument.includes("\0")) throw new Error(`${id}: gate arguments cannot contain NUL bytes`);
    });
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
    "catalog_package", "catalog_alias_package", "catalog_constant_package",
    "catalog_entry_count", "catalog_constant_count", "documentation",
    "native_link", "wasm_registry",
  ], `${id} expected authorities`);
  if (value.catalog_package !== null) repositoryPath(value.catalog_package, `${id} catalog package`);
  if (value.catalog_alias_package !== null) repositoryPath(value.catalog_alias_package, `${id} catalog alias package`);
  if (value.catalog_constant_package !== null) repositoryPath(value.catalog_constant_package, `${id} catalog constant package`);
  integer(value.catalog_entry_count, `${id} catalog entry count`);
  integer(value.catalog_constant_count, `${id} catalog constant count`);
  enumValue(value.documentation, ["catalog", "alias", "none"], `${id} documentation authority`);
  enumValue(value.native_link, ["required", "not-applicable"], `${id} native link`);
  enumValue(value.wasm_registry, ["required", "not-applicable"], `${id} wasm registry`);
}

function validateAuthorityCoherence(value, id) {
  const authorities = value.expected_authorities;
  const callable = value.forms.callable_spellings.length > 0;
  const constant = value.forms.constant_spellings.length > 0;
  const required = (maturity) => value.maturity[maturity].applicability === "required";
  const hasCatalogEntry = authorities.catalog_entry_count > 0;
  const hasCatalogConstant = authorities.catalog_constant_count > 0;
  if ((authorities.catalog_package !== null) !== hasCatalogEntry) {
    throw new Error(`${id}: catalog package and entry count must agree`);
  }
  if ((authorities.catalog_constant_package !== null) !== hasCatalogConstant) {
    throw new Error(`${id}: constant catalog package and count must agree`);
  }
  if (value.public_identity.kind === "alias") {
    if (!callable || constant || hasCatalogEntry || hasCatalogConstant
      || authorities.catalog_package !== null || authorities.catalog_alias_package === null
      || authorities.catalog_constant_package !== null || authorities.documentation !== "alias"
      || authorities.native_link !== "not-applicable"
      || authorities.wasm_registry !== "not-applicable") {
      throw new Error(`${id}: alias authorities must contain only one callable alias declaration`);
    }
  } else {
    if (authorities.catalog_alias_package !== null) {
      throw new Error(`${id}: non-alias identity cannot own an alias declaration package`);
    }
    if (hasCatalogConstant !== constant) {
      throw new Error(`${id}: constant forms and catalog constant authority must agree`);
    }
    if (value.public_identity.kind === "primary") {
      if (hasCatalogEntry !== callable) {
        throw new Error(`${id}: primary callable forms require one catalog entry`);
      }
      if (authorities.documentation !== (hasCatalogEntry ? "catalog" : "none")) {
        throw new Error(`${id}: primary documentation authority differs from its catalog entry`);
      }
    } else {
      if (hasCatalogConstant || constant || authorities.documentation !== "none") {
        throw new Error(`${id}: internal identity cannot own public constant or documentation authority`);
      }
      if (hasCatalogEntry !== callable) {
        throw new Error(`${id}: internal callable forms require one hidden catalog entry`);
      }
    }
    if (callable !== (authorities.native_link === "required")) {
      throw new Error(`${id}: callable implementation and native-link authority must agree`);
    }
  }
  const hasCatalogContract = hasCatalogEntry || hasCatalogConstant
    || authorities.catalog_alias_package !== null;
  if (required("catalog-contract") !== hasCatalogContract) {
    throw new Error(`${id}: catalog-contract maturity differs from expected catalog authority`);
  }
  if (required("documentation") !== (authorities.documentation !== "none")) {
    throw new Error(`${id}: documentation maturity differs from expected documentation authority`);
  }
  if (required("link-reachability") !== (authorities.native_link === "required")) {
    throw new Error(`${id}: link maturity differs from expected native-link authority`);
  }
  if (required("wasm-registry") !== (authorities.wasm_registry === "required")) {
    throw new Error(`${id}: WASM maturity differs from expected registry authority`);
  }
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
