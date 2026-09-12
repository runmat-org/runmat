import { compareCodePoint } from "../constants.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { deepImmutable } from "../immutable.mjs";
import { parseIntegrationProductRegistry } from "../integration-products.mjs";
import { parseFindingDispositions } from "../migration-findings.mjs";
import { assertValidatedTopologyView } from "../topology/freeze.mjs";
import { parseTargetPolicy } from "../target-policy.mjs";
import { digest, exact, kind } from "../schema.mjs";
import { assertValidatedControlOverlayScaffold } from "./scaffold.mjs";
import {
  assertEvidenceDigest, assertScaffoldTopologyBinding, parseExceptionManifestPolicy,
  parseProgramProfile, parseReviewedEvidence, validateStorageTargetCoverage,
} from "./policy-schema.mjs";

export const GLOBAL_CONTROL_REVIEW_KIND = "runmat-builtin-migration-global-control-review";
const PROGRAM = "RM-1064/C00-C07";
const VALIDATED_GLOBAL_REVIEWS = new WeakSet();

export function parseGlobalControlReview(value, scaffoldValue, topology, inventory) {
  const scaffold = assertValidatedControlOverlayScaffold(scaffoldValue);
  assertValidatedTopologyView(topology);
  assertScaffoldTopologyBinding(scaffold, topology);
  kind(value, 2, GLOBAL_CONTROL_REVIEW_KIND, "global control review");
  exact(value, ["schema_version", "kind", "authority", "program", "bindings", "program_profiles", "integration_products", "migration_findings", "exception_manifest", "target_policy", "storage_policy", "review", "digest"], "global control review");
  if (value.authority !== "reviewer-authored-development-input" || value.program !== PROGRAM) throw new Error("global control review has invalid authority or program");
  parseArtifactBindings(value.bindings, scaffold, topology);
  const programProfiles = parseProgramProfiles(value.program_profiles);
  const integrationProducts = parseIntegrationProductRegistry(value.integration_products, inventory);
  const bundles = new Map([...topology.bundles.keys()].map((id) => [id, true]));
  const currentFindings = scaffold.migration_finding_rows.map((entry) => entry.observations);
  const migrationFindings = parseFindingDispositions(value.migration_findings, bundles, currentFindings);
  parseExceptionManifestPolicy(value.exception_manifest, bundles);
  const targetPolicy = parseTargetPolicy(value.target_policy);
  const executionTargets = targetPolicy.migrationExecutionTargets;
  validateStorageTargetCoverage(value.storage_policy, executionTargets);
  parseReviewedEvidence(value.review, "global control review");
  assertEvidenceDigest(value, "global control review");
  const parsed = deepImmutable({
    value, digest: value.digest, programProfiles, integrationProducts, migrationFindings,
    targetPolicy, executionTargets,
  });
  VALIDATED_GLOBAL_REVIEWS.add(parsed);
  return parsed;
}

export function assertValidatedGlobalControlReview(value) {
  if (!VALIDATED_GLOBAL_REVIEWS.has(value)) throw new Error("operation requires an exact validated global control review");
  return value;
}

function parseArtifactBindings(value, scaffold, topology) {
  exact(value, ["scaffold_digest", "topology_digest", "migration_finding_rows_digest"], "global review bindings");
  if (digest(value.scaffold_digest, "global review scaffold digest") !== scaffold.digest) throw new Error("global review does not bind the exact authoring scaffold");
  if (digest(value.topology_digest, "global review topology digest") !== topology.digest) throw new Error("global review does not bind the exact reviewed topology");
  if (digest(value.migration_finding_rows_digest, "global review finding rows digest") !== evidenceDigest(scaffold.migration_finding_rows)) throw new Error("global review does not bind the exact scaffold finding rows");
}

function parseProgramProfiles(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("global program profiles must be an object");
  const ids = Object.keys(value);
  if (JSON.stringify(ids) !== JSON.stringify([...ids].sort(compareCodePoint))) throw new Error("global program profiles must use canonical profile-id order");
  const profiles = new Map();
  const programOwners = new Map();
  for (const id of ids) {
    const profile = parseProgramProfile(value[id], id);
    const programDigest = evidenceDigest(profile.program);
    const prior = programOwners.get(programDigest);
    if (prior) throw new Error(`${id}: program payload duplicates reviewed profile ${prior}`);
    programOwners.set(programDigest, id);
    profiles.set(id, profile);
  }
  return profiles;
}
