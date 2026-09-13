import { compareCodePoint } from "./constants.mjs";
import { evidenceDigest } from "./evidence.mjs";
import { deepImmutable } from "./immutable.mjs";
import {
  array, enumValue, exact, integer, kind, stableId,
} from "./schema.mjs";
import { assertValidatedTopologyView } from "./topology/freeze.mjs";
import {
  canonicalCohorts, canonicalStrings, parseCohort, parseReviewedEvidence,
} from "./topology/schema.mjs";

export const PILOT_POLICY_KIND = "runmat-builtin-migration-pilot-policy";
export const PILOT_POLICY_VERSION = 1;

const MAXIMUM_SAFE_ELAPSED_HOURS = Math.floor(Number.MAX_SAFE_INTEGER / 3_600_000);
const REQUIRED_BELOW_TARGET_FINDINGS = Object.freeze([
  "concrete-limiter", "revised-forecast",
]);
const IDENTITY_DISPOSITIONS = new Set(["canonical", "alias", "internal"]);

export function parsePilotPolicy(value, topologyValue, prerequisitesByBundle) {
  const topology = assertValidatedTopologyView(topologyValue);
  const prerequisites = parsePrerequisiteAuthority(prerequisitesByBundle, topology);
  kind(value, PILOT_POLICY_VERSION, PILOT_POLICY_KIND, "pilot policy");
  exact(value, [
    "schema_version", "kind", "pilot_id", "waves", "derived_counts",
    "admission", "review",
  ], "pilot policy");
  const pilotId = stableId(value.pilot_id, "pilot id");
  const waves = parseWaves(value.waves, topology);
  const waveByBundle = new Map();
  for (const wave of waves) {
    for (const bundleId of wave.bundle_ids) {
      if (waveByBundle.has(bundleId)) {
        throw new Error(`${bundleId}: pilot bundle occurs in more than one wave`);
      }
      waveByBundle.set(bundleId, wave.order);
    }
  }
  validatePrerequisiteClosure(waveByBundle, prerequisites);
  const counts = deriveCounts(topology, waveByBundle);
  parseDerivedCounts(value.derived_counts, counts);
  parseAdmission(value.admission);
  parseReviewedEvidence(value.review, "pilot policy review");
  return deepImmutable({
    value,
    digest: evidenceDigest(value),
    pilotId,
    waves,
    bundleIds: [...waveByBundle.keys()].sort(compareCodePoint),
    waveByBundle,
    counts,
  });
}

function parseWaves(value, topology) {
  const waveIds = new Set();
  return array(value, "pilot waves").map((entry, index) => {
    const label = `pilot wave ${index + 1}`;
    exact(entry, ["wave_id", "order", "bundle_ids"], label);
    const waveId = stableId(entry.wave_id, `${label} id`);
    if (waveIds.has(waveId)) throw new Error("pilot wave ids must be unique");
    waveIds.add(waveId);
    integer(entry.order, `${label} order`, 1);
    if (entry.order !== index + 1) {
      throw new Error("pilot waves must use contiguous order beginning at 1");
    }
    const bundleIds = canonicalStrings(entry.bundle_ids, `${label} bundle ids`)
      .map((bundleId) => stableId(bundleId, `${label} bundle id`));
    for (const bundleId of bundleIds) {
      if (!topology.bundles.has(bundleId)) {
        throw new Error(`${bundleId}: pilot wave references unknown topology bundle`);
      }
    }
    return { wave_id: waveId, order: entry.order, bundle_ids: bundleIds };
  });
}

function parsePrerequisiteAuthority(value, topology) {
  if (!(value instanceof Map)) {
    throw new Error("pilot prerequisite authority must be a topology-complete Map");
  }
  const expected = [...topology.bundles.keys()].sort(compareCodePoint);
  const actual = [...value.keys()].sort(compareCodePoint);
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error("pilot prerequisite authority must cover every topology bundle exactly");
  }
  const result = new Map();
  for (const bundleId of expected) {
    const seen = new Set();
    const rows = array(value.get(bundleId), `${bundleId} pilot prerequisites`, { empty: true })
      .map((entry, index) => {
        exact(entry, ["bundle_id", "kind"], `${bundleId} pilot prerequisite ${index + 1}`);
        const prerequisiteId = stableId(entry.bundle_id, `${bundleId} pilot prerequisite bundle`);
        if (!topology.bundles.has(prerequisiteId)) {
          throw new Error(`${bundleId}: prerequisite references unknown topology bundle ${prerequisiteId}`);
        }
        if (prerequisiteId === bundleId) throw new Error(`${bundleId}: pilot prerequisite cannot be self-referential`);
        if (seen.has(prerequisiteId)) throw new Error(`${bundleId}: pilot prerequisites must be unique`);
        seen.add(prerequisiteId);
        const prerequisiteKind = enumValue(
          entry.kind,
          ["representation", "infrastructure", "semantic", "cohort"],
          `${bundleId} pilot prerequisite kind`,
        );
        return { bundle_id: prerequisiteId, kind: prerequisiteKind };
      });
    const keys = rows.map((entry) => `${entry.bundle_id}\0${entry.kind}`);
    if (JSON.stringify(keys) !== JSON.stringify([...keys].sort(compareCodePoint))) {
      throw new Error(`${bundleId}: pilot prerequisites must use canonical order`);
    }
    result.set(bundleId, rows);
  }
  return result;
}

function validatePrerequisiteClosure(waveByBundle, prerequisites) {
  for (const [bundleId, wave] of waveByBundle) {
    for (const prerequisite of prerequisites.get(bundleId)) {
      const prerequisiteWave = waveByBundle.get(prerequisite.bundle_id);
      if (prerequisiteWave === undefined) {
        throw new Error(`${bundleId}: pilot omits prerequisite ${prerequisite.bundle_id}`);
      }
      if (prerequisiteWave >= wave) {
        throw new Error(`${bundleId}: prerequisite ${prerequisite.bundle_id} must be in an earlier pilot wave`);
      }
    }
  }
}

function deriveCounts(topology, waveByBundle) {
  const cohortCounts = new Map();
  let identities = 0;
  let publicIdentities = 0;
  let internalIdentities = 0;
  for (const bundleId of waveByBundle.keys()) {
    const bundle = topology.bundles.get(bundleId);
    const cohort = parseCohort(bundle.cohort, `${bundleId} pilot cohort`);
    if (!cohortCounts.has(cohort)) {
      cohortCounts.set(cohort, {
        cohort, bundles: 0, identities: 0, public_identities: 0, internal_identities: 0,
      });
    }
    const cohortCount = cohortCounts.get(cohort);
    cohortCount.bundles += 1;
    for (const identityId of bundle.identities) {
      const identity = topology.identities.get(identityId);
      if (!identity || identity.bundle_id !== bundleId || identity.cohort !== cohort) {
        throw new Error(`${bundleId}: topology identity membership is inconsistent for ${identityId}`);
      }
      const disposition = identity.disposition?.kind;
      if (!IDENTITY_DISPOSITIONS.has(disposition)) {
        throw new Error(`${identityId}: pilot identity has unsupported topology disposition`);
      }
      const isInternal = disposition === "internal";
      identities += 1;
      cohortCount.identities += 1;
      if (isInternal) {
        internalIdentities += 1;
        cohortCount.internal_identities += 1;
      } else {
        publicIdentities += 1;
        cohortCount.public_identities += 1;
      }
    }
  }
  const cohorts = [...cohortCounts.values()]
    .sort((left, right) => compareCodePoint(left.cohort, right.cohort));
  return {
    bundles: waveByBundle.size,
    identities,
    public_identities: publicIdentities,
    internal_identities: internalIdentities,
    cohorts,
  };
}

function parseDerivedCounts(value, expected) {
  exact(value, [
    "bundles", "identities", "public_identities", "internal_identities", "cohorts",
  ], "pilot derived counts");
  for (const field of ["bundles", "identities", "public_identities", "internal_identities"]) {
    integer(value[field], `pilot derived ${field}`, 0);
    if (value[field] !== expected[field]) {
      throw new Error(`pilot derived ${field} does not match topology`);
    }
  }
  const cohorts = array(value.cohorts, "pilot derived cohort counts").map((entry, index) => {
    const label = `pilot derived cohort count ${index + 1}`;
    exact(entry, [
      "cohort", "bundles", "identities", "public_identities", "internal_identities",
    ], label);
    parseCohort(entry.cohort, `${label} cohort`);
    for (const field of ["bundles", "identities", "public_identities", "internal_identities"]) {
      integer(entry[field], `${label} ${field}`, 0);
    }
    return entry;
  });
  canonicalCohorts(cohorts.map((entry) => entry.cohort), "pilot derived cohorts");
  if (JSON.stringify(cohorts) !== JSON.stringify(expected.cohorts)) {
    throw new Error("pilot derived cohort counts do not match topology");
  }
}

function parseAdmission(value) {
  exact(value, [
    "minimum_public_identities_per_aggregate_hour", "maximum_elapsed_hours",
    "required_waived_gate_count", "ordinary_gate_policy", "below_target_obligation",
  ], "pilot admission policy");
  positiveFiniteNumber(
    value.minimum_public_identities_per_aggregate_hour,
    "pilot minimum public identities per aggregate hour",
  );
  positiveFiniteNumber(value.maximum_elapsed_hours, "pilot maximum elapsed hours");
  if (value.maximum_elapsed_hours > MAXIMUM_SAFE_ELAPSED_HOURS) {
    throw new Error("pilot maximum elapsed hours must remain safe for millisecond timing arithmetic");
  }
  integer(value.required_waived_gate_count, "pilot required waived gate count", 0);
  if (value.required_waived_gate_count !== 0) {
    throw new Error("pilot required waived gate count must be zero");
  }
  if (value.ordinary_gate_policy !== "all-required-gates-must-pass") {
    throw new Error("pilot must retain all ordinary required gates");
  }
  const obligation = value.below_target_obligation;
  exact(obligation, ["production_transition", "required_findings"], "pilot below-target obligation");
  if (obligation.production_transition !== "requires-reviewer-accepted-obligation") {
    throw new Error("below-target production transition must require reviewer acceptance");
  }
  const findings = canonicalStrings(
    obligation.required_findings,
    "pilot below-target required findings",
  );
  if (JSON.stringify(findings) !== JSON.stringify(REQUIRED_BELOW_TARGET_FINDINGS)) {
    throw new Error("below-target obligation must require a concrete limiter and revised forecast");
  }
}

function positiveFiniteNumber(value, label) {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new Error(`${label} must be a finite positive number`);
  }
  return value;
}
