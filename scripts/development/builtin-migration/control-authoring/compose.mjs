import { compareCodePoint } from "../constants.mjs";
import { validateControlProjection } from "../control-projection-validation.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { deepImmutable } from "../immutable.mjs";
import { parseInventoryEvidence } from "../inventory.mjs";
import { digest, exact, kind } from "../schema.mjs";
import { assertValidatedTopologyView } from "../topology/freeze.mjs";
import { assertValidatedControlReviewSet } from "./review-set.mjs";
import { assertValidatedControlOverlayScaffold } from "./scaffold.mjs";

const PROGRAM = "RM-1064/C00-C07";
const COHORTS = Object.freeze([
  ["C00", "prerequisite"], ["C01", "A"], ["C02", "B"], ["C03", "C"],
  ["C04", "D"], ["C05", "E"], ["C06", "F"], ["C07", "G"],
]);
const VALIDATED_CONTROL_CANDIDATES = new WeakSet();

export function composeControlCandidate({ inventory: inventoryValue, topology, scaffold, reviewSet }) {
  const inventory = parseInventoryEvidence(inventoryValue);
  assertValidatedTopologyView(topology);
  assertValidatedControlOverlayScaffold(scaffold);
  assertValidatedControlReviewSet(reviewSet);
  assertBindings(inventory, topology, scaffold, reviewSet);

  const bundleControls = {};
  const identityControlsById = new Map();
  for (const bundleId of [...topology.bundles.keys()].sort(compareCodePoint)) {
    const review = reviewSet.bundleReviews.get(bundleId);
    if (!review) throw new Error(`${bundleId}: validated control review set has no bundle review`);
    bundleControls[bundleId] = expandBundleControl(review.bundleControl, reviewSet.globalReview);
    for (const identity of topology.bundles.get(bundleId).identities) {
      if (identityControlsById.has(identity)) throw new Error(`${identity}: duplicate reviewed identity control`);
      const control = review.identityControls.get(identity);
      if (!control) throw new Error(`${identity}: bundle review has no identity control`);
      identityControlsById.set(identity, structuredClone(control));
    }
  }
  const expectedIdentities = [...topology.identities.keys()].sort(compareCodePoint);
  if (JSON.stringify([...identityControlsById.keys()].sort(compareCodePoint)) !== JSON.stringify(expectedIdentities)) {
    throw new Error("reviewed identity controls do not exactly cover the topology identity set");
  }
  const identityControls = Object.fromEntries(expectedIdentities.map((identity) => [identity, identityControlsById.get(identity)]));

  const inputs = inputDigests(inventory, topology, scaffold, reviewSet);
  const payload = {
    schema_version: 1,
    kind: "runmat-builtin-migration-control-candidate",
    authority: "deterministically-composed-unreviewed-candidate",
    program: PROGRAM,
    inputs,
    baseline_context: {
      source_digest: inventory.source.digest,
      dispositions_digest: inventory.dispositions_digest,
      migration_findings_digest: inventory.migration_findings_digest,
      compiled_target: {
        operating_system: inventory.compiled_inventory.build.operating_system,
        architecture: inventory.compiled_inventory.build.architecture,
      },
    },
    cohorts: COHORTS.map(([id, semantic], order) => ({ id, semantic, order })),
    bundle_controls: bundleControls,
    identity_controls: identityControls,
    migration_findings: structuredClone(reviewSet.globalReview.value.migration_findings),
    exception_manifest: structuredClone(reviewSet.globalReview.value.exception_manifest),
    execution_targets: structuredClone(reviewSet.globalReview.value.execution_targets),
    storage_policy: structuredClone(reviewSet.globalReview.value.storage_policy),
  };
  validateControlProjection({
    inventory,
    topology,
    baselineContext: payload.baseline_context,
    bundleControls: payload.bundle_controls,
    identityControls: payload.identity_controls,
    migrationFindings: payload.migration_findings,
    exceptionManifest: payload.exception_manifest,
    executionTargets: payload.execution_targets,
    storagePolicy: payload.storage_policy,
  });
  const candidate = deepImmutable({ ...payload, digest: evidenceDigest(payload) });
  VALIDATED_CONTROL_CANDIDATES.add(candidate);
  return candidate;
}

export function parseControlCandidate(value, expected) {
  assertValidatedControlCandidate(expected);
  kind(value, 1, "runmat-builtin-migration-control-candidate", "control candidate");
  exact(value, ["schema_version", "kind", "authority", "program", "inputs", "baseline_context", "cohorts", "bundle_controls", "identity_controls", "migration_findings", "exception_manifest", "execution_targets", "storage_policy", "digest"], "control candidate");
  if (value.authority !== "deterministically-composed-unreviewed-candidate" || value.program !== PROGRAM) {
    throw new Error("control candidate has invalid authority or program");
  }
  digest(value.digest, "control candidate digest");
  const { digest: _ignored, ...payload } = value;
  if (evidenceDigest(payload) !== value.digest) throw new Error("control candidate digest mismatch");
  if (JSON.stringify(value) !== JSON.stringify(expected)) {
    throw new Error("control candidate differs from deterministic recomposition");
  }
  return expected;
}

export function assertValidatedControlCandidate(value) {
  if (!VALIDATED_CONTROL_CANDIDATES.has(value)) {
    throw new Error("operation requires the exact deterministically composed control candidate");
  }
  return value;
}

export function controlCandidateInputDigests(candidate) {
  return structuredClone(candidate.inputs);
}

function expandBundleControl(control, globalReview) {
  const gatePlans = control.gate_plans.map(({ program_profile_id: profileId, ...plan }) => {
    const profile = globalReview.programProfiles.get(profileId);
    if (!profile) throw new Error(`gate plan references unknown reviewed program profile ${profileId}`);
    return { gate: plan.gate, program: structuredClone(profile.program), arguments: [...plan.arguments], working_directory: plan.working_directory, parser: plan.parser, expected_artifact_roles: [...plan.expected_artifact_roles] };
  });
  return {
    prerequisites: structuredClone(control.prerequisites),
    additional_authored_write_set: structuredClone(control.additional_authored_write_set),
    integration_outputs: structuredClone(control.integration_outputs),
    gate_plans: gatePlans,
    owner_role: control.owner_role,
    complexity: structuredClone(control.complexity),
    review: structuredClone(control.review),
  };
}

function inputDigests(inventory, topology, scaffold, reviewSet) {
  return {
    baseline_inventory_digest: inventory.digest,
    control_draft_digest: scaffold.bindings.control_draft_digest,
    reviewed_topology_digest: topology.digest,
    control_scaffold_digest: scaffold.digest,
    review_set_digest: reviewSet.digest,
    global_review_digest: reviewSet.globalReview.digest,
    bundle_review_digests: [...reviewSet.bundleReviews.entries()]
      .sort(([left], [right]) => compareCodePoint(left, right))
      .map(([bundle_id, review]) => ({ bundle_id, digest: review.digest })),
  };
}

function assertBindings(inventory, topology, scaffold, reviewSet) {
  if (scaffold.bindings.inventory_digest !== inventory.digest
    || scaffold.bindings.reviewed_topology_digest !== topology.digest
    || reviewSet.value.bindings.inventory_digest !== inventory.digest
    || reviewSet.value.bindings.scaffold_digest !== scaffold.digest
    || reviewSet.value.bindings.topology_digest !== topology.digest) {
    throw new Error("control candidate inputs do not share the exact inventory, scaffold, and topology bindings");
  }
}
