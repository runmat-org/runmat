import { compareCodePoint } from "./constants.mjs";
import { assertValidatedControl } from "./control.mjs";
import { stableId } from "./schema.mjs";

export function requiredBarrierBundleIds(control, bundleId) {
  assertValidatedControl(control);
  const bundle = control.bundles.get(stableId(bundleId, "barrier bundle id"));
  if (!bundle) throw new Error(`barrier bundle set references unknown bundle ${bundleId}`);
  const result = new Set(bundle.prerequisites.map((entry) => entry.bundle_id));
  const cohortIds = [...new Set(bundle.identities.map((id) => control.identities.get(id)?.cohort))]
    .filter(Boolean);
  if (cohortIds.length !== 1) throw new Error(`${bundle.id}: barrier seal set requires one cohort`);
  const cohort = control.cohorts.get(cohortIds[0]);
  if (!cohort) throw new Error(`${bundle.id}: barrier seal set references an unknown cohort`);
  for (const candidate of control.bundles.values()) {
    const candidates = [...new Set(candidate.identities
      .map((id) => control.identities.get(id)?.cohort))].filter(Boolean);
    if (candidates.length !== 1) continue;
    if (control.cohorts.get(candidates[0])?.order < cohort.order) result.add(candidate.id);
  }
  return [...result].sort(compareCodePoint);
}
