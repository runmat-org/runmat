import { digest, exact, repositoryPath } from "./schema.mjs";

const WORKFLOW_RANK = new Map([
  ["leased", 0], ["submitted", 1], ["integrated", 2], ["verified", 3],
]);

export function parseQueuePredecessor(value) {
  if (value === null) return null;
  exact(value, ["state_path", "state_digest", "checkpoint_path", "checkpoint_digest"], "queue state predecessor");
  return {
    state_path: repositoryPath(value.state_path, "queue predecessor state path"),
    state_digest: digest(value.state_digest, "queue predecessor state digest"),
    checkpoint_path: repositoryPath(value.checkpoint_path, "queue predecessor checkpoint path"),
    checkpoint_digest: digest(value.checkpoint_digest, "queue predecessor checkpoint digest"),
  };
}

export function validateMonotonicQueueTransition(value, predecessor, acceptedSeals, dependencies) {
  if (predecessor === null) {
    if (acceptedSeals.length) throw new Error("initial queue state cannot contain accepted seals");
    return;
  }
  const previous = new Map(predecessor.acceptedSeals.map((entry) => [entry.bundle_id, entry]));
  const current = new Map(acceptedSeals.map((entry) => [entry.bundle_id, entry]));
  for (const [bundleId, reference] of previous) {
    if (JSON.stringify(current.get(bundleId)) !== JSON.stringify(reference)) {
      throw new Error(`queue state removed or replaced accepted seal ${bundleId}`);
    }
  }
  const added = acceptedSeals.filter((entry) => !previous.has(entry.bundle_id));
  if (added.length !== 1) {
    throw new Error("queue state transition must accept exactly one new serialized seal");
  }
  const addition = dependencies.find((entry) => entry.bundleId === added[0].bundle_id);
  validateSerializedSealTransition(predecessor.acceptedSeals, added[0], addition?.accepted ?? null);
  for (const [bundleId, prior] of Object.entries(predecessor.value.bundles)) {
    const next = value.bundles[bundleId];
    if (!next || WORKFLOW_RANK.get(next.state) < WORKFLOW_RANK.get(prior.state)) {
      throw new Error(`${bundleId}: queue workflow state was removed or regressed`);
    }
  }
}

export function validateSerializedSealTransition(previousSeals, nextSeal, declaredAcceptedSeals) {
  if (JSON.stringify(declaredAcceptedSeals) !== JSON.stringify(previousSeals)) {
    throw new Error(`${nextSeal.bundle_id}: accepted seal was produced from a stale queue checkpoint`);
  }
}
