export function recordChildChange(changesByChild, key, bundleId, semantic, prerequisiteOrdered) {
  const priorChanges = changesByChild.get(key) ?? [];
  for (const prior of priorChanges) {
    if (!prerequisiteOrdered(bundleId, prior.bundle_id)) {
      throw new Error(
        `${bundleId}: composition child is also changed by ${prior.bundle_id} and is not prerequisite-ordered`,
      );
    }
    if (semantic && prior.semantic) {
      throw new Error(
        `${bundleId}: composition child is also semantically changed by ${prior.bundle_id}`,
      );
    }
  }
  priorChanges.push({ bundle_id: bundleId, semantic });
  changesByChild.set(key, priorChanges);
}
