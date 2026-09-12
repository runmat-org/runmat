import { compareCodePoint } from "./constants.mjs";
import { evidenceDigest } from "./evidence.mjs";
import { deepImmutable } from "./immutable.mjs";
import {
  array, digest, exact, repositoryPath, stableId,
} from "./schema.mjs";

export function parseSealReferences(value, label) {
  const references = array(value, label, { empty: true }).map((entry) => {
    exact(entry, ["path", "artifact_id", "digest", "bundle_id"], label);
    return {
      path: repositoryPath(entry.path, `${label} path`),
      artifact_id: stableId(entry.artifact_id, `${label} artifact id`),
      digest: digest(entry.digest, `${label} digest`),
      bundle_id: stableId(entry.bundle_id, `${label} bundle id`),
    };
  });
  const keys = references.map((entry) => `${entry.bundle_id}\0${entry.artifact_id}`);
  if (new Set(keys).size !== keys.length
    || JSON.stringify(keys) !== JSON.stringify([...keys].sort(compareCodePoint))) {
    throw new Error(`${label} must be unique and canonically ordered`);
  }
  return references;
}

export function buildAcceptedSealSet(controlManifestDigest, value, label = "accepted") {
  const seals = parseSealReferences(value, label);
  const payload = {
    schema_version: 1,
    kind: "runmat-builtin-migration-accepted-seal-set",
    authority: "derived-from-validated-seals",
    control_manifest_digest: digest(controlManifestDigest, `${label} control manifest digest`),
    seals,
  };
  return deepImmutable({ ...payload, digest: evidenceDigest(payload) });
}

export function buildBarrierSealSet(
  controlManifestDigest, bundleId, value, label = "barrier",
) {
  const seals = parseSealReferences(value, label);
  const payload = {
    schema_version: 1,
    kind: "runmat-builtin-migration-barrier-seal-set",
    authority: "derived-from-validated-seals",
    control_manifest_digest: digest(controlManifestDigest, `${label} control manifest digest`),
    bundle_id: stableId(bundleId, `${label} bundle id`),
    seals,
  };
  return deepImmutable({ ...payload, digest: evidenceDigest(payload) });
}

export function validateAcceptedSealSet(
  controlManifestDigest, value, observedDigest, label = "accepted",
) {
  const set = buildAcceptedSealSet(controlManifestDigest, value, label);
  if (set.digest !== digest(observedDigest, `${label} seal-set digest`)) {
    throw new Error(`${label} seal-set digest mismatch`);
  }
  return set;
}

export function validateBarrierSealSet(
  controlManifestDigest, bundleId, value, observedDigest, label = "barrier",
) {
  const set = buildBarrierSealSet(controlManifestDigest, bundleId, value, label);
  if (set.digest !== digest(observedDigest, `${label} seal-set digest`)) {
    throw new Error(`${label} seal-set digest mismatch`);
  }
  return set;
}
