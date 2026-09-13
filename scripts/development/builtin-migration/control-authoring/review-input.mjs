import { evidenceDigest } from "../evidence.mjs";

export function sealReviewerAuthoredPayload(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  if (Object.hasOwn(value, "digest")) {
    throw new Error(`${label} must not supply its own digest`);
  }
  return { ...value, digest: evidenceDigest(value) };
}
