import { canonicalAuthorityPath } from "../authority-loading/index.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { digest, stableId } from "../schema.mjs";

export function pilotWorkSessionId(pilotIdValue, bundleIdValue) {
  const pilotId = stableId(pilotIdValue, "pilot work-session pilot id");
  const bundleId = stableId(bundleIdValue, "pilot work-session bundle id");
  const identity = evidenceDigest({ pilot_id: pilotId, bundle_id: bundleId })
    .slice("sha256:".length);
  return `pilot-session-${identity}`;
}

export function sessionPaths(
  controlDigestValue, policyDigestValue, pilotIdValue, bundleIdValue,
) {
  const controlDigest = digest(controlDigestValue, "pilot work-session control digest")
    .slice("sha256:".length);
  const policyDigest = digest(policyDigestValue, "pilot work-session policy digest")
    .slice("sha256:".length);
  const sessionId = pilotWorkSessionId(pilotIdValue, bundleIdValue);
  const directory = canonicalAuthorityPath(
    `pilot-work-sessions/${controlDigest}/${policyDigest}/${sessionId}`,
    "pilot work-session directory",
  );
  return Object.freeze({
    sessionId,
    directory,
    start: `${directory}/start.json`,
    completion: `${directory}/completion.json`,
  });
}
