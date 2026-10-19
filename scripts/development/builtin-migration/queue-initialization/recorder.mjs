import path from "node:path";

import {
  assertAuthorityLoadSession, loadJsonArtifact, loadedJsonValue,
  revalidateObservedArtifacts,
} from "../authority-loading/index.mjs";
import {
  EvidenceTargetExistsError, publishEvidenceBytes,
} from "../atomic-evidence-publication.mjs";
import { assertValidatedControl } from "../control.mjs";
import { contentDigest } from "../evidence.mjs";
import { loadQueueAuthority } from "../queue-authority/index.mjs";
import { deriveInitialQueue } from "./derive.mjs";
import { initialQueuePaths } from "./paths.mjs";
import { assertLoadedInitialQueueReview } from "./review.mjs";

export function recordInitialQueue({ session: sessionValue, control: controlValue, review: reviewValue }) {
  const session = assertAuthorityLoadSession(sessionValue);
  const control = assertValidatedControl(controlValue);
  const review = assertLoadedInitialQueueReview(reviewValue, { session, control });
  const derived = deriveInitialQueue({ session, control, review });
  const paths = initialQueuePaths(control.digest);
  revalidateObservedArtifacts(session);
  publishOrVerify(session, paths.state, derived.stateValue);
  revalidateObservedArtifacts(session);
  publishOrVerify(session, paths.checkpoint, derived.checkpointValue);
  return loadQueueAuthority({
    session,
    statePath: paths.state,
    checkpointPath: paths.checkpoint,
    trustedCheckpointDigest: derived.checkpointValue.digest,
    control,
  });
}

function publishOrVerify(session, relativePath, value) {
  const bytes = `${JSON.stringify(value, null, 2)}\n`;
  try {
    publishEvidenceBytes(path.join(session.root.path, ...relativePath.split("/")), bytes, {
      createParentDirectories: true,
    });
  } catch (error) {
    if (!(error instanceof EvidenceTargetExistsError)) throw error;
    const artifact = loadJsonArtifact(session, relativePath, "existing initial queue artifact");
    if (artifact.contentDigest !== contentDigest(Buffer.from(bytes, "utf8"))
      || JSON.stringify(loadedJsonValue(artifact, session.root)) !== JSON.stringify(value)) {
      throw new Error(`${relativePath}: existing initial queue artifact differs from expected bytes`);
    }
  }
}
