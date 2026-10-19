import path from "node:path";

import {
  assertAuthorityLoadSession, loadJsonArtifact, loadedJsonValue,
} from "../authority-loading/index.mjs";
import {
  EvidenceTargetExistsError, publishEvidenceBytes,
} from "../atomic-evidence-publication.mjs";
import { assertValidatedControl } from "../control.mjs";
import { contentDigest } from "../evidence.mjs";
import { assertLoadedPilotEvaluation } from "../pilot-evaluation.mjs";
import { loadPilotTransition } from "./loader.mjs";
import { pilotTransitionPaths } from "./paths.mjs";
import { derivePilotTransition } from "./validation.mjs";

export function recordPilotTransition({ session: sessionValue, control: controlValue, evaluation: value }) {
  const session = assertAuthorityLoadSession(sessionValue);
  const control = assertValidatedControl(controlValue);
  const evaluation = assertLoadedPilotEvaluation(value, { session, control });
  const transition = derivePilotTransition({ session, control, evaluation });
  const paths = pilotTransitionPaths(evaluation.reference.digest);
  publishOrVerify(session, paths.state, transition.state);
  publishOrVerify(session, paths.checkpoint, transition.checkpoint);
  return loadPilotTransition({ session, control, evaluation });
}

function publishOrVerify(session, relativePath, value) {
  const bytes = `${JSON.stringify(value, null, 2)}\n`;
  try {
    publishEvidenceBytes(path.join(session.root.path, ...relativePath.split("/")), bytes, {
      createParentDirectories: true,
    });
  } catch (error) {
    if (!(error instanceof EvidenceTargetExistsError)) throw error;
    const artifact = loadJsonArtifact(session, relativePath, "existing pilot transition artifact");
    if (artifact.contentDigest !== contentDigest(Buffer.from(bytes, "utf8"))
      || JSON.stringify(loadedJsonValue(artifact, session.root)) !== JSON.stringify(value)) {
      throw new Error(`${relativePath}: existing pilot transition artifact differs from expected bytes`);
    }
  }
}
