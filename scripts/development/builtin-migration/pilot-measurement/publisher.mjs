import path from "node:path";

import {
  assertAuthorityLoadSession, revalidateObservedArtifacts,
} from "../authority-loading/index.mjs";
import { publishEvidenceBytes } from "../atomic-evidence-publication.mjs";
import { assertValidatedControl } from "../control.mjs";
import { reconstructPilotMeasurement, derivedResultPayload } from "./reconstruction.mjs";
import { pilotMeasurementResultPath } from "./paths.mjs";
import { loadPilotMeasurementReview } from "./review-loader.mjs";
import { loadPilotMeasurement } from "./result-loader.mjs";
import { withSelfDigest } from "./schema.mjs";

export function publishPilotMeasurement({
  session: sessionValue, manifestReference, control: controlValue, repository,
}) {
  const session = assertAuthorityLoadSession(sessionValue);
  const control = assertValidatedControl(controlValue);
  const review = loadPilotMeasurementReview({
    session, reference: manifestReference, control,
  });
  const reconstructed = reconstructPilotMeasurement({
    review, session, control, repository,
  });
  const value = withSelfDigest(derivedResultPayload(review, reconstructed, control));
  revalidateObservedArtifacts(session);
  const relativePath = pilotMeasurementResultPath(review.reference.digest);
  const target = path.join(session.root.path, ...relativePath.split("/"));
  publishEvidenceBytes(target, `${JSON.stringify(value, null, 2)}\n`, {
    createParentDirectories: true,
  });
  return loadPilotMeasurement({
    session,
    reference: { path: relativePath, digest: value.digest },
    control,
    repository,
  });
}
