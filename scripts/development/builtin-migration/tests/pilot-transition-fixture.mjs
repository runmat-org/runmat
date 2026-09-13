import path from "node:path";

import { recordPilotEvaluation } from "../pilot-evaluation.mjs";
import { derivePilotTransition } from "../pilot-transition/validation.mjs";
import { pilotEvaluationFixture } from "./pilot-evaluation-fixture.mjs";
import { writeJson } from "./pilot-measurement-fixture.mjs";

export function pilotTransitionFixture(options = {}) {
  const fixture = pilotEvaluationFixture(options);
  const evaluation = recordPilotEvaluation({
    session: fixture.session,
    control: fixture.control,
    measurement: fixture.measurement,
    limiterReforecast: fixture.limiter,
    repository: fixture.repository,
  });
  return {
    ...fixture,
    evaluation,
    derived: derivePilotTransition({
      session: fixture.session, control: fixture.control, evaluation,
    }),
  };
}

export function writeTransitionArtifact(fixture, relativePath, value) {
  writeJson(path.join(fixture.root, relativePath), value);
}
