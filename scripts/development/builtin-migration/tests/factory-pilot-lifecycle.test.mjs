import assert from "node:assert/strict";
import test from "node:test";

import { runPilotLifecycleCommand } from "../factory-cli/pilot-lifecycle.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { pilotEvaluationFixture } from "./pilot-evaluation-fixture.mjs";
import { pilotMeasurementFixture } from "./pilot-measurement-fixture.mjs";
import {
  installSuccessor, workSessionFixture,
} from "./pilot-work-session-fixture.mjs";

test("factory lifecycle records session start and completion from persisted authorities", () => {
  const fixture = workSessionFixture();
  const common = {
    authorityRoot: fixture.root,
    lease: fixture.lease.reference.path,
    leaseDigest: fixture.lease.reference.digest,
  };
  const start = runPilotLifecycleCommand({
    options: {
      command: "pilot-session-start",
      ...common,
      state: fixture.queue.observations.state.path,
      queueCheckpoint: fixture.queue.observations.checkpoint.path,
      trustedQueueCheckpointDigest: fixture.queue.observations.checkpoint.semanticDigest,
    },
    control: fixture.control,
    repository: fixture.repository,
  });
  const successor = installSuccessor(fixture);
  const completion = runPilotLifecycleCommand({
    options: {
      command: "pilot-session-complete",
      ...common,
      start: start.reference.path,
      startDigest: start.reference.digest,
      preState: fixture.queue.observations.state.path,
      preQueueCheckpoint: fixture.queue.observations.checkpoint.path,
      preTrustedQueueCheckpointDigest: fixture.queue.observations.checkpoint.semanticDigest,
      successorState: successor.observations.state.path,
      successorQueueCheckpoint: successor.observations.checkpoint.path,
      successorTrustedQueueCheckpointDigest: successor.observations.checkpoint.semanticDigest,
    },
    control: fixture.control,
    repository: fixture.repository,
  });
  assert.equal(start.kind, "runmat-builtin-migration-pilot-work-session-start-command-result");
  assert.equal(completion.kind,
    "runmat-builtin-migration-pilot-work-session-completion-command-result");
  assert.notEqual(completion.reference.digest, start.reference.digest);
  assert.equal(fixture.lease.reference.digest, evidenceDigest(fixture.lease.lease.value));
});

test("factory lifecycle reconstructs and publishes reviewed pilot measurement", () => {
  const fixture = pilotMeasurementFixture({ publish: false });
  const result = runPilotLifecycleCommand({
    options: {
      command: "pilot-measure",
      authorityRoot: fixture.root,
      measurementReview: fixture.manifestReference.path,
      measurementReviewDigest: fixture.manifestReference.digest,
    },
    control: fixture.control,
    repository: fixture.repository,
  });
  assert.equal(result.kind, "runmat-builtin-migration-pilot-measurement-command-result");
  assert.match(result.reference.path, /^pilot-measurements\/[a-f0-9]{64}\//);
});

test("factory lifecycle evaluates and transitions exact pilot authority", () => {
  const fixture = pilotEvaluationFixture();
  const evaluation = runPilotLifecycleCommand({
    options: {
      command: "pilot-evaluate",
      authorityRoot: fixture.root,
      measurement: fixture.measurement.reference.path,
      measurementDigest: fixture.measurement.reference.digest,
      limiter: null,
      limiterDigest: null,
    },
    control: fixture.control,
    repository: fixture.repository,
  });
  assert.equal(evaluation.outcome, "threshold-met");
  assert.match(evaluation.artifact_id, /^pilot-evaluation-[a-f0-9]{64}$/);

  const transition = runPilotLifecycleCommand({
    options: {
      command: "pilot-transition",
      authorityRoot: fixture.root,
      evaluation: evaluation.reference.path,
      evaluationDigest: evaluation.reference.digest,
    },
    control: fixture.control,
    repository: fixture.repository,
  });
  assert.equal(transition.evaluation.artifact_id, evaluation.artifact_id);
  assert.match(transition.state.path, /^pilot-transitions\/[a-f0-9]{64}\//);
  assert.match(transition.checkpoint.path, /^pilot-transitions\/[a-f0-9]{64}\//);
});

test("factory lifecycle carries a reviewed below-target limiter", () => {
  const fixture = pilotEvaluationFixture({ belowTarget: true });
  const result = runPilotLifecycleCommand({
    options: {
      command: "pilot-evaluate",
      authorityRoot: fixture.root,
      measurement: fixture.measurement.reference.path,
      measurementDigest: fixture.measurement.reference.digest,
      limiter: fixture.paths.limiterReforecast,
      limiterDigest: fixture.limiter.reference.digest,
    },
    control: fixture.control,
    repository: fixture.repository,
  });
  assert.equal(result.outcome, "below-target");
});
