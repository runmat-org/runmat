import {
  loadJsonArtifact, loadedJsonValue, openAuthorityLoadSession, openAuthorityRoot,
} from "../authority-loading/index.mjs";
import { loadLeaseAuthority } from "../lease-authority.mjs";
import { loadPilotEvaluation, loadPilotLimiterReforecast, recordPilotEvaluation } from "../pilot-evaluation.mjs";
import { loadPilotMeasurement, publishPilotMeasurement } from "../pilot-measurement/index.mjs";
import { recordPilotTransition } from "../pilot-transition.mjs";
import {
  loadPilotWorkSessionStart, recordPilotWorkSessionCompletion,
  recordPilotWorkSessionStart,
} from "../pilot-work-session/index.mjs";
import { loadQueueAuthority } from "../queue-authority/index.mjs";
import { canonicalCliChildPath } from "../queue-authority/reference.mjs";

export function runPilotLifecycleCommand({ options, control, repository }) {
  const root = openAuthorityRoot(options.authorityRoot);
  const session = openAuthorityLoadSession(root);
  const context = { options, control, repository, root, session };
  if (options.command === "pilot-session-start") return startSession(context);
  if (options.command === "pilot-session-complete") return completeSession(context);
  if (options.command === "pilot-measure") return measure(context);
  if (options.command === "pilot-evaluate") return evaluate(context);
  if (options.command === "pilot-transition") return transition(context);
  throw new Error(`unsupported pilot lifecycle command ${options.command}`);
}

function startSession(context) {
  const queueAuthority = queue(context, "state", "queueCheckpoint",
    "trustedQueueCheckpointDigest");
  const leaseAuthority = lease(context);
  return authorityResult("pilot-work-session-start", recordPilotWorkSessionStart({
    session: context.session, control: context.control, queueAuthority, leaseAuthority,
    repository: context.repository,
  }));
}

function completeSession(context) {
  const start = loadPilotWorkSessionStart({
    session: context.session,
    reference: childReference(context, "start", "startDigest"),
    control: context.control,
    repository: context.repository,
  });
  const finalLeaseAuthority = lease(context);
  const preIntegrationQueue = queue(context, "preState", "preQueueCheckpoint",
    "preTrustedQueueCheckpointDigest");
  const successorQueue = queue(context, "successorState", "successorQueueCheckpoint",
    "successorTrustedQueueCheckpointDigest");
  return authorityResult("pilot-work-session-completion", recordPilotWorkSessionCompletion({
    session: context.session, control: context.control, start, finalLeaseAuthority,
    preIntegrationQueue, successorQueue, repository: context.repository,
  }));
}

function measure(context) {
  const measurement = publishPilotMeasurement({
    session: context.session,
    manifestReference: childReference(
      context, "measurementReview", "measurementReviewDigest",
    ),
    control: context.control,
    repository: context.repository,
  });
  return authorityResult("pilot-measurement", measurement);
}

function evaluate(context) {
  const measurement = loadPilotMeasurement({
    session: context.session,
    reference: childReference(context, "measurement", "measurementDigest"),
    control: context.control,
    repository: context.repository,
  });
  const limiterReforecast = context.options.limiter
    ? loadPilotLimiterReforecast({
      session: context.session,
      reference: childReference(context, "limiter", "limiterDigest"),
      control: context.control,
      measurement,
    })
    : null;
  const evaluation = recordPilotEvaluation({
    session: context.session, control: context.control, measurement,
    limiterReforecast, repository: context.repository,
  });
  return authorityResult("pilot-evaluation", evaluation, {
    artifact_id: evaluation.value.artifact_id,
    outcome: evaluation.value.outcome,
  });
}

function transition(context) {
  const reference = childReference(context, "evaluation", "evaluationDigest");
  const artifact = loadJsonArtifact(context.session, reference.path, "pilot evaluation");
  const serialized = loadedJsonValue(artifact, context.root);
  const evaluation = loadPilotEvaluation({
    session: context.session,
    reference: { ...reference, artifact_id: serialized.artifact_id },
    control: context.control,
    repository: context.repository,
  });
  const result = recordPilotTransition({
    session: context.session, control: context.control, evaluation,
  });
  return {
    schema_version: 1,
    kind: "runmat-builtin-migration-pilot-transition-command-result",
    evaluation: {
      path: evaluation.reference.path,
      artifact_id: evaluation.value.artifact_id,
      digest: evaluation.reference.digest,
    },
    state: observation(result.observations.state),
    checkpoint: observation(result.observations.checkpoint),
  };
}

function lease(context) {
  return loadLeaseAuthority(
    context.session,
    childReference(context, "lease", "leaseDigest"),
    context.control,
    context.repository,
  );
}

function queue(context, stateField, checkpointField, digestField) {
  return loadQueueAuthority({
    session: context.session,
    statePath: childPath(context, stateField),
    checkpointPath: childPath(context, checkpointField),
    trustedCheckpointDigest: context.options[digestField],
    control: context.control,
  });
}

function childReference(context, pathField, digestField) {
  return {
    path: childPath(context, pathField), digest: context.options[digestField],
  };
}

function childPath(context, field) {
  return canonicalCliChildPath(
    context.root.path, context.options[field], `--${field} path`,
  );
}

function authorityResult(kind, value, fields = {}) {
  return {
    schema_version: 1,
    kind: `runmat-builtin-migration-${kind}-command-result`,
    reference: {
      path: value.reference.path,
      digest: value.reference.digest,
      content_digest: value.artifact.contentDigest,
    },
    ...fields,
  };
}

function observation(value) {
  return {
    path: value.path,
    digest: value.semanticDigest,
    content_digest: value.contentDigest,
  };
}
