export const PILOT_COMMANDS = Object.freeze([
  "pilot-session-start", "pilot-session-complete", "pilot-measure",
  "pilot-evaluate", "pilot-transition",
]);

export const PILOT_OPTION_FIELDS = Object.freeze({
  "--authority-root": "authorityRoot",
  "--lease-digest": "leaseDigest",
  "--start": "start",
  "--start-digest": "startDigest",
  "--pre-state": "preState",
  "--pre-queue-checkpoint": "preQueueCheckpoint",
  "--pre-trusted-queue-checkpoint-digest": "preTrustedQueueCheckpointDigest",
  "--successor-state": "successorState",
  "--successor-queue-checkpoint": "successorQueueCheckpoint",
  "--successor-trusted-queue-checkpoint-digest": "successorTrustedQueueCheckpointDigest",
  "--measurement-review": "measurementReview",
  "--measurement-review-digest": "measurementReviewDigest",
  "--measurement": "measurement",
  "--measurement-digest": "measurementDigest",
  "--limiter": "limiter",
  "--limiter-digest": "limiterDigest",
  "--evaluation": "evaluation",
  "--evaluation-digest": "evaluationDigest",
});

export function validatePilotCommandOptions(options) {
  const required = {
    "pilot-session-start": [
      "authorityRoot", "lease", "leaseDigest", "state", "queueCheckpoint",
      "trustedQueueCheckpointDigest",
    ],
    "pilot-session-complete": [
      "authorityRoot", "lease", "leaseDigest", "start", "startDigest",
      "preState", "preQueueCheckpoint", "preTrustedQueueCheckpointDigest",
      "successorState", "successorQueueCheckpoint",
      "successorTrustedQueueCheckpointDigest",
    ],
    "pilot-measure": ["authorityRoot", "measurementReview", "measurementReviewDigest"],
    "pilot-evaluate": ["authorityRoot", "measurement", "measurementDigest"],
    "pilot-transition": ["authorityRoot", "evaluation", "evaluationDigest"],
  }[options.command];
  if (!required) return;
  const missing = required.filter((field) => !options[field]);
  if (missing.length) {
    throw new Error(`${options.command} requires ${missing.map(optionName).join(", ")}`);
  }
  if (options.command === "pilot-evaluate"
    && Boolean(options.limiter) !== Boolean(options.limiterDigest)) {
    throw new Error("pilot-evaluate requires --limiter and --limiter-digest together");
  }
}

function optionName(field) {
  return Object.entries(PILOT_OPTION_FIELDS).find(([, value]) => value === field)?.[0]
    ?? `--${field.replace(/[A-Z]/g, (value) => `-${value.toLowerCase()}`)}`;
}
