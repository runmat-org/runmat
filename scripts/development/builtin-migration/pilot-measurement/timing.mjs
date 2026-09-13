import { integer } from "../schema.mjs";

export function derivePilotTiming(completions) {
  if (!Array.isArray(completions) || completions.length === 0) {
    throw new Error("pilot measurement requires completed work sessions");
  }
  let earliest = Number.POSITIVE_INFINITY;
  let latest = Number.NEGATIVE_INFINITY;
  let aggregate = 0;
  for (const completion of completions) {
    const start = timestampMilliseconds(completion.start.startedAt, "pilot session start");
    const end = timestampMilliseconds(completion.endedAt, "pilot session end");
    const duration = safeDifference(end, start, "pilot session duration");
    aggregate = safeSum(aggregate, duration, "pilot aggregate worker milliseconds");
    earliest = Math.min(earliest, start);
    latest = Math.max(latest, end);
  }
  if (aggregate === 0) {
    throw new Error("pilot aggregate worker milliseconds must be positive");
  }
  const elapsed = safeDifference(latest, earliest, "pilot elapsed milliseconds");
  if (elapsed === 0) throw new Error("pilot elapsed milliseconds must be positive");
  return Object.freeze({
    started_at: new Date(earliest).toISOString(),
    ended_at: new Date(latest).toISOString(),
    elapsed_ms: elapsed,
    aggregate_worker_ms: aggregate,
  });
}

function timestampMilliseconds(value, label) {
  const milliseconds = Date.parse(value);
  if (!Number.isSafeInteger(milliseconds)) {
    throw new Error(`${label} must have a safe-integer millisecond representation`);
  }
  return milliseconds;
}

function safeDifference(end, start, label) {
  const result = end - start;
  integer(result, label, 0);
  return result;
}

function safeSum(left, right, label) {
  const result = left + right;
  integer(result, label, 0);
  return result;
}
