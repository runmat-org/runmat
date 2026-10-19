import { digest } from "../schema.mjs";

export function pilotEvaluationPaths(measurementDigest) {
  const semanticDigest = digest(measurementDigest, "pilot measurement semantic digest");
  const directory = `pilot-evaluations/${semanticDigest.slice("sha256:".length)}`;
  return Object.freeze({
    limiterReforecast: `${directory}/limiter-reforecast.json`,
    evaluation: `${directory}/evaluation.json`,
  });
}
