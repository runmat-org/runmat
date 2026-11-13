import { digest } from "../schema.mjs";

export function pilotMeasurementResultPath(reviewDigest) {
  const validated = digest(reviewDigest, "pilot measurement review digest");
  const hex = validated.slice("sha256:".length);
  return `pilot-measurements/${hex}/measurement.json`;
}
