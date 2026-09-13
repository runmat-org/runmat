import { digest } from "../schema.mjs";

export function initialQueuePaths(controlDigest) {
  const value = digest(controlDigest, "initial queue control digest");
  const directory = `queue-roots/${value.slice("sha256:".length)}/initial`;
  return Object.freeze({
    state: `${directory}/queue-state.json`,
    checkpoint: `${directory}/queue-checkpoint.json`,
  });
}
