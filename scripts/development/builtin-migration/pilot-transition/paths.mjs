import { digest } from "../schema.mjs";

export function pilotTransitionPaths(evaluationDigest) {
  const value = digest(evaluationDigest, "pilot transition evaluation digest");
  const directory = `pilot-transitions/${value.slice("sha256:".length)}`;
  return Object.freeze({
    state: `${directory}/queue-state.json`,
    checkpoint: `${directory}/queue-checkpoint.json`,
  });
}
