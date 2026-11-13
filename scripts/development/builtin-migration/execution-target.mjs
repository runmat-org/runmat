import { array, exact, nonempty } from "./schema.mjs";
import { compareCodePoint } from "./constants.mjs";

export function parseExecutionTarget(value, label = "execution target") {
  exact(value, ["operating_system", "architecture"], label);
  return {
    operating_system: nonempty(value.operating_system, `${label} operating system`),
    architecture: nonempty(value.architecture, `${label} architecture`),
  };
}

export function parseExecutionTargets(value) {
  const rows = array(value, "execution targets").map((entry, position) =>
    parseExecutionTarget(entry, `execution target ${position}`));
  const keys = rows.map(executionTargetKey);
  if (new Set(keys).size !== keys.length) throw new Error("execution targets must be unique");
  if (JSON.stringify(keys) !== JSON.stringify([...keys].sort(compareCodePoint))) {
    throw new Error("execution targets must use canonical ordering");
  }
  return rows;
}

export function executionTargetKey(value) {
  return `${value.operating_system}\0${value.architecture}`;
}

export function sameExecutionTarget(left, right) {
  return executionTargetKey(left) === executionTargetKey(right);
}
