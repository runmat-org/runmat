import { compareCodePoint } from "../constants.mjs";
import { array, exact } from "../schema.mjs";

const FEATURE = /^[a-zA-Z0-9][a-zA-Z0-9_+.-]*$/;
const ARCHITECTURES = Object.freeze(["wasm32"]);
const LEAF_RANK = Object.freeze({ "target-architecture": 0, "cargo-feature": 1, test: 2 });

export function parseCompositionCondition(value, label) {
  if (value?.kind === "always" || value?.kind === "test") {
    exact(value, ["kind"], label);
    return value;
  }
  if (value?.kind === "cargo-feature") {
    exact(value, ["kind", "feature"], label);
    if (typeof value.feature !== "string" || !FEATURE.test(value.feature)) throw new Error(`${label} has an invalid Cargo feature`);
    return value;
  }
  if (value?.kind === "target-architecture") {
    exact(value, ["kind", "architecture"], label);
    if (!ARCHITECTURES.includes(value.architecture)) throw new Error(`${label} has an unsupported target architecture`);
    return value;
  }
  if (value?.kind === "all") {
    exact(value, ["kind", "conditions"], label);
    const conditions = array(value.conditions, `${label} conjunction`).map((entry) => {
      const parsed = parseCompositionCondition(entry, `${label} conjunction member`);
      if (parsed.kind === "always" || parsed.kind === "all") throw new Error(`${label} conjunction members must be closed leaf conditions`);
      return parsed;
    });
    if (conditions.length < 2) throw new Error(`${label} conjunction must contain at least two conditions`);
    canonicalUnique(conditions, label);
    return { kind: "all", conditions };
  }
  throw new Error(`${label} has an unsupported kind`);
}

export function conditionAttribute(condition) {
  const expression = conditionExpression(condition);
  return expression === null ? null : `#[cfg(${expression})]`;
}

export function conditionKey(condition) {
  if (condition.kind === "always") return "0";
  if (condition.kind === "all") return `4:${condition.conditions.map(conditionKey).join("+")}`;
  const detail = condition.kind === "cargo-feature" ? condition.feature
    : condition.kind === "target-architecture" ? condition.architecture : "";
  return `${LEAF_RANK[condition.kind] + 1}:${detail}`;
}

export function conditionImplies(left, right) {
  const leftLeaves = conditionLeaves(left);
  const rightLeaves = conditionLeaves(right);
  return [...rightLeaves].every((entry) => leftLeaves.has(entry));
}

export function parseConditionAttribute(line) {
  if (line === "#[cfg(test)]") return { kind: "test" };
  let match = /^#\[cfg\(feature = "([a-zA-Z0-9][a-zA-Z0-9_+.-]*)"\)\]$/.exec(line);
  if (match) return { kind: "cargo-feature", feature: match[1] };
  match = /^#\[cfg\(target_arch = "(wasm32)"\)\]$/.exec(line);
  if (match) return { kind: "target-architecture", architecture: match[1] };
  match = /^#\[cfg\(all\((.+)\)\)\]$/.exec(line);
  if (!match) return null;
  const conditions = match[1].split(", ").map((part) => parseConditionAttribute(`#[cfg(${part})]`));
  if (conditions.some((entry) => entry === null)) throw new Error("generated cfg conjunction contains unsupported syntax");
  return parseCompositionCondition({ kind: "all", conditions }, "generated cfg condition");
}

function conditionExpression(condition) {
  if (condition.kind === "always") return null;
  if (condition.kind === "test") return "test";
  if (condition.kind === "cargo-feature") return `feature = "${condition.feature}"`;
  if (condition.kind === "target-architecture") return `target_arch = "${condition.architecture}"`;
  return `all(${condition.conditions.map(conditionExpression).join(", ")})`;
}

function conditionLeaves(condition) {
  if (condition.kind === "always") return new Set();
  if (condition.kind === "all") return new Set(condition.conditions.map(conditionKey));
  return new Set([conditionKey(condition)]);
}

function canonicalUnique(values, label) {
  const keys = values.map(conditionKey);
  if (new Set(keys).size !== keys.length) throw new Error(`${label} conjunction must be unique`);
  const ordered = [...values].sort((left, right) => {
    const rank = LEAF_RANK[left.kind] - LEAF_RANK[right.kind];
    return rank || compareCodePoint(conditionKey(left), conditionKey(right));
  });
  if (JSON.stringify(values) !== JSON.stringify(ordered)) throw new Error(`${label} conjunction must use canonical condition order`);
}
