import { compareCodePoint } from "./constants.mjs";

export const MATURITY_EVIDENCE = Object.freeze({
  identity: "catalog-contract", disposition: "catalog-contract", "catalog-contract": "catalog-contract",
  "requested-output": "catalog-contract", "effects-capabilities": "catalog-contract", inference: "catalog-contract",
  "runtime-facts": "catalog-contract", "runtime-binding": "runtime-binding", "native-ir-deopt": "native-link",
  placement: "provider-tests", provider: "provider-tests", fusion: "provider-tests", "link-reachability": "native-link",
  foreign: "foreign-tests", parallel: "parallel-tests", host: "host-tests",
  documentation: "documentation-cutover", "native-example": "native-examples",
  "browser-example": "browser-examples", "browser-runtime": "browser-runtime",
  tests: "focused-tests", "wasm-registry": "wasm-registry",
});

export const MANDATORY_AUDIT_GATES = Object.freeze([
  "architecture", "focused-tests", "format-diff", "strict-clippy",
]);

export const MANDATORY_BUNDLE_GATE_PLANS = Object.freeze([
  "architecture", "deterministic-products", "focused-tests", "format-diff",
  "inventory-delta", "strict-clippy",
]);

export function requiredGateNames(controlled) {
  const result = new Set(MANDATORY_AUDIT_GATES);
  for (const [maturity, state] of Object.entries(controlled.maturity)) {
    if (state.applicability !== "required") continue;
    const gate = MATURITY_EVIDENCE[maturity];
    if (!gate) throw new Error(`${controlled.identity}: required maturity ${maturity} has no evidence gate`);
    result.add(gate);
  }
  if (controlled.expected_authorities.native_link === "required") result.add("native-link");
  if (controlled.expected_authorities.wasm_registry === "required") result.add("wasm-registry");
  return [...result].sort(compareCodePoint);
}

export function requiredGatePlanNames(controlledIdentities) {
  const result = new Set(MANDATORY_BUNDLE_GATE_PLANS);
  for (const controlled of controlledIdentities) {
    for (const gate of requiredGateNames(controlled)) result.add(gate);
  }
  return [...result].sort(compareCodePoint);
}
