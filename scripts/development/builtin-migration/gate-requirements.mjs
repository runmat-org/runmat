import { compareCodePoint } from "./constants.mjs";

export const MATURITY_EVIDENCE = Object.freeze({
  identity: "catalog-contract", disposition: "catalog-contract", "catalog-contract": "catalog-contract",
  "requested-output": "catalog-contract", "effects-capabilities": "catalog-contract", inference: "catalog-contract",
  "runtime-facts": "catalog-contract", "runtime-binding": "runtime-binding", "native-ir-deopt": "native-link",
  placement: "provider-tests", provider: "provider-tests", fusion: "provider-tests", "link-reachability": "native-link",
  "interop-parallel": "foreign-tests", documentation: "documentation-cutover", examples: "native-examples",
  tests: "focused-tests", wasm: "wasm-registry",
});

export function requiredGateNames(controlled) {
  const result = new Set(["architecture"]);
  for (const [maturity, state] of Object.entries(controlled.maturity)) {
    if (state.applicability === "required") result.add(MATURITY_EVIDENCE[maturity]);
  }
  if (controlled.expected_authorities.native_link === "required") result.add("native-link");
  if (controlled.expected_authorities.wasm_registry === "required") result.add("wasm-registry");
  return [...result].filter(Boolean).sort(compareCodePoint);
}
