import assert from "node:assert/strict";
import test from "node:test";

import { MATURITY_GATES } from "../control-authoring/policy-schema.mjs";
import { GATE_PRODUCERS } from "../gate-kinds.mjs";
import { parseGatePlans } from "../gate-plan.mjs";
import {
  MANDATORY_AUDIT_GATES, MANDATORY_BUNDLE_GATE_PLANS, MATURITY_EVIDENCE,
  requiredGateNames, requiredGatePlanNames,
} from "../gate-requirements.mjs";

const DIGEST = `sha256:${"a".repeat(64)}`;

test("maturity vocabulary maps independent obligations to authoritative evidence products", () => {
  assert.deepEqual({
    foreign: MATURITY_EVIDENCE.foreign,
    parallel: MATURITY_EVIDENCE.parallel,
    host: MATURITY_EVIDENCE.host,
    nativeExample: MATURITY_EVIDENCE["native-example"],
    browserExample: MATURITY_EVIDENCE["browser-example"],
    browserRuntime: MATURITY_EVIDENCE["browser-runtime"],
    provider: MATURITY_EVIDENCE.provider,
    wasmRegistry: MATURITY_EVIDENCE["wasm-registry"],
  }, {
    foreign: "foreign-tests",
    parallel: "parallel-tests",
    host: "host-tests",
    nativeExample: "native-examples",
    browserExample: "browser-examples",
    browserRuntime: "browser-examples",
    provider: "provider-tests",
    wasmRegistry: "wasm-registry",
  });
  assert.equal("interop-parallel" in MATURITY_EVIDENCE, false);
  assert.equal("examples" in MATURITY_EVIDENCE, false);
  assert.equal("wasm" in MATURITY_EVIDENCE, false);
});

test("every closed maturity key has a registered evidence producer", () => {
  assert.deepEqual(Object.keys(MATURITY_EVIDENCE).sort(), [...MATURITY_GATES].sort());
  for (const gate of Object.values(MATURITY_EVIDENCE)) {
    assert.equal(typeof GATE_PRODUCERS[gate], "string", `${gate} must have a producer`);
  }
});

test("audit and integration gate plans remain mandatory when every maturity is inapplicable", () => {
  const controlled = identityControl(Object.fromEntries(
    Object.keys(MATURITY_EVIDENCE).map((name) => [name, notApplicable()]),
  ));
  assert.deepEqual(requiredGateNames(controlled), [...MANDATORY_AUDIT_GATES]);
  assert.deepEqual(requiredGatePlanNames([controlled]), [...MANDATORY_BUNDLE_GATE_PLANS].sort());
});

test("required maturity cannot disappear without an evidence-gate mapping", () => {
  const controlled = identityControl({ unrecognized: { applicability: "required" } });
  assert.throws(() => requiredGateNames(controlled), /required maturity unrecognized has no evidence gate/);
});

test("browser execution obligations share one product-backed proof without collapsing policy", () => {
  const maturity = Object.fromEntries(
    Object.keys(MATURITY_EVIDENCE).map((name) => [name, notApplicable()]),
  );
  maturity["browser-example"] = { applicability: "required" };
  maturity["browser-runtime"] = { applicability: "required" };
  const gates = requiredGateNames(identityControl(maturity));
  assert.equal(gates.filter((gate) => gate === "browser-examples").length, 1);
  assert.equal(gates.includes("browser-runtime"), false);
  assert.notEqual("browser-example", "browser-runtime");
});

test("parallel evidence retains a closed exit-status plan", () => {
  const parsed = parseGatePlans([plan("parallel-tests")], "fixture-bundle");
  assert.equal(parsed.get("parallel-tests").parser, "exit_status");
});

function identityControl(maturity) {
  return {
    identity: "fixture",
    maturity,
    expected_authorities: { native_link: "not-applicable", wasm_registry: "not-applicable" },
  };
}

function notApplicable() {
  return { applicability: "not-applicable" };
}

function plan(gate) {
  return {
    gate,
    program: {
      kind: "repository_script",
      path: "scripts/development/check-architecture-boundaries.mjs",
      content_digest: DIGEST,
      approved_toolchains: [{
        operating_system: "macos",
        architecture: "aarch64",
        tools: [{ role: "node", content_digest: DIGEST }],
      }],
    },
    arguments: [],
    working_directory: "repository",
    parser: "exit_status",
    expected_artifact_roles: [],
  };
}
