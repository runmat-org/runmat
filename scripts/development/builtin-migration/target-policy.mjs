import { compareCodePoint } from "./constants.mjs";
import {
  executionTargetKey, parseExecutionTarget, parseExecutionTargets,
} from "./execution-target.mjs";
import {
  array, enumValue, exact, integer, kind, nonempty, stableId, uniqueStrings,
} from "./schema.mjs";

export const R31_REQUIRED_TARGETS = Object.freeze([
  Object.freeze({ operating_system: "linux", architecture: "aarch64" }),
  Object.freeze({ operating_system: "linux", architecture: "x86_64" }),
  Object.freeze({ operating_system: "macos", architecture: "aarch64" }),
  Object.freeze({ operating_system: "macos", architecture: "x86_64" }),
  Object.freeze({ operating_system: "windows", architecture: "x86_64" }),
]);

// R31 closes these product domains as distinct semantic obligations. The
// taxonomy deliberately stops above individual commands and artifacts: those
// remain reviewed evidence, while this list prevents an entire product domain
// from disappearing from the terminal matrix.
export const R31_REQUIRED_LANES = Object.freeze([
  lane("browser-wasm", "optimized-runtime", "Optimized WebAssembly and browser-engine execution"),
  lane("desktop", "native-browser-product", "Desktop native and browser product integration"),
  lane("deterministic-execution", "replay-numerical-policy", "Deterministic replay and platform-accurate numerical policy"),
  lane("foreign-interop", "adapter-compatibility", "Foreign adapter compatibility, lifecycle, and isolation"),
  lane("package-release", "packaging-publication-docs", "Package, release, publication, and user-facing documentation"),
  lane("parallel-distributed", "execution-semantics", "Local, browser, and remote parallel and distributed execution"),
  lane("performance-resource", "regression-budgets", "Performance, memory, scale, and resource regression budgets"),
  lane("provider-gpu", "placement-fusion-device", "Provider placement, fusion, device execution, and loss handling"),
  lane("public-native", "runtime-cli-aot", "Public native runtime, CLI, JIT, AOT, and standalone binaries"),
  lane("security-recovery", "adversarial-resilience", "Security, hostile-input, fencing, failure, and recovery behavior"),
  lane("server-remote-cluster", "execution-domains", "Server, remote execution, cluster, quota, and billing domains"),
  lane("workspace-quality", "source-test-toolchain", "Formatting, lints, tests, dependency, feature, and safety tooling"),
]);

export function parseTargetPolicy(value) {
  kind(value, 1, "runmat-builtin-migration-target-policy", "target policy");
  exact(value, [
    "schema_version", "kind", "migration_execution_targets", "terminal_qualification",
  ], "target policy");
  const migrationExecutionTargets = parseExecutionTargets(value.migration_execution_targets);
  const terminalQualification = parseTerminalQualification(value.terminal_qualification);
  const terminalQualificationMatrix = terminalQualification.lanes.flatMap((lane) =>
    lane.targets.filter((target) => target.applicability === "required").map((target) => ({
      product_id: lane.product_id,
      gate_id: lane.gate_id,
      operating_system: target.operating_system,
      architecture: target.architecture,
      execution_order: target.execution_order,
    })));
  const terminalTargets = new Set(terminalQualificationMatrix.map(executionTargetKey));
  for (const target of migrationExecutionTargets) {
    if (!terminalTargets.has(executionTargetKey(target))) {
      throw new Error("each migration execution target must have a required terminal qualification cell");
    }
  }
  const required = R31_REQUIRED_TARGETS.map(executionTargetKey).sort(compareCodePoint);
  const observed = [...terminalTargets].sort(compareCodePoint);
  if (JSON.stringify(observed) !== JSON.stringify(required)) {
    throw new Error("terminal qualification must require the frozen R31 platform set exactly");
  }
  validateWindowsLast(terminalQualificationMatrix);
  return { migrationExecutionTargets, terminalQualification, terminalQualificationMatrix };
}

function parseTerminalQualification(value) {
  exact(value, ["phase", "lanes"], "terminal qualification policy");
  if (value.phase !== "R31") throw new Error("terminal qualification phase must be R31");
  const lanes = array(value.lanes, "terminal qualification lanes").map((entry, position) => {
    exact(entry, ["product_id", "gate_id", "purpose", "targets"], `terminal qualification lane ${position}`);
    return {
      product_id: stableId(entry.product_id, `terminal qualification lane ${position} product`),
      gate_id: stableId(entry.gate_id, `terminal qualification lane ${position} gate`),
      purpose: nonempty(entry.purpose, `terminal qualification lane ${position} purpose`),
      targets: parseLaneTargets(entry.targets, position),
    };
  });
  const keys = lanes.map((entry) => `${entry.product_id}\0${entry.gate_id}`);
  if (new Set(keys).size !== keys.length) throw new Error("terminal qualification lanes must be unique");
  if (JSON.stringify(keys) !== JSON.stringify([...keys].sort(compareCodePoint))) {
    throw new Error("terminal qualification lanes must use canonical product/gate ordering");
  }
  const expected = R31_REQUIRED_LANES.map((entry) => `${entry.product_id}\0${entry.gate_id}\0${entry.purpose}`);
  const observed = lanes.map((entry) => `${entry.product_id}\0${entry.gate_id}\0${entry.purpose}`);
  if (JSON.stringify(observed) !== JSON.stringify(expected)) {
    throw new Error("terminal qualification lanes must exactly cover the frozen R31 product-domain taxonomy");
  }
  return { phase: "R31", lanes };
}

function parseLaneTargets(value, lanePosition) {
  const rows = array(value, `terminal qualification lane ${lanePosition} targets`).map(
    (entry, targetPosition) => {
      const label = `terminal qualification lane ${lanePosition} target ${targetPosition}`;
      exact(entry, [
        "operating_system", "architecture", "applicability", "execution_order", "reason", "evidence",
      ], label);
      const target = parseExecutionTarget({
        operating_system: entry.operating_system,
        architecture: entry.architecture,
      }, label);
      const applicability = enumValue(entry.applicability, ["required", "not-applicable"], `${label} applicability`);
      const evidence = uniqueStrings(entry.evidence, `${label} evidence`, { empty: true });
      if (applicability === "required") {
        if (entry.reason !== null || evidence.length !== 0) {
          throw new Error(`${label}: required target cannot carry inapplicability evidence`);
        }
        integer(entry.execution_order, `${label} execution order`, 1);
      } else {
        nonempty(entry.reason, `${label} inapplicability reason`);
        if (evidence.length === 0) throw new Error(`${label}: inapplicable target requires evidence`);
        if (entry.execution_order !== null) throw new Error(`${label}: inapplicable target cannot have an execution order`);
      }
      return { ...target, applicability, execution_order: entry.execution_order, reason: entry.reason, evidence };
    },
  );
  const keys = rows.map(executionTargetKey);
  const required = R31_REQUIRED_TARGETS.map(executionTargetKey).sort(compareCodePoint);
  if (JSON.stringify(keys) !== JSON.stringify(required)) {
    throw new Error(`terminal qualification lane ${lanePosition} must classify every frozen R31 target exactly once in canonical order`);
  }
  if (!rows.some((entry) => entry.applicability === "required")) {
    throw new Error(`terminal qualification lane ${lanePosition} must require at least one target`);
  }
  return rows;
}

function validateWindowsLast(matrix) {
  const windows = matrix.filter((entry) => entry.operating_system === "windows");
  const earlier = matrix.filter((entry) => entry.operating_system !== "windows");
  if (!windows.length || !earlier.length) {
    throw new Error("terminal qualification must contain required Windows and non-Windows cells");
  }
  const firstWindows = Math.min(...windows.map((entry) => entry.execution_order));
  const lastEarlier = Math.max(...earlier.map((entry) => entry.execution_order));
  if (firstWindows <= lastEarlier) {
    throw new Error("terminal Windows qualification must execute after every non-Windows cell");
  }
}

function lane(productId, gateId, purpose) {
  return Object.freeze({ product_id: productId, gate_id: gateId, purpose });
}
