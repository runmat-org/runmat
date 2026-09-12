import assert from "node:assert/strict";
import test from "node:test";

import { gatePlanEvidence, parseGatePlans, validateGatePlanTargetCoverage } from "../gate-plan.mjs";

const DIGEST = `sha256:${"a".repeat(64)}`;
const BUILD = { operating_system: "linux", architecture: "x86_64" };

function tools(roles) {
  return [{ ...BUILD, tools: [...roles].sort().map((role) => ({ role, content_digest: DIGEST })) }];
}

function cargoProgram(operation) {
  const roles = operation === "clippy" ? ["cargo", "cargo-clippy", "clippy-driver", "rustc"]
    : operation === "fmt" ? ["cargo", "cargo-fmt", "rustfmt"]
      : operation === "test" ? ["cargo", "rustc", "rustdoc"] : ["cargo", "rustc"];
  return {
    kind: "cargo_operation", operation, manifest_path: "Cargo.toml", manifest_digest: DIGEST,
    approved_toolchains: tools(roles),
  };
}

function plan(operation, argumentsList = []) {
  return {
    gate: operation === "clippy" ? "strict-clippy" : operation === "fmt" ? "format-diff" : "focused-tests",
    program: cargoProgram(operation), arguments: argumentsList, working_directory: "repository",
    parser: "exit_status", expected_artifact_roles: [],
  };
}

test("cargo gate operations have closed commands and exact reviewed tool roles", () => {
  for (const operation of ["check", "clippy", "fmt", "test"]) {
    const parsed = parseGatePlans([plan(operation)], "fixture").values().next().value;
    const evidence = gatePlanEvidence(parsed, BUILD, "/workspace/runmat");
    assert.deepEqual(evidence.arguments.slice(0, 3), [operation, "--manifest-path", "/workspace/runmat/Cargo.toml"]);
    assert.equal(evidence.primary_tool, "cargo");
  }

  const unknown = plan("check");
  unknown.program.operation = "shell";
  assert.throws(() => parseGatePlans([unknown], "fixture"), /Cargo operation/);

  const missing = plan("clippy");
  missing.program.approved_toolchains[0].tools = missing.program.approved_toolchains[0].tools.filter((tool) => tool.role !== "clippy-driver");
  assert.throws(() => parseGatePlans([missing], "fixture"), /allowed and required tools/);

  const duplicate = plan("check");
  duplicate.program.approved_toolchains[0].tools.push({ role: "rustc", content_digest: DIGEST });
  assert.throws(() => parseGatePlans([duplicate], "fixture"), /unique and canonical/);
});

test("cargo gate arguments allow repeated values but cannot replace typed authority", () => {
  assert.doesNotThrow(() => parseGatePlans([plan("test", ["--package", "runmat-vm", "--package", "runmat-runtime"])], "fixture"));
  for (const argumentsList of [
    ["--manifest-path", "other/Cargo.toml"], ["--manifest-path=other/Cargo.toml"],
    ["--target-dir", "/tmp/escape"], ["--target-dir=/tmp/escape"],
    ["--config", "build.target-dir='/tmp/escape'"], ["--config=build.target-dir='/tmp/escape'"],
    ["-Z", "config-include"], ["-Zconfig-include"],
  ]) assert.throws(() => parseGatePlans([plan("check", argumentsList)], "fixture"), /cannot override/);

  assert.throws(() => parseGatePlans([plan("check", ["bad\0argument"])], "fixture"), /NUL/);
});

test("repository scripts retain typed node and auxiliary git identity", () => {
  const repositoryPlan = {
    gate: "documentation-cutover",
    program: {
      kind: "repository_script", path: "scripts/export-docs.mjs", content_digest: DIGEST,
      approved_toolchains: tools(["git", "node"]),
    },
    arguments: [], working_directory: "repository", parser: "documentation_cutover",
    expected_artifact_roles: ["documentation-reconciliation"],
  };
  assert.doesNotThrow(() => parseGatePlans([repositoryPlan], "fixture"));
  repositoryPlan.program.approved_toolchains = tools(["node"]);
  assert.throws(() => parseGatePlans([repositoryPlan], "fixture"), /requires reviewed git tools/);
});

test("every gate toolchain exactly covers the reviewed execution targets", () => {
  const plans = parseGatePlans([plan("check")], "fixture");
  assert.doesNotThrow(() => validateGatePlanTargetCoverage(plans, [BUILD], "fixture"));
  assert.throws(
    () => validateGatePlanTargetCoverage(plans, [BUILD, { operating_system: "windows", architecture: "x86_64" }], "fixture"),
    /exactly cover every execution target/,
  );
});
