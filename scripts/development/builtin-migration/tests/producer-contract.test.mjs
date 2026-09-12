import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { parseGatePlans } from "../gate-plan.mjs";
import { validateGateCheckIds } from "../gate-result.mjs";
import { exampleMaturityGate } from "../producer-adapters/example.mjs";

const DIGEST = `sha256:${"a".repeat(64)}`;
const REPOSITORY = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../..");

test("documentation cutover has one truthful aggregate-export producer contract", () => {
  const valid = repositoryPlan(
    "documentation-cutover", "documentation_cutover", "documentation-reconciliation",
    "scripts/development/builtin-migration/documentation-export-cli.mjs",
    ["cargo", "git", "node", "rustc"],
  );
  assert.doesNotThrow(() => parseGatePlans([valid], "fixture"));
  for (const mutate of [
    (plan) => { plan.program.path = "scripts/development/check-architecture-boundaries.mjs"; },
    (plan) => { plan.arguments = ["--output", "/tmp/result.json"]; },
    (plan) => { plan.program.approved_toolchains[0].tools.pop(); },
  ]) {
    const changed = structuredClone(valid); mutate(changed);
    assert.throws(() => parseGatePlans([changed], "fixture"), /reviewed repository producer|accepts no arguments|exact reviewed/);
  }
});

test("example reconciliation has an stdin-only producer contract", () => {
  const valid = repositoryPlan(
    "native-examples", "example_reconciliation", "example-reconciliation",
    "scripts/development/builtin-migration/example-gate-cli.mjs", ["node"],
  );
  assert.doesNotThrow(() => parseGatePlans([valid], "fixture"));
  const injected = structuredClone(valid);
  injected.arguments = ["--manifest", "/tmp/caller-selected.json"];
  assert.throws(() => parseGatePlans([injected], "fixture"), /accepts no arguments/);
  const auxiliaryTool = structuredClone(valid);
  auxiliaryTool.program.approved_toolchains[0].tools.unshift({ role: "git", content_digest: DIGEST });
  assert.throws(() => parseGatePlans([auxiliaryTool], "fixture"), /exact reviewed/);
});

test("both reviewed producer entrypoints reject caller argv before doing work", () => {
  for (const sourcePath of [
    "scripts/development/builtin-migration/documentation-export-cli.mjs",
    "scripts/development/builtin-migration/example-gate-cli.mjs",
  ]) {
    const result = spawnSync(process.execPath, [path.join(REPOSITORY, sourcePath), "--caller-selected"], {
      cwd: REPOSITORY, encoding: "utf8",
    });
    assert.equal(result.status, 2);
    assert.match(result.stderr, /accepts no arguments/);
  }
});

test("native and browser example gates select their distinct reviewed maturity obligations", () => {
  assert.equal(exampleMaturityGate("native-examples"), "native-example");
  assert.equal(exampleMaturityGate("browser-examples"), "browser-example");
  assert.throws(() => exampleMaturityGate("focused-tests"), /not an example gate/);
});

test("gate check ids are an exact canonical identity projection", () => {
  const checks = [{ id: "focused-tests:alpha" }, { id: "focused-tests:beta" }];
  assert.doesNotThrow(() => validateGateCheckIds(checks, "focused-tests", ["beta", "alpha"]));
  assert.throws(
    () => validateGateCheckIds([...checks, { id: "focused-tests:extra" }], "focused-tests", ["alpha", "beta"]),
    /exactly cover/,
  );
  assert.throws(
    () => validateGateCheckIds([...checks].reverse(), "focused-tests", ["alpha", "beta"]),
    /canonical order/,
  );
});

function repositoryPlan(gate, parser, artifactRole, sourcePath, roles) {
  return {
    gate,
    program: {
      kind: "repository_script", path: sourcePath, content_digest: DIGEST,
      approved_toolchains: [{
        operating_system: "linux", architecture: "x86_64",
        tools: roles.map((role) => ({ role, content_digest: DIGEST })),
      }],
    },
    arguments: [], working_directory: "repository", parser,
    expected_artifact_roles: [artifactRole],
  };
}
