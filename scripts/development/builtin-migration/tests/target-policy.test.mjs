import assert from "node:assert/strict";
import test from "node:test";

import { parseTargetPolicy, R31_REQUIRED_LANES } from "../target-policy.mjs";

test("target policy separates runnable migration targets from exact R31 qualification cells", () => {
  const parsed = parseTargetPolicy(policy());
  assert.deepEqual(parsed.migrationExecutionTargets, [
    { operating_system: "macos", architecture: "aarch64" },
  ]);
  assert.equal(parsed.terminalQualificationMatrix.length, R31_REQUIRED_LANES.length * 5);
  assert.ok(parsed.terminalQualificationMatrix.some((entry) =>
    entry.operating_system === "linux" && entry.architecture === "aarch64"));
  assert.ok(parsed.terminalQualificationMatrix.filter((entry) =>
    entry.operating_system === "windows").every((entry) => entry.execution_order === 2));
});

test("each product/gate lane must classify every terminal target exactly once", () => {
  const omitted = policy();
  omitted.terminal_qualification.lanes[0].targets.splice(0, 1);
  assert.throws(() => parseTargetPolicy(omitted), /must classify every frozen R31 target exactly once/);

  const duplicateCell = policy();
  duplicateCell.terminal_qualification.lanes[0].targets[1] = structuredClone(
    duplicateCell.terminal_qualification.lanes[0].targets[0],
  );
  assert.throws(() => parseTargetPolicy(duplicateCell), /must classify every frozen R31 target exactly once/);

  const duplicateLane = policy();
  duplicateLane.terminal_qualification.lanes.push(structuredClone(
    duplicateLane.terminal_qualification.lanes[0],
  ));
  assert.throws(() => parseTargetPolicy(duplicateLane), /lanes must be unique/);
});

test("terminal qualification cannot omit, replace, or rename an R31 product domain", () => {
  const omitted = policy();
  omitted.terminal_qualification.lanes.splice(1, 1);
  assert.throws(() => parseTargetPolicy(omitted), /frozen R31 product-domain taxonomy/);

  const replaced = policy();
  replaced.terminal_qualification.lanes[0].product_id = "arbitrary-product";
  assert.throws(() => parseTargetPolicy(replaced), /frozen R31 product-domain taxonomy/);

  const renamedPurpose = policy();
  renamedPurpose.terminal_qualification.lanes[0].purpose = "A vague substitute";
  assert.throws(() => parseTargetPolicy(renamedPurpose), /frozen R31 product-domain taxonomy/);
});

test("target-specific inapplicability is explicit and does not imply a Cartesian product", () => {
  const value = policy();
  const windows = value.terminal_qualification.lanes[0].targets.at(-1);
  windows.applicability = "not-applicable";
  windows.execution_order = null;
  windows.reason = "The browser artifact is not distributed on Windows";
  windows.evidence = ["reviewed product support matrix"];
  const parsed = parseTargetPolicy(value);
  assert.equal(parsed.terminalQualificationMatrix.length, R31_REQUIRED_LANES.length * 5 - 1);
  assert.ok(parsed.terminalQualificationMatrix.some((entry) =>
    entry.product_id === "desktop" && entry.operating_system === "windows"));

  const unreviewed = policy();
  const target = unreviewed.terminal_qualification.lanes[0].targets.at(-1);
  target.applicability = "not-applicable";
  target.execution_order = null;
  assert.throws(() => parseTargetPolicy(unreviewed), /inapplicability reason/);
});

test("migration targets need a required terminal cell and Windows remains last", () => {
  const unavailable = policy();
  unavailable.migration_execution_targets = [
    { operating_system: "freebsd", architecture: "x86_64" },
  ];
  assert.throws(() => parseTargetPolicy(unavailable), /required terminal qualification cell/);

  const earlyWindows = policy();
  for (const lane of earlyWindows.terminal_qualification.lanes) {
    lane.targets.at(-1).execution_order = 1;
  }
  assert.throws(() => parseTargetPolicy(earlyWindows), /Windows qualification must execute after/);
});

test("each terminal product lane retains at least one executable qualification target", () => {
  const value = policy();
  for (const target of value.terminal_qualification.lanes[0].targets) {
    target.applicability = "not-applicable";
    target.execution_order = null;
    target.reason = "Unsupported on this target";
    target.evidence = ["reviewed support record"];
  }
  assert.throws(() => parseTargetPolicy(value), /must require at least one target/);
});

function policy() {
  return {
    schema_version: 1,
    kind: "runmat-builtin-migration-target-policy",
    migration_execution_targets: [
      { operating_system: "macos", architecture: "aarch64" },
    ],
    terminal_qualification: {
      phase: "R31",
      lanes: R31_REQUIRED_LANES.map((definition) => lane(definition)),
    },
  };
}

function lane(definition) {
  return {
    ...definition,
    targets: [
      target("linux", "aarch64", 1),
      target("linux", "x86_64", 1),
      target("macos", "aarch64", 1),
      target("macos", "x86_64", 1),
      target("windows", "x86_64", 2),
    ],
  };
}

function target(operatingSystem, architecture, executionOrder) {
  return {
    operating_system: operatingSystem,
    architecture,
    applicability: "required",
    execution_order: executionOrder,
    reason: null,
    evidence: [],
  };
}
