import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";

import { cleanupRepositoryFixtures } from "../../tests/helpers.mjs";
import {
  copyReviewSetAsAuthoringFiles, writeFullControlWorkflow,
} from "../../tests/factory-workflow-fixture.mjs";
import { createTemporaryDirectory } from "../../tests/temporary-directories.mjs";

const cli = path.resolve("scripts/development/builtin-migration-factory.mjs");

test.afterEach(cleanupRepositoryFixtures);

test("review validators emit deterministic pass evidence without changing authoring inputs", () => {
  const fixture = validationFixture();
  const globalBefore = fs.readFileSync(fixture.globalPath);
  const bundleBefore = fs.readFileSync(fixture.bundlePath);

  const firstGlobal = invoke("validate-global-control-review", [
    ...fixture.contextArguments, "--global-review", fixture.globalPath,
  ]);
  assert.equal(firstGlobal.status, 0, firstGlobal.stderr);
  const secondGlobal = invoke("validate-global-control-review", [
    ...fixture.contextArguments, "--global-review", fixture.globalPath,
  ]);
  assert.equal(secondGlobal.status, 0, secondGlobal.stderr);
  assert.equal(secondGlobal.stdout, firstGlobal.stdout);
  const globalReport = JSON.parse(firstGlobal.stdout);
  assert.equal(globalReport.result, "pass");
  assert.deepEqual(globalReport.subject, { kind: "global" });
  assert.equal(globalReport.bindings.global_review_digest, null);

  const firstBundle = invoke("validate-bundle-control-review", [
    ...fixture.contextArguments,
    "--global-review", fixture.globalPath,
    "--bundle-review", fixture.bundlePath,
    "--bundle", fixture.workflow.bundleId,
  ]);
  assert.equal(firstBundle.status, 0, firstBundle.stderr);
  const secondBundle = invoke("validate-bundle-control-review", [
    ...fixture.contextArguments,
    "--global-review", fixture.globalPath,
    "--bundle-review", fixture.bundlePath,
    "--bundle", fixture.workflow.bundleId,
  ]);
  assert.equal(secondBundle.status, 0, secondBundle.stderr);
  assert.equal(secondBundle.stdout, firstBundle.stdout);
  const bundleReport = JSON.parse(firstBundle.stdout);
  assert.equal(bundleReport.result, "pass");
  assert.deepEqual(bundleReport.subject, {
    kind: "bundle", bundle_id: fixture.workflow.bundleId,
  });
  assert.equal(bundleReport.bindings.global_review_digest, globalReport.bindings.review_digest);

  assert.deepEqual(fs.readFileSync(fixture.globalPath), globalBefore);
  assert.deepEqual(fs.readFileSync(fixture.bundlePath), bundleBefore);
});

test("review validators reject ambiguous output, caller digests, drift, and wrong bundle identity", () => {
  const fixture = validationFixture();
  const output = path.join(fixture.directory, "must-not-exist.json");
  const withOutput = invoke("validate-global-control-review", [
    ...fixture.contextArguments, "--global-review", fixture.globalPath,
    "--output", output,
  ]);
  assert.equal(withOutput.status, 2);
  assert.match(withOutput.stderr, /only to stdout; --output is not accepted/);
  assert.equal(fs.existsSync(output), false);

  const unrelatedGlobalOption = invoke("validate-global-control-review", [
    ...fixture.contextArguments, "--global-review", fixture.globalPath,
    "--request", fixture.bundlePath,
  ]);
  assert.equal(unrelatedGlobalOption.status, 2);
  assert.match(
    unrelatedGlobalOption.stderr,
    /validate-global-control-review does not accept --request/,
  );

  const unrelatedBundleOption = invoke("validate-bundle-control-review", [
    ...fixture.contextArguments,
    "--global-review", fixture.globalPath,
    "--bundle-review", fixture.bundlePath,
    "--bundle", fixture.workflow.bundleId,
    "--review-directory", fixture.directory,
  ]);
  assert.equal(unrelatedBundleOption.status, 2);
  assert.match(
    unrelatedBundleOption.stderr,
    /validate-bundle-control-review does not accept --review-directory/,
  );

  const suppliedDigest = readJson(fixture.globalPath);
  suppliedDigest.digest = `sha256:${"0".repeat(64)}`;
  const suppliedDigestPath = writeJson(fixture.directory, "global-with-digest.json", suppliedDigest);
  const digestResult = invoke("validate-global-control-review", [
    ...fixture.contextArguments, "--global-review", suppliedDigestPath,
  ]);
  assert.equal(digestResult.status, 2);
  assert.match(digestResult.stderr, /must not supply its own digest/);

  const drift = readJson(fixture.bundlePath);
  drift.bindings.scaffold_bundle_row_digest = `sha256:${"0".repeat(64)}`;
  const driftPath = writeJson(fixture.directory, "bundle-drift.json", drift);
  const driftResult = invoke("validate-bundle-control-review", [
    ...fixture.contextArguments,
    "--global-review", fixture.globalPath,
    "--bundle-review", driftPath,
    "--bundle", fixture.workflow.bundleId,
  ]);
  assert.equal(driftResult.status, 2);
  assert.match(driftResult.stderr, /scaffold row digest mismatch/);

  const missingProfile = readJson(fixture.bundlePath);
  missingProfile.bundle_control.gate_plans[0].program_profile_id = "missing-profile";
  const missingProfilePath = writeJson(
    fixture.directory, "bundle-missing-profile.json", missingProfile,
  );
  const missingProfileResult = invoke("validate-bundle-control-review", [
    ...fixture.contextArguments,
    "--global-review", fixture.globalPath,
    "--bundle-review", missingProfilePath,
    "--bundle", fixture.workflow.bundleId,
  ]);
  assert.equal(missingProfileResult.status, 2);
  assert.match(missingProfileResult.stderr, /unknown global program profile/);

  const wrongBundle = invoke("validate-bundle-control-review", [
    ...fixture.contextArguments,
    "--global-review", fixture.globalPath,
    "--bundle-review", fixture.bundlePath,
    "--bundle", "c00-not-the-reviewed-bundle",
  ]);
  assert.equal(wrongBundle.status, 2);
  assert.match(wrongBundle.stderr, /not expected bundle/);

  const missingPath = invoke("validate-global-control-review", [
    ...fixture.contextArguments,
  ]);
  assert.equal(missingPath.status, 2);
  assert.match(missingPath.stderr, /requires --global-review/);
});

function validationFixture() {
  const directory = createTemporaryDirectory("runmat-review-validation-");
  const workflow = writeFullControlWorkflow(directory, `git:${"1".repeat(40)}`);
  const authored = path.join(directory, "authored");
  copyReviewSetAsAuthoringFiles(workflow.paths.controlReviewSet, authored);
  return {
    directory,
    workflow,
    globalPath: path.join(authored, "global.json"),
    bundlePath: path.join(authored, "bundles", `${workflow.bundleId}.json`),
    contextArguments: [
      "--baseline-inventory", workflow.paths.inventory,
      ...workflow.topologyArguments,
      "--control-scaffold", workflow.paths.scaffold,
    ],
  };
}

function invoke(command, arguments_) {
  return spawnSync(process.execPath, [cli, command, ...arguments_], { encoding: "utf8" });
}

function readJson(target) {
  return JSON.parse(fs.readFileSync(target, "utf8"));
}

function writeJson(directory, name, value) {
  const target = path.join(directory, name);
  fs.writeFileSync(target, `${JSON.stringify(value, null, 2)}\n`);
  return target;
}
