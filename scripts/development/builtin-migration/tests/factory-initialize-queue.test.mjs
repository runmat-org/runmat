import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import test, { after } from "node:test";

import { evidenceDigest } from "../evidence.mjs";
import {
  cleanFactoryCliRepository, writeRepositoryControlWorkflow,
} from "./factory-workflow-fixture.mjs";
import {
  cleanupTemporaryDirectories, createTemporaryDirectory,
} from "./temporary-directories.mjs";

after(cleanupTemporaryDirectories);

test("initialize-queue CLI publishes authority accepted by lease issuance", () => {
  const directory = createTemporaryDirectory("runmat-initialize-queue-cli-");
  const repository = cleanFactoryCliRepository();
  const workflow = writeRepositoryControlWorkflow(directory, repository);
  const reviewPayload = {
    schema_version: 1,
    kind: "runmat-builtin-migration-initial-queue-review",
    authority: "reviewer-authored-development-input",
    control_manifest_digest: workflow.control.digest,
    review: {
      status: "reviewed", evidence: ["full-chain current queue checkpoint"],
    },
  };
  const review = { ...reviewPayload, digest: evidenceDigest(reviewPayload) };
  const reviewPath = path.join(directory, "initial-queue-review.json");
  const resultPath = path.join(directory, "initial-queue-result.json");
  writeJson(reviewPath, review);
  const cli = path.join(repository, "scripts/development/builtin-migration-factory.mjs");
  const initialized = spawnSync(process.execPath, [
    cli, "initialize-queue", "--authority-root", directory,
    "--initial-queue-review", reviewPath,
    "--initial-queue-review-digest", review.digest,
    "--control", workflow.paths.control,
    "--baseline-inventory", workflow.paths.inventory,
    ...workflow.topologyArguments, ...workflow.controlArguments,
    "--output", resultPath,
  ], { encoding: "utf8" });
  assert.equal(initialized.status, 0, initialized.stderr);
  const result = JSON.parse(fs.readFileSync(resultPath, "utf8"));
  assert.equal(result.kind, "runmat-builtin-migration-initialize-queue-command-result");
  const retryPath = path.join(directory, "initial-queue-retry-result.json");
  const retried = spawnSync(process.execPath, [
    cli, "initialize-queue", "--authority-root", directory,
    "--initial-queue-review", reviewPath,
    "--initial-queue-review-digest", review.digest,
    "--control", workflow.paths.control,
    "--baseline-inventory", workflow.paths.inventory,
    ...workflow.topologyArguments, ...workflow.controlArguments,
    "--output", retryPath,
  ], { encoding: "utf8" });
  assert.equal(retried.status, 0, retried.stderr);
  assert.deepEqual(JSON.parse(fs.readFileSync(retryPath, "utf8")), result);
  const statePath = path.join(directory, result.state.path);
  const checkpointPath = path.join(directory, result.checkpoint.path);

  const lease = spawnSync(process.execPath, [
    cli, "issue-lease", "--request", workflow.paths.leaseRequest,
    "--control", workflow.paths.control,
    "--baseline-inventory", workflow.paths.inventory,
    "--lease-base-inventory", workflow.paths.inventory,
    "--state", statePath, "--queue-checkpoint", checkpointPath,
    "--trusted-queue-checkpoint-digest", result.checkpoint.digest,
    ...workflow.topologyArguments, ...workflow.controlArguments,
  ], { encoding: "utf8" });
  assert.equal(lease.status, 0, lease.stderr);
  assert.equal(JSON.parse(lease.stdout).queue_checkpoint_digest, result.checkpoint.digest);
  assert.deepEqual(publicationResidue(directory), []);
});

test("concurrent conflicting reviews publish exactly one initial checkpoint", async () => {
  const directory = createTemporaryDirectory("runmat-initialize-queue-race-");
  const repository = cleanFactoryCliRepository();
  const workflow = writeRepositoryControlWorkflow(directory, repository);
  const reviews = ["first independent review", "second independent review"].map((evidence, index) => {
    const payload = {
      schema_version: 1,
      kind: "runmat-builtin-migration-initial-queue-review",
      authority: "reviewer-authored-development-input",
      control_manifest_digest: workflow.control.digest,
      review: { status: "reviewed", evidence: [evidence] },
    };
    const value = { ...payload, digest: evidenceDigest(payload) };
    const reviewPath = path.join(directory, `initial-review-${index}.json`);
    writeJson(reviewPath, value);
    return { value, path: reviewPath, result: path.join(directory, `result-${index}.json`) };
  });
  const cli = path.join(repository, "scripts/development/builtin-migration-factory.mjs");
  const runs = await Promise.all(reviews.map((review) => runProcess(process.execPath, [
    cli, "initialize-queue", "--authority-root", directory,
    "--initial-queue-review", review.path,
    "--initial-queue-review-digest", review.value.digest,
    "--control", workflow.paths.control,
    "--baseline-inventory", workflow.paths.inventory,
    ...workflow.topologyArguments, ...workflow.controlArguments,
    "--output", review.result,
  ])));
  assert.deepEqual(runs.map((run) => run.status).sort(), [0, 2]);
  const winner = reviews.find((review) => fs.existsSync(review.result));
  const result = JSON.parse(fs.readFileSync(winner.result, "utf8"));
  const checkpoint = JSON.parse(fs.readFileSync(path.join(directory, result.checkpoint.path), "utf8"));
  assert.deepEqual(checkpoint.review, winner.value.review);
  assert.match(runs.find((run) => run.status === 2).stderr, /differs from expected bytes/);
  assert.deepEqual(publicationResidue(directory), []);
});

function runProcess(command, arguments_) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, arguments_, { encoding: "utf8" });
    let stdout = ""; let stderr = "";
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", reject);
    child.on("close", (status) => resolve({ status, stdout, stderr }));
  });
}

function writeJson(target, value) {
  fs.writeFileSync(target, `${JSON.stringify(value, null, 2)}\n`);
}

function publicationResidue(root) {
  return fs.readdirSync(root, { recursive: true })
    .filter((entry) => entry.includes(".runmat-new-"));
}
