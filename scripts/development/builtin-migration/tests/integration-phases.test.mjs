import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";

import { evidenceDigest } from "../evidence.mjs";
import {
  captureMigrationPhases, validateMigrationPhasePaths, validateSealedMigrationPhases,
} from "../integration-phases.mjs";
import { buildInventory } from "../inventory.mjs";
import {
  cleanupRepositoryFixtures, controlledFixture,
} from "./helpers.mjs";
import { dispositionInputFromControl } from "../dispositions.mjs";

test.afterEach(cleanupRepositoryFixtures);

test("migration phases bind the authored commit separately from integration products", () => {
  const fixture = controlledFixture();
  const runtimePath = "crates/runmat-runtime/src/builtins/math/basic/foo.rs";
  appendAndCommit(fixture.repository, runtimePath, "\n// authored migration\n", "authored");
  const authoredRevision = revision(fixture.repository);
  const productPath = "crates/runmat-runtime/src/builtins/generated_wasm_registry.rs";
  appendAndCommit(fixture.repository, productPath, "\n// regenerated\n", "integration");
  const subject = subjectInventory(fixture);

  const phases = captureMigrationPhases(
    fixture.repository, fixture.lease, fixture.control, subject, authoredRevision,
  );
  assert.deepEqual(phases.authored_changed_paths, [runtimePath]);
  assert.deepEqual(phases.integration_changed_paths, [productPath]);
  assert.equal(phases.lease_base_revision, fixture.lease.value.base_revision);
  assert.equal(phases.authored_revision, authoredRevision);
  assert.equal(phases.integrated_revision, subject.source.revision);
  assert.doesNotThrow(() => validateSealedMigrationPhases(
    fixture.repository, fixture.control, fixture.bundleId, phases,
  ));
});

test("authored and integration phases reject changes owned by the other authority", () => {
  const authoredViolation = controlledFixture();
  appendAndCommit(authoredViolation.repository, "Cargo.toml", "# outside lease\n", "outside");
  assert.throws(() => captureMigrationPhases(
    authoredViolation.repository, authoredViolation.lease, authoredViolation.control,
    subjectInventory(authoredViolation), revision(authoredViolation.repository),
  ), /authored phase changed paths outside/);

  const integrationViolation = controlledFixture();
  const runtimePath = "crates/runmat-runtime/src/builtins/math/basic/foo.rs";
  appendAndCommit(integrationViolation.repository, runtimePath, "\n// authored\n", "authored");
  const authoredRevision = revision(integrationViolation.repository);
  appendAndCommit(integrationViolation.repository, "Cargo.toml", "# not a product\n", "integration");
  assert.throws(() => captureMigrationPhases(
    integrationViolation.repository, integrationViolation.lease, integrationViolation.control,
    subjectInventory(integrationViolation), authoredRevision,
  ), /integration phase changed non-product paths/);
});

test("phase capture rejects an authored revision outside the integrated ancestry", () => {
  const fixture = controlledFixture();
  const base = fixture.lease.value.base_revision.slice("git:".length);
  const tree = execFileSync("git", ["rev-parse", `${base}^{tree}`], {
    cwd: fixture.repository, encoding: "utf8",
  }).trim();
  const sibling = execFileSync("git", ["commit-tree", tree, "-p", base, "-m", "sibling"], {
    cwd: fixture.repository, encoding: "utf8", env: gitEnvironment(),
  }).trim();
  appendAndCommit(
    fixture.repository, "crates/runmat-runtime/src/builtins/math/basic/foo.rs",
    "\n// integrated branch\n", "integrated",
  );
  assert.throws(() => captureMigrationPhases(
    fixture.repository, fixture.lease, fixture.control, subjectInventory(fixture), `git:${sibling}`,
  ), /integrated revision does not descend/);
});

test("seal-time reconstruction rejects forged phase paths and policy metadata", () => {
  const fixture = controlledFixture();
  const phases = captureMigrationPhases(
    fixture.repository, fixture.lease, fixture.control, fixture.inventory,
    fixture.inventory.source.revision,
  );
  const forgedPaths = structuredClone(phases);
  forgedPaths.authored_changed_paths = ["Cargo.toml"];
  assert.throws(() => validateSealedMigrationPhases(
    fixture.repository, fixture.control, fixture.bundleId, forgedPaths,
  ), /differs from the exact repository/);

  const forgedPolicy = structuredClone(phases);
  forgedPolicy.reviewed_integration_outputs = [];
  forgedPolicy.integration_outputs_digest = evidenceDigest([]);
  assert.throws(() => validateSealedMigrationPhases(
    fixture.repository, fixture.control, fixture.bundleId, forgedPolicy,
  ), /differs from the exact repository/);
});

test("integration path policy handles shared products and bundles with no products", () => {
  const sharedProduct = "crates/runmat-runtime/src/builtins/generated_wasm_registry.rs";
  const authored = [{ kind: "tree", path: "crates/runmat-runtime/src/builtins/math/basic" }];
  assert.doesNotThrow(() => validateMigrationPhasePaths({
    authored_write_set: authored,
    integration_outputs: [
      { product_id: "wasm-registry", path: sharedProduct, producer: "integration" },
      { product_id: "wasm-registry-alias", path: sharedProduct, producer: "integration" },
    ],
  }, ["crates/runmat-runtime/src/builtins/math/basic/foo.rs"], [sharedProduct]));
  assert.doesNotThrow(() => validateMigrationPhasePaths({
    authored_write_set: authored, integration_outputs: [],
  }, ["crates/runmat-runtime/src/builtins/math/basic/foo.rs"], []));
  assert.throws(() => validateMigrationPhasePaths({
    authored_write_set: authored, integration_outputs: [],
  }, [], [sharedProduct]), /non-product paths/);
});

function subjectInventory(fixture) {
  return buildInventory(fixture.repository, dispositionInputFromControl(fixture.control), {
    compiledInventory: fixture.compiledInventory,
  });
}

function appendAndCommit(repository, sourcePath, contents, message) {
  fs.appendFileSync(path.join(repository, sourcePath), contents);
  execFileSync("git", ["add", sourcePath], { cwd: repository });
  execFileSync("git", ["-c", "commit.gpgsign=false", "commit", "--quiet", "-m", message], {
    cwd: repository, env: gitEnvironment(),
  });
}

function revision(repository) {
  return `git:${execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: repository, encoding: "utf8",
  }).trim()}`;
}

function gitEnvironment() {
  return {
    ...process.env,
    GIT_AUTHOR_NAME: "RunMat Test", GIT_AUTHOR_EMAIL: "test@runmat.invalid",
    GIT_COMMITTER_NAME: "RunMat Test", GIT_COMMITTER_EMAIL: "test@runmat.invalid",
  };
}
