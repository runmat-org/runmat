import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";

import { evidenceDigest } from "../evidence.mjs";
import { captureMigrationPhases } from "../integration-phases.mjs";
import { deriveEffectiveModuleComposition } from "../module-composition/effective-state.mjs";
import { materializeEffectiveModuleComposition } from "../module-composition/materialize.mjs";
import { validateQueueState } from "../queue.mjs";
import { cleanupRepositoryFixtures } from "./helpers.mjs";
import { acceptFirstBundle } from "./sequential-shared-parent-authority-fixture.mjs";
import {
  buildSubjectInventory, commitFixture, compositionChild, leaseFor,
  sequentialSharedParentFixture, writeFixture,
} from "./sequential-shared-parent-fixture.mjs";
import {
  compiledInventoryWithManifest, verifyGeneratedProducts, writeWasmRegistry,
} from "./sequential-shared-parent-products-fixture.mjs";

test.afterEach(cleanupRepositoryFixtures);

test("a second sealed-plus-active integration retains the first shared-parent child", () => {
  const state = integrateFirstBundle(sequentialSharedParentFixture());
  const second = integrateSecondBundle(state);
  const parent = fs.readFileSync(parentPath(second.fixture), "utf8");
  assert.match(parent, /pub mod alpha_support;/);
  assert.match(parent, /pub mod beta_support;/);
  assert.ok(parent.indexOf("alpha_support") < parent.indexOf("beta_support"));
  assert.deepEqual(second.projection.products
    .find((product) => product.product_id === "runtime-math")
    .children.map((child) => child.module), ["alpha_support", "basic", "beta_support"]);
  assert.deepEqual(second.nativeManifest.entries
    .filter((entry) => entry.kind === "builtin")
    .map((entry) => entry.declaration), ["alpha", "beta"]);
  assert.match(second.wasmSource, /\tbuiltin\talpha\tsome\tdefault\tbuiltins::alpha/);
  assert.match(second.wasmSource, /\tbuiltin\tbeta\tsome\tdefault\tbuiltins::beta/);
  assert.equal(second.products.proof.result, "pass");
  assert.deepEqual(second.products.decoded.native_registration_manifest, {
    schema_version: 1,
    digest: second.nativeManifest.digest,
    counts: second.nativeManifest.counts,
  });
});

test("the successor queue cannot omit the first serialized seal", () => {
  const state = integrateFirstBundle(sequentialSharedParentFixture());
  const value = structuredClone(state.accepted.queueStateValue);
  value.seals = [];
  delete value.digest;
  value.digest = evidenceDigest(value);
  assert.throws(() => validateQueueState(
    value, state.fixture.control, () => null, () => state.fixture.queueState,
  ), /must accept exactly one new serialized seal/);
});

test("the successor queue rejects cloned and fabricated predecessor authority", () => {
  const state = integrateFirstBundle(sequentialSharedParentFixture());
  for (const predecessor of [
    structuredClone(state.fixture.queueState),
    Object.freeze({
      value: state.fixture.queueState.value,
      controlDigest: state.fixture.control.digest,
      stateDigest: state.fixture.queueState.stateDigest,
    }),
  ]) {
    assert.throws(() => validateQueueState(
      state.accepted.queueStateValue,
      state.fixture.control,
      () => state.accepted.seal,
      () => predecessor,
    ), /requires an exact validated queue state/);
  }
});

test("the second materialization rejects parent bytes outside the sealed prior projection", () => {
  const state = integrateFirstBundle(sequentialSharedParentFixture());
  fs.appendFileSync(parentPath(state.fixture), "// unreviewed parent drift\n");
  authorSecondChild(state.fixture);
  assert.throws(() => materializeEffectiveModuleComposition({
    repository: state.fixture.repository,
    control: state.fixture.control,
    queueState: state.accepted.queueState,
    queueCheckpoint: state.accepted.queueCheckpoint,
    lease: state.secondLease.lease,
  }), /parent bytes differ from the canonical prior projection/);
});

test("a beta-only WASM product cannot pass against the retained native manifest", () => {
  const state = integrateFirstBundle(sequentialSharedParentFixture());
  const second = integrateSecondBundle(state);
  const betaOnly = compiledInventoryWithManifest(second.fixture.compiledInventory, ["beta"])
    .snapshot.observed.registration_manifest;
  writeWasmRegistry(second.fixture.repository, betaOnly);
  const products = verifyGeneratedProducts({
    fixture: second.fixture,
    subjectInventory: second.subjectInventory,
    projection: second.projection,
  });
  assert.equal(products.proof.result, "fail");
  const wasm = products.proof.products.find((product) => product.product_id === "wasm-registry");
  assert.equal(wasm.verification.result, "fail");
  assert.notEqual(
    wasm.verification.generated_manifest.digest,
    wasm.verification.native_manifest.digest,
  );
});

function integrateFirstBundle(fixture) {
  writeFixture(
    fixture.repository, compositionChild("alpha").source_path,
    "pub fn alpha_support() {}\n",
  );
  const authoredRevision = commitFixture(fixture.repository, "author alpha child");
  const projection = deriveEffectiveModuleComposition({
    control: fixture.control,
    queueState: fixture.queueState,
    queueCheckpoint: fixture.queueCheckpoint,
    lease: fixture.firstLease.lease,
  });
  materializeEffectiveModuleComposition({
    repository: fixture.repository,
    control: fixture.control,
    queueState: fixture.queueState,
    queueCheckpoint: fixture.queueCheckpoint,
    lease: fixture.firstLease.lease,
  });
  writeWasmRegistry(
    fixture.repository, fixture.compiledInventory.snapshot.observed.registration_manifest,
  );
  commitFixture(fixture.repository, "integrate alpha products");
  const subjectInventory = buildSubjectInventory(fixture);
  assert.equal(verifyGeneratedProducts({ fixture, subjectInventory, projection }).proof.result, "pass");
  const phases = captureMigrationPhases(
    fixture.repository, fixture.firstLease.lease, fixture.control,
    subjectInventory, authoredRevision,
  );
  const accepted = acceptFirstBundle({ fixture, subjectInventory, phases });
  const secondLease = leaseFor({
    repository: fixture.repository,
    control: fixture.control,
    inventory: subjectInventory,
    queueState: accepted.queueState,
    queueCheckpoint: accepted.queueCheckpoint,
    bundleId: fixture.bundleIds[1],
    leaseId: "lease-beta",
  });
  assert.deepEqual(secondLease.lease.value.accepted_seals, [accepted.reference]);
  assert.deepEqual(secondLease.lease.value.barrier_seals, [accepted.reference]);
  return { fixture, subjectInventory, accepted, secondLease };
}

function integrateSecondBundle(state) {
  authorSecondChild(state.fixture);
  const projection = deriveEffectiveModuleComposition({
    control: state.fixture.control,
    queueState: state.accepted.queueState,
    queueCheckpoint: state.accepted.queueCheckpoint,
    lease: state.secondLease.lease,
  });
  materializeEffectiveModuleComposition({
    repository: state.fixture.repository,
    control: state.fixture.control,
    queueState: state.accepted.queueState,
    queueCheckpoint: state.accepted.queueCheckpoint,
    lease: state.secondLease.lease,
  });
  const nativeManifest = state.fixture.compiledInventory.snapshot.observed.registration_manifest;
  const wasmSource = writeWasmRegistry(state.fixture.repository, nativeManifest);
  commitFixture(state.fixture.repository, "integrate beta products");
  const subjectInventory = buildSubjectInventory(state.fixture);
  const products = verifyGeneratedProducts({
    fixture: state.fixture, subjectInventory, projection,
  });
  return {
    fixture: state.fixture, projection, nativeManifest, wasmSource, subjectInventory, products,
  };
}

function authorSecondChild(fixture) {
  writeFixture(
    fixture.repository, compositionChild("beta").source_path,
    "pub fn beta_support() {}\n",
  );
  return commitFixture(fixture.repository, "author beta child");
}

function parentPath(fixture) {
  return path.join(fixture.repository, "crates/runmat-runtime/src/builtins/math/mod.rs");
}
