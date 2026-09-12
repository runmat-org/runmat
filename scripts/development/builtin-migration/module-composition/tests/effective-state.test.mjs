import assert from "node:assert/strict";
import test from "node:test";

import { controlledFixture, cleanupRepositoryFixtures } from "../../tests/helpers.mjs";
import { deriveEffectiveModuleComposition } from "../effective-state.mjs";

test.afterEach(cleanupRepositoryFixtures);

test("effective composition is derived from the reviewed baseline and active lease", () => {
  const fixture = controlledFixture({ composition: true });
  assert.equal(
    fixture.control.moduleComposition.transitions.get(fixture.bundleId).transition_id,
    fixture.bundleId,
  );
  const projection = deriveEffectiveModuleComposition({
    control: fixture.control,
    queueState: fixture.queueState,
    queueCheckpoint: fixture.queueCheckpoint,
    lease: fixture.lease,
  });
  const runtimeMath = projection.products.find((product) => product.product_id === "runtime-math");
  assert.deepEqual(runtimeMath.children.map((child) => child.module), ["basic"]);
  assert.equal(
    runtimeMath.children[0].source_path,
    "crates/runmat-runtime/src/builtins/math/basic/mod.rs",
  );
});

test("baseline-only products do not require an unrelated active bundle transition", () => {
  const fixture = controlledFixture();
  assert.deepEqual(fixture.lease.bundle.integration_outputs.map((entry) => entry.product_id), [
    "wasm-registry",
  ]);
  assert.equal(
    fixture.control.integrationProducts.get("runtime-math").lifecycle.kind,
    "reviewed-baseline-only",
  );
  assert.equal(fixture.control.moduleComposition.transitions.get(fixture.bundleId), null);

  const projection = deriveEffectiveModuleComposition({
    control: fixture.control,
    queueState: fixture.queueState,
    queueCheckpoint: fixture.queueCheckpoint,
    lease: fixture.lease,
  });

  assert.ok(projection.products.every((product) => product.children.length === 0));
});

test("effective composition requires branded queue authority and its exact lease checkpoint", () => {
  const fixture = controlledFixture({ composition: true });
  const request = {
    control: fixture.control,
    queueState: structuredClone(fixture.queueState.value),
    queueCheckpoint: fixture.queueCheckpoint,
    lease: fixture.lease,
  };
  assert.throws(
    () => deriveEffectiveModuleComposition(request),
    /exact validated queue state/,
  );
  assert.throws(
    () => deriveEffectiveModuleComposition({
      ...request,
      queueState: fixture.queueState,
      queueCheckpoint: structuredClone(fixture.queueCheckpoint.value),
    }),
    /exact validated queue checkpoint/,
  );
});

test("effective composition retains lease expiry as an execution-time authority", () => {
  const fixture = controlledFixture({ composition: true });
  assert.throws(
    () => deriveEffectiveModuleComposition({
      control: fixture.control,
      queueState: fixture.queueState,
      queueCheckpoint: fixture.queueCheckpoint,
      lease: fixture.lease,
      clock: () => Date.parse(fixture.lease.value.expires_at),
    }),
    /has expired/,
  );
});
