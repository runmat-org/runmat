import assert from "node:assert/strict";
import test from "node:test";

import { controlledFixture, cleanupRepositoryFixtures } from "../../tests/helpers.mjs";
import { deriveEffectiveModuleComposition } from "../effective-state.mjs";

test.afterEach(cleanupRepositoryFixtures);

test("effective composition is derived from the reviewed baseline and active lease", () => {
  const fixture = controlledFixture({ composition: true });
  const projection = deriveEffectiveModuleComposition({
    control: fixture.control,
    queueState: fixture.queueState,
    queueCheckpoint: fixture.queueCheckpoint,
    lease: fixture.lease,
  });
  assert.deepEqual(projection.products[0].children.map((child) => child.module), ["basic"]);
  assert.equal(
    projection.products[0].children[0].source_path,
    "crates/runmat-runtime/src/builtins/math/basic/mod.rs",
  );
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
