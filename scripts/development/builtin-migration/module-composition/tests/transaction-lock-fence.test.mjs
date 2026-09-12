import assert from "node:assert/strict";
import test from "node:test";

import {
  acquireProcessFence, processFenceIsActive, releaseProcessFence,
} from "../transaction-lock-fence.mjs";

test("fence acquisition and probing terminate timed-out workers", () => {
  for (const operation of [
    (factory) => acquireProcessFence("a".repeat(32), { createWorker: factory, waitMs: 1 }),
    (factory) => processFenceIsActive(12_345, "a".repeat(32), { createWorker: factory, waitMs: 1 }),
  ]) {
    let terminated = 0;
    assert.throws(() => operation(() => ({ terminate() { terminated += 1; } })), /timed out/);
    assert.equal(terminated, 1);
  }
});

test("release terminates the serving worker after a post-publication failure", () => {
  let terminated = 0;
  const fence = acquireProcessFence("b".repeat(32), {
    waitMs: 1,
    createWorker(workerData) {
      const state = new Int32Array(workerData.state);
      Atomics.store(state, 1, 12_345);
      Atomics.store(state, 0, 1);
      return {
        postMessage() { Atomics.store(state, 0, -1); Atomics.notify(state, 0); },
        terminate() { terminated += 1; },
      };
    },
  });
  assert.throws(() => releaseProcessFence(fence), /release failed/);
  assert.equal(terminated, 1);
});
