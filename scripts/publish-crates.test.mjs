import assert from "node:assert/strict";
import test from "node:test";
import { publishWithBackoff, retryDelay } from "./publish-crates.mjs";

const now = Date.parse("2026-11-13T06:00:00Z");
const refusal = (kind = "new crates", date = "Fri, 13 Nov 2026 06:10:00 GMT") =>
  `error: You have published too many ${kind} in a short period of time. Please try again after ${date} or email help@crates.io to have your limit increased.`;

test("uses the registry retry time for each independent quota", () => {
  assert.equal(retryDelay(refusal(), now), 602_000);
  assert.equal(retryDelay(refusal("updates to existing crates"), now), 602_000);
});
test("a past retry time still backs off", () => {
  assert.equal(retryDelay(refusal(), now + 1_000_000), 2_000);
});
test("does not retry authentication, packaging, dependencies or generic HTTP errors", () => {
  for (const output of ["HTTP 401: invalid token", "package too large", "no matching package named runmat-design-tools",
    "HTTP 429", "network timeout", refusal("new crates", "invalid GMT"), refusal("new crates", "Fri, 13 Nov 2037 06:10:00 GMT")]) {
    assert.equal(retryDelay(output, now), null);
  }
});
test("resumes the publisher after an explicit quota refusal", async () => {
  const waits = [];
  let runs = 0;
  await publishWithBackoff({ run: async () => ++runs === 1 ? { code: 101, output: refusal() } : { code: 0, output: "" },
    wait: async delay => waits.push(delay), now: () => now, log: () => {} });
  assert.equal(runs, 2);
  assert.deepEqual(waits, [602_000]);
});
test("stops immediately on an unrelated failure", async () => {
  await assert.rejects(publishWithBackoff({ run: async () => ({ code: 101, output: "invalid token" }),
    wait: async () => assert.fail("must not wait") }), /publication stopped/);
});
test("bounds retries even if the registry keeps refusing", async () => {
  let runs = 0;
  await assert.rejects(publishWithBackoff({ run: async () => { runs++; return { code: 101, output: refusal() }; },
    wait: async () => {}, now: () => now, log: () => {}, maxAttempts: 2 }), /publication stopped/);
  assert.equal(runs, 2);
});
