import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import test from "node:test";

import { evidenceDigest } from "../evidence.mjs";
import {
  finalAuthorityFailuresForBundles, finalIdentityAuthorityFailures,
} from "../inventory-delta.mjs";
import { validateQueueCheckpoint } from "../queue-checkpoint.mjs";
import { validateMonotonicQueueTransition } from "../queue-history.mjs";
import { validateSerializedSealTransition } from "../queue.mjs";
import { cleanupRepositoryFixtures, controlledFixture } from "./helpers.mjs";

test.afterEach(cleanupRepositoryFixtures);

test("trusted current checkpoint rejects replay of an older queue prefix", () => {
  const fixture = controlledFixture();
  assert.equal(
    `git:${execFileSync("git", ["rev-parse", "HEAD"], { cwd: fixture.repository, encoding: "utf8" }).trim()}`,
    fixture.control.baseline.revision,
    "the repository is intentionally rewound to the old prefix's source revision",
  );
  const publishedCurrentCheckpoint = structuredClone(fixture.queueCheckpointValue);
  publishedCurrentCheckpoint.review.evidence = ["durably published current checkpoint"];
  delete publishedCurrentCheckpoint.digest;
  const currentTrustedDigest = evidenceDigest(publishedCurrentCheckpoint);
  assert.throws(
    () => validateQueueCheckpoint(
      fixture.queueCheckpointValue,
      currentTrustedDigest,
      fixture.queueState,
      fixture.control,
    ),
    /differs from the trusted current checkpoint digest/,
  );
});

test("serialized finalization rejects a same-base peer seal after the queue advances", () => {
  const first = sealReference("first");
  const peer = sealReference("peer");
  assert.doesNotThrow(() => validateSerializedSealTransition([], first, []));
  assert.throws(
    () => validateSerializedSealTransition([first], peer, []),
    /peer: accepted seal was produced from a stale queue checkpoint/,
  );
  assert.doesNotThrow(() => validateSerializedSealTransition([first], peer, [first]));
});

test("queue history cannot omit an accepted seal or regress workflow state", () => {
  const first = sealReference("first");
  const predecessor = {
    acceptedSeals: [first],
    value: { bundles: { first: { artifact: "seal-first", state: "verified" } } },
  };
  assert.throws(
    () => validateMonotonicQueueTransition(
      { bundles: {} }, predecessor, [], [],
    ),
    /removed or replaced accepted seal first/,
  );
  assert.throws(
    () => validateMonotonicQueueTransition(
      { bundles: { first: { artifact: "seal-first", state: "leased" } } },
      predecessor,
      [first, sealReference("second")],
      [{ bundleId: "second", accepted: [first] }],
    ),
    /workflow state was removed or regressed/,
  );
});

test("sealed final authority detects loss of a required WASM registration", () => {
  const fixture = controlledFixture();
  const controlled = structuredClone(fixture.control.identities.get(fixture.id));
  controlled.expected_authorities.wasm_registry = "required";
  const current = structuredClone(fixture.inventory.identities[0]);
  current.registrations.wasm = [controlled.implementation.callable.bindings[0].function];
  assert.deepEqual(finalIdentityAuthorityFailures(fixture.id, current, controlled), []);
  current.registrations.wasm = [];
  assert.match(
    finalIdentityAuthorityFailures(fixture.id, current, controlled).join("\n"),
    /WASM registration set differs from review/,
  );
  controlled.expected_authorities.wasm_registry = "not-applicable";
  current.registrations.wasm = [controlled.implementation.callable.bindings[0].function];
  assert.match(
    finalIdentityAuthorityFailures(fixture.id, current, controlled).join("\n"),
    /WASM registration set differs from review/,
    "not-applicable must reject an unreviewed registration rather than ignoring it",
  );
  controlled.expected_authorities.wasm_registry = "required";
  current.registrations.wasm = [];
  const priorSealedInventory = structuredClone(fixture.inventory);
  priorSealedInventory.identities[0] = current;
  const priorControl = {
    bundles: fixture.control.bundles,
    identities: new Map(fixture.control.identities).set(fixture.id, controlled),
  };
  assert.match(
    finalAuthorityFailuresForBundles(
      priorSealedInventory, priorControl, [fixture.bundleId],
    ).join("\n"),
    /WASM registration set differs from review/,
  );
});

function sealReference(bundleId) {
  return {
    path: `seals/${bundleId}.json`,
    artifact_id: `seal-${bundleId}`,
    digest: `sha256:${(bundleId === "first" ? "1" : "2").repeat(64)}`,
    bundle_id: bundleId,
  };
}
