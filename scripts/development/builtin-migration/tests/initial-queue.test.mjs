import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test, { after } from "node:test";

import {
  openAuthorityLoadSession, openAuthorityRoot,
} from "../authority-loading/index.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { issueLease } from "../lease.mjs";
import { loadLeaseAuthority } from "../lease-authority.mjs";
import {
  deriveInitialQueue, initialQueuePaths, loadInitialQueueReview, recordInitialQueue,
} from "../queue-initialization/index.mjs";
import { recordPilotWorkSessionStart } from "../pilot-work-session/index.mjs";
import { assertLoadedQueueAuthority } from "../queue-authority/index.mjs";
import { buildQueue } from "../queue.mjs";
import { controlledFixture } from "./helpers.mjs";
import {
  cleanupTemporaryDirectories, createTemporaryDirectory,
} from "./temporary-directories.mjs";

after(cleanupTemporaryDirectories);

test("initial queue publication commits an exact empty pilot root and is idempotent", () => {
  const fixture = initialFixture();
  const first = recordInitialQueue(fixture);
  const second = recordInitialQueue(fixture);
  const third = recordInitialQueue(reopenFixture(fixture));
  assert.deepEqual(second.observations, first.observations);
  assert.deepEqual(third.observations, first.observations);
  assertLoadedQueueAuthority(first, { session: fixture.session, control: fixture.control });
  assert.deepEqual(first.state.value, fixture.controlled.queueState.value);
  assert.deepEqual(first.checkpoint.value, fixture.controlled.queueCheckpoint.value);
  assert.equal(first.checkpoint.value.predecessor_checkpoint_digest, null);
  assert.equal(first.checkpoint.value.head_event, null);
  assert.equal(first.checkpoint.value.source_revision, fixture.control.baseline.revision);
  assert.equal(first.checkpoint.value.inventory_digest,
    fixture.control.baseline.inventory_digest);
  const queue = buildQueue(fixture.controlled.inventory, fixture.control, first.state);
  assert.equal(queue.phase, "pilot");
  assert.equal(queue.summary.bundles, fixture.control.bundles.size);
});

test("an exact state-only partial publication resumes at the checkpoint commit point", () => {
  const fixture = initialFixture();
  const derived = deriveInitialQueue(fixture);
  const paths = initialQueuePaths(fixture.control.digest);
  writeJson(path.join(fixture.root, paths.state), derived.stateValue);
  assert.equal(fs.existsSync(path.join(fixture.root, paths.checkpoint)), false);
  const queue = recordInitialQueue(reopenFixture(fixture));
  assert.equal(queue.checkpoint.value.digest, derived.checkpointValue.digest);
  assert.deepEqual(publicationResidue(fixture.root), []);
});

test("an exact checkpoint-only interrupted result reconstructs and reloads its state", () => {
  const fixture = initialFixture();
  const derived = deriveInitialQueue(fixture);
  const paths = initialQueuePaths(fixture.control.digest);
  writeJson(path.join(fixture.root, paths.checkpoint), derived.checkpointValue);
  const queue = recordInitialQueue(reopenFixture(fixture));
  assert.equal(queue.state.value.digest, derived.stateValue.digest);
  assert.equal(queue.checkpoint.value.digest, derived.checkpointValue.digest);
  assert.deepEqual(publicationResidue(fixture.root), []);
});

test("existing divergent state or checkpoint bytes fail without repair", () => {
  for (const divergent of ["state", "checkpoint"]) {
    const fixture = initialFixture();
    const derived = deriveInitialQueue(fixture);
    const paths = initialQueuePaths(fixture.control.digest);
    if (divergent === "checkpoint") {
      writeJson(path.join(fixture.root, paths.state), derived.stateValue);
    }
    const changed = structuredClone(
      divergent === "state" ? derived.stateValue : derived.checkpointValue,
    );
    changed.digest = `sha256:${"f".repeat(64)}`;
    writeJson(path.join(fixture.root, paths[divergent]), changed);
    assert.throws(() => recordInitialQueue(fixture), /differs from expected bytes/);
    if (divergent === "state") {
      assert.equal(fs.existsSync(path.join(fixture.root, paths.checkpoint)), false);
    }
    assert.deepEqual(readJson(path.join(fixture.root, paths[divergent])), changed);
    assert.deepEqual(publicationResidue(fixture.root), []);
  }
});

test("review authority rejects malformed, stale, and rebound inputs", () => {
  for (const [mutate, pattern] of [
    [(value) => { value.schema_version = 2; }, /must use schema_version 1/],
    [(value) => { value.extra = true; }, /fields must be exactly/],
    [(value) => { value.authority = "derived"; }, /invalid authority/],
    [(value) => { value.review.status = "unreviewed"; }, /status must be reviewed/],
    [(value) => { value.review.evidence = []; }, /must be a nonempty array/],
    [(value) => { value.review.evidence = ["review:zeta", "review:alpha"]; },
      /canonically ordered/],
    [(value) => { value.control_manifest_digest = `sha256:${"a".repeat(64)}`; },
      /another control manifest/],
    [(value) => { value.digest = `sha256:${"b".repeat(64)}`; }, /digest mismatch/],
  ]) {
    const controlled = controlledFixture();
    const root = createTemporaryDirectory("runmat-initial-queue-invalid-");
    const value = reviewValue(controlled.control);
    mutate(value);
    if (!pattern.source.includes("digest mismatch")) reseal(value);
    writeJson(path.join(root, "review.json"), value);
    const session = openAuthorityLoadSession(openAuthorityRoot(root));
    assert.throws(() => loadInitialQueueReview({
      session, reference: { path: "review.json", digest: value.digest },
      control: controlled.control,
    }), pattern);
  }
});

test("raw, cross-session, escaped, and changed review authority fails closed", () => {
  const fixture = initialFixture();
  assert.throws(() => deriveInitialQueue({
    session: fixture.session, control: fixture.control,
    review: Object.freeze({ ...fixture.review }),
  }), /exact loaded initial queue review/);
  const otherSession = openAuthorityLoadSession(openAuthorityRoot(fixture.root));
  assert.throws(() => deriveInitialQueue({
    session: otherSession, control: fixture.control, review: fixture.review,
  }), /exact loaded initial queue review/);
  assert.throws(() => loadInitialQueueReview({
    session: otherSession, reference: { path: "../review.json", digest: fixture.review.value.digest },
    control: fixture.control,
  }), /canonical relative POSIX path/);
  assert.throws(() => loadInitialQueueReview({
    session: fixture.session,
    reference: { path: "review.json", digest: `sha256:${"e".repeat(64)}` },
    control: fixture.control,
  }), /path was rebound/);
  const changed = structuredClone(fixture.review.value);
  changed.review.evidence = ["replacement review"];
  reseal(changed);
  fs.writeFileSync(path.join(fixture.root, "review.json"), `${JSON.stringify(changed)}\n`);
  assert.throws(() => recordInitialQueue(fixture), /changed after observation/);
});

test("a second review cannot fork one control's deterministic initial root", () => {
  const first = initialFixture();
  recordInitialQueue(first);
  const alternate = reviewValue(first.control);
  alternate.review.evidence = ["alternate reviewed evidence"];
  reseal(alternate);
  writeJson(path.join(first.root, "alternate-review.json"), alternate);
  const session = openAuthorityLoadSession(openAuthorityRoot(first.root));
  const review = loadInitialQueueReview({
    session, reference: { path: "alternate-review.json", digest: alternate.digest },
    control: first.control,
  });
  assert.throws(() => recordInitialQueue({
    session, control: first.control, review,
  }), /existing initial queue artifact differs from expected bytes/);
  assert.deepEqual(publicationResidue(first.root), []);
});

test("the published root authorizes lease issuance and a pilot session start", () => {
  const fixture = initialFixture();
  const queue = recordInitialQueue(fixture);
  const leaseValue = issueLease(
    fixture.controlled.leaseRequest, fixture.control, fixture.controlled.repository,
    fixture.controlled.inventory, queue.state, queue.checkpoint,
  );
  writeJson(path.join(fixture.root, "lease.json"), leaseValue);
  const lease = loadLeaseAuthority(
    fixture.session, { path: "lease.json", digest: evidenceDigest(leaseValue) },
    fixture.control, fixture.controlled.repository,
  );
  const start = recordPilotWorkSessionStart({
    session: fixture.session, control: fixture.control, queueAuthority: queue,
    leaseAuthority: lease, repository: fixture.controlled.repository,
  });
  assert.equal(start.value.initial_queue.checkpoint.semantic_digest,
    queue.observations.checkpoint.semanticDigest);
});

function initialFixture() {
  const controlled = controlledFixture();
  const root = createTemporaryDirectory("runmat-initial-queue-");
  const value = reviewValue(controlled.control);
  writeJson(path.join(root, "review.json"), value);
  const session = openAuthorityLoadSession(openAuthorityRoot(root));
  const review = loadInitialQueueReview({
    session, reference: { path: "review.json", digest: value.digest },
    control: controlled.control,
  });
  return { root, session, review, control: controlled.control, controlled };
}

function reopenFixture(fixture) {
  const session = openAuthorityLoadSession(openAuthorityRoot(fixture.root));
  const review = loadInitialQueueReview({
    session,
    reference: { path: "review.json", digest: fixture.review.value.digest },
    control: fixture.control,
  });
  return { ...fixture, session, review };
}

function reviewValue(control) {
  const payload = {
    schema_version: 1,
    kind: "runmat-builtin-migration-initial-queue-review",
    authority: "reviewer-authored-development-input",
    control_manifest_digest: control.digest,
    review: {
      status: "reviewed", evidence: ["fixture current queue checkpoint review"],
    },
  };
  return { ...payload, digest: evidenceDigest(payload) };
}

function reseal(value) {
  delete value.digest;
  value.digest = evidenceDigest(value);
}

function writeJson(target, value) {
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, `${JSON.stringify(value, null, 2)}\n`);
}

function readJson(target) {
  return JSON.parse(fs.readFileSync(target, "utf8"));
}

function publicationResidue(root) {
  return fs.readdirSync(root, { recursive: true })
    .filter((entry) => entry.includes(".runmat-new-"));
}
