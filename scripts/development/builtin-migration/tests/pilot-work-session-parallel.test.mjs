import assert from "node:assert/strict";
import path from "node:path";
import test from "node:test";

import { loadQueueAuthority } from "../queue-authority/index.mjs";
import { cleanupRepositoryFixtures } from "./helpers.mjs";
import { parallelPilotSessionFixture } from "./pilot-parallel-session-fixture.mjs";
import {
  installAcceptedFiles, installAndLoadLease, loadQueue, recordCompletion,
  recordStart, startRecorderInput, workSessionFixture, writeJson,
} from "./pilot-work-session-fixture.mjs";
import {
  recordPilotWorkSessionCompletion, recordPilotWorkSessionStart,
} from "../pilot-work-session/index.mjs";

test.afterEach(cleanupRepositoryFixtures);

test("parallel starts retain one logical session across an intervening seal and refresh", () => {
  const chain = installedParallelChain();
  const alphaStart = recordStart(chain.fixture, chain.alphaLease);
  const betaStart = recordStart(chain.fixture, chain.initialBetaLease);
  const alphaCompletion = recordCompletion(
    chain.fixture, alphaStart, chain.alphaLease, chain.fixture.queue, chain.queue1,
  );
  const betaCompletion = recordCompletion(
    chain.fixture, betaStart, chain.refreshedBetaLease, chain.queue1, chain.queue2,
  );
  assert.equal(betaCompletion.start, betaStart);
  assert.equal(betaCompletion.finalLeaseAuthority, chain.refreshedBetaLease);
  assert.deepEqual(
    betaCompletion.preIntegrationQueue.observations, chain.queue1.observations,
  );
  assert.deepEqual(betaCompletion.successorQueue.observations, chain.queue2.observations);
  assert.equal(
    betaCompletion.seal.leaseId,
    chain.refreshedBetaLease.lease.value.lease_id,
  );
  assert.ok(Date.parse(betaStart.startedAt) <= Date.parse(alphaCompletion.endedAt));
  assert.ok(Date.parse(alphaCompletion.endedAt) <= Date.parse(betaCompletion.endedAt));
  assert.throws(() => recordPilotWorkSessionStart({
    ...startRecorderInput(chain.fixture, chain.refreshedBetaLease),
    queueAuthority: chain.queue1,
  }), /evidence target already exists/);
});

test("completion rejects stale refresh and missing final lease authority", () => {
  const chain = installedParallelChain();
  const betaStart = recordStart(chain.fixture, chain.initialBetaLease);
  assert.throws(() => recordCompletion(
    chain.fixture, betaStart, chain.initialBetaLease, chain.queue1, chain.queue2,
  ), /exact queue checkpoint/);
  assert.throws(() => recordPilotWorkSessionCompletion({
    session: chain.fixture.session,
    control: chain.fixture.control,
    start: betaStart,
    preIntegrationQueue: chain.queue1,
    successorQueue: chain.queue2,
    repository: chain.fixture.repository,
  }), /fields must be exactly/);
});

test("completion rejects another owner, nonancestor queue, and an already sealed target", () => {
  const chain = installedParallelChain();
  const betaStart = recordStart(chain.fixture, chain.initialBetaLease);
  assert.throws(() => recordCompletion(
    chain.fixture, betaStart, chain.alphaLease, chain.fixture.queue, chain.queue1,
  ), /final lease belongs to another bundle/);
  const wrongOwner = installAndLoadLease(
    chain.fixture, "leases/lease-beta-wrong-owner.json",
    chain.chain.wrongOwnerBetaLease,
  );
  assert.throws(() => recordCompletion(
    chain.fixture, betaStart, wrongOwner, chain.queue1, chain.queue2,
  ), /another reviewed owner/);

  const alternate = alternateInitialQueue(chain.fixture);
  assert.throws(() => recordCompletion(
    chain.fixture, betaStart, chain.initialBetaLease, alternate, chain.queue1,
  ), /does not descend/);
  assert.throws(() => recordCompletion(
    chain.fixture, betaStart, chain.refreshedBetaLease, chain.queue2, chain.queue2,
  ), /already sealed|target is already sealed/);
});

test("a second bundle start cannot replace the deterministic logical session", () => {
  const chain = installedParallelChain();
  recordStart(chain.fixture, chain.initialBetaLease);
  assert.throws(() => recordPilotWorkSessionStart({
    ...startRecorderInput(chain.fixture, chain.refreshedBetaLease),
    queueAuthority: chain.queue1,
  }), /evidence target already exists/);
});

function installedParallelChain() {
  const chain = parallelPilotSessionFixture();
  const fixture = workSessionFixture({ fixture: chain.fixture });
  const initialBetaLease = installAndLoadLease(
    fixture, "leases/lease-beta-initial.json", chain.initialBetaLease,
  );
  installAcceptedFiles(fixture.root, chain.alphaAccepted, 1, "alpha");
  const queue1 = loadQueue(fixture, 1, chain.alphaAccepted.checkpointValue.digest);
  const refreshedBetaLease = installAndLoadLease(
    fixture, "leases/lease-beta-refreshed.json", chain.refreshedBetaLease,
  );
  installAcceptedFiles(fixture.root, chain.betaAccepted, 2, "beta");
  const queue2 = loadQueue(fixture, 2, chain.betaAccepted.checkpointValue.digest);
  return {
    chain,
    fixture,
    alphaLease: fixture.lease,
    initialBetaLease,
    refreshedBetaLease,
    queue1,
    queue2,
  };
}

function alternateInitialQueue(fixture) {
  writeJson(
    path.join(fixture.root, "alternate/queue-state.json"),
    fixture.fixture.queueState.value,
  );
  writeJson(
    path.join(fixture.root, "alternate/queue-checkpoint.json"),
    fixture.fixture.queueCheckpointValue,
  );
  return loadQueueAuthority({
    session: fixture.session,
    statePath: "alternate/queue-state.json",
    checkpointPath: "alternate/queue-checkpoint.json",
    trustedCheckpointDigest: fixture.fixture.queueCheckpointValue.digest,
    control: fixture.control,
  });
}
