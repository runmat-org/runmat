import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";

import {
  openAuthorityLoadSession, openAuthorityRoot,
} from "../authority-loading/index.mjs";
import { evidenceDigest } from "../evidence.mjs";
import {
  assertLoadedPilotEvaluation, assertLoadedPilotLimiterReforecast,
  loadPilotEvaluation, loadPilotLimiterReforecast, recordPilotEvaluation,
  revalidatePilotEvaluation,
} from "../pilot-evaluation.mjs";
import { cleanupRepositoryFixtures, controlledFixture } from "./helpers.mjs";
import {
  limiterPayload, pilotEvaluationFixture,
} from "./pilot-evaluation-fixture.mjs";
import { writeJson } from "./pilot-measurement-fixture.mjs";

test.afterEach(cleanupRepositoryFixtures);

test("records a threshold-met evaluation from exact persisted measurement authority", () => {
  const fixture = pilotEvaluationFixture();
  const evaluation = record(fixture);
  assert.equal(evaluation.value.outcome, "threshold-met");
  assert.equal(evaluation.value.admission_comparison.threshold_met, true);
  assert.equal(evaluation.value.admission_comparison.rate.measured_operand, "3600000");
  assert.equal(
    evaluation.value.admission_comparison.rate.required_operand,
    fixture.measurement.value.timing.aggregate_worker_ms.toString(),
  );
  assert.equal(evaluation.value.limiter_reforecast, null);
  assert.equal(
    evaluation.value.seal_set_digest,
    fixture.finalQueue.checkpoint.value.accepted_seal_set_digest,
  );
  assert.match(evaluation.reference.artifact_id, /^pilot-evaluation-[a-f0-9]{64}$/);
  assert.equal(evaluation.reference.artifact_id, evaluation.value.artifact_id);
  assert.equal(assertLoadedPilotEvaluation(evaluation, fixture), evaluation);
  assert.equal(revalidatePilotEvaluation(evaluation, fixture), evaluation);
  assert.throws(() => record(fixture), /evidence target already exists/);
  assert.throws(
    () => recordPilotEvaluation({ ...recorderInput(fixture), outcome: "threshold-met" }),
    /accepts only/,
  );
});

test("below-target evaluation requires and retains exact reviewed limiter authority", () => {
  const fixture = pilotEvaluationFixture({ belowTarget: true });
  assert.throws(() => recordPilotEvaluation({
    ...recorderInput(fixture), limiterReforecast: null,
  }), /requires a reviewed limiter and reforecast/);
  const evaluation = record(fixture);
  assert.equal(evaluation.value.outcome, "below-target");
  assert.equal(evaluation.value.admission_comparison.rate.meets_minimum, false);
  assert.equal(evaluation.limiterReforecast, fixture.limiter);
  assert.deepEqual(evaluation.value.limiter_reforecast, {
    path: fixture.limiter.observation.path,
    semantic_digest: fixture.limiter.observation.semanticDigest,
    content_digest: fixture.limiter.observation.contentDigest,
  });
  assert.equal(
    assertLoadedPilotLimiterReforecast(fixture.limiter, {
      session: fixture.session,
      control: fixture.control,
      measurement: fixture.measurement,
    }),
    fixture.limiter,
  );
  assert.throws(
    () => assertLoadedPilotLimiterReforecast({ ...fixture.limiter }, {
      session: fixture.session,
      control: fixture.control,
      measurement: fixture.measurement,
    }),
    /exact loaded pilot limiter reforecast/,
  );
});

test("threshold-met evaluation rejects limiter data", () => {
  const fixture = pilotEvaluationFixture({ loadLimiter: true });
  assert.throws(() => record(fixture), /must not include limiter data/);
});

test("persisted evaluation rejects authored facts and schema drift", async (context) => {
  const scenarios = [
    ["legacy schema", (value) => { value.schema_version = 0; }, /schema_version 1/],
    ["future schema", (value) => { value.schema_version = 2; }, /schema_version 1/],
    ["extra decision input", (value) => { value.rate = 1; }, /fields must be exactly/],
    ["pilot identity", (value) => { value.pilot_id = "different-pilot"; }, /another control, policy, or pilot/],
    ["count", (value) => { value.counts.public_identities = 0; }, /exact recomputed evidence/],
    ["timing", (value) => { value.timing.aggregate_worker_ms += 1; }, /exact recomputed evidence/],
    ["final queue", (value) => { value.final_queue.checkpoint.content_digest = fakeDigest("c"); }, /exact recomputed evidence/],
    ["final source", (value) => { value.source.inventory_digest = fakeDigest("c"); }, /exact recomputed evidence/],
    ["seal reference", (value) => { value.seals[0].digest = fakeDigest("c"); }, /exact recomputed evidence/],
    ["comparison", (value) => { value.admission_comparison.rate.meets_minimum = false; }, /exact recomputed evidence/],
    ["outcome", (value) => { value.outcome = "below-target"; }, /exact recomputed evidence/],
    ["seal set", (value) => { value.seal_set_digest = fakeDigest("d"); }, /exact recomputed evidence/],
    ["policy", (value) => { value.policy.maximum_elapsed_milliseconds -= 1; }, /exact recomputed evidence/],
  ];
  for (const [name, mutate, pattern] of scenarios) {
    await context.test(name, () => {
      const fixture = pilotEvaluationFixture();
      const evaluation = record(fixture);
      const changed = structuredClone(evaluation.value);
      mutate(changed);
      reseal(changed);
      replaceEvaluation(fixture, evaluation, changed);
      assert.throws(() => reload(fixture, changed), pattern);
    });
  }
});

test("limiter rejects stale bindings, open taxonomy, missing evidence, and invalid forecasts", async (context) => {
  const scenarios = [
    ["legacy schema", (value) => { value.schema_version = 0; }, /schema_version 1/],
    ["future schema", (value) => { value.schema_version = 2; }, /schema_version 1/],
    ["extra field", (value) => { value.notes = []; }, /fields must be exactly/],
    ["policy drift", (value) => { value.pilot_policy_digest = fakeDigest("b"); }, /another control, policy, or pilot/],
    ["measurement drift", (value) => { value.measurement.content_digest = fakeDigest("c"); }, /measurement observation mismatch/],
    ["category", (value) => { value.limiter.category = "miscellaneous"; }, /limiter category/],
    ["missing concrete evidence", (value) => { value.limiter.evidence = ["   "]; }, /nonempty string/],
    ["zero worker forecast", (value) => { value.reforecast.aggregate_worker_milliseconds = 0; }, /integer >= 1/],
    ["unsafe elapsed forecast", (value) => { value.reforecast.elapsed_milliseconds = Number.MAX_SAFE_INTEGER + 1; }, /integer >= 1/],
  ];
  for (const [name, mutate, pattern] of scenarios) {
    await context.test(name, () => {
      const fixture = pilotEvaluationFixture({ belowTarget: true, loadLimiter: false });
      const value = limiterPayload(fixture);
      mutate(value);
      reseal(value);
      writeJson(path.join(fixture.root, fixture.paths.limiterReforecast), value);
      assert.throws(() => loadPilotLimiterReforecast({
        session: fixture.session,
        reference: { path: fixture.paths.limiterReforecast, digest: value.digest },
        control: fixture.control,
        measurement: fixture.measurement,
      }), pattern);
    });
  }
});

test("capability, reference, path, session, control, and observation drift fail closed", () => {
  const fixture = pilotEvaluationFixture();
  const evaluation = record(fixture);
  assert.throws(
    () => assertLoadedPilotEvaluation({ ...evaluation }, fixture),
    /exact loaded pilot evaluation/,
  );
  const otherSession = openAuthorityLoadSession(openAuthorityRoot(fixture.root));
  assert.throws(
    () => assertLoadedPilotEvaluation(evaluation, {
      session: otherSession, control: fixture.control,
    }),
    /exact loaded pilot evaluation/,
  );
  assert.throws(() => loadPilotEvaluation({
    ...loaderInput(fixture, evaluation),
    reference: { ...evaluation.reference, artifact_id: "pilot-evaluation-forged" },
  }), /path was rebound|reference/);
  const other = controlledFixture();
  assert.throws(
    () => assertLoadedPilotEvaluation(evaluation, {
      session: fixture.session, control: other.control,
    }),
    /exact loaded pilot evaluation/,
  );
  const wrongDigestSession = openAuthorityLoadSession(openAuthorityRoot(fixture.root));
  assert.throws(() => loadPilotEvaluation({
    session: wrongDigestSession,
    reference: { ...evaluation.reference, digest: fakeDigest("f") },
    control: fixture.control,
    repository: fixture.repository,
  }), /reference does not match/);
  const relocatedPath = "other/evaluation.json";
  writeJson(path.join(fixture.root, relocatedPath), evaluation.value);
  const relocatedSession = openAuthorityLoadSession(openAuthorityRoot(fixture.root));
  assert.throws(() => loadPilotEvaluation({
    session: relocatedSession,
    reference: { ...evaluation.reference, path: relocatedPath },
    control: fixture.control,
    repository: fixture.repository,
  }), /not deterministic for its measurement/);
  fs.writeFileSync(
    path.join(fixture.root, fixture.measurement.reference.path),
    `${JSON.stringify(fixture.measurement.value)}\n`,
  );
  assert.throws(() => revalidatePilotEvaluation(evaluation, fixture), /changed after observation/);
});

test("evaluation and below-target limiter mutation or replacement fail final revalidation", () => {
  const evaluationFixture = pilotEvaluationFixture();
  const evaluation = record(evaluationFixture);
  fs.writeFileSync(
    path.join(evaluationFixture.root, evaluation.reference.path),
    `${JSON.stringify(evaluation.value)}\n`,
  );
  assert.throws(
    () => revalidatePilotEvaluation(evaluation, evaluationFixture),
    /changed after observation/,
  );

  const limiterFixture = pilotEvaluationFixture({ belowTarget: true });
  const belowTarget = record(limiterFixture);
  const limiterPath = path.join(limiterFixture.root, limiterFixture.limiter.reference.path);
  fs.renameSync(limiterPath, `${limiterPath}.replaced`);
  writeJson(limiterPath, limiterFixture.limiter.value);
  assert.throws(
    () => revalidatePilotEvaluation(belowTarget, limiterFixture),
    /changed after observation/,
  );
});

function record(fixture) {
  return recordPilotEvaluation({
    ...recorderInput(fixture), limiterReforecast: fixture.limiter,
  });
}

function recorderInput(fixture) {
  return {
    session: fixture.session,
    control: fixture.control,
    measurement: fixture.measurement,
    repository: fixture.repository,
  };
}

function loaderInput(fixture, evaluation) {
  return {
    session: fixture.session,
    reference: evaluation.reference,
    control: fixture.control,
    repository: fixture.repository,
  };
}

function reload(fixture, value) {
  const session = openAuthorityLoadSession(openAuthorityRoot(fixture.root));
  return loadPilotEvaluation({
    session,
    reference: {
      path: fixture.paths.evaluation,
      artifact_id: value.artifact_id,
      digest: value.digest,
    },
    control: fixture.control,
    repository: fixture.repository,
  });
}

function replaceEvaluation(fixture, evaluation, value) {
  const target = path.join(fixture.root, evaluation.reference.path);
  fs.renameSync(target, `${target}.prior`);
  writeJson(target, value);
}

function reseal(value) {
  delete value.digest;
  value.digest = evidenceDigest(value);
}

function fakeDigest(character) {
  return `sha256:${character.repeat(64)}`;
}
