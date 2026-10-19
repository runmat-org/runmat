import assert from "node:assert/strict";
import test from "node:test";

import {
  parsePilotPolicy, pilotAdmissionComparison, pilotElapsedWithinMaximum,
  pilotRateMeetsMinimum,
} from "../pilot-policy.mjs";
import {
  cohortCount, pilotFixture, prerequisite,
} from "./pilot-policy-fixture.mjs";

test("pilot policy derives exact public, internal, bundle, identity, and cohort counts", () => {
  const fixture = pilotFixture();
  const parsed = parsePilotPolicy(fixture.policy, fixture.topology, fixture.prerequisites);
  assert.equal(parsed.pilotId, "rm-1064-accelerated");
  assert.deepEqual(parsed.bundleIds, ["pilot-c01", "pilot-c02", "pilot-c03", "pilot-c04", "pilot-c05", "pilot-c06", "pilot-c07"]);
  assert.deepEqual(parsed.counts, fixture.policy.derived_counts);
  assert.equal(parsed.waveByBundle.get("pilot-c07"), 7);
});

test("pilot waves reject unknown, duplicate, unsafe, and noncanonical bundle membership", () => {
  const unknown = pilotFixture();
  unknown.policy.waves[0].bundle_ids = ["missing-bundle"];
  assert.throws(
    () => parsePilotPolicy(unknown.policy, unknown.topology, unknown.prerequisites),
    /unknown topology bundle/,
  );

  const duplicate = pilotFixture();
  duplicate.policy.waves[1].bundle_ids = ["pilot-c01", "pilot-c02"];
  assert.throws(
    () => parsePilotPolicy(duplicate.policy, duplicate.topology, duplicate.prerequisites),
    /occurs in more than one wave/,
  );

  const unsafe = pilotFixture();
  unsafe.policy.waves[0].wave_id = "Wave 1";
  assert.throws(
    () => parsePilotPolicy(unsafe.policy, unsafe.topology, unsafe.prerequisites),
    /safe stable identifier/,
  );

  const noncanonical = pilotFixture();
  noncanonical.policy.waves[0].bundle_ids = ["pilot-c02", "pilot-c01"];
  assert.throws(
    () => parsePilotPolicy(noncanonical.policy, noncanonical.topology, noncanonical.prerequisites),
    /must use canonical order/,
  );

  const duplicateWave = pilotFixture();
  duplicateWave.policy.waves[1].wave_id = duplicateWave.policy.waves[0].wave_id;
  assert.throws(
    () => parsePilotPolicy(duplicateWave.policy, duplicateWave.topology, duplicateWave.prerequisites),
    /wave ids must be unique/,
  );

  const skippedOrder = pilotFixture();
  skippedOrder.policy.waves[1].order = 3;
  assert.throws(
    () => parsePilotPolicy(skippedOrder.policy, skippedOrder.topology, skippedOrder.prerequisites),
    /contiguous order beginning at 1/,
  );
});

test("pilot policy checks every derived count rather than trusting authored values", () => {
  for (const field of ["bundles", "identities", "public_identities", "internal_identities"]) {
    const fixture = pilotFixture();
    fixture.policy.derived_counts[field] += 1;
    assert.throws(
      () => parsePilotPolicy(fixture.policy, fixture.topology, fixture.prerequisites),
      new RegExp(`derived ${field} does not match topology`),
    );
  }

  const cohort = pilotFixture();
  cohort.policy.derived_counts.cohorts[1].public_identities += 1;
  assert.throws(
    () => parsePilotPolicy(cohort.policy, cohort.topology, cohort.prerequisites),
    /cohort counts do not match topology/,
  );
});

test("pilot schema accepts an explicitly reviewed topology subset and authored rate and window", () => {
  const fixture = pilotFixture();
  fixture.policy.waves = fixture.policy.waves.slice(0, 2);
  fixture.policy.derived_counts = {
    bundles: 2,
    identities: 3,
    public_identities: 2,
    internal_identities: 1,
    cohorts: [
      cohortCount("C01", 1, 1, 1, 0),
      cohortCount("C02", 1, 2, 1, 1),
    ],
  };
  fixture.policy.admission.minimum_public_identities_per_aggregate_hour = {
    numerator: 25, denominator: 4,
  };
  fixture.policy.admission.maximum_elapsed_milliseconds = 45_000_000;
  const parsed = parsePilotPolicy(fixture.policy, fixture.topology, fixture.prerequisites);
  assert.deepEqual(parsed.counts, fixture.policy.derived_counts);
  assert.deepEqual(parsed.admission.rate, { numerator: 25, denominator: 4 });
  assert.equal(parsed.admission.maximumElapsedMilliseconds, 45_000_000);
});

test("pilot prerequisite authority is complete and every selected prerequisite is in an earlier wave", () => {
  const incompleteAuthority = pilotFixture();
  incompleteAuthority.prerequisites.delete("unselected-c01");
  assert.throws(
    () => parsePilotPolicy(incompleteAuthority.policy, incompleteAuthority.topology, incompleteAuthority.prerequisites),
    /cover every topology bundle exactly/,
  );

  const missing = pilotFixture();
  missing.prerequisites.set("pilot-c02", [prerequisite("unselected-c01")]);
  assert.throws(
    () => parsePilotPolicy(missing.policy, missing.topology, missing.prerequisites),
    /pilot omits prerequisite unselected-c01/,
  );

  const sameWave = pilotFixture();
  sameWave.policy.waves[0].bundle_ids = ["pilot-c01", "pilot-c02"];
  sameWave.policy.waves.splice(1, 1);
  sameWave.policy.waves.forEach((wave, index) => { wave.order = index + 1; });
  assert.throws(
    () => parsePilotPolicy(sameWave.policy, sameWave.topology, sameWave.prerequisites),
    /must be in an earlier pilot wave/,
  );

  const laterWave = pilotFixture();
  laterWave.prerequisites.set("pilot-c01", [prerequisite("pilot-c02")]);
  assert.throws(
    () => parsePilotPolicy(laterWave.policy, laterWave.topology, laterWave.prerequisites),
    /must be in an earlier pilot wave/,
  );

  const unknown = pilotFixture();
  unknown.prerequisites.set("pilot-c02", [prerequisite("missing-bundle")]);
  assert.throws(
    () => parsePilotPolicy(unknown.policy, unknown.topology, unknown.prerequisites),
    /unknown topology bundle/,
  );
});

test("pilot admission requires positive bounded measurements, zero waivers, full gates, and a typed below-target obligation", () => {
  for (const rate of [
    { numerator: 0, denominator: 1 },
    { numerator: 1, denominator: 0 },
    { numerator: 9, denominator: 2, scale: 1 },
    4.5,
  ]) {
    const fixture = pilotFixture();
    fixture.policy.admission.minimum_public_identities_per_aggregate_hour = rate;
    assert.throws(
      () => parsePilotPolicy(fixture.policy, fixture.topology, fixture.prerequisites),
      /minimum public identities per aggregate hour/,
    );
  }

  const unreducedRate = pilotFixture();
  unreducedRate.policy.admission.minimum_public_identities_per_aggregate_hour = {
    numerator: 18, denominator: 4,
  };
  assert.throws(
    () => parsePilotPolicy(
      unreducedRate.policy, unreducedRate.topology, unreducedRate.prerequisites,
    ),
    /must use a reduced positive rational/,
  );

  for (const elapsed of [0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1]) {
    const fixture = pilotFixture();
    fixture.policy.admission.maximum_elapsed_milliseconds = elapsed;
    assert.throws(
      () => parsePilotPolicy(fixture.policy, fixture.topology, fixture.prerequisites),
      /maximum elapsed milliseconds must be an integer >= 1/,
    );
  }

  const waiver = pilotFixture();
  waiver.policy.admission.required_waived_gate_count = 1;
  assert.throws(
    () => parsePilotPolicy(waiver.policy, waiver.topology, waiver.prerequisites),
    /waived gate count must be zero/,
  );

  const gates = pilotFixture();
  gates.policy.admission.ordinary_gate_policy = "pilot-subset";
  assert.throws(
    () => parsePilotPolicy(gates.policy, gates.topology, gates.prerequisites),
    /retain all ordinary required gates/,
  );

  const unreviewedTransition = pilotFixture();
  unreviewedTransition.policy.admission.below_target_obligation.production_transition = "automatic";
  assert.throws(
    () => parsePilotPolicy(
      unreviewedTransition.policy,
      unreviewedTransition.topology,
      unreviewedTransition.prerequisites,
    ),
    /must require reviewer acceptance/,
  );

  for (const findings of [["concrete-limiter"], ["revised-forecast"], ["concrete-limiter", "other"]]) {
    const fixture = pilotFixture();
    fixture.policy.admission.below_target_obligation.required_findings = findings;
    assert.throws(
      () => parsePilotPolicy(fixture.policy, fixture.topology, fixture.prerequisites),
      /must require a concrete limiter and revised forecast/,
    );
  }
});

test("pilot policy rejects legacy and future schema versions and requires exact reviewed authority", () => {
  for (const version of [1, 3]) {
    const fixture = pilotFixture();
    fixture.policy.schema_version = version;
    assert.throws(
      () => parsePilotPolicy(fixture.policy, fixture.topology, fixture.prerequisites),
      /must use schema_version 2/,
    );
  }

  const review = pilotFixture();
  review.policy.review.status = "unreviewed";
  assert.throws(
    () => parsePilotPolicy(review.policy, review.topology, review.prerequisites),
    /status must be reviewed/,
  );

  const extra = pilotFixture();
  extra.policy.selector = "all";
  assert.throws(
    () => parsePilotPolicy(extra.policy, extra.topology, extra.prerequisites),
    /fields must be exactly/,
  );

  const lookalike = pilotFixture();
  assert.throws(
    () => parsePilotPolicy(
      lookalike.policy,
      { ...lookalike.topology },
      lookalike.prerequisites,
    ),
    /deterministically validated topology view/,
  );
});

test("validated pilot policy is deeply immutable and does not retain caller-owned data", () => {
  const fixture = pilotFixture();
  const parsed = parsePilotPolicy(fixture.policy, fixture.topology, fixture.prerequisites);
  fixture.policy.waves[0].bundle_ids[0] = "changed";
  fixture.prerequisites.get("pilot-c02")[0].bundle_id = "changed";
  assert.equal(parsed.value.waves[0].bundle_ids[0], "pilot-c01");
  assert.equal(parsed.waves[0].bundle_ids[0], "pilot-c01");
  assert.throws(() => parsed.value.waves.push({}), TypeError);
  assert.throws(() => parsed.waveByBundle.set("other", 1), /immutable/);
});

test("pilot admission compares elapsed time and throughput with exact integer arithmetic", () => {
  const fixture = pilotFixture();
  const parsed = parsePilotPolicy(fixture.policy, fixture.topology, fixture.prerequisites);
  assert.equal(pilotRateMeetsMinimum(parsed, 9, 7_200_000), true);
  assert.equal(pilotRateMeetsMinimum(parsed, 9, 7_200_001), false);
  assert.equal(pilotRateMeetsMinimum(parsed, 18, 14_400_000), true);
  assert.equal(pilotElapsedWithinMaximum(parsed, 172_800_000), true);
  assert.equal(pilotElapsedWithinMaximum(parsed, 172_800_001), false);
  assert.throws(
    () => pilotRateMeetsMinimum({ ...parsed }, 9, 7_200_000),
    /exact validated pilot policy/,
  );
});

test("pilot admission exposes the canonical exact comparison and preserves helper parity", () => {
  const fixture = pilotFixture();
  const policy = parsePilotPolicy(fixture.policy, fixture.topology, fixture.prerequisites);
  const exactBoundary = pilotAdmissionComparison(policy, {
    publicIdentities: 9,
    aggregateWorkerMilliseconds: 7_200_000,
    elapsedMilliseconds: 172_800_000,
  });
  assert.deepEqual(exactBoundary, {
    rate: {
      public_identities: 9,
      aggregate_worker_milliseconds: 7_200_000,
      minimum_rate_numerator: 9,
      minimum_rate_denominator: 2,
      milliseconds_per_hour: 3_600_000,
      measured_operand: "64800000",
      required_operand: "64800000",
      meets_minimum: true,
    },
    elapsed: {
      elapsed_milliseconds: 172_800_000,
      maximum_elapsed_milliseconds: 172_800_000,
      within_maximum: true,
    },
    threshold_met: true,
  });
  assert.equal(
    pilotRateMeetsMinimum(policy, 9, 7_200_000),
    exactBoundary.rate.meets_minimum,
  );
  assert.equal(
    pilotElapsedWithinMaximum(policy, 172_800_000),
    exactBoundary.elapsed.within_maximum,
  );

  const missesBoth = pilotAdmissionComparison(policy, {
    publicIdentities: 9,
    aggregateWorkerMilliseconds: 7_200_001,
    elapsedMilliseconds: 172_800_001,
  });
  assert.equal(missesBoth.rate.meets_minimum, false);
  assert.equal(missesBoth.elapsed.within_maximum, false);
  assert.equal(missesBoth.threshold_met, false);

  const rateOnly = pilotAdmissionComparison(policy, {
    publicIdentities: 9,
    aggregateWorkerMilliseconds: 7_200_000,
    elapsedMilliseconds: 172_800_001,
  });
  assert.equal(rateOnly.rate.meets_minimum, true);
  assert.equal(rateOnly.elapsed.within_maximum, false);
  assert.equal(rateOnly.threshold_met, false);

  const elapsedOnly = pilotAdmissionComparison(policy, {
    publicIdentities: 9,
    aggregateWorkerMilliseconds: 7_200_001,
    elapsedMilliseconds: 172_800_000,
  });
  assert.equal(elapsedOnly.rate.meets_minimum, false);
  assert.equal(elapsedOnly.elapsed.within_maximum, true);
  assert.equal(elapsedOnly.threshold_met, false);
});

test("pilot admission keeps products exact at the safe-integer boundary", () => {
  const fixture = pilotFixture();
  const policy = parsePilotPolicy(fixture.policy, fixture.topology, fixture.prerequisites);
  const comparison = pilotAdmissionComparison(policy, {
    publicIdentities: Number.MAX_SAFE_INTEGER,
    aggregateWorkerMilliseconds: Number.MAX_SAFE_INTEGER,
    elapsedMilliseconds: Number.MAX_SAFE_INTEGER,
  });
  assert.equal(
    comparison.rate.measured_operand,
    (BigInt(Number.MAX_SAFE_INTEGER) * 2n * 3_600_000n).toString(),
  );
  assert.equal(
    comparison.rate.required_operand,
    (BigInt(Number.MAX_SAFE_INTEGER) * 9n).toString(),
  );
  assert.throws(
    () => pilotAdmissionComparison(policy, {
      publicIdentities: Number.MAX_SAFE_INTEGER + 1,
      aggregateWorkerMilliseconds: 1,
      elapsedMilliseconds: 1,
    }),
    /public identities must be an integer/,
  );
});
