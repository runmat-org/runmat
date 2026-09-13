import assert from "node:assert/strict";
import test from "node:test";

import { parsePilotPolicy } from "../pilot-policy.mjs";
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
  fixture.policy.admission.minimum_public_identities_per_aggregate_hour = 6.25;
  fixture.policy.admission.maximum_elapsed_hours = 12.5;
  const parsed = parsePilotPolicy(fixture.policy, fixture.topology, fixture.prerequisites);
  assert.deepEqual(parsed.counts, fixture.policy.derived_counts);
  assert.equal(parsed.value.admission.minimum_public_identities_per_aggregate_hour, 6.25);
  assert.equal(parsed.value.admission.maximum_elapsed_hours, 12.5);
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
  for (const rate of [0, -1, Number.NaN, Number.POSITIVE_INFINITY, "4.5"]) {
    const fixture = pilotFixture();
    fixture.policy.admission.minimum_public_identities_per_aggregate_hour = rate;
    assert.throws(
      () => parsePilotPolicy(fixture.policy, fixture.topology, fixture.prerequisites),
      /minimum public identities per aggregate hour must be a finite positive number/,
    );
  }

  for (const elapsed of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
    const fixture = pilotFixture();
    fixture.policy.admission.maximum_elapsed_hours = elapsed;
    assert.throws(
      () => parsePilotPolicy(fixture.policy, fixture.topology, fixture.prerequisites),
      /maximum elapsed hours must be a finite positive number/,
    );
  }

  const unsafeElapsed = pilotFixture();
  unsafeElapsed.policy.admission.maximum_elapsed_hours = Number.MAX_SAFE_INTEGER;
  assert.throws(
    () => parsePilotPolicy(
      unsafeElapsed.policy,
      unsafeElapsed.topology,
      unsafeElapsed.prerequisites,
    ),
    /safe for millisecond timing arithmetic/,
  );

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
  for (const version of [0, 2]) {
    const fixture = pilotFixture();
    fixture.policy.schema_version = version;
    assert.throws(
      () => parsePilotPolicy(fixture.policy, fixture.topology, fixture.prerequisites),
      /must use schema_version 1/,
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
