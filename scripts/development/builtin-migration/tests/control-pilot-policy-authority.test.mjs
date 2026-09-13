import assert from "node:assert/strict";
import test from "node:test";

import { assertValidatedPilotPolicy } from "../pilot-policy.mjs";
import { cleanupRepositoryFixtures, controlledFixture } from "./helpers.mjs";

test.afterEach(cleanupRepositoryFixtures);

test("validated control preserves the exact pilot-policy authority capability", () => {
  const { control } = controlledFixture();
  assert.equal(assertValidatedPilotPolicy(control.pilotPolicy), control.pilotPolicy);
  assert.equal(control.pilotPolicyDigest, control.pilotPolicy.digest);
  assert.throws(
    () => assertValidatedPilotPolicy({ ...control.pilotPolicy }),
    /exact validated pilot policy/,
  );
  assert.throws(() => { control.pilotPolicy.admission.rate.numerator = 2; }, TypeError);
});
