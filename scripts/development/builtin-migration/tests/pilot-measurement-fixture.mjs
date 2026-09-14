import fs from "node:fs";
import path from "node:path";

import { evidenceDigest } from "../evidence.mjs";
import { publishPilotMeasurement } from "../pilot-measurement.mjs";
import { queueBindingFromAuthority, withSelfDigest } from "../pilot-work-session/schema.mjs";
import { acceptFirstBundle } from "./sequential-shared-parent-authority-fixture.mjs";
import { controlledFixture } from "./helpers.mjs";
import {
  installAcceptedFiles, loadQueue, recordCompletion, recordStart, workSessionFixture,
} from "./pilot-work-session-fixture.mjs";

export function pilotMeasurementFixture({ publish = true } = {}) {
  const controlled = controlledFixture();
  const core = compatibleFixture(controlled);
  const fixture = workSessionFixture({ fixture: core });
  const start = recordStart(fixture);
  awaitNextMillisecond(start.startedAt);
  const accepted = acceptFirstBundle({
    fixture: core,
    subjectInventory: controlled.inventory,
    phases: unchangedPhases(controlled),
  });
  installAcceptedFiles(fixture.root, accepted);
  const finalQueue = loadQueue(fixture, 1, accepted.checkpointValue.digest);
  const completion = recordCompletion(
    fixture, start, fixture.lease, fixture.queue, finalQueue,
  );
  const reviewValue = measurementReview({
    control: controlled.control,
    initialQueue: fixture.queue,
    finalQueue,
    completion,
  });
  const reviewPath = "reviews/pilot-measurement.json";
  writeJson(path.join(fixture.root, reviewPath), reviewValue);
  const manifestReference = { path: reviewPath, digest: reviewValue.digest };
  const base = {
    ...controlled,
    root: fixture.root,
    session: fixture.session,
    initialQueue: fixture.queue,
    finalQueue,
    lease: fixture.lease,
    start,
    completion,
    reviewValue,
    reviewPath,
    manifestReference,
  };
  return publish
    ? { ...base, measurement: publishPilotMeasurement({
      session: fixture.session,
      manifestReference,
      control: controlled.control,
      repository: controlled.repository,
    }) }
    : base;
}

export function measurementReview({ control, initialQueue, finalQueue, completion }) {
  return withSelfDigest({
    schema_version: 1,
    kind: "runmat-builtin-migration-pilot-measurement-review",
    authority: "reviewer-authored-development-input",
    control_manifest_digest: control.digest,
    pilot_policy_digest: control.pilotPolicyDigest,
    initial_queue: queueBindingFromAuthority(initialQueue),
    final_queue: queueBindingFromAuthority(finalQueue),
    completed_sessions: [{ ...completion.reference }],
    review: { status: "reviewed", evidence: ["fixture measurement review"] },
  });
}

export function writeJson(target, value) {
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, `${JSON.stringify(value, null, 2)}\n`);
}

function compatibleFixture(controlled) {
  return {
    ...controlled,
    bundleIds: [controlled.bundleId],
    identities: [controlled.id],
    firstLease: { value: controlled.leaseValue, lease: controlled.lease },
  };
}

function unchangedPhases(controlled) {
  const bundle = controlled.control.bundles.get(controlled.bundleId);
  const reviewedOutputs = bundle.integration_outputs.map(
    ({ product_id, path: outputPath, producer }) => ({
      product_id, path: outputPath, producer,
    }),
  );
  return {
    lease_base_revision: controlled.inventory.source.revision,
    authored_revision: controlled.inventory.source.revision,
    integrated_revision: controlled.inventory.source.revision,
    authored_changed_paths: [],
    integration_changed_paths: [],
    reviewed_authored_write_set: structuredClone(bundle.authored_write_set),
    reviewed_source_migrations: structuredClone(bundle.source_migrations),
    reviewed_integration_outputs: reviewedOutputs,
    authored_write_set_digest: evidenceDigest(bundle.authored_write_set),
    source_migrations_digest: evidenceDigest(bundle.source_migrations),
    integration_outputs_digest: evidenceDigest(reviewedOutputs),
  };
}

function awaitNextMillisecond(timestamp) {
  while (Date.now() <= Date.parse(timestamp)) {
    // Keep production clock observation while making positive-work assertions deterministic.
  }
}
