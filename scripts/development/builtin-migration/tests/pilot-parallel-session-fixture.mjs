import path from "node:path";

import { captureMigrationPhases } from "../integration-phases.mjs";
import { deriveEffectiveModuleComposition } from "../module-composition/effective-state.mjs";
import { renderModuleCompositionProduct } from "../module-composition/generate.mjs";
import {
  acceptBundle, acceptFirstBundle,
} from "./sequential-shared-parent-authority-fixture.mjs";
import {
  buildSubjectInventory, commitFixture, compositionChild, leaseFor,
  sequentialSharedParentFixture, writeFixture,
} from "./sequential-shared-parent-fixture.mjs";
import { writeWasmRegistry } from "./sequential-shared-parent-products-fixture.mjs";

export function parallelPilotSessionFixture() {
  const fixture = sequentialSharedParentFixture({ parallelBundles: true });
  const initialBetaLease = leaseForBundle(
    fixture, fixture.inventory, fixture.queueState, fixture.queueCheckpoint,
    fixture.bundleIds[1], "lease-beta-initial",
  );
  const alpha = integrateBundle({
    fixture,
    identity: fixture.identities[0],
    leaseRecord: fixture.firstLease,
    queueState: fixture.queueState,
    queueCheckpoint: fixture.queueCheckpoint,
  });
  const alphaAccepted = acceptFirstBundle({
    fixture, subjectInventory: alpha.subjectInventory, phases: alpha.phases,
  });
  const refreshedBetaLease = leaseForBundle(
    fixture, alpha.subjectInventory, alphaAccepted.queueState,
    alphaAccepted.queueCheckpoint, fixture.bundleIds[1], "lease-beta-refreshed",
  );
  const wrongOwnerBetaLease = leaseFor({
    repository: fixture.repository,
    control: fixture.control,
    inventory: alpha.subjectInventory,
    queueState: alphaAccepted.queueState,
    queueCheckpoint: alphaAccepted.queueCheckpoint,
    bundleId: fixture.bundleIds[1],
    leaseId: "lease-beta-wrong-owner",
    owner: "another-reviewed-owner",
  });
  const beta = integrateBundle({
    fixture,
    identity: fixture.identities[1],
    leaseRecord: refreshedBetaLease,
    queueState: alphaAccepted.queueState,
    queueCheckpoint: alphaAccepted.queueCheckpoint,
  });
  const betaAccepted = acceptBundle({
    fixture,
    subjectInventory: beta.subjectInventory,
    phases: beta.phases,
    leaseRecord: refreshedBetaLease,
    predecessorState: alphaAccepted.queueState,
    predecessorStatePath: "queue-state-1.json",
    predecessorCheckpoint: alphaAccepted.queueCheckpoint,
    predecessorCheckpointPath: "queue-checkpoint-1.json",
    checkpointHistory: new Map([
      [fixture.queueCheckpoint.digest, fixture.queueCheckpointValue],
      [alphaAccepted.queueCheckpoint.digest, alphaAccepted.checkpointValue],
    ]),
    sealId: "seal-beta",
    sealPath: "test-artifacts/seal-beta.json",
  });
  return {
    fixture,
    initialAlphaLease: fixture.firstLease,
    initialBetaLease,
    alphaSubjectInventory: alpha.subjectInventory,
    alphaAccepted,
    refreshedBetaLease,
    wrongOwnerBetaLease,
    betaSubjectInventory: beta.subjectInventory,
    betaAccepted,
  };
}

function integrateBundle({
  fixture, identity, leaseRecord, queueState, queueCheckpoint,
}) {
  writeFixture(
    fixture.repository,
    compositionChild(identity).source_path,
    `pub fn ${identity}_support() {}\n`,
  );
  const authoredRevision = commitFixture(fixture.repository, `author ${identity} child`);
  const projection = deriveEffectiveModuleComposition({
    control: fixture.control,
    queueState,
    queueCheckpoint,
    lease: leaseRecord.lease,
  });
  installProjectedParents(fixture.repository, projection);
  writeWasmRegistry(
    fixture.repository, fixture.compiledInventory.snapshot.observed.registration_manifest,
  );
  commitFixture(fixture.repository, `integrate ${identity} products`);
  const subjectInventory = buildSubjectInventory(fixture);
  const phases = captureMigrationPhases(
    fixture.repository, leaseRecord.lease, fixture.control,
    subjectInventory, authoredRevision,
  );
  return { projection, subjectInventory, phases };
}

function installProjectedParents(repository, projection) {
  for (const product of projection.products) {
    if (product.state !== "present") continue;
    writeFixture(
      repository, path.posix.normalize(product.path),
      renderModuleCompositionProduct(product),
    );
  }
}

function leaseForBundle(
  fixture, inventory, queueState, queueCheckpoint, bundleId, leaseId,
) {
  return leaseFor({
    repository: fixture.repository,
    control: fixture.control,
    inventory,
    queueState,
    queueCheckpoint,
    bundleId,
    leaseId,
  });
}
