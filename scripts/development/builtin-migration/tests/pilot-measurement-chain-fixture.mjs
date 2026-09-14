import { execFileSync } from "node:child_process";
import path from "node:path";

import { compareCodePoint } from "../constants.mjs";
import { parseControlManifest } from "../control.mjs";
import { validateControlReviewChain } from "../control-authoring/authority.mjs";
import { composeControlCandidate, controlCandidateInputDigests } from "../control-authoring/compose.mjs";
import { buildControlOverlayScaffold } from "../control-authoring/scaffold.mjs";
import { buildControlDraft } from "../control-draft.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { captureMigrationPhases } from "../integration-phases.mjs";
import { buildInventory } from "../inventory.mjs";
import { deriveEffectiveModuleComposition } from "../module-composition/effective-state.mjs";
import { renderModuleCompositionProduct } from "../module-composition/generate.mjs";
import { publishPilotMeasurement } from "../pilot-measurement.mjs";
import { emptyQueueState, validateQueueState } from "../queue.mjs";
import { validateQueueCheckpoint } from "../queue-checkpoint.mjs";
import { acceptBundle } from "./sequential-shared-parent-authority-fixture.mjs";
import {
  buildSubjectInventory, commitFixture, leaseFor, writeFixture,
} from "./sequential-shared-parent-fixture.mjs";
import {
  SEQUENTIAL_BUNDLES, SEQUENTIAL_FAMILY, SEQUENTIAL_IDENTITIES, sequentialBundleControls,
  sequentialDispositionInput, sequentialIdentityControls,
} from "./sequential-shared-parent-definition-fixture.mjs";
import { sequentialReviewedControlSet } from "./sequential-shared-parent-review-fixture.mjs";
import { sequentialReviewedTopology } from "./sequential-shared-parent-topology-fixture.mjs";
import { writeWasmRegistry } from "./sequential-shared-parent-products-fixture.mjs";
import {
  installAcceptedFiles, installAndLoadLease, loadQueue, recordCompletion,
  recordStart, workSessionFixture, writeJson,
} from "./pilot-work-session-fixture.mjs";
import { measurementReview } from "./pilot-measurement-fixture.mjs";
import {
  compiledInventoryFixture, initialQueueCheckpointValue, repositoryFixture,
} from "./helpers.mjs";

export function pilotMeasurementChainFixture({
  publish = true, duplicateInitialLeaseId = false,
} = {}) {
  const chain = reverseParallelChain({ duplicateInitialLeaseId });
  const fixture = workSessionFixture({ fixture: chain.fixture });
  const initialBetaLease = installAndLoadLease(
    fixture, "leases/lease-beta-initial.json", chain.initialBetaLease,
  );
  const alphaStart = recordStart(fixture, fixture.lease);
  const betaStart = recordStart(fixture, initialBetaLease);
  awaitNextMillisecond(betaStart.startedAt);
  installAcceptedFiles(fixture.root, chain.betaAccepted, 1, "beta");
  const queue1 = loadQueue(fixture, 1, chain.betaAccepted.checkpointValue.digest);
  const betaCompletion = recordCompletion(
    fixture, betaStart, initialBetaLease, fixture.queue, queue1,
  );
  awaitNextMillisecond(betaCompletion.endedAt);
  const refreshedAlphaLease = installAndLoadLease(
    fixture, "leases/lease-alpha-refreshed.json", chain.refreshedAlphaLease,
  );
  installAcceptedFiles(fixture.root, chain.alphaAccepted, 2, "alpha");
  const queue2 = loadQueue(fixture, 2, chain.alphaAccepted.checkpointValue.digest);
  const alphaCompletion = recordCompletion(
    fixture, alphaStart, refreshedAlphaLease, queue1, queue2,
  );
  const history = [betaCompletion, alphaCompletion];
  const canonicalReferences = history.map((entry) => ({ ...entry.reference }))
    .sort((left, right) => compareCodePoint(referenceKey(left), referenceKey(right)));
  const reviewValue = measurementReview({
    control: fixture.control, initialQueue: fixture.queue,
    finalQueue: queue2, completion: alphaCompletion,
  });
  reviewValue.completed_sessions = canonicalReferences;
  reseal(reviewValue);
  const reviewPath = "reviews/pilot-chain.json";
  writeJson(path.join(fixture.root, reviewPath), reviewValue);
  const manifestReference = { path: reviewPath, digest: reviewValue.digest };
  const base = {
    ...fixture, chain, queue1, queue2, initialBetaLease, refreshedAlphaLease,
    alphaStart, betaStart, alphaCompletion, betaCompletion, history,
    canonicalReferences, reviewValue, reviewPath, manifestReference,
  };
  return publish
    ? { ...base, measurement: publishPilotMeasurement({
      session: fixture.session, manifestReference, control: fixture.control,
      repository: fixture.repository,
    }) }
    : base;
}

function reverseParallelChain({ duplicateInitialLeaseId }) {
  const fixture = independentParallelFixture();
  const initialBetaLease = leaseForBundle(
    fixture, fixture.inventory, fixture.queueState, fixture.queueCheckpoint,
    fixture.bundleIds[1],
    duplicateInitialLeaseId ? fixture.firstLease.value.lease_id : "lease-beta-initial",
  );
  const beta = integrateBundle(
    fixture, 1, initialBetaLease, fixture.queueState, fixture.queueCheckpoint,
  );
  const betaAccepted = acceptBundle({
    fixture, subjectInventory: beta.subjectInventory, phases: beta.phases,
    leaseRecord: initialBetaLease, predecessorState: fixture.queueState,
    predecessorStatePath: "test-artifacts/queue-state-0.json",
    predecessorCheckpoint: fixture.queueCheckpoint,
    predecessorCheckpointPath: "test-artifacts/queue-checkpoint-0.json",
    checkpointHistory: new Map([[fixture.queueCheckpoint.digest, fixture.queueCheckpointValue]]),
    sealId: "seal-beta", sealPath: "test-artifacts/seal-beta.json",
  });
  const refreshedAlphaLease = leaseForBundle(
    fixture, beta.subjectInventory, betaAccepted.queueState, betaAccepted.queueCheckpoint,
    fixture.bundleIds[0], "lease-alpha-refreshed",
  );
  const alpha = integrateBundle(
    fixture, 0, refreshedAlphaLease, betaAccepted.queueState, betaAccepted.queueCheckpoint,
  );
  const alphaAccepted = acceptBundle({
    fixture, subjectInventory: alpha.subjectInventory, phases: alpha.phases,
    leaseRecord: refreshedAlphaLease, predecessorState: betaAccepted.queueState,
    predecessorStatePath: "queue-state-1.json",
    predecessorCheckpoint: betaAccepted.queueCheckpoint,
    predecessorCheckpointPath: "queue-checkpoint-1.json",
    checkpointHistory: new Map([
      [fixture.queueCheckpoint.digest, fixture.queueCheckpointValue],
      [betaAccepted.queueCheckpoint.digest, betaAccepted.checkpointValue],
    ]),
    sealId: "seal-alpha", sealPath: "test-artifacts/seal-alpha.json",
  });
  return {
    fixture, initialBetaLease, betaAccepted, refreshedAlphaLease, alphaAccepted,
  };
}

function integrateBundle(fixture, index, leaseRecord, queueState, queueCheckpoint) {
  const identity = fixture.identities[index];
  writeFixture(
    fixture.repository, `crates/runmat-runtime/src/builtins/math/${SEQUENTIAL_FAMILY}/${identity}.rs`,
    `#[runtime_builtin(name = "${identity}")]\nfn ${identity}_builtin() {}\n`,
  );
  const authoredRevision = commitFixture(fixture.repository, `author ${identity} child`);
  if (index === 0) {
    const projection = deriveEffectiveModuleComposition({
      control: fixture.control, queueState, queueCheckpoint, lease: leaseRecord.lease,
    });
    const product = projection.products
      .find((entry) => entry.product_id === "runtime-math-reduction");
    writeFixture(
      fixture.repository,
      fixture.control.integrationProducts.get("runtime-math-reduction").path,
      renderModuleCompositionProduct(product),
    );
  } else {
    writeWasmRegistry(
      fixture.repository,
      fixture.compiledInventory.snapshot.observed.registration_manifest,
    );
  }
  commitFixture(fixture.repository, `integrate ${identity} products`);
  const subjectInventory = buildSubjectInventory(fixture);
  const phases = captureMigrationPhases(
    fixture.repository, leaseRecord.lease, fixture.control,
    subjectInventory, authoredRevision,
  );
  return { subjectInventory, phases };
}

function independentParallelFixture() {
  const repository = repositoryFixture({
    identities: SEQUENTIAL_IDENTITIES,
    composition: true,
    compositionBaseChild: true,
    family: SEQUENTIAL_FAMILY,
    familyCompositionProduct: true,
  });
  const revision = `git:${execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: repository, encoding: "utf8",
  }).trim()}`;
  const compiledInventory = compiledInventoryFixture(SEQUENTIAL_IDENTITIES, {
    family: SEQUENTIAL_FAMILY,
  });
  const dispositions = sequentialDispositionInput();
  const inventory = buildInventory(repository, dispositions, { revision, compiledInventory });
  const draft = buildControlDraft(inventory);
  const topology = sequentialReviewedTopology(inventory, draft.digest);
  const scaffold = buildControlOverlayScaffold(inventory, draft, topology);
  const bundleControls = independentBundleControls(inventory);
  const identityControls = sequentialIdentityControls();
  const reviewSet = sequentialReviewedControlSet({
    repository, inventory, topology, scaffold, bundleControls, identityControls,
  });
  const candidate = composeControlCandidate({ inventory, topology, scaffold, reviewSet });
  const attestationPayload = {
    schema_version: 1, kind: "runmat-builtin-migration-control-attestation",
    authority: "reviewer-authored-development-input", program: "RM-1064/C00-C07",
    candidate_digest: candidate.digest, input_digests: controlCandidateInputDigests(candidate),
    review: { status: "reviewed", evidence: ["independent pilot fixture review"] },
  };
  const attestation = { ...attestationPayload, digest: evidenceDigest(attestationPayload) };
  const reviewedControl = validateControlReviewChain(candidate, attestation, candidate);
  const control = parseControlManifest(reviewedControl.controlValue, {
    inventory, reviewedTopology: topology, reviewedControl,
  });
  const queueState = validateQueueState(emptyQueueState(control), control, () => null);
  const queueCheckpointValue = initialQueueCheckpointValue(control, queueState);
  const queueCheckpoint = validateQueueCheckpoint(
    queueCheckpointValue, queueCheckpointValue.digest, queueState, control,
  );
  const firstLease = leaseFor({
    repository, control, inventory, queueState, queueCheckpoint,
    bundleId: SEQUENTIAL_BUNDLES[0], leaseId: "lease-alpha",
  });
  return {
    repository, inventory, compiledInventory, dispositions, control, queueState,
    queueCheckpoint, queueCheckpointValue, firstLease,
    bundleIds: [...SEQUENTIAL_BUNDLES], identities: [...SEQUENTIAL_IDENTITIES],
  };
}

function independentBundleControls(inventory) {
  const controls = sequentialBundleControls(inventory, { parallelBundles: true });
  return new Map([...controls].map(([bundleId, value], index) => [bundleId, {
    ...value,
    additional_authored_write_set: index === 0
      ? value.additional_authored_write_set
      : value.additional_authored_write_set.filter(
        (entry) => !entry.path.includes(`${SEQUENTIAL_IDENTITIES[index]}_support`),
      ),
    integration_product_refs: [index === 0 ? "runtime-math-reduction" : "wasm-registry"],
    module_composition_transition: index === 0 ? value.module_composition_transition : null,
  }]));
}

function leaseForBundle(fixture, inventory, queueState, queueCheckpoint, bundleId, leaseId) {
  return leaseFor({
    repository: fixture.repository, control: fixture.control, inventory,
    queueState, queueCheckpoint, bundleId, leaseId,
  });
}

function awaitNextMillisecond(timestamp) {
  while (Date.now() <= Date.parse(timestamp)) {
    // Keep production clock observation while making interval assertions deterministic.
  }
}

function referenceKey(value) { return `${value.path}\0${value.digest}`; }
function reseal(value) { delete value.digest; value.digest = evidenceDigest(value); }
