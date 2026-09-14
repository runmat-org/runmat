import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

import { parseControlManifest } from "../control.mjs";
import { validateControlReviewChain } from "../control-authoring/authority.mjs";
import {
  composeControlCandidate, controlCandidateInputDigests,
} from "../control-authoring/compose.mjs";
import { buildControlOverlayScaffold } from "../control-authoring/scaffold.mjs";
import { buildControlDraft } from "../control-draft.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { buildInventory } from "../inventory.mjs";
import { issueLease, parseLease } from "../lease.mjs";
import { acceptedSealSet, barrierSealSet, emptyQueueState, validateQueueState } from "../queue.mjs";
import { validateQueueCheckpoint } from "../queue-checkpoint.mjs";
import {
  compiledInventoryFixture, initialQueueCheckpointValue, leaseBaseInventoryBinding,
  repositoryFixture,
} from "./helpers.mjs";
import {
  SEQUENTIAL_BUNDLES, SEQUENTIAL_FAMILY, SEQUENTIAL_IDENTITIES, compositionChild,
  sequentialBundleControls, sequentialDispositionInput, sequentialIdentityControls,
} from "./sequential-shared-parent-definition-fixture.mjs";
import { sequentialReviewedControlSet } from "./sequential-shared-parent-review-fixture.mjs";
import { sequentialReviewedTopology } from "./sequential-shared-parent-topology-fixture.mjs";

export { compositionChild };

export function sequentialSharedParentFixture({ parallelBundles = false } = {}) {
  const repository = repositoryFixture({
    identities: SEQUENTIAL_IDENTITIES, composition: true, compositionBaseChild: true,
    family: SEQUENTIAL_FAMILY, familyCompositionProduct: true,
  });
  const revision = repositoryRevision(repository);
  const compiledInventory = compiledInventoryFixture(SEQUENTIAL_IDENTITIES, {
    family: SEQUENTIAL_FAMILY,
  });
  const dispositions = sequentialDispositionInput();
  const inventory = buildInventory(repository, dispositions, { revision, compiledInventory });
  const draft = buildControlDraft(inventory);
  const topology = sequentialReviewedTopology(inventory, draft.digest);
  const scaffold = buildControlOverlayScaffold(inventory, draft, topology);
  const bundleControls = sequentialBundleControls(inventory, { parallelBundles });
  const identityControls = sequentialIdentityControls();
  const reviewSet = sequentialReviewedControlSet({
    repository, inventory, topology, scaffold, bundleControls, identityControls,
  });
  const candidate = composeControlCandidate({ inventory, topology, scaffold, reviewSet });
  const attestationPayload = {
    schema_version: 1,
    kind: "runmat-builtin-migration-control-attestation",
    authority: "reviewer-authored-development-input",
    program: "RM-1064/C00-C07",
    candidate_digest: candidate.digest,
    input_digests: controlCandidateInputDigests(candidate),
    review: { status: "reviewed", evidence: ["sequential fixture independent review"] },
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

export function leaseFor({
  repository, control, inventory, queueState, queueCheckpoint, bundleId, leaseId,
  owner = "sequential-fixture",
}) {
  const accepted = acceptedSealSet(queueState, control);
  const barriers = barrierSealSet(queueState, control, bundleId);
  const request = {
    schema_version: 5,
    kind: "runmat-builtin-migration-lease-request",
    authority: "reviewed-development-request",
    control_manifest_digest: control.digest,
    bundle_id: bundleId,
    lease_id: leaseId,
    owner,
    base_revision: inventory.source.revision,
    lease_base_inventory: leaseBaseInventoryBinding(inventory),
    queue_checkpoint_digest: queueCheckpoint.digest,
    queue_phase: queueState.value.phase,
    accepted_seals: accepted.value.seals,
    accepted_seal_set_digest: accepted.value.digest,
    barrier_seals: barriers.value.seals,
    barrier_seal_set_digest: barriers.value.digest,
    issued_at: "2020-01-01T00:00:00.000Z",
    expires_at: "2099-01-01T00:00:00.000Z",
    review: { status: "reviewed", evidence: ["sequential fixture lease review"] },
  };
  const value = issueLease(
    request, control, repository, inventory, queueState, queueCheckpoint,
  );
  return { request, value, lease: parseLease(value, control, repository) };
}

export function buildSubjectInventory(fixture, compiledInventory = fixture.compiledInventory) {
  return buildInventory(fixture.repository, fixture.dispositions, {
    revision: repositoryRevision(fixture.repository), compiledInventory,
  });
}

export function commitFixture(repository, message) {
  execFileSync("git", ["add", "."], { cwd: repository });
  execFileSync("git", [
    "-c", "user.name=RunMat Test", "-c", "user.email=test@runmat.invalid",
    "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", message,
  ], { cwd: repository });
  return repositoryRevision(repository);
}

export function writeFixture(repository, relative, contents) {
  const target = path.join(repository, relative);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, contents);
}

function repositoryRevision(repository) {
  return `git:${execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: repository, encoding: "utf8",
  }).trim()}`;
}
