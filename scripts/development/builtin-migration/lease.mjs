import { pathAllowed } from "./path-scope.mjs";
import { subjectAuthorityPathFailures } from "./authority-paths.mjs";
import { assertControlSubject, assertValidatedControl } from "./control.mjs";
import { evidenceDigest } from "./evidence.mjs";
import { deepImmutable } from "./immutable.mjs";
import { finalAuthorityFailuresForBundles } from "./inventory-delta.mjs";
import {
  assertIssuableIntegrationBase, assertRecordedIntegrationBase,
  leaseBaseInventoryBinding, parseLeaseBaseInventoryBinding,
  validateAcceptedIntegrationCheckpoint, validateLeaseBaseInventory,
} from "./lease-base.mjs";
import { acceptedSealSet, barrierSealSet } from "./queue.mjs";
import { assertValidatedQueueCheckpoint } from "./queue-checkpoint.mjs";
import {
  parseSealReferences, validateAcceptedSealSet, validateBarrierSealSet,
} from "./seal-set-schema.mjs";
import {
  array, digest, enumValue, exact, kind, nonempty, sourceRevision, stableId, uniqueStrings,
} from "./schema.mjs";

const VALIDATED_LEASES = new WeakSet();

export { validateAcceptedIntegrationCheckpoint } from "./lease-base.mjs";

export function issueLease(
  requestValue, control, repository, baseInventory, queueState, queueCheckpoint,
  clock = Date.now,
) {
  assertValidatedControl(control);
  const request = parseLeaseRequest(requestValue, control);
  assertActiveInterval(request, clock, "lease request");
  assertIssuableIntegrationBase(repository, request.base_revision, control.baseline.revision);
  const base = validateLeaseBaseInventory(repository, baseInventory, control, request);
  const bundle = control.bundles.get(request.bundle_id);
  const accepted = acceptedSealSet(queueState, control);
  const barriers = barrierSealSet(queueState, control, bundle.id);
  const checkpoint = assertValidatedQueueCheckpoint(queueCheckpoint, control, queueState);
  if (request.queue_checkpoint_digest !== checkpoint.digest) {
    throw new Error("lease request differs from the trusted current queue checkpoint");
  }
  if (request.queue_phase !== queueState.value.phase
    || request.queue_phase !== checkpoint.value.phase) {
    throw new Error("lease request queue phase differs from the validated queue authority");
  }
  if (request.queue_phase === "pilot" && !control.pilotPolicy.waveByBundle.has(bundle.id)) {
    throw new Error(`${bundle.id}: bundle is not admitted by the reviewed pilot policy`);
  }
  if (request.base_revision !== checkpoint.value.source_revision
    || request.lease_base_inventory.source_digest !== checkpoint.value.source_digest
    || request.lease_base_inventory.inventory_digest !== checkpoint.value.inventory_digest) {
    throw new Error("lease base differs from the trusted current queue checkpoint source");
  }
  if (JSON.stringify(request.accepted_seals) !== JSON.stringify(accepted.value.seals)
    || request.accepted_seal_set_digest !== accepted.value.digest) {
    throw new Error("lease request accepted seal set differs from the validated queue state");
  }
  if (JSON.stringify(request.barrier_seals) !== JSON.stringify(barriers.value.seals)
    || request.barrier_seal_set_digest !== barriers.value.digest) {
    throw new Error("lease request barrier seal set differs from the validated queue barriers");
  }
  if (accepted.sealedBundles.some((entry) => entry.reference.bundle_id === bundle.id)) {
    throw new Error(`${bundle.id}: cannot issue a lease for an already sealed bundle`);
  }
  validateRollingIntegrationBase(repository, request, base, control, accepted.sealedBundles);
  const sealedAuthorityFailures = subjectAuthorityPathFailures(
    control, base, accepted.sealedBundles.map((entry) => entry.reference.bundle_id),
  );
  if (sealedAuthorityFailures.length) {
    throw new Error(`lease base does not preserve sealed authority: ${sealedAuthorityFailures.join("; ")}`);
  }
  const semanticFailures = finalAuthorityFailuresForBundles(
    base, control, accepted.sealedBundles.map((sealed) => sealed.reference.bundle_id),
  );
  if (semanticFailures.length) {
    throw new Error(`lease base does not preserve sealed semantic authority: ${semanticFailures.join("; ")}`);
  }
  const payload = {
    schema_version: 6,
    kind: "runmat-builtin-migration-authored-lease",
    authority: "derived-from-reviewed-control",
    request: structuredClone(requestValue),
    control_manifest_digest: control.digest,
    bundle_id: bundle.id,
    lease_id: request.lease_id,
    owner: request.owner,
    base_revision: request.base_revision,
    lease_base_inventory: structuredClone(request.lease_base_inventory),
    queue_checkpoint_digest: request.queue_checkpoint_digest,
    queue_phase: request.queue_phase,
    accepted_seals: structuredClone(request.accepted_seals),
    accepted_seal_set_digest: request.accepted_seal_set_digest,
    barrier_seals: structuredClone(request.barrier_seals),
    barrier_seal_set_digest: request.barrier_seal_set_digest,
    authored_write_set: bundle.authored_write_set,
    source_migrations: bundle.source_migrations,
    forbidden_integration_outputs: bundle.integration_outputs.map(({ product_id, path, producer }) => ({ product_id, path, producer })),
    issued_at: request.issued_at,
    expires_at: request.expires_at,
  };
  return { ...payload, digest: evidenceDigest(payload) };
}

export function parseLeaseRequest(value, control) {
  assertValidatedControl(control);
  kind(value, 5, "runmat-builtin-migration-lease-request", "lease request");
  exact(value, ["schema_version", "kind", "authority", "control_manifest_digest", "bundle_id", "lease_id", "owner", "base_revision", "lease_base_inventory", "queue_checkpoint_digest", "queue_phase", "accepted_seals", "accepted_seal_set_digest", "barrier_seals", "barrier_seal_set_digest", "issued_at", "expires_at", "review"], "lease request");
  if (value.authority !== "reviewed-development-request") throw new Error("lease request has invalid authority");
  if (value.control_manifest_digest !== control.digest) throw new Error("lease request was reviewed for another control manifest");
  const bundleId = stableId(value.bundle_id, "lease request bundle id");
  if (!control.bundles.has(bundleId)) throw new Error(`lease request references unknown bundle ${bundleId}`);
  stableId(value.lease_id, "lease request id"); nonempty(value.owner, "lease request owner");
  sourceRevision(value.base_revision, "lease request base revision");
  const base = parseLeaseBaseInventoryBinding(value.lease_base_inventory, "lease request base inventory");
  digest(value.queue_checkpoint_digest, "lease request queue checkpoint digest");
  const queuePhase = enumValue(value.queue_phase, ["pilot", "production"], "lease request queue phase");
  if (base.source_revision !== value.base_revision) throw new Error("lease request base inventory revision differs from its base revision");
  validateAcceptedSealSet(
    control.digest, value.accepted_seals, value.accepted_seal_set_digest,
    "lease request accepted seals",
  );
  validateBarrierSealSet({
    controlManifestDigest: control.digest,
    bundleId,
    queuePhase,
    seals: value.barrier_seals,
    observedDigest: value.barrier_seal_set_digest,
    label: "lease request barrier seals",
  });
  const issued = timestamp(value.issued_at, "lease request issued_at");
  const expires = timestamp(value.expires_at, "lease request expires_at");
  if (expires <= issued) throw new Error("lease request expiry must follow issuance");
  exact(value.review, ["status", "evidence"], "lease request review");
  if (value.review.status !== "reviewed") throw new Error("lease request must be reviewed");
  uniqueStrings(value.review.evidence, "lease request review evidence");
  return value;
}

export function parseLease(value, control, repository, changedPaths = null) {
  assertValidatedControl(control);
  kind(value, 6, "runmat-builtin-migration-authored-lease", "authored lease");
  exact(value, ["schema_version", "kind", "authority", "request", "control_manifest_digest", "bundle_id", "lease_id", "owner", "base_revision", "lease_base_inventory", "queue_checkpoint_digest", "queue_phase", "accepted_seals", "accepted_seal_set_digest", "barrier_seals", "barrier_seal_set_digest", "authored_write_set", "source_migrations", "forbidden_integration_outputs", "issued_at", "expires_at", "digest"], "authored lease");
  if (value.authority !== "derived-from-reviewed-control") throw new Error("authored lease has invalid authority");
  const request = parseLeaseRequest(value.request, control);
  if (value.control_manifest_digest !== control.digest) throw new Error("lease was issued for another control manifest");
  const bundleId = stableId(value.bundle_id, "lease bundle id");
  const bundle = control.bundles.get(bundleId);
  if (!bundle) throw new Error(`lease references unknown bundle ${bundleId}`);
  stableId(value.lease_id, "lease id");
  nonempty(value.owner, "lease owner");
  sourceRevision(value.base_revision, "lease base revision");
  if (value.base_revision !== request.base_revision) throw new Error("lease base revision differs from the reviewed request");
  parseLeaseBaseInventoryBinding(value.lease_base_inventory, "lease base inventory");
  digest(value.queue_checkpoint_digest, "lease queue checkpoint digest");
  const queuePhase = enumValue(value.queue_phase, ["pilot", "production"], "lease queue phase");
  validateAcceptedSealSet(
    control.digest, value.accepted_seals, value.accepted_seal_set_digest,
    "lease accepted seals",
  );
  validateBarrierSealSet({
    controlManifestDigest: control.digest,
    bundleId: bundle.id,
    queuePhase,
    seals: value.barrier_seals,
    observedDigest: value.barrier_seal_set_digest,
    label: "lease barrier seals",
  });
  const acceptedByBundle = new Map(value.accepted_seals.map((entry) => [entry.bundle_id, entry]));
  for (const barrier of value.barrier_seals) {
    if (JSON.stringify(acceptedByBundle.get(barrier.bundle_id)) !== JSON.stringify(barrier)) {
      throw new Error("lease barrier set is not an exact subset of its accepted seal set");
    }
  }
  if (JSON.stringify(value.lease_base_inventory) !== JSON.stringify(request.lease_base_inventory)
    || value.queue_checkpoint_digest !== request.queue_checkpoint_digest
    || value.queue_phase !== request.queue_phase
    || JSON.stringify(value.accepted_seals) !== JSON.stringify(request.accepted_seals)
    || value.accepted_seal_set_digest !== request.accepted_seal_set_digest
    || JSON.stringify(value.barrier_seals) !== JSON.stringify(request.barrier_seals)
    || value.barrier_seal_set_digest !== request.barrier_seal_set_digest) {
    throw new Error("lease base inventory, queue checkpoint, or seal set differs from the reviewed request");
  }
  if (value.bundle_id !== request.bundle_id || value.lease_id !== request.lease_id || value.owner !== request.owner || value.issued_at !== request.issued_at || value.expires_at !== request.expires_at) throw new Error("lease fields differ from the reviewed request");
  const authored = array(value.authored_write_set, "lease authored write set");
  const sourceMigrations = array(value.source_migrations, "lease source migrations", { empty: true });
  const generated = array(value.forbidden_integration_outputs, "lease forbidden integration outputs", { empty: true });
  if (JSON.stringify(authored) !== JSON.stringify(bundle.authored_write_set)) throw new Error("lease authored write set differs from reviewed bundle scope");
  if (JSON.stringify(sourceMigrations) !== JSON.stringify(bundle.source_migrations)) throw new Error("lease source migrations differ from reviewed bundle control");
  if (JSON.stringify(generated) !== JSON.stringify(bundle.integration_outputs.map(({ product_id, path, producer }) => ({ product_id, path, producer })))) {
    throw new Error("lease integration-output exclusions differ from reviewed bundle scope");
  }
  const issued = timestamp(value.issued_at, "lease issued_at");
  const expires = timestamp(value.expires_at, "lease expires_at");
  if (expires <= issued) throw new Error("lease expiry must follow issuance");
  digest(value.digest, "lease digest");
  const { digest: _ignored, ...payload } = value;
  if (evidenceDigest(payload) !== value.digest) throw new Error("authored lease digest mismatch");
  assertRecordedIntegrationBase(repository, value.base_revision, control.baseline.revision);
  if (changedPaths) validateBundleDiff(bundle, changedPaths);
  const parsed = deepImmutable({ value, bundle });
  VALIDATED_LEASES.add(parsed);
  return parsed;
}

export function assertValidatedLease(value, control) {
  assertValidatedControl(control);
  if (!VALIDATED_LEASES.has(value)) throw new Error("operation requires the exact validated authored lease");
  if (value.value.control_manifest_digest !== control.digest) {
    throw new Error("authored lease was validated for another control manifest");
  }
  return value;
}

export function assertActiveLease(value, control, clock = Date.now) {
  const lease = assertValidatedLease(value, control);
  assertActiveInterval(lease.value, clock, "authored lease");
  return lease;
}

export function validateLeaseDiff(lease, control, changedPaths) {
  assertValidatedLease(lease, control);
  validateBundleDiff(lease.bundle, changedPaths);
}

export function assertLeaseBaseInventory(lease, control, inventory) {
  assertValidatedLease(lease, control);
  assertControlSubject(control, inventory);
  const observed = leaseBaseInventoryBinding(inventory);
  if (JSON.stringify(observed) !== JSON.stringify(lease.value.lease_base_inventory)) {
    throw new Error("lease base inventory differs from the exact inventory bound at issuance");
  }
  return inventory;
}

function validateBundleDiff(bundle, changedPaths) {
  const violations = [];
  for (const changed of changedPaths) {
    const sourcePath = typeof changed === "string" ? changed : changed.path;
    if (bundle.integration_outputs.some((entry) => entry.path === sourcePath)) {
      violations.push({ code: "integration-output-authored-by-lane", path: sourcePath });
    } else if (!pathAllowed(bundle.authored_write_set, sourcePath)) {
      violations.push({ code: "path-outside-authored-lease", path: sourcePath });
    }
  }
  if (violations.length) {
    const error = new Error(`authored lease violation: ${violations.map((entry) => entry.path).join(", ")}`);
    error.violations = violations;
    throw error;
  }
}

function timestamp(value, label) {
  const text = nonempty(value, label);
  const parsed = Date.parse(text);
  if (!Number.isFinite(parsed) || new Date(parsed).toISOString() !== text) throw new Error(`${label} must be an ISO-8601 UTC timestamp`);
  return parsed;
}

function assertActiveInterval(value, clock, label) {
  if (typeof clock !== "function") throw new Error(`${label} clock must be a function`);
  const observed = clock();
  const now = observed instanceof Date ? observed.getTime() : observed;
  if (!Number.isFinite(now)) throw new Error(`${label} clock must return a finite Unix timestamp or Date`);
  const issued = Date.parse(value.issued_at);
  const expires = Date.parse(value.expires_at);
  if (now < issued) throw new Error(`${label} is not active yet`);
  if (now >= expires) throw new Error(`${label} has expired`);
}

function validateRollingIntegrationBase(repository, request, base, control, sealedBundles) {
  validateAcceptedIntegrationCheckpoint(
    repository, request.base_revision, base,
    { revision: control.baseline.revision, inventory_digest: control.baseline.inventory_digest },
    sealedBundles,
  );
  const acceptedIds = new Set(sealedBundles.map((entry) => entry.reference.bundle_id));
  const currentFindings = new Set(base.migration_findings.map(evidenceDigest));
  const reviewedFindings = new Set(control.migrationFindings.map((entry) => entry.finding_digest));
  for (const finding of currentFindings) {
    if (!reviewedFindings.has(finding)) throw new Error(`${finding}: lease base contains an unreviewed migration finding`);
  }
  for (const finding of control.migrationFindings) {
    const shouldRemain = finding.disposition === "reviewed-no-action"
      || !acceptedIds.has(finding.bundle_id);
    if (currentFindings.has(finding.finding_digest) !== shouldRemain) {
      throw new Error(`${finding.finding_digest}: lease base finding state differs from the accepted seal set`);
    }
  }
}
