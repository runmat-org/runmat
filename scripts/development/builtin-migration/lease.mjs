import { pathAllowed } from "./control-graph.mjs";
import { assertValidatedControl } from "./control.mjs";
import { evidenceDigest } from "./evidence.mjs";
import { deepImmutable } from "./immutable.mjs";
import { array, digest, exact, kind, nonempty, sourceRevision, stableId, uniqueStrings } from "./schema.mjs";

const VALIDATED_LEASES = new WeakSet();

export function issueLease(requestValue, control) {
  assertValidatedControl(control);
  const request = parseLeaseRequest(requestValue, control);
  const bundle = control.bundles.get(request.bundle_id);
  const payload = {
    schema_version: 1,
    kind: "runmat-builtin-migration-authored-lease",
    authority: "derived-from-reviewed-control",
    request: requestValue,
    control_manifest_digest: control.digest,
    bundle_id: bundle.id,
    lease_id: request.lease_id,
    owner: request.owner,
    base_revision: control.baseline.revision,
    authored_write_set: bundle.authored_write_set,
    forbidden_integration_outputs: bundle.integration_outputs.map(({ product_id, path, producer }) => ({ product_id, path, producer })),
    issued_at: request.issued_at,
    expires_at: request.expires_at,
  };
  return { ...payload, digest: evidenceDigest(payload) };
}

export function parseLeaseRequest(value, control) {
  assertValidatedControl(control);
  kind(value, 1, "runmat-builtin-migration-lease-request", "lease request");
  exact(value, ["schema_version", "kind", "authority", "control_manifest_digest", "bundle_id", "lease_id", "owner", "issued_at", "expires_at", "review"], "lease request");
  if (value.authority !== "reviewed-development-request") throw new Error("lease request has invalid authority");
  if (value.control_manifest_digest !== control.digest) throw new Error("lease request was reviewed for another control manifest");
  const bundleId = stableId(value.bundle_id, "lease request bundle id");
  if (!control.bundles.has(bundleId)) throw new Error(`lease request references unknown bundle ${bundleId}`);
  nonempty(value.lease_id, "lease request id"); nonempty(value.owner, "lease request owner");
  const issued = timestamp(value.issued_at, "lease request issued_at");
  const expires = timestamp(value.expires_at, "lease request expires_at");
  if (expires <= issued) throw new Error("lease request expiry must follow issuance");
  exact(value.review, ["status", "evidence"], "lease request review");
  if (value.review.status !== "reviewed") throw new Error("lease request must be reviewed");
  uniqueStrings(value.review.evidence, "lease request review evidence");
  return value;
}

export function parseLease(value, control, changedPaths = null) {
  assertValidatedControl(control);
  kind(value, 1, "runmat-builtin-migration-authored-lease", "authored lease");
  exact(value, ["schema_version", "kind", "authority", "request", "control_manifest_digest", "bundle_id", "lease_id", "owner", "base_revision", "authored_write_set", "forbidden_integration_outputs", "issued_at", "expires_at", "digest"], "authored lease");
  if (value.authority !== "derived-from-reviewed-control") throw new Error("authored lease has invalid authority");
  const request = parseLeaseRequest(value.request, control);
  if (value.control_manifest_digest !== control.digest) throw new Error("lease was issued for another control manifest");
  const bundleId = stableId(value.bundle_id, "lease bundle id");
  const bundle = control.bundles.get(bundleId);
  if (!bundle) throw new Error(`lease references unknown bundle ${bundleId}`);
  nonempty(value.lease_id, "lease id");
  nonempty(value.owner, "lease owner");
  sourceRevision(value.base_revision, "lease base revision");
  if (value.base_revision !== control.baseline.revision) throw new Error("lease base revision differs from the reviewed control baseline");
  if (value.bundle_id !== request.bundle_id || value.lease_id !== request.lease_id || value.owner !== request.owner || value.issued_at !== request.issued_at || value.expires_at !== request.expires_at) throw new Error("lease fields differ from the reviewed request");
  const authored = array(value.authored_write_set, "lease authored write set");
  const generated = array(value.forbidden_integration_outputs, "lease forbidden integration outputs", { empty: true });
  if (JSON.stringify(authored) !== JSON.stringify(bundle.authored_write_set)) throw new Error("lease authored write set differs from reviewed bundle scope");
  if (JSON.stringify(generated) !== JSON.stringify(bundle.integration_outputs.map(({ product_id, path, producer }) => ({ product_id, path, producer })))) {
    throw new Error("lease integration-output exclusions differ from reviewed bundle scope");
  }
  const issued = timestamp(value.issued_at, "lease issued_at");
  const expires = timestamp(value.expires_at, "lease expires_at");
  if (expires <= issued) throw new Error("lease expiry must follow issuance");
  digest(value.digest, "lease digest");
  const { digest: _ignored, ...payload } = value;
  if (evidenceDigest(payload) !== value.digest) throw new Error("authored lease digest mismatch");
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

export function validateLeaseDiff(lease, control, changedPaths) {
  assertValidatedLease(lease, control);
  validateBundleDiff(lease.bundle, changedPaths);
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
