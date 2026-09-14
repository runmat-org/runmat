import { compareCodePoint } from "./constants.mjs";
import { assertValidatedControl } from "./control.mjs";
import { deepImmutable } from "./immutable.mjs";
import { requiredGateNames } from "./gate-requirements.mjs";
import {
  assertCompleteOrdinaryGateProof, parseOrdinaryGateProof,
} from "./ordinary-gate-proof.mjs";
import { parseMigrationPhases } from "./integration-phases.mjs";
import {
  validateAcceptedSealSet, validateBarrierSealSet,
} from "./seal-set-schema.mjs";
import {
  SAFE_IDENTITY, array, digest, enumValue, exact, integer, kind, nonempty,
  repositoryPath, sourceRevision, stableId, uniqueStrings,
} from "./schema.mjs";

export function parseVerificationResult(value, control, { requirePassing = false } = {}) {
  assertValidatedControl(control);
  kind(value, 8, "runmat-builtin-migration-verification-result", "verification result");
  exact(value, [
    "schema_version", "kind", "authority", "artifact_id", "source_revision",
    "source_digest", "control_baseline_inventory_digest", "lease_base_inventory_digest",
    "subject_inventory_digest", "control_manifest_digest", "bundle_id", "lease_id",
    "lease_digest", "queue_phase", "accepted_seals", "accepted_seal_set_digest",
    "barrier_seals", "barrier_seal_set_digest", "identities", "phases", "inputs",
    "ordinary_gate_proof", "summary", "result", "global_failures", "identity_results",
  ], "verification result");
  if (value.authority !== "development-verification-evidence-only") {
    throw new Error("verification result has invalid authority");
  }
  stableId(value.artifact_id, "verification result artifact id");
  sourceRevision(value.source_revision, "verification result source revision");
  for (const [field, label] of [
    ["source_digest", "source"],
    ["control_baseline_inventory_digest", "control baseline inventory"],
    ["lease_base_inventory_digest", "lease base inventory"],
    ["subject_inventory_digest", "subject inventory"],
    ["control_manifest_digest", "control manifest"],
    ["lease_digest", "lease"],
  ]) digest(value[field], `verification result ${label} digest`);
  if (value.control_manifest_digest !== control.digest) {
    throw new Error("verification result belongs to another control manifest");
  }
  const bundleId = stableId(value.bundle_id, "verification result bundle id");
  const bundle = control.bundles.get(bundleId);
  if (!bundle) throw new Error(`${bundleId}: verification result references unknown bundle`);
  stableId(value.lease_id, "verification result lease id");
  const queuePhase = enumValue(
    value.queue_phase, ["pilot", "production"], "verification result queue phase",
  );
  validateAcceptedSealSet(
    control.digest, value.accepted_seals, value.accepted_seal_set_digest,
    "verification result accepted seals",
  );
  validateBarrierSealSet({
    controlManifestDigest: control.digest,
    bundleId,
    queuePhase,
    seals: value.barrier_seals,
    observedDigest: value.barrier_seal_set_digest,
    label: "verification result barrier seals",
  });
  const identities = uniqueStrings(
    value.identities, "verification result identities", { pattern: SAFE_IDENTITY, lower: true },
  ).sort(compareCodePoint);
  const expectedIdentities = [...bundle.identities].sort(compareCodePoint);
  if (JSON.stringify(identities) !== JSON.stringify(expectedIdentities)) {
    throw new Error("verification result identities differ from the reviewed bundle");
  }
  const phases = parseMigrationPhases(value.phases);
  if (value.source_revision !== phases.integrated_revision) {
    throw new Error("verification result source revision differs from its integrated phase");
  }
  const inputs = parseInputs(value.inputs);
  const ordinaryGateProof = parseOrdinaryGateProof(
    value.ordinary_gate_proof, control, bundleId, queuePhase,
  );
  const proofReferences = [...ordinaryGateProof.gates.values()]
    .map((entry) => entry.reference)
    .sort((left, right) => compareCodePoint(left.artifact_id, right.artifact_id));
  if (proofReferences.some((reference) => !inputs.gates.some((input) =>
    JSON.stringify(input) === JSON.stringify(reference)))) {
    throw new Error("ordinary gate proof references evidence outside the verification inputs");
  }
  const identityResults = parseIdentityResults(value.identity_results, control, identities);
  const globalFailures = parseFailures(value.global_failures, "verification global failures");
  exact(value.summary, ["identities", "passed", "failed", "global_failures"], "verification summary");
  for (const field of ["identities", "passed", "failed", "global_failures"]) {
    integer(value.summary[field], `verification summary ${field}`, 0);
  }
  const passed = identityResults.filter((entry) => entry.result === "pass").length;
  const derivedResult = passed === identityResults.length && globalFailures.length === 0
    ? "pass" : "fail";
  enumValue(value.result, ["pass", "fail"], "verification result status");
  if (value.summary.identities !== identityResults.length
    || value.summary.passed !== passed
    || value.summary.failed !== identityResults.length - passed
    || value.summary.global_failures !== globalFailures.length
    || value.result !== derivedResult) {
    throw new Error("verification result or summary is inconsistent");
  }
  if (value.result === "pass") {
    assertCompleteOrdinaryGateProof(ordinaryGateProof);
    if (JSON.stringify(inputs.gates) !== JSON.stringify(proofReferences)) {
      throw new Error("passing verification gate inputs differ from its ordinary gate proof");
    }
  }
  if (requirePassing && value.result !== "pass") throw new Error("verification result is not passing");
  return deepImmutable({
    value, bundle, queuePhase, identities, phases, inputs, ordinaryGateProof,
    identityResults, globalFailures,
  });
}

function parseInputs(value) {
  exact(value, ["audit", "gates"], "verification inputs");
  const audit = parseReference(value.audit, "verification audit input");
  const references = array(value.gates, "verification gate inputs").map((entry) =>
    parseReference(entry, "verification gate input"));
  const ids = references.map((entry) => entry.artifact_id);
  if (new Set(ids).size !== ids.length
    || JSON.stringify(ids) !== JSON.stringify([...ids].sort(compareCodePoint))) {
    throw new Error("verification gate inputs must be unique and canonically ordered");
  }
  return { audit, gates: references };
}

function parseIdentityResults(value, control, identities) {
  const rows = array(value, "verification identity results").map((entry) => {
    exact(entry, [
      "identity", "required_gates", "observed_gates", "result", "failures",
    ], "verification identity result");
    const identity = stableId(entry.identity, "verification result identity");
    const required = uniqueStrings(entry.required_gates, `${identity} required gates`)
      .sort(compareCodePoint);
    const expected = control.identities.get(identity);
    if (!expected) throw new Error(`${identity}: verification result identity is not controlled`);
    const expectedRequired = requiredGateNames(expected);
    if (JSON.stringify(required) !== JSON.stringify(expectedRequired)) {
      throw new Error(`${identity}: verification required gates differ from reviewed control`);
    }
    const observed = uniqueStrings(entry.observed_gates, `${identity} observed gates`, { empty: true })
      .sort(compareCodePoint);
    if (observed.some((gate) => !required.includes(gate))) {
      throw new Error(`${identity}: verification observed an unrequired gate`);
    }
    const failures = parseFailures(entry.failures, `${identity} verification failures`);
    const missing = required.filter((gate) => !observed.includes(gate));
    const expectedFailures = missing.map((gate) => ({ code: "required-gate-missing", detail: gate }));
    const result = enumValue(entry.result, ["pass", "fail"], `${identity} verification result`);
    if (JSON.stringify(failures) !== JSON.stringify(expectedFailures)
      || result !== (missing.length ? "fail" : "pass")) {
      throw new Error(`${identity}: verification result conflicts with required gate coverage`);
    }
    return { identity, required_gates: required, observed_gates: observed, result, failures };
  });
  if (JSON.stringify(rows.map((entry) => entry.identity)) !== JSON.stringify(identities)) {
    throw new Error("verification identity results do not exactly cover the bundle identities");
  }
  return rows;
}

function parseFailures(value, label) {
  return array(value, label, { empty: true }).map((entry) => {
    exact(entry, ["code", "detail"], label);
    return { code: nonempty(entry.code, `${label} code`), detail: nonempty(entry.detail, `${label} detail`) };
  });
}

function parseReference(value, label) {
  exact(value, ["path", "artifact_id", "digest"], label);
  return {
    path: repositoryPath(value.path, `${label} path`),
    artifact_id: stableId(value.artifact_id, `${label} artifact id`),
    digest: digest(value.digest, `${label} digest`),
  };
}
