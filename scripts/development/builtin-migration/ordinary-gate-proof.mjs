import { compareCodePoint } from "./constants.mjs";
import { assertValidatedControl } from "./control.mjs";
import { evidenceDigest } from "./evidence.mjs";
import { requiredGateNames } from "./gate-requirements.mjs";
import { deepImmutable } from "./immutable.mjs";
import {
  array, digest, enumValue, exact, kind, repositoryPath, stableId, uniqueStrings,
} from "./schema.mjs";

const KIND = "runmat-builtin-migration-ordinary-gate-proof";

export function buildOrdinaryGateProof(control, bundleId, queuePhase, gates) {
  assertValidatedControl(control);
  const bundle = control.bundles.get(stableId(bundleId, "ordinary gate proof bundle id"));
  if (!bundle) throw new Error(`${bundleId}: ordinary gate proof references unknown bundle`);
  const phase = enumValue(queuePhase, ["pilot", "production"], "ordinary gate proof queue phase");
  const required = requiredOrdinaryGateNames(control, bundle);
  const gateResults = [...gates.values()].map((gate) => ({
    gate: gate.gate,
    path: gate.reference.path,
    artifact_id: gate.reference.artifact_id,
    digest: gate.reference.digest,
  })).sort((left, right) => compareCodePoint(left.gate, right.gate));
  const payload = {
    schema_version: 1,
    kind: KIND,
    authority: "derived-from-validated-ordinary-gates",
    control_manifest_digest: control.digest,
    bundle_id: bundle.id,
    queue_phase: phase,
    required_gate_names: required,
    gate_results: gateResults,
  };
  return deepImmutable({ ...payload, digest: evidenceDigest(payload) });
}

export function parseOrdinaryGateProof(value, control, bundleId, queuePhase) {
  assertValidatedControl(control);
  kind(value, 1, KIND, "ordinary gate proof");
  exact(value, [
    "schema_version", "kind", "authority", "control_manifest_digest", "bundle_id",
    "queue_phase", "required_gate_names", "gate_results", "digest",
  ], "ordinary gate proof");
  if (value.authority !== "derived-from-validated-ordinary-gates") {
    throw new Error("ordinary gate proof has invalid authority");
  }
  const expectedBundle = stableId(bundleId, "ordinary gate proof expected bundle id");
  if (value.control_manifest_digest !== control.digest || value.bundle_id !== expectedBundle) {
    throw new Error("ordinary gate proof belongs to another control or bundle");
  }
  const phase = enumValue(queuePhase, ["pilot", "production"], "ordinary gate proof expected queue phase");
  if (value.queue_phase !== phase) throw new Error("ordinary gate proof queue phase mismatch");
  const bundle = control.bundles.get(expectedBundle);
  if (!bundle) throw new Error(`${expectedBundle}: ordinary gate proof references unknown bundle`);
  const required = uniqueStrings(value.required_gate_names, "ordinary required gate names")
    .sort(compareCodePoint);
  if (JSON.stringify(required) !== JSON.stringify(requiredOrdinaryGateNames(control, bundle))) {
    throw new Error("ordinary gate proof required gates differ from reviewed control");
  }
  const gates = new Map();
  for (const entry of array(value.gate_results, "ordinary gate result references", { empty: true })) {
    exact(entry, ["gate", "path", "artifact_id", "digest"], "ordinary gate result reference");
    const gate = stableId(entry.gate, "ordinary gate result name");
    if (gates.has(gate)) throw new Error(`ordinary gate proof duplicates ${gate}`);
    gates.set(gate, {
      gate,
      reference: {
        path: repositoryPath(entry.path, `${gate} ordinary gate path`),
        artifact_id: stableId(entry.artifact_id, `${gate} ordinary gate artifact id`),
        digest: digest(entry.digest, `${gate} ordinary gate digest`),
      },
    });
  }
  const observed = [...gates.keys()];
  if (JSON.stringify(observed) !== JSON.stringify([...observed].sort(compareCodePoint))) {
    throw new Error("ordinary gate proof results must use canonical gate order");
  }
  if (observed.some((gate) => !required.includes(gate))) {
    throw new Error("ordinary gate proof contains an unrequired gate");
  }
  digest(value.digest, "ordinary gate proof digest");
  const { digest: _ignored, ...payload } = value;
  if (evidenceDigest(payload) !== value.digest) throw new Error("ordinary gate proof digest mismatch");
  return deepImmutable({ value, requiredGateNames: required, gates, digest: value.digest });
}

export function assertCompleteOrdinaryGateProof(proof) {
  const observed = [...proof.gates.keys()].sort(compareCodePoint);
  if (JSON.stringify(observed) !== JSON.stringify(proof.requiredGateNames)) {
    throw new Error("ordinary gate proof does not cover every required reviewed gate");
  }
  return proof;
}

function requiredOrdinaryGateNames(control, bundle) {
  const required = new Set();
  for (const identity of bundle.identities) {
    for (const gate of requiredGateNames(control.identities.get(identity))) required.add(gate);
  }
  return [...required].sort(compareCodePoint);
}
