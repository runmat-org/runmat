import { evidenceDigest } from "./evidence.mjs";
import { assertValidatedControl } from "./control.mjs";
import { parseGateResult } from "./gate-result.mjs";
import { array, digest, exact, kind, repositoryPath, sourceRevision, stableId, uniqueStrings, SAFE_IDENTITY } from "./schema.mjs";

export function sealBundle(manifest, verification, gateValues, prerequisiteValues = [], control = null, repository = null) {
  assertValidatedControl(control);
  const parsed = parseSealManifest(manifest);
  const failures = [];
  if (control.digest !== parsed.control_manifest_digest || !control.bundles.has(parsed.bundle_id)) throw new Error("seal requires its exact parsed control manifest");
  if (parsed.baseline_inventory_digest !== control.baseline.inventory_digest) throw new Error("seal baseline inventory differs from the reviewed control baseline");
  const requiredPrerequisites = control.bundles.get(parsed.bundle_id).prerequisites.map((entry) => entry.bundle_id).sort();
  const declaredPrerequisites = parsed.prerequisite_seals.map((entry) => entry.bundle_id).sort();
  if (JSON.stringify(requiredPrerequisites) !== JSON.stringify(declaredPrerequisites)) throw new Error("seal prerequisite references do not exactly match the control DAG");
  reconcileVerification(parsed, verification, failures);
  const expected = {
    source_revision: parsed.source_revision, source_digest: parsed.source_digest,
    baseline_source_revision: control.baseline.revision,
    baseline_inventory_digest: parsed.baseline_inventory_digest,
    subject_inventory_digest: parsed.subject_inventory_digest,
    control_manifest_digest: parsed.control_manifest_digest, bundle_id: parsed.bundle_id,
    gate_plans: control.bundles.get(parsed.bundle_id).gate_plans, compiled_build: control.baseline.compiled_target,
    repository,
  };
  if (!repository) throw new Error("seal requires the trusted repository path for gate-plan verification");
  const gates = new Map();
  for (const reference of parsed.integration_gates) {
    const loaded = gateValues.find((entry) => entry.reference.artifact_id === reference.artifact_id);
    if (!loaded || evidenceDigest(loaded.value) !== reference.digest) { failures.push(issue("integration-gate-reference-invalid", reference.artifact_id)); continue; }
    try {
      const gate = parseGateResult(loaded.value, expected);
      if (gate.artifact_id !== reference.artifact_id || gate.result !== "pass") throw new Error(`${gate.gate}: evidence is not passing`);
      if (JSON.stringify(gate.identities) !== JSON.stringify(parsed.identities)) throw new Error(`${gate.gate}: identities differ from bundle`);
      if (gates.has(gate.gate)) throw new Error(`duplicate integration gate ${gate.gate}`);
      gates.set(gate.gate, gate);
    } catch (error) { failures.push(issue("integration-gate-invalid", error.message)); }
  }
  for (const required of ["deterministic-products", "inventory-delta"]) if (!gates.has(required)) failures.push(issue("integration-gate-missing", required));
  for (const reference of parsed.prerequisite_seals) {
    const loaded = prerequisiteValues.find((entry) => entry.reference.artifact_id === reference.artifact_id);
    if (!loaded || evidenceDigest(loaded.value) !== reference.digest || loaded.value.schema_version !== 2 || loaded.value.kind !== "runmat-builtin-migration-seal-result" || loaded.value.result !== "pass" || loaded.value.seal_id !== reference.artifact_id || loaded.value.bundle_id !== reference.bundle_id || loaded.value.control_manifest_digest !== parsed.control_manifest_digest) failures.push(issue("prerequisite-seal-invalid", reference.artifact_id));
  }
  return {
    schema_version: 2, kind: "runmat-builtin-migration-seal-result", authority: "development-integration-evidence-only",
    seal_id: parsed.seal_id, bundle_id: parsed.bundle_id, identities: parsed.identities,
    source_revision: parsed.source_revision, source_digest: parsed.source_digest,
    baseline_inventory_digest: parsed.baseline_inventory_digest,
    subject_inventory_digest: parsed.subject_inventory_digest,
    control_manifest_digest: parsed.control_manifest_digest, verification_digest: parsed.verification.digest,
    prerequisite_seals: parsed.prerequisite_seals, integration_gate_artifacts: [...gates.values()].map((entry) => entry.artifact_id).sort(),
    result: failures.length ? "fail" : "pass", failures,
  };
}

export function parseSealManifest(value) {
  kind(value, 2, "runmat-builtin-migration-seal-manifest", "seal manifest");
  exact(value, ["schema_version", "kind", "authority", "seal_id", "bundle_id", "identities", "source_revision", "source_digest", "baseline_inventory_digest", "subject_inventory_digest", "control_manifest_digest", "verification", "integration_gates", "prerequisite_seals", "review"], "seal manifest");
  if (value.authority !== "reviewed-integration-request") throw new Error("seal manifest has invalid authority");
  const result = {
    ...value, seal_id: stableId(value.seal_id, "seal id"), bundle_id: stableId(value.bundle_id, "seal bundle id"),
    identities: uniqueStrings(value.identities, "seal identities", { pattern: SAFE_IDENTITY, lower: true }).sort(),
    source_revision: sourceRevision(value.source_revision, "seal source revision"), source_digest: digest(value.source_digest, "seal source digest"),
    baseline_inventory_digest: digest(value.baseline_inventory_digest, "seal baseline inventory digest"),
    subject_inventory_digest: digest(value.subject_inventory_digest, "seal subject inventory digest"),
    control_manifest_digest: digest(value.control_manifest_digest, "seal control digest"),
    verification: reference(value.verification, "seal verification"),
    integration_gates: array(value.integration_gates, "seal integration gates").map((entry) => reference(entry, "seal integration gate")),
    prerequisite_seals: array(value.prerequisite_seals, "prerequisite seals", { empty: true }).map(prerequisiteReference),
  };
  exact(value.review, ["status", "evidence"], "seal review");
  if (value.review.status !== "reviewed") throw new Error("seal request must be reviewed");
  uniqueStrings(value.review.evidence, "seal review evidence");
  return result;
}

function reconcileVerification(manifest, loaded, failures) {
  if (!loaded || evidenceDigest(loaded) !== manifest.verification.digest) { failures.push(issue("verification-reference-invalid", manifest.verification.artifact_id)); return; }
  if (loaded.schema_version !== 3 || loaded.kind !== "runmat-builtin-migration-verification-result" || loaded.result !== "pass") failures.push(issue("verification-not-passing", manifest.verification.artifact_id));
  else if (loaded.artifact_id !== manifest.verification.artifact_id || loaded.bundle_id !== manifest.bundle_id || JSON.stringify(loaded.identities) !== JSON.stringify(manifest.identities) || loaded.source_revision !== manifest.source_revision || loaded.source_digest !== manifest.source_digest || loaded.baseline_inventory_digest !== manifest.baseline_inventory_digest || loaded.subject_inventory_digest !== manifest.subject_inventory_digest || loaded.control_manifest_digest !== manifest.control_manifest_digest) failures.push(issue("verification-provenance-mismatch", manifest.verification.artifact_id));
}

function reference(value, label) { exact(value, ["path", "artifact_id", "digest"], label); return { path: repositoryPath(value.path, `${label} path`), artifact_id: stableId(value.artifact_id, `${label} artifact id`), digest: digest(value.digest, `${label} digest`) }; }
function prerequisiteReference(value) { exact(value, ["path", "artifact_id", "digest", "bundle_id"], "prerequisite seal"); return { ...reference({ path: value.path, artifact_id: value.artifact_id, digest: value.digest }, "prerequisite seal"), bundle_id: stableId(value.bundle_id, "prerequisite bundle id") }; }
function issue(code, detail) { return { code, detail }; }
