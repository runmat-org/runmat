import { evidenceDigest } from "./evidence.mjs";
import { parseGateResult } from "./gate-result.mjs";
import { parseVerificationManifest, validateAudit } from "./verify-schema.mjs";

export function verifyBatch(manifestValue, loadedAudit, loadedGates) {
  const manifest = parseVerificationManifest(manifestValue);
  const failures = [];
  const audit = reconcileReference(manifest.audit, loadedAudit, "audit", failures);
  if (audit) {
    try { validateAudit(audit, { ...manifest.batch, audit_artifact_id: manifest.audit.artifact_id }); }
    catch (error) { failures.push(issue("audit-invalid", error.message)); }
  }
  const expected = {
    source_revision: manifest.batch.source_revision, source_digest: manifest.batch.source_digest,
    baseline_inventory_digest: manifest.batch.baseline_inventory_digest,
    subject_inventory_digest: manifest.batch.subject_inventory_digest,
    control_manifest_digest: manifest.batch.control_manifest_digest,
    bundle_id: manifest.batch.bundle_id,
  };
  const gates = new Map();
  for (const reference of manifest.gate_results) {
    const loaded = loadedGates.find((entry) => entry.reference.artifact_id === reference.artifact_id);
    const raw = reconcileReference(reference, loaded, "gate", failures);
    if (!raw) continue;
    try {
      const parsed = parseGateResult(raw, expected);
      if (parsed.artifact_id !== reference.artifact_id) throw new Error("gate artifact id differs from reference");
      if (parsed.result !== "pass") throw new Error(`${parsed.gate}: result is ${parsed.result}`);
      if (JSON.stringify(parsed.identities) !== JSON.stringify(manifest.batch.identities)) throw new Error(`${parsed.gate}: identities differ from batch`);
      if (gates.has(parsed.gate)) throw new Error(`duplicate evidence for gate ${parsed.gate}`);
      gates.set(parsed.gate, parsed);
    } catch (error) { failures.push(issue("gate-invalid", error.message)); }
  }
  if (audit) {
    const auditedArtifacts = [...audit.evidence.gate_artifacts].sort();
    const verifiedArtifacts = [...gates.values()].map((entry) => entry.artifact_id).sort();
    if (JSON.stringify(auditedArtifacts) !== JSON.stringify(verifiedArtifacts)) failures.push(issue("audit-gate-set-mismatch", "verification gates differ from the gate set accepted by audit"));
  }
  const identities = manifest.expectations.map((expectation) => {
    const missing = expectation.required_gates.filter((gate) => !gates.has(gate));
    return { identity: expectation.identity, required_gates: expectation.required_gates, observed_gates: expectation.required_gates.filter((gate) => gates.has(gate)), result: missing.length ? "fail" : "pass", failures: missing.map((gate) => issue("required-gate-missing", gate)) };
  });
  const passed = identities.filter((entry) => entry.result === "pass").length;
  return {
    schema_version: 3, kind: "runmat-builtin-migration-verification-result", authority: "development-verification-evidence-only",
    artifact_id: manifest.batch.artifact_id, source_revision: manifest.batch.source_revision, source_digest: manifest.batch.source_digest,
    baseline_inventory_digest: manifest.batch.baseline_inventory_digest,
    subject_inventory_digest: manifest.batch.subject_inventory_digest,
    control_manifest_digest: manifest.batch.control_manifest_digest,
    bundle_id: manifest.batch.bundle_id, identities: manifest.batch.identities,
    inputs: { audit: manifest.audit, gates: manifest.gate_results },
    summary: { identities: identities.length, passed, failed: identities.length - passed, global_failures: failures.length },
    result: passed === identities.length && !failures.length ? "pass" : "fail", global_failures: failures, identity_results: identities,
  };
}

function reconcileReference(reference, loaded, label, failures) {
  if (!loaded || loaded.reference.path !== reference.path || loaded.reference.artifact_id !== reference.artifact_id || loaded.reference.digest !== reference.digest) {
    failures.push(issue(`${label}-reference-invalid`, "loaded reference differs from reviewed manifest"));
    return null;
  }
  if (evidenceDigest(loaded.value) !== reference.digest) {
    failures.push(issue(`${label}-digest-mismatch`, reference.artifact_id));
    return null;
  }
  return loaded.value;
}

function issue(code, detail) { return { code, detail }; }
