import { sorted } from "./constants.mjs";
import { reconcileExampleEvidence } from "./verify-examples.mjs";
import { parseVerificationManifest, validateFactoryAudit } from "./verify-schema.mjs";

export function verifyBatch(manifestValue, loadedFactory, loadedReports) {
  const manifest = parseVerificationManifest(manifestValue);
  const globalFailures = [];
  let factory = null;
  let examples = null;
  try {
    factory = reconcileFactory(manifest, loadedFactory);
  } catch (error) {
    globalFailures.push(failure("factory-evidence-invalid", error));
  }
  try {
    examples = reconcileExampleEvidence(manifest, loadedReports);
  } catch (error) {
    globalFailures.push(failure("example-evidence-invalid", error));
  }
  const identities = manifest.batch.identities.map((identity) => {
    const failures = [];
    const factoryResult = factory?.value.identities.find((entry) => entry.identity === identity);
    const exampleResult = examples?.identities.find((entry) => entry.identity === identity);
    if (!factoryResult) failures.push({ code: "factory-identity-missing" });
    else if (factoryResult.result !== "pass") failures.push({ code: "factory-audit-failed", audit_failures: factoryResult.failures });
    if (!exampleResult) failures.push({ code: "example-evidence-unavailable" });
    else failures.push(...exampleResult.failures);
    return {
      identity,
      result: failures.length ? "fail" : "pass",
      factory_result: factoryResult?.result ?? "missing",
      required_lanes: exampleResult?.required_lanes ?? [],
      observed_lanes: exampleResult?.observed_lanes ?? [],
      example_keys: exampleResult?.example_keys ?? [],
      failures,
    };
  });
  if (factory && factory.value.result !== "pass") globalFailures.push({ code: "factory-audit-not-passing" });
  if (factory && factory.value.global_diagnostics?.length) globalFailures.push({ code: "factory-global-errors", count: factory.value.global_diagnostics.length });
  const result = !globalFailures.length && identities.every((entry) => entry.result === "pass") ? "pass" : "fail";
  return {
    schema_version: 1,
    kind: "runmat-builtin-migration-verification-result",
    authority: "development-verification-evidence-only",
    metadata: {
      batch: { artifact: manifest.batch.artifact, source: manifest.batch.source },
      inventories: { factory: manifest.batch.factory_inventory_digest, examples: manifest.batch.example_inventory_digest },
      artifacts: {
        factory: manifest.factory_audit.artifact,
        example_combined: manifest.batch.combined_example_artifact,
        example_inputs: examples?.inputArtifacts ?? manifest.example_reports.map((entry) => entry.artifact),
        example_constituents: examples?.constituentArtifacts ?? [],
      },
    },
    summary: {
      identities: identities.length,
      passed: identities.filter((entry) => entry.result === "pass").length,
      failed: identities.filter((entry) => entry.result === "fail").length,
      global_failures: globalFailures.length,
    },
    result,
    global_failures: globalFailures,
    identities,
  };
}

function reconcileFactory(manifest, loaded) {
  if (!loaded || loaded.reference.path !== manifest.factory_audit.path
      || loaded.reference.artifact !== manifest.factory_audit.artifact) {
    throw new Error("Factory evidence reference changed during loading");
  }
  const audit = validateFactoryAudit(loaded.value);
  if (audit.artifact !== manifest.factory_audit.artifact) throw new Error("Factory audit artifact mismatch");
  if (audit.source !== manifest.batch.source) throw new Error("Stale factory audit source");
  if (audit.inventory_digest !== manifest.batch.factory_inventory_digest) throw new Error("Stale factory inventory");
  const requested = sorted(audit.value.requested_identities.map((entry) => entry.toLowerCase()));
  if (JSON.stringify(requested) !== JSON.stringify(manifest.batch.identities)) throw new Error("Factory audit identities do not match batch");
  return audit;
}

function failure(code, error) {
  return { code, detail: error instanceof Error ? error.message : String(error) };
}
