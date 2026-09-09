import { combineMachineReports, validateMachineReport } from "../../runtime/builtin-example-verifier/reporting.mjs";
import { usesBrowserLane, usesNativeLane } from "../../runtime/builtin-example-verifier/lanes.mjs";
import { sorted } from "./constants.mjs";
import { classifyExampleReport } from "./verify-schema.mjs";

export function reconcileExampleEvidence(manifest, loadedReports) {
  if (loadedReports.length !== manifest.example_reports.length) throw new Error("Loaded example report count does not match manifest");
  const reports = loadedReports.map(({ reference, value }, index) => {
    const expected = manifest.example_reports[index];
    if (reference.path !== expected.path || reference.artifact !== expected.artifact) throw new Error("Example evidence reference order changed during loading");
    const report = classifyExampleReport(value);
    if (report.artifact !== reference.artifact) throw new Error(`Example artifact mismatch for ${reference.path}`);
    if (report.source !== manifest.batch.source) throw new Error(`Stale example source for ${reference.artifact}`);
    if (report.inventory_digest !== manifest.batch.example_inventory_digest) throw new Error(`Stale example inventory for ${reference.artifact}`);
    return report;
  });
  const kinds = new Set(reports.map((entry) => entry.kind));
  if (kinds.has("unknown") || kinds.size !== 1) throw new Error("Example evidence must be exactly one combined report or one complete shard set");
  let combined;
  if (reports[0].kind === "combined") {
    if (reports.length !== 1) throw new Error("Only one combined example report may be supplied");
    combined = reports[0].value;
  } else {
    combined = combineMachineReports(reports.map((entry) => entry.value), manifest.batch.source, manifest.batch.combined_example_artifact);
  }
  if (combined.metadata.artifact !== manifest.batch.combined_example_artifact) throw new Error("Combined example artifact mismatch");
  return {
    report: combined,
    inputArtifacts: reports.map((entry) => entry.artifact),
    constituentArtifacts: combined.metadata.constituentArtifacts,
    identities: manifest.expectations.map((expectation) => reconcileIdentity(expectation, combined.results)),
  };
}

function reconcileIdentity(expectation, results) {
  const owned = results.filter((entry) => String(entry.builtin).toLowerCase() === expectation.identity);
  const actualKeys = sorted(owned.map((entry) => entry.key));
  const failures = [];
  if (JSON.stringify(actualKeys) !== JSON.stringify(sorted(expectation.example_keys))) {
    failures.push({ code: "example-set-mismatch", expected: sorted(expectation.example_keys), actual: actualKeys });
  }
  const byKey = new Map(owned.map((entry) => [entry.key, entry]));
  const observedLanes = new Set();
  for (const key of expectation.example_keys) {
    const result = byKey.get(key);
    if (!result) continue;
    if (usesBrowserLane(result.harness)) observedLanes.add("browser");
    if (usesNativeLane(result.harness)) observedLanes.add("native");
    if (!result.matches) failures.push({ code: "example-failed", example_key: key });
  }
  for (const lane of expectation.required_lanes) {
    if (!observedLanes.has(lane)) failures.push({ code: "required-lane-missing", lane });
  }
  return {
    identity: expectation.identity,
    result: failures.length ? "fail" : "pass",
    required_lanes: expectation.required_lanes,
    observed_lanes: sorted(observedLanes),
    example_keys: expectation.example_keys,
    failures,
  };
}
