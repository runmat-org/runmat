import { parseExampleGateProof } from "../example-gate.mjs";

export function exampleMaturityGate(gate) {
  if (gate === "native-examples") return "native-example";
  if (gate === "browser-examples") return "browser-example";
  throw new Error(`${gate}: not an example gate`);
}

export function parseExampleProducer(stdout, status, identities, gate, context, services) {
  if (status !== 0) return services.failedProducer(gate, identities);
  const artifactOutput = services.producerArtifactOutput(context, gate);
  const proof = parseExampleGateProof(JSON.parse(stdout), {
    source_revision: context.subject.source.revision,
    identities,
    manifest_evidence: context.manifestEvidence,
    evidence_root: context.evidenceRoot,
  });
  const rows = new Map(proof.rows.map((entry) => [entry.identity, entry]));
  const maturityGate = exampleMaturityGate(gate);
  const checks = identities.map((identity) => {
    const row = rows.get(identity);
    const controlled = context.control.identities.get(identity);
    const passed = exampleProofSatisfiesMaturity(proof, row, controlled, gate, maturityGate);
    return { id: `${gate}:${identity}`, result: passed ? "pass" : "fail", evidence_digest: row.evidence_digest };
  });
  return {
    checks,
    artifacts: [services.writeProducerArtifact(artifactOutput, "example-reconciliation", stdout)],
  };
}

export function exampleProofSatisfiesMaturity(proof, row, controlled, gate, maturityGate = exampleMaturityGate(gate)) {
  const exampleRequired = controlled.maturity[maturityGate].applicability === "required";
  const runtimeRequired = gate === "browser-examples"
    && controlled.maturity["browser-runtime"].applicability === "required";
  const runtimePassed = !runtimeRequired || proof.products.includes("browser-wasm");
  return runtimePassed && (row.status === "passed" || (!exampleRequired && row.status === "absent"));
}
