import { parseCompiledInventory } from "../compiled-inventory.mjs";
import { canonicalJson } from "../evidence.mjs";
import { buildInventoryDeltaProof, inventoryDeltaChecks } from "../inventory-delta.mjs";

export function parseCompiledInventoryProducer(stdout, status, identities, gate, context, services) {
  if (status !== 0) return services.failedProducer(gate, identities);
  const artifactOutput = services.producerArtifactOutput(context, gate);
  const compiled = parseCompiledInventory(JSON.parse(stdout));
  if (compiled.digest !== context.subject.compiled_inventory.digest) {
    throw new Error(`${gate}: producer output differs from the subject compiled inventory`);
  }
  return {
    checks: identities.map((identity) => ({
      id: `${gate}:${identity}`,
      result: compiledCheck(gate, authority(compiled, identity)) ? "pass" : "fail",
      evidence_digest: compiled.digest,
    })),
    artifacts: [services.writeProducerArtifact(artifactOutput, "compiled-inventory", stdout)],
  };
}

export function parseInventoryDeltaProducer(stdout, status, identities, gate, context, services) {
  if (status !== 0) return services.failedProducer(gate, identities);
  const artifactOutput = services.producerArtifactOutput(context, gate);
  const compiled = parseCompiledInventory(JSON.parse(stdout));
  if (compiled.digest !== context.subject.compiled_inventory.digest) {
    throw new Error(`${gate}: producer output differs from the subject compiled inventory`);
  }
  const proof = buildInventoryDeltaProof(
    services.repository, context.leaseBase, context.subject, context.control, context.bundle.id,
  );
  if (JSON.stringify(proof.identities.map((entry) => entry.identity)) !== JSON.stringify(identities)) {
    throw new Error(`${gate}: inventory delta identity coverage differs from the reviewed bundle`);
  }
  return {
    checks: inventoryDeltaChecks(proof),
    artifacts: [services.writeProducerArtifact(artifactOutput, "inventory-delta", `${canonicalJson(proof)}\n`)],
  };
}

function compiledCheck(gate, authorityValue) {
  if (gate === "catalog-contract") return authorityValue.catalog_entries.length === 1 || authorityValue.constants.length > 0;
  if (gate === "runtime-binding") return authorityValue.runtime_constants.length > 0
    || (authorityValue.runtime_bindings.length > 0 && authorityValue.implementation_provenance.some((entry) => entry.authority === "canonical_binding"));
  return false;
}

function authority(compiled, identity) {
  const matches = (rows, selector = (entry) => entry.name) => rows.filter((entry) => selector(entry).toLowerCase() === identity.toLowerCase());
  return {
    catalog_entries: matches(compiled.snapshot.declared.catalog_entries, (entry) => entry.identity.name),
    constants: matches(compiled.snapshot.declared.constants),
    runtime_constants: matches(compiled.snapshot.observed.runtime_constants),
    runtime_bindings: matches(compiled.snapshot.observed.runtime_bindings),
    implementation_provenance: matches(compiled.snapshot.observed.implementation_provenance),
  };
}
