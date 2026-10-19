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
      result: compiledInventoryAuthorityCheck(
        gate,
        authority(compiled, identity),
        context.control.identities.get(identity),
      ) ? "pass" : "fail",
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

export function compiledInventoryAuthorityCheck(gate, authorityValue, controlled) {
  if (!controlled) return false;
  if (gate === "catalog-contract") {
    if (controlled.public_identity.kind === "alias") {
      const aliases = authorityValue.catalog_aliases;
      return aliases.length === 1
        && aliases[0].alias.name.toLowerCase()
          === controlled.public_identity.alias_spelling.spelling.toLowerCase()
        && aliases[0].canonical.name
          === controlled.public_identity.canonical_identity;
    }
    return authorityValue.catalog_aliases.length === 0
      && authorityValue.catalog_entries.length
        === controlled.expected_authorities.catalog_entry_count
      && authorityValue.constants.length
        === controlled.expected_authorities.catalog_constant_count;
  }
  if (gate === "runtime-binding") return authorityValue.runtime_constants.length > 0
    || (authorityValue.runtime_bindings.length > 0
      && authorityValue.implementation_provenance
        .some((entry) => entry.authority === "canonical_binding"));
  if (gate === "native-link") {
    const expected = controlled.implementation.callable.kind === "owned"
      ? controlled.implementation.callable.bindings
        .map((entry) => `${entry.variant}\0${entry.native_symbol}`).sort()
      : [];
    const runtime = authorityValue.runtime_bindings
      .map((entry) => `${entry.variant}\0${entry.native_symbol}`).sort();
    const catalogVariants = authorityValue.catalog_entries.flatMap((entry) => entry.bindings
      .map((binding) => binding.variant)).sort();
    const expectedVariants = controlled.implementation.callable.kind === "owned"
      ? controlled.implementation.callable.bindings.map((entry) => entry.variant).sort()
      : [];
    return expected.length > 0
      && JSON.stringify(runtime) === JSON.stringify(expected)
      && JSON.stringify(catalogVariants) === JSON.stringify(expectedVariants);
  }
  return false;
}

function authority(compiled, identity) {
  const matches = (rows, selector = (entry) => entry.name) => rows.filter((entry) => selector(entry).toLowerCase() === identity.toLowerCase());
  return {
    catalog_entries: matches(compiled.snapshot.declared.catalog_entries, (entry) => entry.identity.name),
    catalog_aliases: matches(compiled.snapshot.declared.catalog_aliases, (entry) => entry.alias.name),
    constants: matches(compiled.snapshot.declared.constants),
    runtime_constants: matches(compiled.snapshot.observed.runtime_constants),
    runtime_bindings: matches(compiled.snapshot.observed.runtime_bindings),
    implementation_provenance: matches(compiled.snapshot.observed.implementation_provenance),
  };
}
