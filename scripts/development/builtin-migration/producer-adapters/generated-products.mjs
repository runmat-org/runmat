import { generatedProductChecks, parseGeneratedProductsProof } from "../generated-products.mjs";

export function parseGeneratedProductsProducer(stdout, status, identities, gate, context, services) {
  if (status !== 0) return services.failedProducer(gate, identities);
  const artifactOutput = services.producerArtifactOutput(context, gate);
  const proof = parseGeneratedProductsProof(JSON.parse(stdout), {
    integration_products: context.integrationProducts,
    source_files: context.subject.source.files,
    native_registration_manifest: context.subject.compiled_inventory.snapshot.observed.registration_manifest,
  });
  return {
    checks: generatedProductChecks(proof, identities),
    artifacts: [services.writeProducerArtifact(artifactOutput, "generated-products", stdout)],
  };
}
