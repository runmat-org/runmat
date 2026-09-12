import { execFileSync } from "node:child_process";
import { buildDocumentationCutoverArtifact, documentationCutoverChecks, parseDocumentationCutoverArtifact } from "../documentation-cutover.mjs";
import { absolutePath, exact } from "../schema.mjs";
import { sourceFieldBaselineSource } from "../source-fields.mjs";

export function parseDocumentationProducer(stdout, status, identities, gate, context, services) {
  if (status !== 0) return services.failedProducer(gate, identities);
  if (!context.input.inputs) throw new Error(`${gate}: documentation producer requires reviewed source disposition inputs`);
  exact(context.input.inputs, ["source_dispositions", "artifact_output"], "documentation producer inputs");
  const artifact = buildDocumentationCutoverArtifact({
    catalog_export: JSON.parse(stdout), catalog_export_bytes: stdout,
    source_dispositions: context.input.inputs.source_dispositions,
    expected_sources: Object.fromEntries(identities.map((identity) => {
      const row = context.controlBaseline.identities.find((entry) => entry.identity === identity);
      const ownership = row?.lexical_observations?.ownership;
      const paths = [...(ownership?.sidecars ?? []), ...(ownership?.runtime_documentation_shadows ?? [])].sort();
      return [identity, paths.map((sourcePath) => {
        const frozen = context.controlBaseline.source.files.find((entry) => entry.path === sourcePath);
        if (!frozen) throw new Error(`${sourcePath}: documentation source is absent from the frozen inventory`);
        const git = context.tools.find((entry) => entry.role === "git")?.path;
        if (!git) throw new Error(`${gate}: reviewed git tool is required to read baseline documentation`);
        const bytes = execFileSync(git, ["show", `${context.controlBaseline.source.revision.slice("git:".length)}:${sourcePath}`], { cwd: services.repository });
        return sourceFieldBaselineSource(sourcePath, bytes, frozen.content_digest);
      })];
    })),
    provenance: {
      source_revision: context.subject.source.revision, source_digest: context.subject.source.digest,
      compiled_inventory_digest: context.subject.compiled_inventory.digest,
      control_manifest_digest: context.control.digest, bundle_id: context.bundle.id, identities,
    },
  });
  parseDocumentationCutoverArtifact(artifact, {
    source_revision: context.subject.source.revision, source_digest: context.subject.source.digest,
    compiled_inventory_digest: context.subject.compiled_inventory.digest,
    control_manifest_digest: context.control.digest, bundle_id: context.bundle.id, identities,
    catalog_export_digest: artifact.catalog_export.content_digest,
    source_dispositions: context.input.inputs.source_dispositions,
  });
  const output = services.canonicalPotentialPath(absolutePath(context.input.inputs.artifact_output, "documentation evidence output"));
  if (services.isWithin(services.repository, output)) throw new Error("documentation evidence output must be outside the canonical repository");
  return {
    checks: documentationCutoverChecks(artifact),
    artifacts: [services.writeProducerArtifact(output, "documentation-reconciliation", `${JSON.stringify(artifact, null, 2)}\n`)],
  };
}
