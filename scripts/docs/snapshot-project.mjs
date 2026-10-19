import { mkdirSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { hash, safePath, fileIdentity } from "./manifest.mjs";
import {
  FEA_RUN_DATASET_SCHEMA_VERSION, FEA_RUN_DATASET_KIND, FEA_DATASET_ARTIFACT_KIND,
  FEA_OBJECT_ARTIFACT_METADATA_SCHEMA_VERSION, FEA_RUN_KIND, FEA_RUN_CELL_ID,
  FEA_RUN_MANIFEST_METADATA_SCHEMA_VERSION, feaRunArtifactRefId,
} from "../../bindings/ts/dist/fea-contracts.js";

/** Use the same public FEA wire constants and content-addressed object layout
 * consumed by Desktop's ordinary run-history and cold descriptor hydration. */
export function writeSnapshotProject(project, example, snapshot) {
  const fieldBindings = snapshot.results.field_descriptors.map(descriptor => ({
    stepId: snapshot.publication.steps[0].procedure_step.step_id, runId: snapshot.run_id, descriptor,
  }));
  const dataset = {
    schemaVersion: FEA_RUN_DATASET_SCHEMA_VERSION, kind: FEA_RUN_DATASET_KIND,
    sessionId: snapshot.run_id, docPath: "/" + example.entry,
    study: { path: "/" + example.entry, documentKind: "study", valid: true,
      evidenceArtifactPaths: [snapshot.publication.evidence_artifact_path] },
    physicsFamily: snapshot.publication.steps[0].procedure_step.family, run: snapshot.publication, resultSummary: snapshot.results.summary,
    fieldBindings, fieldPaging: { mode: "paged", policy: "lazy_materialization" },
    assessments: snapshot.publication.assessments ?? [], diagnostics: snapshot.results.diagnostics ?? [],
  };
  const bytes = JSON.stringify(dataset, null, 2) + "\n";
  const digest = hash(bytes); const object = `sha256:${digest}`;
  const directory = `.artifacts/objects/${digest.slice(0, 2)}`;
  const src = `${directory}/${digest}.fea-dataset.json`;
  mkdirSync(join(project, directory), { recursive: true });
  writeFileSync(join(project, src), bytes);
  const metadata = { schemaVersion: FEA_OBJECT_ARTIFACT_METADATA_SCHEMA_VERSION,
    artifactId: object, src, bytes: Buffer.byteLength(bytes), mime: "application/json", kind: FEA_DATASET_ARTIFACT_KIND };
  writeFileSync(join(project, directory, digest + ".json"), JSON.stringify(metadata, null, 2) + "\n");
  // History's figure slots are persisted references, never live renderer handles.
  const figures = (snapshot.views ?? []).map((view, index) => {
    const sceneSrc = ".snapshot/" + safePath(view.scene);
    const previewSrc = ".snapshot/" + safePath(view.poster);
    return { handle: index + 1, sceneSrc, previewSrc,
      sceneObject: "sha256:" + fileIdentity(join(project, sceneSrc)).sha256,
      previewObject: "sha256:" + fileIdentity(join(project, previewSrc)).sha256 };
  });
  const manifest = {
    schemaVersion: 2, runId: snapshot.run_id, docPath: "/" + example.entry,
    createdAt: new Date().toISOString(), status: "completed", cancelled: false, runKind: FEA_RUN_KIND,
    fea: { schemaVersion: FEA_RUN_MANIFEST_METADATA_SCHEMA_VERSION, datasetObject: object, datasetSrc: src },
    cells: { [FEA_RUN_CELL_ID]: { figures, kept: [{ id: feaRunArtifactRefId("dataset"),
      kind: FEA_DATASET_ARTIFACT_KIND, src, bytes: metadata.bytes, artifactId: object }] } },
  };
  const manifestPath = ".artifacts/runs/docs-example.json";
  mkdirSync(dirname(join(project, manifestPath)), { recursive: true });
  writeFileSync(join(project, manifestPath), JSON.stringify(manifest, null, 2) + "\n");
}
