import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { writeSnapshotProject } from "./snapshot-project.mjs";
import { hash } from "./manifest.mjs";
for (const family of ["structural", "thermal"]) test(`${family} project uses shared FEA contracts and authentic dataset references`, () => {
  const root = mkdtempSync(join(tmpdir(), "runmat-snapshot-project-test-"));
  try {
    const descriptor = { field_id: family === "thermal" ? "thermal.temperature.0" : "structural.displacement", dtype: "float64", shape: [4, 3], class_name: "fea.VectorField" };
    const publication = { steps: [{ procedure_step: { step_id: "static", family } }], evidence_artifact_path: ".artifacts/evidence.json", assessments: [] };
    mkdirSync(join(root, ".snapshot"));
    writeFileSync(join(root, ".snapshot/mesh.scene.json"), "saved-scene");
    writeFileSync(join(root, ".snapshot/mesh.png"), "saved-poster");
    writeSnapshotProject(root, { entry: "study.fea" }, { run_id: "saved-run", publication, views: [{ scene: "mesh.scene.json", poster: "mesh.png" }], results: { field_descriptors: [descriptor], summary: { fixture: true }, diagnostics: [] } });
    const manifest = JSON.parse(readFileSync(join(root, ".artifacts/runs/docs-example.json")));
    const bytes = readFileSync(join(root, manifest.fea.datasetSrc));
    assert.equal(manifest.fea.datasetObject, `sha256:${hash(bytes)}`);
    const dataset = JSON.parse(bytes);
    assert.equal(dataset.kind, "finite_element_run_dataset");
    assert.equal(dataset.physicsFamily, family);
    assert.deepEqual(dataset.fieldBindings, [{ stepId: "static", runId: "saved-run", descriptor }]);
    assert.deepEqual(dataset.run, publication);
    assert.equal(manifest.cells.__fea__.kept[0].kind, "finite_element_dataset");
    assert.equal(manifest.cells.__fea__.kept[0].bytes, bytes.length);
    assert.equal(manifest.cells.__fea__.figures[0].sceneSrc, ".snapshot/mesh.scene.json");
    assert.equal(manifest.cells.__fea__.figures[0].sceneObject, "sha256:" + hash("saved-scene"));
  } finally { rmSync(root, { recursive: true, force: true }); }
});
