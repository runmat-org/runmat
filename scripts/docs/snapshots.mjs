import { mkdirSync, cpSync, readFileSync, writeFileSync, utimesSync } from "node:fs";
import { join, dirname } from "node:path";
import { hash, inventory, inputFile, safePath } from "./manifest.mjs";
import { readJson } from "./documents.mjs";
import { execFileSync } from "node:child_process";
export function collectSnapshot(root, output, example, packagePath) {
  if (!root) throw new Error("Produce the declared example snapshots before exporting docs");
  const project = join(root, safePath(example.id), "project");
  const snapshot = readJson(inputFile(project, ".snapshot/snapshot.json"));
  if (snapshot.schema_version !== 1 || !snapshot.publication.numerically_acceptable || !snapshot.views.length) throw new Error("Invalid example snapshot");
  if (typeof snapshot.run_id !== "string" || !snapshot.run_id || snapshot.views.some(view =>
    typeof view.id !== "string" || !view.id || typeof view.label !== "string" || !view.label ||
    typeof view.unit !== "string" || !view.unit || (view.scalar_scale !== undefined && (!Number.isFinite(view.scalar_scale) || view.scalar_scale <= 0)) || !Number.isFinite(view.deformation_scale) || view.deformation_scale < 0 ||
    ![view.minimum, view.maximum].every(value => value === null || Number.isFinite(value)))) throw new Error("Invalid saved-view metadata");
  for (const view of snapshot.views) validateSavedCamera(view.initial_camera);
  if (new Set(snapshot.views.map(view => view.id)).size !== snapshot.views.length) throw new Error("Duplicate saved view");
  // Bind captured inputs to the current canonical example before publishing old results.
  for (const file of example.files) {
    if (readFileSync(inputFile(project, file.path)).toString("utf8") !== file.content) throw new Error("Snapshot source differs from its canonical example");
  }
  const files = inventory(project);
  if (files.length > 4096 || files.some(file => file.bytes > 16 * 1024 * 1024) || files.reduce((sum, file) => sum + file.bytes, 0) > 64 * 1024 * 1024) throw new Error("Example project exceeds import limits");
  const manifest = { schema_version: 1, id: example.id, entry: example.entry, run_manifest: ".artifacts/runs/docs-example.json", files };
  if (!files.some(file => file.path === manifest.run_manifest)) throw new Error("Example has no durable replay manifest");
  const projectPath = packagePath + "/project";
  for (const file of files) {
    const destination = join(output, projectPath, "files", file.path); mkdirSync(dirname(destination), { recursive: true });
    cpSync(inputFile(project, file.path), destination); utimesSync(destination, new Date("1980-01-01T00:00:00Z"), new Date("1980-01-01T00:00:00Z"));
  }
  const bytes = JSON.stringify(manifest, null, 2) + "\n";
  writeFileSync(join(output, projectPath, "project.json"), bytes);
  execFileSync("zip", ["-q", "-X", join(output, packagePath, "project.zip"), ...files.map(file => file.path)], { cwd: join(output, projectPath, "files"), env: { ...process.env, TZ: "UTC" } });
  const views = snapshot.views.map(view => {
    const path = ".snapshot/" + safePath(view.scene); const scene = files.find(file => file.path === path);
    const poster = ".snapshot/" + safePath(view.poster);
    // The light-theme poster beside the dark one, when the capture drew one.
    const posterLight = view.poster_light === undefined ? null : ".snapshot/" + safePath(view.poster_light);
    if (!scene || !files.some(file => file.path === poster) || (posterLight && !files.some(file => file.path === posterLight))) throw new Error("Snapshot view has missing assets");
    const url = file => "/docs-assets/" + projectPath + "/files/" + file;
    return { ...view, scene: { url: url(path), sha256: scene.sha256, bytes: scene.bytes }, poster: url(poster), ...(posterLight ? { poster_light: url(posterLight) } : {}) };
  });
  return { run_id: snapshot.run_id, views, project: { manifest_path: projectPath + "/project.json", sha256: hash(bytes) }, publication: snapshot.publication };
}

export function snapshotIdentity(root, id) {
  if (!root) throw new Error("Produce the declared example snapshots before exporting docs");
  return hash(JSON.stringify(inventory(join(root, safePath(id), "project"))));
}

function validateSavedCamera(camera) {
  if (camera === undefined) return; // Legacy captures use the renderer's fit policy.
  const vector = value => Array.isArray(value) && value.length === 3 && value.every(Number.isFinite);
  if (!Number.isInteger(camera.activeAxes) || !Array.isArray(camera.axes) || camera.activeAxes < 0 || camera.activeAxes >= camera.axes.length || !camera.axes.every(axis =>
    vector(axis.position) && vector(axis.target) && vector(axis.up) &&
    axis.up.some(value => value !== 0) && axis.position.some((value, index) => value !== axis.target[index]) &&
    Number.isFinite(axis.zoom) && axis.zoom > 0 && Number.isFinite(axis.aspectRatio) && axis.aspectRatio > 0 &&
    axis.projection?.kind === "perspective" && Number.isFinite(axis.projection.fov) && axis.projection.fov > 0 && axis.projection.fov < Math.PI &&
    Number.isFinite(axis.projection.near) && axis.projection.near > 0 && Number.isFinite(axis.projection.far) && axis.projection.far > axis.projection.near))
    throw new Error("Invalid saved camera metadata");
}
