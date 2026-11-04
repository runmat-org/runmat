import { copyFileSync, readFileSync, constants, cpSync, mkdirSync, existsSync, mkdtempSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { join, dirname, basename } from "node:path";
import { execFileSync } from "node:child_process";
import { readJson } from "./documents.mjs";
import { inputFile, safePath } from "./manifest.mjs";
import { writeSnapshotProject } from "./snapshot-project.mjs";

export function snapshotExamples(source, selectedIds = null) {
  const examples = readJson(join(source, "docs/simulation/examples/catalog.json")).examples.filter(example => example.snapshot);
  if (selectedIds && selectedIds.some(id => !examples.some(example => example.id === id))) throw new Error("Unknown snapshot example");
  return selectedIds ? examples.filter(example => selectedIds.includes(example.id)) : examples;
}

export function requireCompleteSnapshot(snapshot) {
  if (snapshot.schema_version !== 1 || !snapshot.publication?.numerically_acceptable || !Array.isArray(snapshot.views) || snapshot.views.length === 0) throw new Error("Example snapshot is incomplete");
}

export function retainCaptureExecutable(executable, stage) {
  const directory = join(stage, ".native-executable");
  mkdirSync(directory, { recursive: true });
  const retained = join(directory, basename(executable));
  // All current_exe() workers keep this build, even if Cargo replaces the source.
  copyFileSync(executable, retained, constants.COPYFILE_FICLONE);
  return retained;
}

/** Produce completed projects without reading or replacing the shared inventory.
 * Independent captures may run concurrently; the coordinator serializes publication.
 * Neither a failed nor a running capture is publishable. */
export function captureExampleSnapshots(source, executable, selectedIds, output, failedCapture = null, resumeCaptures = null) {
  const examples = snapshotExamples(source, selectedIds);
  if (existsSync(output)) throw new Error("Capture output already exists");
  if (failedCapture && (failedCapture === output || existsSync(failedCapture))) throw new Error("Failed capture destination must be separate and unused");
  mkdirSync(dirname(output), { recursive: true });
  const stage = mkdtempSync(join(dirname(output), ".example-captures-"));
  try {
    const retained = examples.length ? retainCaptureExecutable(executable, stage) : null;
    for (const example of examples) {
      const project = join(stage, safePath(example.id), "project");
      mkdirSync(join(project, ".snapshot"), { recursive: true });
      if (resumeCaptures) {
        const original = join(resumeCaptures, safePath(example.id), "project");
        for (const path of example.files) {
          if (!readFileSync(inputFile(original, path)).equals(readFileSync(inputFile(source, "docs/simulation/examples/" + path)))) {
            throw new Error("Saved capture inputs differ from the canonical example");
          }
        }
        cpSync(original, project, { recursive: true, mode: constants.COPYFILE_FICLONE });
      }
      for (const path of example.files) {
        const destination = join(project, safePath(path));
        mkdirSync(dirname(destination), { recursive: true });
        cpSync(inputFile(source, "docs/simulation/examples/" + path), destination);
      }
      if (resumeCaptures) {
        const checkpoint = existsSync(join(project, ".snapshot/accepted-solve.json")) ? ".snapshot/accepted-solve.json" : ".snapshot/snapshot.json";
        if (!existsSync(join(project, checkpoint))) throw new Error("Capture has no accepted numerical checkpoint to render");
        // Explicit presentation resume: native authenticates the original saved
        // study and field metadata. This does not qualify changed numerical code.
        execFileSync(retained, ["render-snapshot", checkpoint, "--out", join(project, ".snapshot/snapshot.json")], { cwd: project, stdio: "inherit" });
        rmSync(join(project, ".snapshot/accepted-solve.json"), { force: true });
      } else {
        execFileSync(retained, ["snapshot", join(project, example.entry), "--out", join(project, ".snapshot/snapshot.json")], { cwd: project, stdio: "inherit" });
      }
      const snapshot = readJson(join(project, ".snapshot/snapshot.json"));
      requireCompleteSnapshot(snapshot);
      writeSnapshotProject(project, example, snapshot);
      // A separate process reads every original field page from this project's
      // relative artifact root. Warm solver/cache state cannot satisfy this gate.
      execFileSync(retained, ["verify-snapshot", ".snapshot/snapshot.json", "--out", ".snapshot/cold-field-replay.json"], { cwd: project, stdio: "inherit" });
    }
    rmSync(join(stage, ".native-executable"), { recursive: true, force: true });
    renameSync(stage, output);
    return Object.fromEntries(examples.map(example => [example.id, join(output, safePath(example.id), "project")]));
  } catch (error) {
    if (failedCapture) {
      retainFailedCapture(stage, failedCapture, error);
      throw new Error(`Native capture failed; diagnostic project retained at ${failedCapture}`, { cause: error });
    }
    rmSync(stage, { recursive: true, force: true });
    throw error;
  }
}

/** Preserve expensive native artifacts at their original relative project paths.
 * A failed capture is explicitly excluded from publication, even if an earlier
 * example completed. Executable copies are discarded after native teardown. */
export function retainFailedCapture(stage, destination, error) {
  if (existsSync(destination)) throw new Error("Failed capture destination already exists", { cause: error });
  mkdirSync(dirname(destination), { recursive: true });
  writeFileSync(join(stage, "capture-failure.json"), JSON.stringify({
    schemaVersion: 1, publishable: false, diagnostic: String(error).slice(0, 4096),
  }, null, 2));
  rmSync(join(stage, ".native-executable"), { recursive: true, force: true });
  renameSync(stage, destination);
}
