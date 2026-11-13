import { cpSync, readFileSync, mkdirSync, existsSync, renameSync, rmSync, mkdtempSync } from "node:fs";
import { join, dirname } from "node:path";
import { readJson } from "./documents.mjs";
import { inputFile, safePath } from "./manifest.mjs";
import { requireCompleteSnapshot, snapshotExamples } from "./snapshot-capture.mjs";

function copyCompletedProject(source, example, project, stage) {
  requireCompleteSnapshot(readJson(inputFile(project, ".snapshot/snapshot.json")));
  for (const path of example.files) {
    if (!readFileSync(inputFile(project, path)).equals(readFileSync(inputFile(source, "docs/simulation/examples/" + path)))) throw new Error("Snapshot source differs from its canonical example");
  }
  cpSync(project, join(stage, safePath(example.id), "project"), { recursive: true });
}

/** The coordinator serializes this short transaction, never the expensive capture.
 * Merge with the latest inventory; the final exporter requires every declared example. */
export function publishExampleSnapshots(source, captures = {}) {
  const examples = snapshotExamples(source);
  if (Object.keys(captures).some(id => !examples.some(example => example.id === id))) throw new Error("Unknown snapshot example");
  const output = join(source, "target/docs/example-snapshots");
  mkdirSync(dirname(output), { recursive: true });
  const stage = mkdtempSync(join(dirname(output), ".example-snapshots-"));
  try {
    for (const example of examples) {
      const project = captures[example.id] ?? join(output, safePath(example.id), "project");
      if (!existsSync(project)) {
        if (captures[example.id]) throw new Error("Completed capture project is missing");
        continue;
      }
      copyCompletedProject(source, example, project, stage);
    }
    const backup = output + ".previous";
    rmSync(backup, { recursive: true, force: true });
    if (existsSync(output)) renameSync(output, backup);
    try { renameSync(stage, output); } catch (error) { if (existsSync(backup)) renameSync(backup, output); throw error; }
    rmSync(backup, { recursive: true, force: true });
    return output;
  } catch (error) {
    rmSync(stage, { recursive: true, force: true });
    throw error;
  }
}

export function completedCaptures(source, directory) {
  if (existsSync(join(directory, "capture-failure.json"))) throw new Error("Failed captures cannot be published");
  const captures = Object.fromEntries(snapshotExamples(source)
    .filter(example => existsSync(join(directory, safePath(example.id), "project")))
    .map(example => [example.id, join(directory, safePath(example.id), "project")]));
  if (!Object.keys(captures).length) throw new Error("No completed capture projects found");
  return captures;
}
