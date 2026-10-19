import { mkdtempSync, mkdirSync, rmSync } from "node:fs";
import { resolve, join } from "node:path";
import { fileURLToPath } from "node:url";
import { captureExampleSnapshots } from "./snapshot-capture.mjs";
import { publishExampleSnapshots, completedCaptures } from "./snapshot-inventory.mjs";
export { retainCaptureExecutable } from "./snapshot-capture.mjs";

export function buildExampleSnapshots(source, executable, selectedIds = null) {
  const parent = join(source, "target/docs");
  mkdirSync(parent, { recursive: true });
  const temporary = mkdtempSync(join(parent, ".snapshot-build-"));
  try {
    const captures = captureExampleSnapshots(source, executable, selectedIds, join(temporary, "captures"));
    return publishExampleSnapshots(source, captures);
  } finally { rmSync(temporary, { recursive: true, force: true }); }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const source = resolve(import.meta.dirname, "../..");
  const args = process.argv.slice(2);
  const selectedIds = [];
  let captureOnly, publishCaptures, failedCapture, resumeCaptures;
  for (let i = 0; i < args.length; i += 2) {
    if (!args[i + 1]) throw new Error("Missing snapshot argument");
    if (args[i] === "--example") selectedIds.push(args[i + 1]);
    else if (args[i] === "--capture-only" && !captureOnly) captureOnly = resolve(args[i + 1]);
    else if (args[i] === "--failed-capture" && !failedCapture) failedCapture = resolve(args[i + 1]);
    else if (args[i] === "--resume-captures" && !resumeCaptures) resumeCaptures = resolve(args[i + 1]);
    else if (args[i] === "--publish-captures" && !publishCaptures) publishCaptures = resolve(args[i + 1]);
    else throw new Error("Usage: build-example-snapshots.mjs [--example ID ...] [--capture-only DIR | --publish-captures DIR]");
  }
  if (resumeCaptures && !captureOnly) throw new Error("Presentation resume requires --capture-only; it does not establish new numerical execution");
  if (failedCapture && !captureOnly) throw new Error("Failed capture retention requires --capture-only");
  if (publishCaptures && (captureOnly || selectedIds.length)) throw new Error("Publication cannot select or execute studies");
  const executable = resolve(process.env.RUNMAT_DOCS_NATIVE_EXECUTABLE || join(source, "target/release/runmat"));
  if (publishCaptures) console.log(publishExampleSnapshots(source, completedCaptures(source, publishCaptures)));
  else if (captureOnly) console.log(captureExampleSnapshots(source, executable, selectedIds.length ? selectedIds : null, captureOnly, failedCapture, resumeCaptures));
  else console.log(buildExampleSnapshots(source, executable, selectedIds.length ? selectedIds : null));
}
