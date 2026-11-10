import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync, existsSync, cpSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execFileSync } from "node:child_process";
import { buildExampleSnapshots, retainCaptureExecutable } from "./build-example-snapshots.mjs";
import { captureExampleSnapshots, retainFailedCapture } from "./snapshot-capture.mjs";
import { publishExampleSnapshots, completedCaptures } from "./snapshot-inventory.mjs";

function fixture() {
  const root = mkdtempSync(join(tmpdir(), "runmat-snapshot-selection-test-"));
  mkdirSync(join(root, "docs/simulation/examples"), { recursive: true });
  writeFileSync(join(root, "docs/simulation/examples/study.fea"), "canonical study");
  writeFileSync(join(root, "docs/simulation/examples/catalog.json"), JSON.stringify({ examples: [{ id: "saved", snapshot: true, files: ["study.fea"], entry: "study.fea" }] }));
  const project = join(root, "target/docs/example-snapshots/saved/project");
  mkdirSync(join(project, ".snapshot"), { recursive: true });
  writeFileSync(join(project, "study.fea"), "canonical study");
  writeFileSync(join(project, ".snapshot/snapshot.json"), JSON.stringify({ schema_version: 1, publication: { numerically_acceptable: true }, views: [{ id: "geometry" }] }));
  return { root, project };
}
test("explicit selection retains completed source-matching examples without executing them", () => {
  const { root, project } = fixture();
  try {
    buildExampleSnapshots(root, "/not-an-executable", []);
    assert.equal(readFileSync(join(project, "study.fea"), "utf8"), "canonical study");
    assert.equal(existsSync(join(root, "target/docs/example-snapshots.previous")), false);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
test("stale retained sources and unknown selections reject without replacing completed output", () => {
  const { root, project } = fixture();
  try {
    assert.throws(() => buildExampleSnapshots(root, "/not-an-executable", ["unknown"]), /Unknown snapshot example/);
    writeFileSync(join(root, "docs/simulation/examples/study.fea"), "changed study");
    assert.throws(() => buildExampleSnapshots(root, "/not-an-executable", []), /source differs/);
    assert.equal(readFileSync(join(project, "study.fea"), "utf8"), "canonical study");
  } finally { rmSync(root, { recursive: true, force: true }); }
});
test("a capture keeps its executable when a development rebuild replaces the source", () => {
  const root = mkdtempSync(join(tmpdir(), "runmat-capture-executable-test-"));
  try {
    const executable = join(root, "capture-program.mjs");
    writeFileSync(executable, "process.stdout.write('original build');\n", { mode: 0o755 });
    const retained = retainCaptureExecutable(executable, join(root, "stage"));
    writeFileSync(executable, "process.stdout.write('replacement build');\n");
    assert.equal(execFileSync(process.execPath, [retained], { encoding: "utf8" }), "original build");
    assert.equal(execFileSync(process.execPath, [executable], { encoding: "utf8" }), "replacement build");
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test("independent completed captures merge with the latest inventory without re-execution", () => {
  const { root, project } = fixture();
  try {
    const example = { id: "second", snapshot: true, files: ["second.fea"], entry: "second.fea" };
    const catalog = JSON.parse(readFileSync(join(root, "docs/simulation/examples/catalog.json")));
    catalog.examples.push(example);
    writeFileSync(join(root, "docs/simulation/examples/catalog.json"), JSON.stringify(catalog));
    writeFileSync(join(root, "docs/simulation/examples/second.fea"), "second canonical study");
    const captures = join(root, "completed");
    const captured = join(captures, "second/project");
    mkdirSync(join(captured, ".snapshot"), { recursive: true });
    cpSync(join(project, ".snapshot/snapshot.json"), join(captured, ".snapshot/snapshot.json"));
    writeFileSync(join(captured, "second.fea"), "second canonical study");
    // Another capture finishes after this one was prepared. Publication must
    // retain its current project, not a copy taken before lengthy native work.
    writeFileSync(join(project, "latest-result.txt"), "latest completed result");
    publishExampleSnapshots(root, completedCaptures(root, captures));
    assert.equal(readFileSync(join(project, "latest-result.txt"), "utf8"), "latest completed result");
    assert.equal(readFileSync(join(root, "target/docs/example-snapshots/second/project/second.fea"), "utf8"), "second canonical study");
    assert.equal(existsSync(join(root, "target/docs/example-snapshots/.native-executable")), false);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test("failed capture and stale captured source preserve the published inventory", () => {
  const { root, project } = fixture();
  try {
    const destination = join(root, "independent-capture");
    assert.throws(() => captureExampleSnapshots(root, "/not-an-executable", ["saved"], destination));
    assert.equal(existsSync(destination), false);
    assert.equal(readdirSync(root).some(name => name.startsWith(".example-captures-")), false);
    const captured = join(root, "stale/project");
    cpSync(project, captured, { recursive: true });
    writeFileSync(join(captured, "study.fea"), "stale study");
    assert.throws(() => publishExampleSnapshots(root, { saved: captured }), /source differs/);
    assert.equal(readFileSync(join(project, "study.fea"), "utf8"), "canonical study");
    assert.equal(existsSync(join(root, "target/docs/example-snapshots.previous")), false);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test("failed capture recovery keeps relative field objects but cannot publish", () => {
  const { root, project } = fixture();
  try {
    const stage = join(root, "stage");
    const destination = join(root, "failed");
    const field = join(stage, "saved/project/.artifacts/objects/field");
    mkdirSync(join(stage, "saved/project/.artifacts/objects"), { recursive: true });
    mkdirSync(join(stage, ".native-executable"));
    writeFileSync(field, "original field bytes");
    writeFileSync(join(stage, ".native-executable/runmat"), "temporary executable");
    retainFailedCapture(stage, destination, new Error("poster failed after accepted solve"));
    assert.equal(readFileSync(join(destination, "saved/project/.artifacts/objects/field"), "utf8"), "original field bytes");
    assert.equal(existsSync(join(destination, ".native-executable")), false);
    assert.throws(() => completedCaptures(root, destination), /Failed captures cannot be published/);
    assert.equal(readFileSync(join(project, "study.fea"), "utf8"), "canonical study");
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test("explicit presentation resume preserves original artifacts and never invokes a solve", () => {
  const { root, project } = fixture();
  try {
    const original = join(root, "saved-capture");
    const savedProject = join(original, "saved/project");
    cpSync(project, savedProject, { recursive: true });
    mkdirSync(join(savedProject, ".artifacts/objects"), { recursive: true });
    writeFileSync(join(savedProject, ".artifacts/objects/original"), "original physical fields");
    const snapshot = { schema_version: 1, run_id: "saved-run", publication: { numerically_acceptable: true, steps: [{ procedure_step: { step_id: "steady", family: "thermal" } }] }, results: { field_descriptors: [], summary: {}, diagnostics: [] }, views: [] };
    writeFileSync(join(savedProject, ".snapshot/accepted-solve.json"), JSON.stringify(snapshot));
    const executable = join(root, "native-dispatch-fixture.mjs");
    writeFileSync(executable, `#!${process.execPath}\nimport { readFileSync, writeFileSync, appendFileSync } from 'node:fs';\nconst args = process.argv.slice(2);\nif (!['render-snapshot', 'verify-snapshot'].includes(args[0])) process.exit(99);\nif (readFileSync('.artifacts/objects/original', 'utf8') !== 'original physical fields') process.exit(98);\nappendFileSync('invocations.txt', args[0] + '\\n');\nif (args[0] === 'render-snapshot') {\n const snapshot = JSON.parse(readFileSync(args[1], 'utf8'));\n snapshot.views = [{ id: 'geometry', scene: 'geometry.scene.json', poster: 'geometry.png' }];\n writeFileSync('.snapshot/geometry.scene.json', 'scene fixture');\n writeFileSync('.snapshot/geometry.png', 'poster fixture');\n writeFileSync(args[3], JSON.stringify(snapshot));\n} else writeFileSync(args[3], '{}');\n`, { mode: 0o755 });
    const captured = captureExampleSnapshots(root, executable, ["saved"], join(root, "resumed"), null, original).saved;
    assert.equal(readFileSync(join(captured, "invocations.txt"), "utf8"), "render-snapshot\nverify-snapshot\n");
    assert.equal(readFileSync(join(captured, ".artifacts/objects/original"), "utf8"), "original physical fields");
    assert.equal(existsSync(join(captured, ".snapshot/accepted-solve.json")), false);
    assert.equal(existsSync(join(root, "resumed/.native-executable")), false);
    writeFileSync(join(root, "docs/simulation/examples/study.fea"), "changed physical input");
    assert.throws(() => captureExampleSnapshots(root, executable, ["saved"], join(root, "stale-resume"), null, original), /inputs differ/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
