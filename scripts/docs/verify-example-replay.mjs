/** Native-produced artifact inspection and cold graphics import. This does not
 * execute a study, initialize a session, or assert browser/GPU rendering quality. */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { inputFile } from "./manifest.mjs";
import { requireGraphicsReplayBootstrap } from "./graphics.mjs";
import { requireCompleteSnapshot, snapshotExamples } from "./snapshot-capture.mjs";

export async function verifyExampleReplay(source, captures, graphics, selectedIds = null) {
  const examples = snapshotExamples(source, selectedIds);
  assert(examples.length > 0, "No saved examples selected");
  const bootstrap = readFileSync(inputFile(graphics, "runmat_wasm_web.js"), "utf8");
  requireGraphicsReplayBootstrap(bootstrap);
  // The generated web module has no Node package boundary. A data URL preserves
  // its ESM exports; supplying the WASM bytes avoids URL-relative fetching.
  const renderer = await import("data:text/javascript;base64," + Buffer.from(bootstrap).toString("base64"));
  await renderer.default({ module_or_path: new Uint8Array(readFileSync(inputFile(graphics, "runmat_wasm_web_bg.wasm"))) });
  let checked = 0;
  for (const example of examples) {
    const root = join(captures, example.id, "project/.snapshot");
    const snapshot = JSON.parse(readFileSync(inputFile(root, "snapshot.json"), "utf8"));
    requireCompleteSnapshot(snapshot);
    assert.equal(snapshot.views.length, 3, `${example.id}: incomplete guide views`);
    for (const view of snapshot.views) {
      const label = `${example.id}/${view.id}`;
      const bytes = new Uint8Array(readFileSync(inputFile(root, view.scene)));
      const scene = JSON.parse(new TextDecoder().decode(bytes));
      assert(scene.figure.plots.some(plot => plot.vertices?.length > 0 && plot.triangles?.length > 0), `${label}: missing mesh`);
      assert(view.initial_camera.axes.length > 0, `${label}: missing camera`);
      if (view.field) {
        const colours = scene.figure.plots.flatMap(plot => plot.vertex_colors_rgba ?? []);
        assert(new Set(colours.map(colour => JSON.stringify(colour))).size > 1, `${label}: missing field colours`);
        const legend = scene.figure.geometryOverlay.scalar_legend;
        const displayUnit = view.display_unit ?? view.unit;
        assert.equal(legend.unit, displayUnit, `${label}: legend unit differs`);
        // Field extrema remain in the source unit; presentation may use the
        // assessment's unit. Use the same Rust conversion authority as the CLI.
        const [minimum, maximum] = displayUnit === view.unit ? [view.minimum, view.maximum]
          : JSON.parse(execFileSync("cargo", ["run", "--locked", "--quiet", "-p", "runmat-types",
            "--example", "convert_physical_values", "--", view.unit, displayUnit,
            String(view.minimum), String(view.maximum)], { cwd: source, encoding: "utf8" }));
        assert.equal(legend.minimum, minimum, `${label}: legend minimum differs`);
        assert.equal(legend.maximum, maximum, `${label}: legend maximum differs`);
        assert(Number.isFinite(view.minimum) && Number.isFinite(view.maximum) && view.maximum > view.minimum, `${label}: invalid field range`);
      }
      const handle = renderer.importGeometryScene(bytes);
      assert(Number.isInteger(handle) && handle >= 0, `${label}: cold import failed`);
      try { checked++; } finally { assert.equal(renderer.closeGeometryScene(handle), true, `${label}: scene release failed`); }
      console.log(`PASS ${label}: saved mesh, field metadata and cold graphics import`);
    }
  }
  return checked;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const source = resolve(import.meta.dirname, "../..");
  let captures = join(source, "target/docs/example-snapshots");
  let graphics = join(source, "target/docs/graphics-runtime");
  const selectedIds = [];
  const args = process.argv.slice(2);
  for (let index = 0; index < args.length; index += 2) {
    if (!args[index + 1]) throw new Error("Missing replay verification argument");
    if (args[index] === "--example") selectedIds.push(args[index + 1]);
    else if (args[index] === "--captures") captures = resolve(args[index + 1]);
    else if (args[index] === "--graphics-runtime") graphics = resolve(args[index + 1]);
    else throw new Error("Usage: verify-example-replay.mjs [--example ID ...] [--captures DIR] [--graphics-runtime DIR]");
  }
  const checked = await verifyExampleReplay(source, captures, graphics, selectedIds.length ? selectedIds : null);
  console.log(`PASS ${checked} original saved views; no execution runtime or solve`);
}
