import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { buildMachineReport } from "./reporting.mjs";
import { createInventory, shardInventory } from "./sharding.mjs";

const cli = fileURLToPath(new URL("../combine-builtin-example-reports.mjs", import.meta.url));

test("combiner CLI requires and preserves explicit product and constituent artifacts", (context) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-example-combiner-"));
  context.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const cases = ["foo#one", "foo#two"].map((exampleKey, id) => ({
    id, exampleIndex: id, exampleKey, builtin: "foo", authority: "catalog", compatibility: "Matlab",
    harness: "Portable", input: `x=${id}`, expectedOutput: "", hasExpectedOutput: false,
  }));
  const inventory = createInventory(cases);
  const inputs = [0, 1].map((index) => {
    const shard = { index, count: 2 };
    const selected = shardInventory(inventory, shard);
    const rows = selected.cases.map((testCase) => ({
      testCase, normalizedExpected: "", normalizedActual: "", imageRelPath: "", imageError: "", matches: true,
    }));
    const report = buildMachineReport({ rows, inventory, shard, range: selected.range, source: "git:test", artifact: `part-${index}` });
    const target = path.join(root, `part-${index}.json`);
    fs.writeFileSync(target, JSON.stringify(report));
    return target;
  });
  const output = path.join(root, "combined.json");
  const completed = spawnSync(process.execPath, [cli, "--output", output, "--artifact", "product-release", "--source", "git:test", ...inputs.reverse()], { encoding: "utf8" });
  assert.equal(completed.status, 0, completed.stderr);
  const combined = JSON.parse(fs.readFileSync(output, "utf8"));
  assert.equal(combined.schemaVersion, "runmat.builtin-example-report.v2");
  assert.equal(combined.metadata.artifact, "product-release");
  assert.deepEqual(combined.metadata.constituentArtifacts, ["part-0", "part-1"]);

  const missingArtifact = spawnSync(process.execPath, [cli, "--output", output, "--source", "git:test", ...inputs], { encoding: "utf8" });
  assert.notEqual(missingArtifact.status, 0);
  assert.match(missingArtifact.stderr, /--artifact/);
});
