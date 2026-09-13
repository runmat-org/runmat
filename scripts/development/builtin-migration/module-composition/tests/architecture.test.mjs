import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const productionFiles = [
  "authority.mjs", "baseline-authority.mjs", "baseline-candidate.mjs", "baseline-observation.mjs",
  "baseline-review.mjs", "baseline-schema.mjs", "baseline-source-head.mjs", "binding.mjs", "bootstrap.mjs", "child-state.mjs", "condition.mjs", "control.mjs", "durability.mjs",
  "effective-state.mjs", "generate.mjs", "handwritten-aggregation.mjs", "handwritten-parser.mjs", "handwritten.mjs", "index.mjs", "materialize.mjs",
  "prior-state.mjs", "projection.mjs", "registry.mjs", "repository-state.mjs", "rust-schema.mjs", "schema.mjs", "surface.mjs",
  "transaction-journal-schema.mjs", "transaction-journal.mjs", "transaction-lock-fence-worker.mjs", "transaction-lock-fence.mjs", "transaction-lock-recovery.mjs", "transaction-recovery-reservation.mjs",
  "transaction-lock-owner.mjs", "transaction-lock.mjs", "transaction-plan.mjs", "transaction.mjs", "verify.mjs",
];
const filesystemLeaves = new Set([
  "baseline-observation.mjs", "baseline-source-head.mjs", "child-state.mjs", "durability.mjs", "repository-state.mjs",
  "transaction-journal.mjs", "transaction-lock-owner.mjs", "transaction-lock-recovery.mjs", "transaction-lock.mjs", "transaction-recovery-reservation.mjs", "transaction.mjs",
]);

test("module composition stays bounded and cannot discover source from the filesystem", () => {
  for (const file of productionFiles) {
    const source = fs.readFileSync(path.join(root, file), "utf8");
    assert.ok(source.split("\n").length <= 192, `${file} exceeds the production leaf ceiling`);
    if (!filesystemLeaves.has(file)) {
      assert.doesNotMatch(source, /node:fs|node:child_process|\breaddir(?:Sync)?\s*\(|\bglob(?:Sync)?\s*\(|\bwalk(?:Dir|Sync)?\s*\(/);
    }
  }
});
