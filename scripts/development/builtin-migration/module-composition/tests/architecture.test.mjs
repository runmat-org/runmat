import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const productionFiles = ["binding.mjs", "generate.mjs", "index.mjs", "projection.mjs", "schema.mjs", "verify.mjs"];

test("module composition stays bounded and cannot discover source from the filesystem", () => {
  for (const file of productionFiles) {
    const source = fs.readFileSync(path.join(root, file), "utf8");
    assert.ok(source.split("\n").length <= 192, `${file} exceeds the production leaf ceiling`);
    assert.doesNotMatch(source, /node:fs|node:child_process|\breaddir(?:Sync)?\s*\(|\bglob(?:Sync)?\s*\(|\bwalk(?:Dir|Sync)?\s*\(/);
  }
});
