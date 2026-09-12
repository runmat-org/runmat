import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { verifyModuleCompositionProduct } from "../verify.mjs";

const repository = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../../..");
const cli = path.join(repository, "scripts/development/generate-builtin-module-composition.mjs");

test("composition CLI writes the exact typed parent atomically", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-module-composition-cli-"));
  try {
    const output = path.join(directory, "mod.rs");
    const product = catalogProduct();
    const completed = spawnSync(process.execPath, [cli, "--output", output], {
      cwd: repository,
      encoding: "utf8",
      input: `${JSON.stringify(product)}\n`,
    });
    assert.equal(completed.status, 0, completed.stderr);
    assert.doesNotThrow(() => verifyModuleCompositionProduct(product, fs.readFileSync(output, "utf8")));
    assert.deepEqual(fs.readdirSync(directory), ["mod.rs"]);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

function catalogProduct() {
  return {
    product_id: "catalog-math",
    crate_role: "catalog",
    path: "crates/runmat-builtins/src/catalog/entries/math/mod.rs",
    module_path: "crate::catalog::entries::math",
    aggregations: ["entries"],
    children: [{
      module: "arithmetic",
      source_kind: "directory",
      source_path: "crates/runmat-builtins/src/catalog/entries/math/arithmetic/mod.rs",
      role: "group",
      visibility: "private",
      declaration_condition: { kind: "always" },
      macro_use: false,
      reexports: [{ kind: "glob", visibility: "public", condition: { kind: "always" }, doc_hidden: false }],
      aggregation_sources: [{ role: "entries", kind: "slice", order: 0 }],
    }],
  };
}
