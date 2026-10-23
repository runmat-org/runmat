import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { productionDocumentation } from "./production-documentation.mjs";

const full = {
  schema_version: 2,
  inventory: { documents: 2, catalog_identities: 2, legacy_sidecars: 0, missing_catalog_documentation: [] },
  builtins: [
    { key: "abs", module_stem: "abs", authority: "catalog" },
    { key: "plot", module_stem: "plot", authority: "catalog" },
  ],
};
const scoped = {
  schema_version: 2,
  inventory: { documents: 1, catalog_identities: 1, legacy_sidecars: 0, missing_catalog_documentation: [] },
  builtins: [full.builtins[0]],
};

test("production docs reject a self-consistent scoped or stale supplied export", () => {
  const repository = mkdtempSync(join(tmpdir(), "runmat-doc-product-"));
  try {
    writeFileSync(join(repository, "scoped.json"), JSON.stringify(scoped));
    writeFileSync(join(repository, "full.json"), JSON.stringify({
      builtins: full.builtins, inventory: full.inventory, schema_version: full.schema_version,
    }));
    const calls = [];
    const spawn = (command, args) => {
      calls.push({ command, args });
      return { status: 0, stdout: JSON.stringify(command === "cargo" ? full : scoped), stderr: "" };
    };
    assert.deepEqual(productionDocumentation(repository, { exportPath: "full.json" }, spawn), full);
    assert.throws(() => productionDocumentation(repository, { exportPath: "scoped.json" }, spawn), /current full catalog export/);
    assert.throws(() => productionDocumentation(repository, { binary: "stale-exporter" }, spawn), /current full catalog export/);
    assert.deepEqual(productionDocumentation(repository, {}, spawn), full);
    assert.equal(calls.filter(call => call.command === "cargo").length, 4);
    for (const call of calls.filter(call => call.command === "cargo")) {
      assert.deepEqual(call.args, ["run", "--quiet", "-p", "runmat-builtins", "--bin", "export_builtin_documentation", "--"]);
    }
    assert.throws(() => productionDocumentation(repository, { mode: "transition" }, spawn), /strict canonical/);
  } finally {
    rmSync(repository, { recursive: true, force: true });
  }
});
