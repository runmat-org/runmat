import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtempSync, writeFileSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { stageAotRuntimePack } from "./stage-aot-runtime-pack.mjs";

test("publication names the exact version and target and checks the copied pack", () => {
  const directory = mkdtempSync(join(tmpdir(), "runmat-aot-stage-"));
  try {
    // This is publication metadata, not a fixture for native pack admission.
    const bytes = Buffer.from("pack publication fixture");
    const fields = { pack: join(directory, "pack.zip"), manifest: join(directory, "manifest.json"),
      output: join(directory, "assets"), version: "0.7.2", target: "aarch64-apple-darwin" };
    writeFileSync(fields.pack, bytes);
    const metadata = { schema_version: 1, runtime: { runmat_version: fields.version, target_triple: fields.target },
      pack_bytes: bytes.length, pack_digest: `sha256:${createHash("sha256").update(bytes).digest("hex")}` };
    writeFileSync(fields.manifest, JSON.stringify(metadata));
    const staged = stageAotRuntimePack(fields);
    assert.equal(staged.pack, join(fields.output, "runmat-aot-runtime-0.7.2-aarch64-apple-darwin.zip"));
    assert.deepEqual(readFileSync(staged.pack), bytes);
    assert.throws(() => stageAotRuntimePack(fields), /EEXIST/);
    assert.throws(() => stageAotRuntimePack({ ...fields, version: "0.7.1" }));
    writeFileSync(fields.pack, Buffer.alloc(bytes.length));
    assert.throws(() => stageAotRuntimePack({ ...fields, output: join(directory, "corrupt") }), /digest mismatch/);
  } finally { rmSync(directory, { recursive: true, force: true }); }
});
