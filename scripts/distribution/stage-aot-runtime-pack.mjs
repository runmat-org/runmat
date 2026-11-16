#!/usr/bin/env node
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

// Publication staging checks bytes and naming. The CLI's embedded manifest
// remains the authority for native compatibility and pack admission.
export function stageAotRuntimePack({ pack, manifest, output, version, target }) {
  assert.match(version, /^\d+\.\d+\.\d+(?:-[a-zA-Z0-9.-]+)?$/);
  assert.match(target, /^(?:x86_64|aarch64)-[a-z0-9]+(?:-[a-z0-9]+)+$/);
  assert(fs.lstatSync(manifest).isFile(), "Pack manifest must be a regular file");
  assert(fs.lstatSync(manifest).size <= 2 * 1024 * 1024, "Pack manifest is too large");
  const metadata = JSON.parse(fs.readFileSync(manifest, "utf8"));
  assert.equal(metadata.schema_version, 1);
  assert.equal(metadata.runtime.runmat_version, version);
  assert.equal(metadata.runtime.target_triple, target);
  assert.match(metadata.pack_digest, /^sha256:[a-f0-9]{64}$/);
  assert(Number.isSafeInteger(metadata.pack_bytes) && metadata.pack_bytes > 0
    && metadata.pack_bytes <= 1024 * 1024 * 1024 + 16 * 1024 * 1024, "Invalid pack byte bound");
  assert(fs.lstatSync(pack).isFile(), "Runtime pack must be a regular file");
  assert.equal(fs.statSync(pack).size, metadata.pack_bytes);
  fs.mkdirSync(output, { recursive: true });
  assert(fs.lstatSync(output).isDirectory(), "Asset output must be a real directory");
  const base = `runmat-aot-runtime-${version}-${target}`;
  const packOutput = path.join(output, `${base}.zip`);
  const manifestOutput = path.join(output, `${base}.json`);
  fs.copyFileSync(pack, packOutput, fs.constants.COPYFILE_EXCL);
  assert.equal(hashFile(packOutput), metadata.pack_digest, "Staged runtime pack digest mismatch");
  fs.copyFileSync(manifest, manifestOutput, fs.constants.COPYFILE_EXCL);
  assert.deepEqual(JSON.parse(fs.readFileSync(manifestOutput, "utf8")), metadata);
  return { pack: packOutput, manifest: manifestOutput };
}

function hashFile(filename) {
  const hash = createHash("sha256");
  const buffer = Buffer.alloc(64 * 1024);
  const file = fs.openSync(filename, "r");
  try {
    for (;;) {
      const count = fs.readSync(file, buffer, 0, buffer.length, null);
      if (count === 0) break;
      hash.update(buffer.subarray(0, count));
    }
    return `sha256:${hash.digest("hex")}`;
  } finally { fs.closeSync(file); }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  assert.equal(process.argv.length, 7, "Expected pack ZIP, manifest, output directory, version and target");
  const [pack, manifest, output, version, target] = process.argv.slice(2);
  console.log(JSON.stringify(stageAotRuntimePack({ pack, manifest, output, version, target })));
}
