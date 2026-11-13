import assert from "node:assert/strict";
import "fake-indexeddb/auto";
import { readFile } from "node:fs/promises";
import { initRunMat, createInMemoryFsProvider, createIndexedDbFsHandle } from "../dist/index.js";
const wasmModule = await readFile(new URL("../dist/pkg-web/runmat_wasm_web_bg.wasm", import.meta.url));
const encoder = new TextEncoder();
const original = 'import model supports from "./supports.fea"\nmodel assembly { include model.supports }\n';
const proposed = 'import model supports from "./shared.fea"\nmodel assembly { include model.supports }\n';
const request = { base: null, changes: [
  { type: "rename", from: "supports.fea", to: "shared.fea" },
  { type: "write", path: "main.fea", source: proposed },
] };
const open = filesystem => initRunMat({ enableGpu: false, enableJit: false, telemetryConsent: false, fsProvider: filesystem, wasmModule });
const text = async (provider, path) => new TextDecoder().decode(await provider.readFile(path));
for (const storage of ["memory", "indexeddb"]) {
  const dbName = `fea-wasm-${crypto.randomUUID()}`;
  let handle = storage === "indexeddb" ? await createIndexedDbFsHandle({ dbName }) : null;
  let provider = handle?.provider ?? createInMemoryFsProvider();
  await provider.writeFile("/main.fea", encoder.encode(original));
  await provider.writeFile("/supports.fea", encoder.encode("model supports {}\n"));
  let session = await open(provider);
  try {
    const staged = await session.stageFeaChanges(request);
    assert.equal(staged.publication_valid, true);
    assert.equal(await text(provider, "/main.fea"), original);
    const review = await session.reviewFeaChanges({ id: staged.id, change_index: 1, before_offset: 0, after_offset: 0 });
    assert.equal(review.before.text, original);
    assert.equal(review.after.text, proposed);
    await session.dispose();
    if (handle) { handle.close(); handle = await createIndexedDbFsHandle({ dbName }); provider = handle.provider; }
    session = await open(provider);
    const restored = await session.inspectFeaChanges(staged.id);
    assert.equal(restored.publication_valid, true);
    assert.deepEqual(restored.id, staged.id);
    assert.equal(await text(provider, "/main.fea"), original);
    await assert.rejects(async () => provider.readFile("/shared.fea"));
    const published = await session.publishFeaChanges(staged.id);
    assert.equal(published.type, "published");
    assert.equal(await text(provider, "/main.fea"), proposed);
    assert.equal(await text(provider, "/shared.fea"), "model supports {}\n");
    await provider.writeFile("/main.fea", encoder.encode("model teammate {}\n"));
    assert.deepEqual(await session.publishFeaChanges(staged.id), published);
    assert.equal(await text(provider, "/main.fea"), "model teammate {}\n");
    const bad = await session.stageFeaChanges({ base: null, changes: [{ type: "write", path: "main.fea", source: "invalid syntax" }] });
    assert.equal(bad.publication_valid, false);
    assert.equal((await session.inspectFeaChanges(bad.id)).publication_valid, false);
    const repaired = await session.stageFeaChanges({ base: bad.id, changes: [
      { type: "write", path: "main.fea", source: "model repaired {}\n" },
    ] });
    assert.equal((await session.inspectFeaChanges(repaired.id)).publication_valid, true);
    assert.equal(await text(provider, "/main.fea"), "model teammate {}\n");
    assert.equal((await session.publishFeaChanges(bad.id)).type, "invalid");
    assert.equal(await text(provider, "/main.fea"), "model teammate {}\n");
    const stale = await session.stageFeaChanges({ base: null, changes: [{ type: "write", path: "main.fea", source: "model next {}\n" }] });
    await provider.writeFile("/main.fea", encoder.encode("model newer {}\n"));
    await assert.rejects(() => session.inspectFeaChanges(stale.id));
    assert.equal((await session.publishFeaChanges(stale.id)).type, "conflict");
    assert.equal(await text(provider, "/main.fea"), "model newer {}\n");
    await provider.createDirAll("/project");
    const nested = await session.stageFeaChanges({ base: null, changes: [{ type: "write", path: "project/main.fea", source: "model nested {}\n" }] });
    assert.equal(nested.publication_valid, true);
    assert.equal((await session.publishFeaChanges(nested.id)).type, "published");
    assert.equal(await text(provider, "/project/main.fea"), "model nested {}\n");
    assert.equal(await text(provider, "/main.fea"), "model newer {}\n");
    for (const path of ["/outside.fea", "../outside.fea", "."]) {
      await assert.rejects(() => session.stageFeaChanges({ base: null, changes: [{ type: "write", path, source: "model outside {}\n" }] }));
    }
    console.log(`${storage}: actual optimized WASM source review, restart publication, retry, invalid, conflict and scoped path checks passed`);
  } finally { await session.dispose(); handle?.close(); }
}
