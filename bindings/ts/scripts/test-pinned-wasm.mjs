// Integration smoke test against the actual optimized WASM and public bindings.
import assert from "node:assert/strict";
import "fake-indexeddb/auto";
import { readFile } from "node:fs/promises";
import { initRunMat, createInMemoryFsProvider, createIndexedDbPackageCache, createBrowserProjectSession } from "../dist/index.js";

const filesystem = createInMemoryFsProvider();
await filesystem.writeFile("/main.m", new TextEncoder().encode('error("live source was executed")'));
const captured = new TextEncoder().encode(`captured_value = 7;
f = fopen('/output.txt', 'w');
fprintf(f, 'first');
fclose(f);
f = fopen('/output.txt', 'a');
fprintf(f, ' second');
fclose(f);`);
const heads = new Map();
const versions = new Map();
let sequence = 0;
filesystem.publishOutput = async (path, bytes, expected) => {
  assert.equal(heads.get(path) ?? null, expected, "conditional output version changed");
  const version = `output-${++sequence}`;
  await filesystem.writeFile(path, bytes);
  heads.set(path, version);
  versions.set(version, bytes.slice());
  return version;
};
filesystem.removeOutput = async (path, expected) => {
  assert.equal(heads.get(path), expected, "conditional output version changed");
  await filesystem.removeFile(path);
  heads.delete(path);
};
const reads = [];
const capturedPackageBytes = new Map();
let unavailable = false;
filesystem.readVersion = async (path, versionId, size, snapshotId) => {
  reads.push({ path, versionId, size, snapshotId });
  if (capturedPackageBytes.has(versionId)) {
    const bytes = capturedPackageBytes.get(versionId);
    assert.equal(bytes.byteLength, size);
    return bytes;
  }
  if (versions.has(versionId)) {
    assert.equal(snapshotId, "");
    const bytes = versions.get(versionId);
    assert.equal(bytes.byteLength, size);
    return bytes;
  }
  if (unavailable) throw new Error("Captured version is unavailable");
  return captured;
};
const session = await initRunMat({
  applicationSettings: "isolatedMemory",
  enableGpu: false, enableJit: false, telemetryConsent: false,
  fsProvider: filesystem,
  wasmModule: await readFile(new URL("../dist/pkg-web/runmat_wasm_web_bg.wasm", import.meta.url)),
});
const request = {
  source: { kind: "path", path: "/main.m" },
  inputManifest: { snapshotId: "capture", entries: [{ path: "/main.m", content: { kind: "file", versionId: "old", size: captured.length } }] },
};
try {
  const result = await session.executeRequest(request);
  assert.equal(result.error, undefined);
  assert.deepEqual(result.workspace.values.find(value => value.name === "captured_value")?.preview?.values, [7]);
  assert.ok(reads.length > 0);
  assert.ok(reads.filter(read => read.path === "/main.m").every(read => read.path === "/main.m" && read.versionId === "old" && read.snapshotId === "capture" && read.size === captured.length));
  assert.equal(new TextDecoder().decode(await filesystem.readFile("/output.txt")), "first second");
  heads.set("/output.txt", "peer-version");
  await filesystem.writeFile("/output.txt", new TextEncoder().encode("teammate"));
  const conflicted = await session.executeRequest(request);
  assert.ok(conflicted.error, "A peer output version must reject the next run write");
  assert.equal(new TextDecoder().decode(await filesystem.readFile("/output.txt")), "teammate");
  const runDataset = (text) => session.executeRequest({ ...request, source: { kind: "text", name: "/dataset.m", text } });
  const dataset = await runDataset(`
    schema = struct("arrays", struct("samples", struct("dtype", "f64", "shape", [2 2], "chunk", [1 1])));
    ds = data.create("/result.data", schema);
    samples = Dataset.array(ds, "samples");
    DataArray.write(samples, [1 2; 3 4]);
  `);
  assert.equal(dataset.error, undefined, JSON.stringify(dataset.error));
  const manifestPath = "/result.data/manifest.json";
  const manifest = JSON.parse(new TextDecoder().decode(await filesystem.readFile(manifestPath)));
  const indexPath = "/result.data/" + manifest.arrays.samples.chunk_index_path;
  const indexBytes = await filesystem.readFile(indexPath);
  const index = JSON.parse(new TextDecoder().decode(indexBytes));
  const oldObjects = await Promise.all(index.chunks.map(async chunk => ["/result.data/" + chunk.data_path, await filesystem.readFile("/result.data/" + chunk.data_path)]));
  const peerManifest = new TextEncoder().encode(JSON.stringify({ ...manifest, txn_sequence: manifest.txn_sequence + 1, attrs: { owner: "teammate" } }));
  heads.set(manifestPath, "peer-dataset-version");
  await filesystem.writeFile(manifestPath, peerManifest);
  const failedDataset = await runDataset("DataArray.write(samples, [9 9; 9 9]);");
  assert.ok(failedDataset.error, "Dataset publication must reject a changed manifest head");
  assert.deepEqual(await filesystem.readFile(manifestPath), peerManifest);
  assert.deepEqual(await filesystem.readFile(indexPath), indexBytes);
  for (const [path, bytes] of oldObjects) assert.deepEqual(await filesystem.readFile(path), bytes);
  const settings = await session.executeRequest({ source: { kind: "text", name: "/settings.m", text: "setpref('PinnedSmoke', 'Value', 7);" } });
  assert.equal(settings.error, undefined, JSON.stringify(settings.error));
  session.resetSession();
  const reset = await session.executeRequest({ source: { kind: "text", name: "/settings-reset.m", text: "assert(getpref('PinnedSmoke', 'Value') == 7);" } });
  assert.equal(reset.error, undefined, JSON.stringify(reset.error));
  const packageEntries = [];
  for (const [path, text] of [
    ["/runmat.toml", "[package]\nname='application'\nversion='1.0.0'\n[sources]\nroots=['src']\n[dependencies]\nhelper={path='vendor/helper'}\n"],
    ["/src/main.m", "package_answer = helper();"],
    ["/vendor/helper/runmat.toml", "[package]\nname='helper'\nversion='1.0.0'\n[sources]\nroots=['src']\n"],
    ["/vendor/helper/src/helper.m", "function y = helper(); y = 42; end"],
  ]) {
    const bytes = new TextEncoder().encode(text);
    const versionId = `captured-package-${packageEntries.length}`;
    capturedPackageBytes.set(versionId, bytes);
    packageEntries.push({ path, content: { kind: "file", versionId, size: bytes.length } });
  }
  const cache = await createIndexedDbPackageCache({ dbName: "captured-run-smoke" });
  const project = await createBrowserProjectSession({
    session, filesystem, packageCache: cache.provider,
    gitGateway: { baseUrl: "https://unused.test" },
  });
  try {
    const inputManifest = { snapshotId: "package-capture", entries: packageEntries };
    await project.resolve({
      manifestPath: "/runmat.toml", inputManifest,
      options: {
        target: "wasm32-unknown-unknown", default_server_origin: "https://unused.test",
        default_registry_index: "https://unused.test", groups: ["runtime"],
        root_features: [], host_capabilities: [], source_intent: "execute", source_policy: {},
      },
    });
    const packageRun = await session.executeRequest({ inputManifest, source: { kind: "path", path: "/src/main.m" } });
    assert.equal(packageRun.error, undefined, JSON.stringify(packageRun.error));
    assert.equal(typeof packageRun.dependencyLock, "string", "captured package provenance must survive the public API");
    assert.ok(packageRun.dependencyLock.length > 0);
    assert.deepEqual(packageRun.workspace.values.find(value => value.name === "package_answer")?.preview?.values, [42]);
    await assert.rejects(async () => filesystem.metadata("/runmat.lock"), /Path not found/);
  } finally {
    await project.dispose();
    cache.close();
  }
  unavailable = true;
  const rejected = await session.executeRequest(request);
  assert.ok(rejected.error, "Unavailable captured inputs must reject execution");
  console.log("PASS: compiled WASM pins inputs, flushes versioned output handles, preserves dataset objects after publication conflicts, resolves captured package sources, and rejects stale writes and unavailable versions");
} finally { session.dispose(); }
