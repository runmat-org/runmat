const assert = require("node:assert/strict");
const { mkdtempSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync, existsSync } = require("node:fs");
const { tmpdir } = require("node:os");
const { join } = require("node:path");
const test = require("node:test");

const { syncWasmArtifacts } = require("./sync-wasm-artifacts.cjs");
const { NOTICE_FILE_NAME, thirdPartyNoticeBytes } = require("../../../scripts/distribution/third-party-notices.cjs");

test("wasm sync requires a real runtime package and retires only stale generated outputs", async () => {
  const root = mkdtempSync(join(tmpdir(), "runmat-wasm-sync-"));
  try {
    const dist = join(root, "dist");
    mkdirSync(join(dist, "pkg-web"), { recursive: true });
    mkdirSync(join(dist, "lsp"));
    writeFileSync(join(dist, "pkg-web", "old.wasm"), "old runtime");
    writeFileSync(join(dist, "lsp", "old.js"), "old lsp");
    writeFileSync(join(dist, "keep.txt"), "unrelated");
    await assert.rejects(syncWasmArtifacts(root), /Required wasm artifacts are missing/);
    assert.equal(readFileSync(join(dist, "pkg-web", "old.wasm"), "utf8"), "old runtime");
    assert.equal(readFileSync(join(dist, "lsp", "old.js"), "utf8"), "old lsp");

    const runtime = join(root, "pkg-web");
    mkdirSync(runtime);
    writeFileSync(join(runtime, "package.json"), JSON.stringify({ name: "fixture", version: "0.0.0", files: ["runtime.js"] }));
    writeFileSync(join(runtime, "runtime.js"), "new runtime");
    writeFileSync(join(runtime, "runtime_bg.wasm"), "wasm bytes");
    await syncWasmArtifacts(root);
    assert.equal(readFileSync(join(dist, "pkg-web", "runtime.js"), "utf8"), "new runtime");
    assert.equal(existsSync(join(dist, "pkg-web", "old.wasm")), false);
    assert.equal(existsSync(join(dist, "lsp")), false);
    assert.equal(readFileSync(join(dist, "keep.txt"), "utf8"), "unrelated");
    assert.deepEqual(readFileSync(join(runtime, NOTICE_FILE_NAME)), thirdPartyNoticeBytes());
    assert.deepEqual(readFileSync(join(dist, "pkg-web", NOTICE_FILE_NAME)), thirdPartyNoticeBytes());
    assert.deepEqual(readFileSync(join(dist, NOTICE_FILE_NAME)), thirdPartyNoticeBytes());
    assert.ok(JSON.parse(readFileSync(join(runtime, "package.json"), "utf8")).files.includes(NOTICE_FILE_NAME));

    const lsp = join(root, "lsp-web");
    mkdirSync(lsp);
    writeFileSync(join(lsp, "lsp.js"), "new lsp");
    await syncWasmArtifacts(root);
    assert.equal(readFileSync(join(dist, "lsp", "lsp.js"), "utf8"), "new lsp");

    const outside = join(root, "outside");
    mkdirSync(outside);
    const linkType = process.platform === "win32" ? "junction" : "dir";
    symlinkSync(outside, join(runtime, "linked"), linkType);
    await assert.rejects(syncWasmArtifacts(root), /must not be a symlink/);
    assert.equal(readFileSync(join(dist, "pkg-web", "runtime.js"), "utf8"), "new runtime");
    rmSync(join(runtime, "linked"));
    rmSync(lsp, { recursive: true });
    symlinkSync(runtime, lsp, linkType);
    await assert.rejects(syncWasmArtifacts(root), /must not be a symlink/);
    assert.equal(readFileSync(join(dist, "lsp", "lsp.js"), "utf8"), "new lsp");
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
