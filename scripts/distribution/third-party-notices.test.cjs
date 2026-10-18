const assert = require("node:assert/strict");
const { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } = require("node:fs");
const { tmpdir } = require("node:os");
const { join } = require("node:path");
const test = require("node:test");
const { execFileSync } = require("node:child_process");
const { NOTICE_FILE_NAME, thirdPartyNoticeBytes, stageThirdPartyNotices, stageWasmPackageNotices, verifyThirdPartyNotices } = require("./third-party-notices.cjs");

test("distribution notices survive native and npm archives without changing package metadata", () => {
  const root = mkdtempSync(join(tmpdir(), "runmat-notices-"));
  try {
    const product = join(root, "native");
    mkdirSync(product);
    stageThirdPartyNotices(product);
    stageThirdPartyNotices(product);
    assert.deepEqual(readFileSync(join(product, NOTICE_FILE_NAME)), thirdPartyNoticeBytes());
    const nativeArchive = join(root, "native.tar.gz");
    execFileSync("tar", ["-czf", nativeArchive, "-C", product, NOTICE_FILE_NAME]);
    assert.deepEqual(execFileSync("tar", ["-xOf", nativeArchive, NOTICE_FILE_NAME]), thirdPartyNoticeBytes());
    writeFileSync(join(product, NOTICE_FILE_NAME), "stale");
    assert.throws(() => verifyThirdPartyNotices(product), /notice bytes differ/);
    assert.throws(() => stageThirdPartyNotices(product), /notice bytes differ/);
    assert.equal(readFileSync(join(product, NOTICE_FILE_NAME), "utf8"), "stale");

    const wasm = join(root, "wasm");
    mkdirSync(wasm);
    const manifestPath = join(wasm, "package.json");
    const original = { name: "runmat-notice-fixture", version: "0.0.0", files: ["module.js"], sideEffects: false };
    writeFileSync(manifestPath, JSON.stringify(original));
    writeFileSync(join(wasm, "module.js"), "export {};\n");
    writeFileSync(join(wasm, ".gitignore"), "*\n");
    stageWasmPackageNotices(wasm);
    const staged = readFileSync(manifestPath);
    assert.deepEqual(JSON.parse(staged), { ...original, files: [...original.files, NOTICE_FILE_NAME] });
    assert.deepEqual(readFileSync(join(wasm, NOTICE_FILE_NAME)), thirdPartyNoticeBytes());
    stageWasmPackageNotices(wasm);
    assert.deepEqual(readFileSync(manifestPath), staged);
    const options = {
      cwd: wasm,
      env: { ...process.env, npm_config_cache: join(root, "npm-cache"), npm_config_update_notifier: "false" },
      encoding: "utf8",
    };
    const packed = process.platform === "win32"
      ? execFileSync("cmd.exe", ["/d", "/s", "/c", "npm pack --ignore-scripts --offline --json"], options)
      : execFileSync("npm", ["pack", "--ignore-scripts", "--offline", "--json"], options);
    const [archive] = JSON.parse(packed);
    assert.deepEqual(execFileSync("tar", ["-xOf", join(wasm, archive.filename), `package/${NOTICE_FILE_NAME}`]), thirdPartyNoticeBytes());
    writeFileSync(manifestPath, JSON.stringify({ ...original, files: "invalid" }));
    assert.throws(() => stageWasmPackageNotices(wasm), /must declare its published files/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
