const assert = require("node:assert/strict");
const { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } = require("node:fs");
const { tmpdir } = require("node:os");
const { join, resolve } = require("node:path");
const test = require("node:test");
const { execFileSync } = require("node:child_process");
const { NOTICE_FILE_NAME, thirdPartyNoticeBytes, stageThirdPartyNotices, stageWasmPackageNotices, verifyThirdPartyNotices } = require("./third-party-notices.cjs");
const { LICENSE_FILE_NAMES, stageRuntimeLicenses, verifyRuntimeLicenses } = require("./runtime-licenses.cjs");

test("distribution notices survive native and npm archives without changing package metadata", () => {
  const root = mkdtempSync(join(tmpdir(), "runmat-notices-"));
  try {
    const runtime = resolve(__dirname, "../../crates/runmat-runtime");
    const expected = Buffer.concat([
      readFileSync(join(runtime, "src/builtins/math/linalg/ops/linear_operator/LAPACK-LICENSE.txt")),
      Buffer.from("\n"),
      readFileSync(join(runtime, "src/builtins/math/linalg/ops/sparse_factor/managed_lu/FAER-LICENSE.txt")),
      Buffer.from("\n"),
      readFileSync(join(runtime, "licenses/UNICODE-LICENSE.txt")),
      Buffer.from("\n"),
      readFileSync(join(runtime, "licenses/CLARABEL-LICENSE.txt")),
      Buffer.from("\n"),
      readFileSync(join(runtime, "licenses/CLARABEL-DEPENDENCIES-NOTICES.txt")),
      Buffer.from("\n"),
      readFileSync(join(runtime, "licenses/SOBOL-NOTICES.txt")),
    ]);
    assert.deepEqual(thirdPartyNoticeBytes(), expected);
    assert.deepEqual(readFileSync(join(runtime, "licenses/CLARABEL-LICENSE.txt")),
      readFileSync(resolve(runtime, "../runmat-clarabel/LICENSE.md")), "vendored upstream license must survive native/npm delivery byte-for-byte");
    // The native/AOT owner deliberately uses only literal include_str! and
    // newline concatenation. Check that exact assembly, not a second notice
    // inventory that can pass while the actual Rust payload silently drifts.
    const native = readFileSync(join(runtime, "src/third_party_notices.rs"), "utf8");
    const assembly = native.match(/concat!\(([\s\S]*?)\)\s*\.as_bytes\(\)/)?.[1];
    assert.ok(assembly, "native notice owner must expose its literal assembly");
    assert.equal(assembly.replace(/include_str!\("[^"]+"\)|"\\n"|[\s,]/g, ""), "", "unrecognized native notice assembly");
    const nativeParts = [...assembly.matchAll(/include_str!\("([^"]+)"\)|"\\n"/g)].map(part =>
      part[1] ? readFileSync(resolve(runtime, "src", part[1])) : Buffer.from("\n"));
    assert.deepEqual(Buffer.concat(nativeParts), expected, "native/AOT and JS delivery must retain identical notice bytes");
    const product = join(root, "native");
    mkdirSync(product);
    stageThirdPartyNotices(product);
    stageThirdPartyNotices(product);
    stageRuntimeLicenses(product);
    stageRuntimeLicenses(product);
    assert.deepEqual(readFileSync(join(product, NOTICE_FILE_NAME)), thirdPartyNoticeBytes());
    const nativeArchive = join(root, "native.tar.gz");
    execFileSync("tar", ["-czf", nativeArchive, "-C", product, NOTICE_FILE_NAME, ...LICENSE_FILE_NAMES]);
    assert.deepEqual(execFileSync("tar", ["-xOf", nativeArchive, NOTICE_FILE_NAME]), thirdPartyNoticeBytes());
    for (const name of LICENSE_FILE_NAMES) {
      const expectedLicense = readFileSync(resolve(__dirname, "../..", name));
      assert.deepEqual(execFileSync("tar", ["-xOf", nativeArchive, name]), expectedLicense);
    }
    writeFileSync(join(product, "FREE-USE.md"), "unrelated bytes");
    assert.throws(() => verifyRuntimeLicenses(product), /Runtime license bytes differ/);
    assert.throws(() => stageRuntimeLicenses(product), /Runtime license bytes differ/);
    assert.equal(readFileSync(join(product, "FREE-USE.md"), "utf8"), "unrelated bytes");
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
