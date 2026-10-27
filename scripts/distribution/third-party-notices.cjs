// One byte-for-byte notice payload for native and WebAssembly distribution.
const { lstatSync, readFileSync, writeFileSync } = require("node:fs");
const { join, resolve } = require("node:path");

const NOTICE_FILE_NAME = "THIRD_PARTY_NOTICES.txt";
const NOTICE_SOURCES = [
  resolve(__dirname, "../../crates/runmat-runtime/src/builtins/math/linalg/ops/linear_operator/LAPACK-LICENSE.txt"),
  resolve(__dirname, "../../crates/runmat-runtime/src/builtins/math/linalg/ops/sparse_factor/managed_lu/FAER-LICENSE.txt"),
  resolve(__dirname, "../../crates/runmat-runtime/licenses/UNICODE-LICENSE.txt"),
  resolve(__dirname, "../../crates/runmat-runtime/licenses/CLARABEL-LICENSE.txt"),
  resolve(__dirname, "../../crates/runmat-runtime/licenses/CLARABEL-DEPENDENCIES-NOTICES.txt"),
  resolve(__dirname, "../../crates/runmat-runtime/licenses/SOBOL-NOTICES.txt"),
];

function thirdPartyNoticeBytes() {
  const parts = [];
  for (const source of NOTICE_SOURCES) {
    regularFile(source);
    if (parts.length) parts.push(Buffer.from("\n"));
    parts.push(readFileSync(source));
  }
  return Buffer.concat(parts);
}

function regularFile(path) {
  const stat = lstatSync(path);
  if (!stat.isFile() || stat.isSymbolicLink()) throw new Error(`Notice input must be a regular file: ${path}`);
}

function noticePath(root) {
  const directory = resolve(root);
  const stat = lstatSync(directory);
  if (!stat.isDirectory() || stat.isSymbolicLink()) throw new Error(`Notice destination must be a real directory: ${directory}`);
  return join(directory, NOTICE_FILE_NAME);
}

function verifyThirdPartyNotices(root) {
  const destination = noticePath(root);
  regularFile(destination);
  if (!readFileSync(destination).equals(thirdPartyNoticeBytes())) {
    throw new Error(`Third-party notice bytes differ from the runtime source: ${destination}`);
  }
}

function stageThirdPartyNotices(root) {
  const destination = noticePath(root);
  try {
    writeFileSync(destination, thirdPartyNoticeBytes(), { flag: "wx" });
  } catch (error) {
    if (error.code !== "EEXIST") throw error;
    // Repeated staging is idempotent, but never overwrites unrelated or stale bytes.
    verifyThirdPartyNotices(root);
  }
  verifyThirdPartyNotices(root);
}

function stageWasmPackageNotices(root) {
  const manifestPath = join(root, "package.json");
  regularFile(manifestPath);
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  if (!Array.isArray(manifest.files) || !manifest.files.every(file => typeof file === "string")) {
    throw new Error(`wasm-pack package must declare its published files: ${manifestPath}`);
  }
  stageThirdPartyNotices(root);
  if (!manifest.files.includes(NOTICE_FILE_NAME)) {
    manifest.files.push(NOTICE_FILE_NAME);
    writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  }
}

module.exports = { NOTICE_FILE_NAME, thirdPartyNoticeBytes, stageThirdPartyNotices, stageWasmPackageNotices, verifyThirdPartyNotices };

if (require.main === module) {
  const [action, root, ...extra] = process.argv.slice(2);
  if (action !== "verify" || !root || extra.length) throw new Error("Usage: third-party-notices.cjs verify <product-directory>");
  verifyThirdPartyNotices(root);
}
