#!/usr/bin/env node

const INCLUDE_BUNDLER_ARTIFACTS = false;

const fs = require("node:fs/promises");
const path = require("node:path");
const { stageThirdPartyNotices, stageWasmPackageNotices } = require("../../../scripts/distribution/third-party-notices.cjs");
const { stageRuntimeLicenses } = require("../../../scripts/distribution/runtime-licenses.cjs");

async function artifactStat(source, optional = false) {
  let stat;
  try {
    stat = await fs.lstat(source);
  } catch (error) {
    if (optional && error.code === "ENOENT") return null;
    throw error;
  }
  if (stat.isSymbolicLink()) throw new Error(`Wasm artifact path must not be a symlink: ${source}`);
  if (!stat.isDirectory() && !stat.isFile()) throw new Error(`Wasm artifact path is not a regular file or directory: ${source}`);
  return stat;
}

async function validateSourceTree(source, optional = false) {
  const stat = await artifactStat(source, optional);
  if (stat === null) return false;
  if (!stat.isDirectory()) throw new Error(`Wasm artifact source must be a real directory: ${source}`);
  for (const name of await fs.readdir(source)) {
    const child = path.join(source, name);
    if ((await artifactStat(child)).isDirectory()) await validateSourceTree(child);
  }
  return true;
}

async function validateDestination(root, destination) {
  const rootStat = await artifactStat(root, true);
  if (rootStat !== null && !rootStat.isDirectory()) {
    throw new Error(`Wasm artifact destination must be a real directory: ${root}`);
  }
  const targetStat = await artifactStat(destination, true);
  if (targetStat !== null && !targetStat.isDirectory()) {
    throw new Error(`Generated artifact destination must be a real directory: ${destination}`);
  }
}

async function copyTree(source, destination) {
  if (!(await artifactStat(source)).isDirectory()) throw new Error(`Wasm artifact source must be a real directory: ${source}`);
  await fs.mkdir(destination);
  for (const name of await fs.readdir(source)) {
    const from = path.join(source, name);
    const to = path.join(destination, name);
    const stat = await artifactStat(from);
    if (stat.isDirectory()) {
      await copyTree(from, to);
    } else if (name !== "package.json" && name !== ".gitignore" && name !== ".npmignore") {
      await fs.copyFile(from, to);
    }
  }
}

async function syncWasmArtifacts(packageRoot) {
  const root = path.resolve(packageRoot);
  if (!(await artifactStat(root)).isDirectory()) throw new Error(`Package root must be a real directory: ${root}`);
  const dist = path.join(root, "dist");
  const copies = [
    ...(INCLUDE_BUNDLER_ARTIFACTS ? [{ source: path.join(root, "pkg"), destination: path.join(dist, "pkg"), notices: true, required: true }] : []),
    { source: path.join(root, "pkg-web"), destination: path.join(dist, "pkg-web"), notices: true, required: true },
    { source: path.join(root, "lsp-web"), destination: path.join(dist, "lsp"), notices: false, required: false },
  ];

  // Reject a missing runtime and unsafe trees before modifying dist or the
  // wasm-pack manifest. The optional LSP is absent in the runtime-only build.
  for (const copy of copies) {
    copy.present = await validateSourceTree(copy.source, true);
    if (copy.required && !copy.present) throw new Error(`Required wasm artifacts are missing: ${copy.source}`);
    await validateDestination(dist, copy.destination);
  }

  for (const copy of copies) {
    if (copy.notices) stageWasmPackageNotices(copy.source);
  }
  await fs.mkdir(dist, { recursive: true });
  for (const copy of copies) {
    // Retire only this exact generated output. A missing optional LSP cannot
    // leave an older dist/lsp to be packaged with the current runtime.
    await validateDestination(dist, copy.destination);
    await fs.rm(copy.destination, { recursive: true, force: true });
    if (copy.present) await copyTree(copy.source, copy.destination);
  }
  stageThirdPartyNotices(dist);
  stageRuntimeLicenses(root);
}

module.exports = { syncWasmArtifacts };

if (require.main === module) {
  syncWasmArtifacts(path.join(__dirname, "..")).catch((error) => {
    console.error("[runmat] failed to copy wasm artifacts:", error);
    process.exitCode = 1;
  });
}
