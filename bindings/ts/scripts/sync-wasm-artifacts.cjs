#!/usr/bin/env node

const fs = require("node:fs/promises");
const fsSync = require("node:fs");
const path = require("node:path");

async function copyTree(src, dest) {
  const srcStat = await fs.stat(src).catch(() => null);
  if (!srcStat) {
    console.warn(`[runmat] wasm artifacts missing at ${src}; did the wasm build succeed?`);
    return false;
  }
  if (srcStat.isFile()) {
    await fs.mkdir(path.dirname(dest), { recursive: true });
    await fs.copyFile(src, dest);
    return true;
  }
  await fs.rm(dest, { recursive: true, force: true });
  await fs.mkdir(dest, { recursive: true });
  const entries = await fs.readdir(src, { withFileTypes: true });
  for (const entry of entries) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      await copyTree(srcPath, destPath);
    } else if (entry.isFile()) {
      if (entry.name === "package.json" || entry.name === ".gitignore" || entry.name === ".npmignore") {
        continue;
      }
      await fs.copyFile(srcPath, destPath);
    }
  }
  return true;
}

async function main() {
  const repoRoot = path.join(__dirname, "..");
  const snapshotSrcCandidates = [
    path.join(repoRoot, "artifacts", "stdlib.snapshot"),
    path.join(repoRoot, "stdlib.snapshot")
  ];
  const snapshotSrc = snapshotSrcCandidates.find((candidate) => fsSync.existsSync(candidate));
  const copies = [
    {
      src: path.join(repoRoot, "pkg"),
      dest: path.join(repoRoot, "dist", "pkg"),
    },
    {
      src: path.join(repoRoot, "pkg-web"),
      dest: path.join(repoRoot, "dist", "pkg-web"),
    },
    {
      src: path.join(repoRoot, "lsp-web"),
      dest: path.join(repoRoot, "dist", "lsp"),
    },
    {
      src: snapshotSrc ?? snapshotSrcCandidates[0],
      dest: path.join(repoRoot, "dist", "runtime", "stdlib.snapshot"),
      required: true,
    },
  ];

  let copiedAny = false;
  let missingRequired = false;
  for (const { src, dest, required } of copies) {
    const ok = await copyTree(src, dest);
    copiedAny = copiedAny || ok;
    if (ok === false && required) {
      missingRequired = true;
    }
  }

  if (!copiedAny || missingRequired) {
    process.exitCode = 1;
  }
}

main().catch((err) => {
  console.error("[runmat] failed to copy wasm artifacts:", err);
  process.exitCode = 1;
});
