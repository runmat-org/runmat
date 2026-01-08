#!/usr/bin/env node

const fs = require("node:fs/promises");
const path = require("node:path");

async function copyTree(src, dest) {
  const srcStat = await fs.stat(src).catch(() => null);
  if (!srcStat) {
    console.warn(`[runmat] wasm artifacts missing at ${src}; did the wasm build succeed?`);
    return false;
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
  ];

  let copiedAny = false;
  for (const { src, dest } of copies) {
    const ok = await copyTree(src, dest);
    copiedAny = copiedAny || ok;
  }

  if (!copiedAny) {
    process.exitCode = 1;
  }
}

main().catch((err) => {
  console.error("[runmat] failed to copy wasm artifacts:", err);
  process.exitCode = 1;
});

