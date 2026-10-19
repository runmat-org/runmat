import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const temporaryDirectories = new Set();

export function createTemporaryDirectory(prefix) {
  const directory = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), prefix)));
  temporaryDirectories.add(directory);
  return directory;
}

export function cleanupTemporaryDirectories() {
  for (const directory of temporaryDirectories) {
    fs.rmSync(directory, { recursive: true, force: true });
  }
  temporaryDirectories.clear();
}
