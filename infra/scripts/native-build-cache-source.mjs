import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, lstatSync, readFileSync, utimesSync, writeFileSync } from "node:fs";
import { isAbsolute, join, resolve, win32 } from "node:path";
import { pathToFileURL } from "node:url";

const sha = bytes => createHash("sha256").update(bytes).digest("hex");
const validPath = name => typeof name === "string" && name.length > 0 && !isAbsolute(name)
  && !win32.isAbsolute(name) && !name.split(/[\\/]/).includes("..");

export function captureSource(root, source, paths) {
  const files = paths.map(name => {
    if (!validPath(name)) throw new Error("Invalid tracked source path");
    const filename = join(root, name), stat = lstatSync(filename);
    if (!stat.isFile() || stat.isSymbolicLink()) throw new Error(`Nonregular source: ${name}`);
    return { path: name, bytes: stat.size, mode: stat.mode & 0o777,
      sha256: sha(readFileSync(filename)), mtimeMs: stat.mtimeMs };
  });
  return { schema: 1, source, root: resolve(root), files };
}

export function restoreSource(root, source, snapshot, paths) {
  if (snapshot.schema !== 1 || !Array.isArray(snapshot.files)) throw new Error("Invalid source snapshot");
  // A different commit can change Git-derived build metadata even when its
  // Rust files match. Let Cargo re-evaluate that checkout normally.
  if (snapshot.source !== source) return { restored: 0, reason: "different-source" };
  if (snapshot.root !== resolve(root)) return { restored: 0, reason: "different-checkout" };
  const allowed = new Set(paths), seen = new Set();
  for (const row of snapshot.files) {
    if (!validPath(row.path) || seen.has(row.path) || !Number.isFinite(row.mtimeMs)
        || row.mtimeMs < 0 || !Number.isSafeInteger(row.bytes) || row.bytes < 0
        || !Number.isInteger(row.mode) || row.mode < 0 || row.mode > 0o777
        || !/^[a-f0-9]{64}$/.test(row.sha256)) throw new Error("Invalid source snapshot record");
    seen.add(row.path);
  }
  let restored = 0, changed = 0;
  for (const row of snapshot.files) {
    const filename = join(root, row.path);
    if (!allowed.has(row.path) || !existsSync(filename)) { changed++; continue; }
    const stat = lstatSync(filename);
    if (!stat.isFile() || stat.isSymbolicLink() || stat.size !== row.bytes
        || (stat.mode & 0o777) !== row.mode || sha(readFileSync(filename)) !== row.sha256) {
      changed++; continue;
    }
    utimesSync(filename, stat.atimeMs / 1000, row.mtimeMs / 1000);
    restored++;
  }
  return { restored, changed };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [mode, filename] = process.argv.slice(2), root = process.cwd();
  const source = execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim();
  const paths = execFileSync("git", ["ls-files", "-z"], { maxBuffer: 16 * 1024 * 1024 })
    .toString().split("\0").filter(Boolean);
  if (mode === "capture") {
    execFileSync("git", ["diff", "--quiet", "HEAD", "--"], { stdio: "inherit" });
    writeFileSync(filename, JSON.stringify(captureSource(root, source, paths)) + "\n");
  } else if (mode === "restore") {
    console.log(existsSync(filename)
      ? restoreSource(root, source, JSON.parse(readFileSync(filename)), paths)
      : { restored: 0, reason: "no-source-snapshot" });
  } else throw new Error("Expected capture or restore and a snapshot path");
}
