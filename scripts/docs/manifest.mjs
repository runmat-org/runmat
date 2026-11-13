import { createHash } from "node:crypto";
import { readFileSync, readdirSync, lstatSync, realpathSync, writeFileSync, openSync, readSync, closeSync } from "node:fs";
import { join, relative, resolve, isAbsolute } from "node:path";
export const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
/** File hashing owns one bounded buffer, independent of artifact size. */
export function fileIdentity(path) {
    const descriptor = openSync(path, "r");
    try {
        const digest = createHash("sha256");
        const buffer = Buffer.allocUnsafe(64 * 1024);
        let bytes = 0;
        for (;;) {
            const count = readSync(descriptor, buffer, 0, buffer.length, null);
            if (count === 0) break;
            digest.update(buffer.subarray(0, count));
            bytes += count;
        }
        return { sha256: digest.digest("hex"), bytes };
    } finally {
        closeSync(descriptor);
    }
}
export function safePath(path) {
    if (typeof path !== "string" || !path || isAbsolute(path) || path.includes("\\") || path.split("/").some(part => !part || part === ".." || part === "."))
        throw new Error(`Unsafe docs path: ${path}`);
    return path;
}
export function inputFile(root, path) {
    const actual = realpathSync(join(root, safePath(path)));
    if (!actual.startsWith(realpathSync(root) + "/"))
        throw new Error(`Source file escapes root: ${path}`);
    return actual;
}
export function inventory(root) {
    const files = [];
    function walk(dir) {
        for (const name of readdirSync(dir).sort()) {
            const absolute = join(dir, name);
            const path = relative(root, absolute).replaceAll("\\", "/");
            if (path === "manifest.json")
                continue;
            const stat = lstatSync(absolute);
            if (stat.isSymbolicLink())
                throw new Error("Docs artifacts cannot contain symlinks");
            if (stat.isDirectory())
                walk(absolute);
            else {
                files.push({ path, ...fileIdentity(inputFile(root, path)) });
            }
        }
    }
    walk(root);
    return files.sort((left, right) => left.path.localeCompare(right.path));
}
export function writeManifest(root, source) {
    const files = inventory(root);
    const content_hash = hash(JSON.stringify({ source, files }));
    const manifest = { schema_version: 1, source, files, content_hash };
    writeFileSync(join(root, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
    return manifest;
}
export function verifyArtifact(root) {
    const manifest = JSON.parse(readFileSync(join(root, "manifest.json"), "utf8"));
    if (manifest.schema_version !== 1 || !Array.isArray(manifest.files))
        throw new Error("Unsupported docs manifest");
    const actual = inventory(root);
    if (JSON.stringify(actual) !== JSON.stringify(manifest.files) || hash(JSON.stringify({ source: manifest.source, files: actual })) !== manifest.content_hash)
        throw new Error("Docs artifact inventory/hash mismatch");
    return manifest;
}
