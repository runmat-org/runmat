// @ts-check

import { createHash } from "node:crypto";
import { existsSync, lstatSync, mkdirSync, writeFileSync } from "node:fs";
import { relative, resolve, sep } from "node:path";

/**
 * Materialize one validated catalog filesystem fixture below a newly-created
 * isolated root. The caller owns and removes the root's parent.
 *
 * @param {string} root
 * @param {import("../../../metadata/BuiltinMetadataSpecification").BuiltinFilesystemFixture} fixture
 */
export function materializeFilesystemFixture(root, fixture) {
    const absoluteRoot = resolve(root);
    if (existsSync(absoluteRoot)) throw new Error(`Fixture root already exists: ${absoluteRoot}`);
    mkdirSync(absoluteRoot);
    const records = [];
    for (const entry of fixture.entries) {
        if ("Directory" in entry) {
            const path = fixturePath(absoluteRoot, entry.Directory.relative_path);
            mkdirSync(path);
            records.push({ kind: "directory", relativePath: entry.Directory.relative_path });
            continue;
        }
        const relativePath = entry.File.relative_path;
        const path = fixturePath(absoluteRoot, relativePath);
        const content = "Utf8" in entry.File.content
            ? Buffer.from(entry.File.content.Utf8, "utf8")
            : Buffer.from(entry.File.content.Bytes);
        writeFileSync(path, content, { flag: "wx" });
        records.push({
            kind: "file",
            relativePath,
            byteLength: content.byteLength,
            sha256: createHash("sha256").update(content).digest("hex")
        });
    }
    return {
        fixtureId: fixture.id.local_name,
        root: absoluteRoot,
        entries: records
    };
}

export function fixturePath(root, relativePath) {
    if (!isNormalizedRelativePath(relativePath)) {
        throw new Error(`Invalid fixture relative path: ${String(relativePath)}`);
    }
    const absoluteRoot = resolve(root);
    const path = resolve(absoluteRoot, relativePath);
    if (!path.startsWith(`${absoluteRoot}${sep}`)) {
        throw new Error(`Fixture path escapes its isolated root: ${relativePath}`);
    }
    assertNoSymlinkAncestor(absoluteRoot, path);
    return path;
}

export function isNormalizedRelativePath(value) {
    return typeof value === "string"
        && value.length > 0
        && !value.startsWith("/")
        && !value.includes("\\")
        && !value.includes("\0")
        && value.split("/").every((part) => part.length > 0 && part !== "." && part !== "..");
}

function assertNoSymlinkAncestor(root, target) {
    const relativePath = relative(root, target);
    let current = root;
    for (const component of relativePath.split(sep).slice(0, -1)) {
        current = resolve(current, component);
        if (existsSync(current) && lstatSync(current).isSymbolicLink()) {
            throw new Error(`Fixture path crosses a symbolic link: ${current}`);
        }
    }
}
