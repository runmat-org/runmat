// @ts-check

import { chmodSync, copyFileSync, lstatSync, mkdirSync, readdirSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { sha256 } from "./identity.mjs";
import { readFileSync } from "node:fs";

export function stageNativeProduct(fields) {
    const source = resolve(fields.binary);
    const destination = resolve(fields.destination);
    const sourceStat = lstatSync(source);
    if (!sourceStat.isFile() || sourceStat.isSymbolicLink()) throw new Error("Native product binary must be a regular file");
    ensureEmptyDirectory(destination);
    const executableName = fields.executableName ?? basename(source);
    validateLeafName(executableName, "native executable name");
    const stagedExecutable = join(destination, executableName);
    copyFileSync(source, stagedExecutable);
    chmodSync(stagedExecutable, sourceStat.mode & 0o777);

    const staged = new Map([[executableName.toLowerCase(), sha256(readFileSync(source))]]);
    for (const directory of fields.dependencyDirectories ?? []) {
        const root = resolve(directory);
        const stat = lstatSync(root);
        if (!stat.isDirectory() || stat.isSymbolicLink()) throw new Error(`Native dependency directory is not a real directory: ${directory}`);
        for (const entry of readdirSync(root, { withFileTypes: true })) {
            if (!entry.isFile() || entry.isSymbolicLink() || !entry.name.toLowerCase().endsWith(".dll")) continue;
            validateLeafName(entry.name, "native dependency name");
            const dependency = join(root, entry.name);
            const key = entry.name.toLowerCase();
            const identity = sha256(readFileSync(dependency));
            const prior = staged.get(key);
            if (prior && prior !== identity) throw new Error(`Conflicting native dependency bytes for ${entry.name}`);
            if (!prior) copyFileSync(dependency, join(destination, entry.name));
            staged.set(key, identity);
        }
    }
    return { destination, executableName, fileCount: staged.size };
}

function ensureEmptyDirectory(path) {
    mkdirSync(path, { recursive: true });
    const stat = lstatSync(path);
    if (!stat.isDirectory() || stat.isSymbolicLink()) throw new Error(`Native product destination must be a real directory: ${path}`);
    if (readdirSync(path).length !== 0) throw new Error(`Native product destination is not empty: ${path}`);
}

function validateLeafName(value, label) {
    if (typeof value !== "string" || !value || value !== basename(value) || value === "." || value === "..") throw new Error(`${label} must be a file name`);
}
