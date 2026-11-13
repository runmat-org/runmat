// @ts-check

import { chmodSync, copyFileSync, lstatSync, mkdirSync, readdirSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { sha256 } from "./identity.mjs";
import { readFileSync } from "node:fs";
import { NOTICE_FILE_NAME, stageThirdPartyNotices } from "../../distribution/third-party-notices.cjs";
import { LICENSE_FILE_NAMES, stageRuntimeLicenses } from "../../distribution/runtime-licenses.cjs";

export function stageNativeProduct(fields) {
    const source = resolve(fields.binary);
    const destination = resolve(fields.destination);
    const sourceStat = lstatSync(source);
    if (!sourceStat.isFile() || sourceStat.isSymbolicLink()) throw new Error("Native product binary must be a regular file");
    ensureEmptyDirectory(destination);
    const executableName = fields.executableName ?? basename(source);
    validateLeafName(executableName, "native executable name");
    if ([NOTICE_FILE_NAME, ...LICENSE_FILE_NAMES].some(name => executableName.toLowerCase() === name.toLowerCase())) throw new Error("Native executable name is reserved for distribution assets");
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
    // Platform redistributables retain their publisher notices separately
    // from the canonical notice payload embedded by the Rust runtime.
    for (const directory of fields.dependencyNoticeDirectories ?? []) {
        const root = resolve(directory);
        const stat = lstatSync(root);
        if (!stat.isDirectory() || stat.isSymbolicLink()) throw new Error(`Native dependency notice directory is not real: ${directory}`);
        const noticeRoot = join(destination, "THIRD_PARTY_NOTICES.d");
        mkdirSync(noticeRoot, { recursive: true });
        for (const entry of readdirSync(root, { withFileTypes: true })) {
            if (!entry.isFile() || entry.isSymbolicLink()) throw new Error(`Native dependency notice must be a regular file: ${entry.name}`);
            validateLeafName(entry.name, "native dependency notice name");
            const source = join(root, entry.name);
            const identity = sha256(readFileSync(source));
            const key = `third_party_notices.d/${entry.name.toLowerCase()}`;
            const prior = staged.get(key);
            if (prior && prior !== identity) throw new Error(`Conflicting native dependency notice bytes for ${entry.name}`);
            if (!prior) copyFileSync(source, join(noticeRoot, entry.name));
            staged.set(key, identity);
        }
    }
    stageThirdPartyNotices(destination);
    stageRuntimeLicenses(destination);
    return { destination, executableName, fileCount: staged.size + 1 + LICENSE_FILE_NAMES.length };
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
