// @ts-check

import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { compareUtf8, sha256 } from "./identity.mjs";

export function repositoryRoot() {
    let current = dirname(fileURLToPath(import.meta.url));
    while (dirname(current) !== current) {
        if (existsSync(join(current, "rust-toolchain.toml"))) return current;
        current = dirname(current);
    }
    throw new Error("Could not locate the RunMat repository root");
}

export function readJson(path) {
    return JSON.parse(readFileSync(resolve(path), "utf8"));
}

export function writeJson(path, value) {
    const target = resolve(path);
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

export function sourceIdentity(root = repositoryRoot()) {
    const sourceRevision = execFileSync("git", ["rev-parse", "HEAD"], { cwd: root, encoding: "utf8" }).trim();
    const status = execFileSync("git", ["status", "--porcelain", "--untracked-files=all"], { cwd: root, encoding: "utf8" });
    return { sourceRevision, sourceState: status.trim() ? "dirty" : "clean" };
}

export function computeRunnerDigest(root = repositoryRoot()) {
    const entrypoint = join(root, "scripts", "runtime", "verify-builtin-examples.mjs");
    const moduleRoot = join(root, "scripts", "runtime", "builtin-example-verifier");
    const files = [entrypoint, ...walk(moduleRoot)]
        .filter((path) => path.endsWith(".mjs") && !path.endsWith(".test.mjs"))
        .sort(compareUtf8);
    return sha256(files.map((path) => ({ path: relative(root, path).replaceAll("\\", "/"), digest: sha256(readFileSync(path)) })));
}

function walk(root) {
    return readdirSync(root, { withFileTypes: true }).flatMap((entry) => {
        const path = join(root, entry.name);
        return entry.isDirectory() ? walk(path) : entry.isFile() ? [path] : [];
    });
}
