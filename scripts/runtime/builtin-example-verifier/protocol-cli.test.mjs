// @ts-check

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { ARTIFACT_SCHEMA } from "./artifacts.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
const entrypoint = join(root, "scripts/runtime/verify-builtin-examples.mjs");
const revision = "0123456789abcdef0123456789abcdef01234567";

test("protocol CLI builds native and browser manifests from exact package bytes", () => {
    const directory = mkdtempSync(join(tmpdir(), "runmat-example-artifact-cli-"));
    try {
        const nativeRoot = join(directory, "native-product");
        mkdirSync(nativeRoot);
        const binary = join(nativeRoot, "runmat");
        const dependency = join(nativeRoot, "runmat_runtime.dll");
        const nativeManifest = join(directory, "native.json");
        writeFileSync(binary, "native");
        writeFileSync(dependency, "runtime");
        run(["artifacts", "--product", "native-cli", "--profile", "embedded-aot", "--source-revision", revision, "--root", nativeRoot, "--entrypoint", "runmat-binary=runmat", "--output", nativeManifest], 0);
        const native = JSON.parse(readFileSync(nativeManifest, "utf8"));
        assert.equal(native.schema, ARTIFACT_SCHEMA);
        assert.equal(native.entrypoints[0].path, "runmat");
        assert.deepEqual(native.files.map((file) => file.path), ["runmat", "runmat_runtime.dll"]);

        const compatibilityRoot = join(directory, "native-file-compatibility");
        mkdirSync(compatibilityRoot);
        const compatibilityBinary = join(compatibilityRoot, "runmat");
        const compatibilityManifest = join(compatibilityRoot, "artifacts.json");
        writeFileSync(compatibilityBinary, "native");
        writeFileSync(join(compatibilityRoot, "runtime.dll"), "runtime");
        run(["artifacts", "--product", "native-cli", "--profile", "embedded-aot", "--source-revision", revision, "--file", `runmat-binary=${compatibilityBinary}`, "--output", compatibilityManifest], 0);
        const compatible = JSON.parse(readFileSync(compatibilityManifest, "utf8"));
        assert.equal(compatible.layout.root, ".");
        assert.deepEqual(compatible.files.map((file) => file.path), ["runmat", "runtime.dll"]);

        const javascript = join(directory, "runmat_wasm_web.js");
        const wasm = join(directory, "runmat_wasm_web_bg.wasm");
        const browserManifest = join(directory, "browser.json");
        writeFileSync(javascript, "js");
        writeFileSync(wasm, "wasm");
        run(["artifacts", "--product", "browser-wasm", "--profile", "web", "--source-revision", revision,
            "--file", `wasm-js=${javascript}`, "--file", `wasm-binary=${wasm}`, "--output", browserManifest], 0);
        const browser = JSON.parse(readFileSync(browserManifest, "utf8"));
        assert.deepEqual(browser.entrypoints.map((artifact) => artifact.role), ["wasm-binary", "wasm-js"]);
    } finally {
        rmSync(directory, { recursive: true, force: true });
    }
});

test("protocol CLI refuses to execute an available lane without its product artifact", () => {
    const directory = mkdtempSync(join(tmpdir(), "runmat-example-protocol-cli-"));
    try {
        const exported = join(directory, "export.json");
        const inventory = join(directory, "inventory.json");
        const plan = join(directory, "plan.json");
        const result = join(directory, "foreign.shard-result.json");
        writeFileSync(exported, JSON.stringify({ schema_version: 1, builtins: [{
            key: "python-example",
            authority: "catalog",
            category: "language/foreign",
            examples: [{ id: "python", input: "disp(1)", output: "1", compatibility: "RunMat", harness: "NativeForeignRuntime", verification: "Succeeds" }]
        }] }));
        run(["inventory", "--export", exported, "--source-revision", revision, "--source-state", "clean", "--output", inventory], 0);
        run(["plan", "--inventory", inventory, "--output", plan], 0);
        run(["run", "--inventory", inventory, "--plan", plan, "--lane", "native-foreign-runtime", "--shard-index", "0", "--result-out", result], 1);
        assert.throws(() => readFileSync(result, "utf8"), /ENOENT/u);
    } finally {
        rmSync(directory, { recursive: true, force: true });
    }
});

test("run accepts one artifact manifest as a file path, not an array", () => {
    const directory = mkdtempSync(join(tmpdir(), "runmat-example-run-manifest-cli-"));
    try {
        const exported = join(directory, "export.json");
        const inventory = join(directory, "inventory.json");
        const plan = join(directory, "plan.json");
        const manifest = join(directory, "invalid-manifest.json");
        writeFileSync(exported, JSON.stringify({ schema_version: 1, builtins: [{
            key: "fixture", authority: "catalog", category: "test",
            examples: [{ id: "fixture", input: "disp(1)", compatibility: "RunMat", harness: "Native", verification: "Succeeds" }]
        }] }));
        writeFileSync(manifest, "{}");
        run(["inventory", "--export", exported, "--source-revision", revision, "--source-state", "clean", "--output", inventory], 0);
        run(["plan", "--inventory", inventory, "--output", plan], 0);
        const result = run(["run", "--inventory", inventory, "--plan", plan, "--lane", "native-host", "--result-out", join(directory, "result.json"), "--artifact-manifest", manifest], 1);
        assert.match(result.stderr, /Unsupported builtin example artifact manifest schema/u);
        assert.doesNotMatch(result.stderr, /Received an instance of Array/u);
    } finally {
        rmSync(directory, { recursive: true, force: true });
    }
});

test("protocol CLI rejects repeated singleton arguments before file access", () => {
    for (const [command, flag] of [
        ["run", "--artifact-manifest"], ["run", "--inventory"],
        ["run", "--plan"], ["run", "--lane"], ["run", "--shard-index"],
        ["probe", "--artifact-manifest"], ["inventory", "--output"],
        ["plan", "--inventory"], ["matrix", "--plan"],
        ["reconcile", "--inventory"], ["artifacts", "--root"]
    ]) {
        const result = run([command, flag, "first", flag, "second"], 1);
        assert.match(result.stderr, new RegExp(`${flag} may only be specified once for ${command}`, "u"));
        assert.doesNotMatch(result.stderr, /ENOENT/u);
    }
});

test("reconcile retains repeatable manifests and inventory retains repeatable builtin scopes", () => {
    const reconcile = run(["reconcile", "--artifact-manifest", "first", "--artifact-manifest", "second", "--product-probe", "first", "--product-probe", "second", "--result", "first", "--result", "second"], 1);
    assert.match(reconcile.stderr, /--inventory is required/u);
    assert.doesNotMatch(reconcile.stderr, /may only be specified once/u);
    const inventory = run(["inventory", "--builtin", "first", "--builtin", "second"], 1);
    assert.match(inventory.stderr, /--output is required/u);
    assert.doesNotMatch(inventory.stderr, /may only be specified once/u);
});

function run(args, expectedStatus) {
    const completed = spawnSync(process.execPath, [entrypoint, ...args], { cwd: root, encoding: "utf8", env: { ...process.env, TMPDIR: "/private/tmp" } });
    assert.equal(completed.status, expectedStatus, `${completed.stdout}\n${completed.stderr}`);
    return completed;
}
