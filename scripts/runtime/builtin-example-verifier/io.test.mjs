import assert from "node:assert/strict";
import test from "node:test";
import { execFileSync } from "node:child_process";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { computeRunnerDigest, sourceIdentity } from "./io.mjs";

test("runner identity includes shared validation/export owners and staged notice inputs", () => {
    const root = mkdtempSync(join(tmpdir(), "runmat-runner-inputs-"));
    const files = [
        "scripts/runtime/verify-builtin-examples.mjs",
        "scripts/runtime/builtin-example-verifier/inventory.mjs",
        "scripts/metadata/BuiltinExampleFixtureSchema.mjs",
        "scripts/distribution/third-party-notices.cjs",
        "scripts/distribution/runtime-licenses.cjs",
        "LICENSE.md",
        "FREE-USE.md",
        "scripts/docs/production-documentation.mjs",
        "scripts/development/builtin-migration/documentation-export.mjs",
        "crates/runmat-runtime/src/builtins/math/linalg/ops/linear_operator/LAPACK-LICENSE.txt",
        "crates/runmat-runtime/src/builtins/math/linalg/ops/sparse_factor/managed_lu/FAER-LICENSE.txt",
        "crates/runmat-runtime/licenses/UNICODE-LICENSE.txt",
    ];
    try {
        for (const file of files) {
            mkdirSync(dirname(join(root, file)), { recursive: true });
            writeFileSync(join(root, file), file);
        }
        const original = computeRunnerDigest(root);
        writeFileSync(join(root, "scripts/runtime/builtin-example-verifier/fixture.test.mjs"), "test only");
        assert.equal(computeRunnerDigest(root), original);
        for (const file of files) {
            writeFileSync(join(root, file), `${file}\nchanged`);
            assert.notEqual(computeRunnerDigest(root), original, file);
            writeFileSync(join(root, file), file);
        }
        assert.equal(computeRunnerDigest(root), original);
    } finally {
        rmSync(root, { recursive: true, force: true });
    }
});

test("source identity detects tracked and nested untracked changes without ignoring source", () => {
    const root = mkdtempSync(join(tmpdir(), "runmat-source-identity-"));
    const git = (...args) => execFileSync("git", args, { cwd: root, stdio: "pipe" });
    try {
        git("init", "--quiet");
        writeFileSync(join(root, "source.m"), "x = 1;\n");
        writeFileSync(join(root, ".gitignore"), "ignored/\n");
        git("add", "source.m", ".gitignore");
        git("-c", "user.name=RunMat test", "-c", "user.email=fixture@example.invalid",
            "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "fixture");
        git("config", "status.showUntrackedFiles", "all");
        const clean = sourceIdentity(root);
        assert.equal(clean.sourceState, "clean");
        mkdirSync(join(root, "ignored"));
        writeFileSync(join(root, "ignored", "artifact"), "ignored");
        assert.deepEqual(sourceIdentity(root), clean);
        mkdirSync(join(root, "nested"));
        writeFileSync(join(root, "nested", "new.m"), "y = 2;\n");
        assert.deepEqual(sourceIdentity(root), { ...clean, sourceState: "dirty" });
        git("add", "nested/new.m");
        assert.equal(sourceIdentity(root).sourceState, "dirty");
        git("-c", "user.name=RunMat test", "-c", "user.email=fixture@example.invalid",
            "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "nested fixture");
        assert.equal(sourceIdentity(root).sourceState, "clean");
        writeFileSync(join(root, "source.m"), "x = 3;\n");
        assert.equal(sourceIdentity(root).sourceState, "dirty");
    } finally {
        rmSync(root, { recursive: true, force: true });
    }
});
