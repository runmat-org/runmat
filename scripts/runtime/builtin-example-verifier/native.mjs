// @ts-check

import { execFileSync, spawnSync } from "child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "fs";
import { join, resolve } from "path";
import { tmpdir } from "os";

export function runNativeCases(repository, cases, timeoutMs) {
    const binary = resolveNativeBinary(repository);
    const root = mkdtempSync(join(tmpdir(), "runmat-documentation-examples-"));
    try {
        return cases.map((testCase) => runCase(binary, root, testCase, timeoutMs));
    } finally {
        rmSync(root, { recursive: true, force: true });
    }
}

function resolveNativeBinary(repository) {
    const override = process.env.RUNMAT_EXAMPLE_NATIVE_BINARY;
    if (override) {
        const binary = resolve(override);
        if (!existsSync(binary)) throw new Error(`RUNMAT_EXAMPLE_NATIVE_BINARY does not exist: ${binary}`);
        return binary;
    }
    execFileSync("cargo", ["build", "--quiet", "-p", "runmat"], {
        cwd: repository,
        env: { ...process.env, CARGO_INCREMENTAL: process.env.CARGO_INCREMENTAL ?? "0" },
        stdio: "inherit"
    });
    const metadata = JSON.parse(execFileSync("cargo", ["metadata", "--no-deps", "--format-version", "1"], {
        cwd: repository,
        encoding: "utf8",
        maxBuffer: 16 * 1024 * 1024
    }));
    const executable = process.platform === "win32" ? "runmat.exe" : "runmat";
    const binary = join(metadata.target_directory, "debug", executable);
    if (!existsSync(binary)) throw new Error(`Native RunMat binary was not produced at ${binary}`);
    return binary;
}

function runCase(binary, root, testCase, timeoutMs) {
    const caseDirectory = join(root, String(testCase.id));
    const artifactsDirectory = join(caseDirectory, "artifacts");
    const manifest = join(artifactsDirectory, "run_manifest.json");
    mkdirSync(caseDirectory, { recursive: true });
    const source = join(caseDirectory, "example.m");
    const config = join(caseDirectory, "runmat.toml");
    writeFileSync(source, `${testCase.input.trimEnd()}\n`, "utf8");
    writeFileSync(
        config,
        `[runtime.language]\ncompat = "${testCase.compatibility.toLowerCase()}"\n\n[runtime.accelerate]\nenabled = false\n\n[runtime.telemetry]\nenabled = false\n`,
        "utf8"
    );
    const completed = spawnSync(
        binary,
        [
            "--color=never",
            "--no-jit",
            "--config",
            config,
            "--artifacts-dir",
            artifactsDirectory,
            "--artifacts-manifest",
            manifest,
            "--capture-figures=off",
            "run",
            source
        ],
        {
            cwd: caseDirectory,
            encoding: "utf8",
            env: { ...process.env, NO_COLOR: "1" },
            timeout: timeoutMs,
            maxBuffer: 16 * 1024 * 1024
        }
    );
    const stdoutText = completed.stdout ?? "";
    const stderrText = completed.stderr ?? "";
    if (completed.error) {
        return failure(testCase.id, stdoutText, `native lane: ${completed.error.message}`, "");
    }
    if (!existsSync(manifest)) {
        return failure(
            testCase.id,
            stdoutText,
            `native lane produced no structured execution manifest; ${stderrText.trim() || `process exited with status ${completed.status}`}`,
            ""
        );
    }
    let outcome;
    try {
        outcome = parseNativeArtifactManifest(readFileSync(manifest, "utf8"));
    } catch (error) {
        return failure(
            testCase.id,
            stdoutText,
            `native lane produced an invalid execution manifest: ${error instanceof Error ? error.message : String(error)}`,
            ""
        );
    }
    if (completed.status === 0 && outcome.success) {
        return { id: testCase.id, stdoutText, valueText: "", errorText: "", errorIdentifier: "" };
    }
    return failure(
        testCase.id,
        stdoutText,
        `native lane: ${stderrText.trim() || `process exited with status ${completed.status}`}`,
        outcome.errorIdentifier
    );
}

export function parseNativeArtifactManifest(source) {
    const manifest = JSON.parse(source);
    if (!manifest || typeof manifest !== "object" || manifest.schema_version !== "runmat.artifacts.v1") {
        throw new Error("unsupported or missing run artifact schema");
    }
    if (typeof manifest.success !== "boolean") {
        throw new Error("run artifact success must be a boolean");
    }
    if (manifest.error_identifier !== null && typeof manifest.error_identifier !== "string") {
        throw new Error("run artifact error_identifier must be a string or null");
    }
    return {
        success: manifest.success,
        errorIdentifier: manifest.error_identifier ?? ""
    };
}

function failure(id, stdoutText, errorText, errorIdentifier) {
    return { id, stdoutText, valueText: "", errorText, errorIdentifier };
}
