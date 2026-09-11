// @ts-check

import { execFileSync } from "child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "fs";
import { join, resolve } from "path";
import { tmpdir } from "os";

import { materializeFilesystemFixture } from "./adapters/filesystem.mjs";
import { prepareForeignFixture } from "./adapters/foreign.mjs";
import { runInteraction } from "./adapters/interaction.mjs";
import { withLoopbackFixture } from "./adapters/loopback.mjs";
import { runBoundedProcess } from "./adapters/process.mjs";

export async function runNativeCases(repository, cases, timeoutMs) {
    const binary = resolveNativeBinary(repository);
    const root = mkdtempSync(join(tmpdir(), "runmat-documentation-examples-"));
    try {
        const results = [];
        for (const testCase of cases) results.push(await runCase(binary, root, testCase, timeoutMs));
        return results;
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
    execFileSync("cargo", ["build", "--quiet", "-p", "runmat", "--no-default-features"], {
        cwd: repository,
        env: {
            ...process.env,
            CARGO_INCREMENTAL: process.env.CARGO_INCREMENTAL ?? "0",
            CARGO_PROFILE_DEV_DEBUG: process.env.CARGO_PROFILE_DEV_DEBUG ?? "0"
        },
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

async function runCase(binary, root, testCase, timeoutMs) {
    const caseDirectory = join(root, String(testCase.id));
    const workspaceDirectory = join(caseDirectory, "workspace");
    const artifactsDirectory = join(caseDirectory, "artifacts");
    mkdirSync(caseDirectory, { recursive: true });
    if (testCase.fixture && typeof testCase.fixture === "object" && "DesktopHostOnly" in testCase.fixture) {
        return unavailable(
            testCase.id,
            "the native CLI product cannot provide the declared RunMat Desktop host scenario"
        );
    }
    try {
        prepareWorkspace(workspaceDirectory, testCase.fixture);
    } catch (error) {
        return failure(
            testCase.id,
            "",
            `native lane fixture preparation failed: ${error instanceof Error ? error.message : String(error)}`,
            ""
        );
    }
    let foreignConfig = "";
    let foreignEnvironment = {};
    if (testCase.fixture && typeof testCase.fixture === "object" && "ForeignAdapter" in testCase.fixture) {
        const prepared = await prepareForeignFixture(binary, workspaceDirectory, testCase.fixture.ForeignAdapter, timeoutMs);
        if (!prepared.available) return unavailable(testCase.id, prepared.reason);
        if (prepared.error) return failure(testCase.id, "", `native foreign fixture: ${prepared.error}`, "");
        foreignConfig = prepared.config;
        foreignEnvironment = prepared.environment;
    }
    const config = join(workspaceDirectory, "runmat.toml");
    writeFileSync(
        config,
        `[runtime.language]\ncompat = "${testCase.compatibility.toLowerCase()}"\n\n[runtime.accelerate]\nenabled = false\n\n[runtime.telemetry]\nenabled = false\n\n${foreignConfig}`,
        "utf8"
    );
    if (testCase.fixture && typeof testCase.fixture === "object" && "Loopback" in testCase.fixture) {
        try {
            const outcome = await withLoopbackFixture(
                testCase.input,
                testCase.fixture.Loopback,
                ({ program }) => executeProgram(binary, caseDirectory, workspaceDirectory, artifactsDirectory, config, program, testCase, timeoutMs, null, foreignEnvironment)
            );
            return outcome.value;
        } catch (error) {
            return failure(testCase.id, "", `native loopback fixture failed: ${error instanceof Error ? error.message : String(error)}`, "");
        }
    }
    const transcript = testCase.fixture && typeof testCase.fixture === "object" && "CliInteraction" in testCase.fixture
        ? testCase.fixture.CliInteraction.transcript
        : null;
    return executeProgram(binary, caseDirectory, workspaceDirectory, artifactsDirectory, config, testCase.input, testCase, timeoutMs, transcript, foreignEnvironment);
}

async function executeProgram(binary, caseDirectory, workspaceDirectory, artifactsDirectory, config, program, testCase, timeoutMs, transcript = null, environmentOverrides = {}) {
    const source = join(workspaceDirectory, "example.m");
    const manifest = join(artifactsDirectory, "run_manifest.json");
    writeFileSync(source, `${program.trimEnd()}\n`, "utf8");
    if (testCase.requirements.engine === "Aot") {
        return executeAotProgram(binary, caseDirectory, workspaceDirectory, config, source, testCase, timeoutMs, environmentOverrides);
    }
    const args = [
        "--color=never",
        ...nativeEngineArguments(testCase.requirements.engine),
        "--config",
        config,
        "--artifacts-dir",
        artifactsDirectory,
        "--artifacts-manifest",
        manifest,
        "--capture-figures=off",
        "run",
        source
    ];
    const options = {
        cwd: workspaceDirectory,
        env: { ...process.env, ...environmentOverrides, NO_COLOR: "1" },
        timeout: timeoutMs,
        maxBuffer: 16 * 1024 * 1024
    };
    const completed = transcript
        ? await runInteraction(binary, args, options, transcript)
        : await runBoundedProcess(binary, args, options);
    const stdoutText = completed.stdout;
    const stderrText = completed.stderr;
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
    if (testCase.requirements.engine === "Interpreter" && outcome.usedJit) {
        return failure(testCase.id, stdoutText, "native interpreter example executed through the JIT", "");
    }
    if (testCase.requirements.engine === "Jit" && !outcome.usedJit) {
        return failure(testCase.id, stdoutText, "native JIT example did not execute through the JIT", "");
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

export function nativeEngineArguments(engine) {
    switch (engine) {
        case "Default": return [];
        case "Interpreter": return ["--no-jit"];
        case "Jit": return [];
        case "Aot": throw new Error("AOT execution uses the native compile adapter");
        default: throw new Error(`unsupported native execution engine requirement: ${engine}`);
    }
}

async function executeAotProgram(binary, caseDirectory, workspaceDirectory, config, source, testCase, timeoutMs, environmentOverrides) {
    const executable = join(caseDirectory, process.platform === "win32" ? "example-aot.exe" : "example-aot");
    const environment = { ...process.env, ...environmentOverrides, NO_COLOR: "1" };
    const compile = await runBoundedProcess(binary, [
        "--color=never",
        "--config",
        config,
        "compile",
        source,
        "--output",
        executable,
        "--force"
    ], {
        cwd: workspaceDirectory,
        env: environment,
        timeout: timeoutMs,
        maxBuffer: 16 * 1024 * 1024
    });
    if (compile.error || compile.status !== 0) {
        const detail = compile.error?.message ?? (compile.stderr.trim() || `compiler exited with status ${compile.status}`);
        return failure(testCase.id, compile.stdout, `native AOT compile: ${detail}`, "");
    }
    if (!existsSync(executable)) {
        return failure(testCase.id, compile.stdout, "native AOT compile produced no executable", "");
    }
    const completed = await runBoundedProcess(executable, [], {
        cwd: workspaceDirectory,
        env: environment,
        timeout: timeoutMs,
        maxBuffer: 16 * 1024 * 1024
    });
    const stdoutText = completed.stdout;
    if (completed.error) return failure(testCase.id, stdoutText, `native AOT execution: ${completed.error.message}`, "");
    if (completed.status === 0) return { id: testCase.id, stdoutText, valueText: "", errorText: "", errorIdentifier: "" };
    return failure(
        testCase.id,
        stdoutText,
        `native AOT execution: ${completed.stderr.trim() || `process exited with status ${completed.status}`}`,
        ""
    );
}

function prepareWorkspace(workspaceDirectory, fixture) {
    if (fixture === undefined
        || fixture === "None"
        || (fixture && typeof fixture === "object" && ("Loopback" in fixture || "CliInteraction" in fixture))) {
        mkdirSync(workspaceDirectory);
        return;
    }
    if (fixture && typeof fixture === "object" && "ForeignAdapter" in fixture) {
        materializeFilesystemFixture(workspaceDirectory, fixture.ForeignAdapter.files);
        return;
    }
    if (!fixture || typeof fixture !== "object" || !("Filesystem" in fixture)) {
        throw new Error("selected native adapter cannot materialize the declared fixture");
    }
    materializeFilesystemFixture(workspaceDirectory, fixture.Filesystem);
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
    if (typeof manifest.used_jit !== "boolean") {
        throw new Error("run artifact used_jit must be a boolean");
    }
    return {
        success: manifest.success,
        errorIdentifier: manifest.error_identifier ?? "",
        usedJit: manifest.used_jit
    };
}

function failure(id, stdoutText, errorText, errorIdentifier) {
    return { id, stdoutText, valueText: "", errorText, errorIdentifier };
}

function unavailable(id, reason) {
    return { id, stdoutText: "", valueText: "", errorText: reason, errorIdentifier: "", availability: "unavailable" };
}
