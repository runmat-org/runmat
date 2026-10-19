// @ts-check

import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { delimiter, dirname, join } from "node:path";

import { runBoundedProcess } from "./process.mjs";
import { prepareNativeInterface } from "./native-library.mjs";

const MAX_PROCESS_OUTPUT_BYTES = 16 * 1024 * 1024;

export async function prepareForeignFixture(binary, workspaceDirectory, fixture, timeoutMs) {
    const [kind, preparation] = Object.entries(fixture.preparation)[0];
    if (kind === "Mex") return prepareMex(binary, workspaceDirectory, preparation, timeoutMs);
    if (kind === "NativeFfi") return prepareNativeInterface(binary, workspaceDirectory, preparation, preparation.isolation, timeoutMs);
    if (kind === "Java") return prepareJava(workspaceDirectory, preparation, timeoutMs);
    if (kind === "Python") return preparePython(workspaceDirectory, preparation, preparation.isolation, timeoutMs);
    return unavailable(`the native example verifier does not support foreign preparation ${kind}`);
}

async function prepareMex(binary, workspaceDirectory, preparation, timeoutMs) {
    let extension;
    try {
        extension = mexExtension(process.platform, process.arch);
    } catch (error) {
        return unavailable(error instanceof Error ? error.message : String(error));
    }
    const completed = await runBoundedProcess(binary, buildMexArguments(preparation), processOptions(workspaceDirectory, timeoutMs));
    const failed = processFailure("MEX preparation", completed);
    if (failed) return failed;
    const modulePath = `${preparation.module_name}.${extension}`;
    const manifestPath = `${modulePath}.runmat.json`;
    if (!existsSync(join(workspaceDirectory, modulePath)) || !existsSync(join(workspaceDirectory, manifestPath))) {
        return failure("MEX preparation did not produce the declared module and manifest");
    }
    return ready(`[mex-artifacts.${tomlKey(preparation.module_name)}]\nmanifest = ${tomlString(manifestPath)}\nmodule = ${tomlString(modulePath)}\n`);
}

async function prepareJava(workspaceDirectory, preparation, timeoutMs) {
    const root = ".runmat-example/java";
    const classes = join(workspaceDirectory, root, "classes");
    const artifactRelativePath = `${root}/${preparation.artifact_name}.jar`;
    mkdirSync(classes, { recursive: true });
    const compileArguments = buildJavaCompileArguments(workspaceDirectory, classes, preparation);
    const compiled = await runBoundedProcess("javac", compileArguments, processOptions(workspaceDirectory, timeoutMs));
    const compileFailure = processFailure("Java compilation", compiled, true);
    if (compileFailure) return compileFailure;
    for (const resource of preparation.resources) {
        const destination = join(classes, resource.artifact_path);
        mkdirSync(dirname(destination), { recursive: true });
        copyFileSync(join(workspaceDirectory, resource.source_path), destination);
    }
    const archived = await runBoundedProcess(
        "jar",
        ["--create", "--file", join(workspaceDirectory, artifactRelativePath), "-C", classes, "."],
        processOptions(workspaceDirectory, timeoutMs)
    );
    const archiveFailure = processFailure("Java archive construction", archived, true);
    if (archiveFailure) return archiveFailure;
    return ready(javaArtifactConfiguration(preparation.artifact_name, artifactRelativePath));
}

async function preparePython(workspaceDirectory, preparation, isolation, timeoutMs) {
    const requested = process.env.RUNMAT_EXAMPLE_PYTHON ?? "python3";
    const probe = await runBoundedProcess(
        requested,
        ["-c", "import json,sys,sysconfig;print(json.dumps({'executable':sys.executable,'implementation':sys.implementation.name,'major':sys.version_info[0],'minor':sys.version_info[1],'abi_tag':sysconfig.get_config_var('SOABI') or sys.implementation.cache_tag or '','platform_tag':sysconfig.get_platform()}))"],
        processOptions(workspaceDirectory, timeoutMs)
    );
    const probeFailure = processFailure("Python capability probe", probe, true);
    if (probeFailure) return probeFailure;
    let observed;
    try {
        observed = JSON.parse(probe.stdout.trim());
    } catch (error) {
        return failure(`Python capability probe returned invalid JSON: ${error instanceof Error ? error.message : String(error)}`);
    }
    const incompatibility = pythonProbeIncompatibility(preparation, observed);
    if (incompatibility) return unavailable(incompatibility);
    return ready(...pythonExecutionConfiguration(workspaceDirectory, preparation, isolation, observed, process.env.PYTHONPATH));
}

export function buildJavaCompileArguments(workspaceDirectory, classesDirectory, preparation) {
    const arguments_ = ["--release", String(preparation.release), "-d", classesDirectory];
    if (preparation.compile_classpath.length > 0) {
        arguments_.push("-classpath", preparation.compile_classpath.map((path) => join(workspaceDirectory, path)).join(delimiter));
    }
    arguments_.push(...preparation.source_files.map((path) => join(workspaceDirectory, path)));
    return arguments_;
}

export function javaArtifactConfiguration(artifactName, artifactRelativePath) {
    return `[java-artifacts.${tomlKey(artifactName)}]\npath = ${tomlString(artifactRelativePath)}\n`;
}

export function pythonExecutionConfiguration(workspaceDirectory, preparation, isolation, observed, inheritedPythonPath = "") {
    const mode = isolation === "OutOfProcess" ? "out_of_process" : "in_process";
    let config = `[runtime.foreign.python]\nexecutable = ${tomlString(observed.executable)}\nversion = ${tomlString(`${observed.major}.${observed.minor}`)}\nexecution_mode = ${tomlString(mode)}\n`;
    const environment = {};
    if ("SourceTree" in preparation.artifact) {
        const moduleRoot = join(workspaceDirectory, preparation.artifact.SourceTree.module_root);
        environment.PYTHONPATH = inheritedPythonPath ? `${moduleRoot}${delimiter}${inheritedPythonPath}` : moduleRoot;
    } else {
        const wheel = preparation.artifact.Wheel;
        config += `\n[python-artifacts.${tomlKey(wheel.artifact_name)}]\npath = ${tomlString(wheel.relative_path)}\nmodule = ${tomlString(wheel.module)}\n`;
    }
    return [config, environment];
}

export function pythonProbeIncompatibility(preparation, observed) {
    if (observed.implementation !== "cpython" || observed.major !== preparation.environment.major || observed.minor !== preparation.environment.minor || typeof observed.executable !== "string" || !observed.executable) {
        return `required CPython ${preparation.environment.major}.${preparation.environment.minor} is not the probed interpreter`;
    }
    if ("Wheel" in preparation.artifact && "Native" in preparation.artifact.Wheel.compatibility) {
        const required = preparation.artifact.Wheel.compatibility.Native;
        if (observed.abi_tag !== required.abi_tag || observed.platform_tag !== required.platform_tag) {
            return `required Python ABI/platform ${required.abi_tag}/${required.platform_tag} does not match the probed interpreter`;
        }
    }
    return "";
}

export function buildMexArguments(preparation) {
    const command = preparation.translation_units.some((unit) => unit.language === "Cuda") ? "mexcuda" : "mex";
    const api = {
        R2017b: "--R2017b",
        R2018a: "--R2018a",
        LargeArrayDims: "--largeArrayDims",
        CompatibleArrayDims: "--compatibleArrayDims"
    }[preparation.api];
    if (!api) throw new Error(`unsupported MEX API: ${preparation.api}`);
    return [
        "--color=never", command, api,
        "--output", preparation.module_name,
        "--out-dir", ".",
        ...preparation.include_directories.map((directory) => `-I${directory}`),
        ...preparation.definitions.map((definition) => `-D${definition.name}${definition.value === null ? "" : `=${definition.value}`}`),
        ...preparation.translation_units.map((unit) => unit.relative_path)
    ];
}

export function mexExtension(platform, architecture) {
    if (platform === "darwin" && architecture === "arm64") return "mexmaca64";
    if (platform === "darwin" && architecture === "x64") return "mexmaci64";
    if (platform === "linux" && architecture === "x64") return "mexa64";
    if (platform === "win32" && architecture === "x64") return "mexw64";
    throw new Error(`MEX fixture preparation is unavailable on ${platform}/${architecture}`);
}

function processOptions(cwd, timeout) {
    return { cwd, env: { ...process.env, NO_COLOR: "1" }, timeout, maxBuffer: MAX_PROCESS_OUTPUT_BYTES };
}

function processFailure(label, completed, missingIsUnavailable = false) {
    if (completed.error) {
        if (missingIsUnavailable && completed.error.code === "ENOENT") return unavailable(`${label} tool is unavailable`);
        return failure(`${label} failed: ${completed.error.message}`);
    }
    if (completed.status !== 0) return failure(`${label} failed: ${completed.stderr.trim() || `process exited with status ${completed.status}`}`);
    return null;
}

function ready(config = "", environment = {}) {
    return { available: true, error: "", config, environment };
}

function failure(error) {
    return { available: true, error, config: "", environment: {} };
}

function unavailable(reason) {
    return { available: false, reason, config: "", environment: {} };
}

function tomlKey(value) {
    return tomlString(value);
}

function tomlString(value) {
    return JSON.stringify(value);
}
