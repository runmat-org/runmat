// @ts-check

import { existsSync, mkdirSync } from "node:fs";
import { basename, join } from "node:path";

import { runBoundedProcess } from "./process.mjs";

const MAX_PROCESS_OUTPUT_BYTES = 16 * 1024 * 1024;

export async function prepareNativeInterface(binary, workspaceDirectory, preparation, isolation, timeoutMs) {
    const target = nativeLibraryTarget(process.platform, process.arch, preparation.library_name);
    if (!target.available) return unavailable(target.reason);
    const plan = nativeLibraryBuildPlan(preparation, target, process.env);
    if (!plan.available) return unavailable(plan.reason);
    const buildRoot = join(workspaceDirectory, ".runmat-example", "native");
    mkdirSync(buildRoot, { recursive: true });
    const deadline = Date.now() + timeoutMs;
    const frontend = process.env.RUNMAT_EXAMPLE_CLANG ?? process.env.CLANG ?? "clang";
    const probedFrontend = await runBoundedProcess(
        frontend,
        ["--version"],
        processOptions(workspaceDirectory, remaining(deadline))
    );
    const frontendFailure = processFailure("native interface compiler frontend", probedFrontend, true);
    if (frontendFailure) return frontendFailure;
    for (const command of plan.commands) {
        const completed = await runBoundedProcess(
            command.executable,
            command.arguments,
            processOptions(workspaceDirectory, remaining(deadline))
        );
        const failed = processFailure(command.label, completed, true);
        if (failed) return failed;
    }
    if (!existsSync(join(workspaceDirectory, target.relativePath))) {
        return failure("native library preparation did not produce the declared shared library");
    }
    const manifestPath = `${target.relativePath}.runmat.json`;
    const prepared = await runBoundedProcess(
        binary,
        buildNativeInterfaceArguments(preparation, target.relativePath, manifestPath, frontend),
        processOptions(workspaceDirectory, remaining(deadline))
    );
    const preparationFailure = processFailure("native interface preparation", prepared);
    if (preparationFailure) return preparationFailure;
    if (!existsSync(join(workspaceDirectory, manifestPath))) {
        return failure("native interface preparation did not produce the declared manifest");
    }
    return ready(nativeInterfaceConfiguration(
        preparation.interface.interface_name,
        target.relativePath,
        manifestPath,
        isolation
    ));
}

export function nativeInterfaceConfiguration(interfaceName, libraryPath, manifestPath, isolation) {
    const isolationMode = isolation === "OutOfProcess" ? "process" : "in_process";
    return `[runtime.foreign.native]\nisolation = ${tomlString(isolationMode)}\n\n` +
        `[native-interfaces.${tomlKey(interfaceName)}]\n` +
        `manifest = ${tomlString(manifestPath)}\n` +
        `library = ${tomlString(libraryPath)}\n`;
}

export function nativeLibraryTarget(platform, architecture, libraryName) {
    if (platform === "darwin" && ["arm64", "x64"].includes(architecture)) {
        return { available: true, relativePath: `.runmat-example/native/lib${libraryName}.dylib`, objectSuffix: "o", linkerFlag: "-dynamiclib", toolchain: "gnu" };
    }
    if (platform === "linux" && architecture === "x64") {
        return { available: true, relativePath: `.runmat-example/native/lib${libraryName}.so`, objectSuffix: "o", linkerFlag: "-shared", toolchain: "gnu" };
    }
    if (platform === "win32" && architecture === "x64") {
        return { available: true, relativePath: `.runmat-example/native/${libraryName}.dll`, objectSuffix: "obj", linkerFlag: "/DLL", toolchain: "msvc" };
    }
    return { available: false, reason: `native FFI fixture preparation is unavailable on ${platform}/${architecture}` };
}

export function nativeLibraryBuildPlan(preparation, target, environment = {}) {
    if (!target.available) return target;
    const languages = new Set(preparation.translation_units.map((unit) => unit.language));
    if (target.toolchain === "msvc" && (languages.has("Fortran") || languages.has("Cuda"))) {
        return { available: false, reason: "Windows native FFI fixtures currently require C or C++ translation units" };
    }
    const objects = preparation.translation_units.map((_, index) => `.runmat-example/native/unit-${String(index).padStart(4, "0")}.${target.objectSuffix}`);
    const commands = preparation.translation_units.map((unit, index) => compileCommand(
        unit,
        objects[index],
        preparation.include_directories,
        preparation.definitions,
        target,
        environment
    ));
    commands.push(linkCommand(languages, objects, target, environment));
    return { available: true, commands };
}

export function buildNativeInterfaceArguments(preparation, libraryPath, manifestPath, frontend = "clang") {
    const interfacePreparation = preparation.interface;
    return [
        "--color=never", "native-interface", "prepare",
        "--library", libraryPath,
        "--library-name", preparation.library_name,
        "--header", interfacePreparation.primary_header,
        "--interface-name", interfacePreparation.interface_name,
        "--frontend", frontend,
        "--output", manifestPath,
        ...interfacePreparation.additional_headers.flatMap((path) => ["--add-header", path]),
        ...interfacePreparation.include_directories.flatMap((path) => ["-I", path]),
        ...interfacePreparation.definitions.flatMap((definition) => ["-D", definitionArgument(definition)])
    ];
}

function compileCommand(unit, object, includes, definitions, target, environment) {
    const executable = compilerFor(unit.language, target.toolchain, environment);
    const includeArguments = includes.flatMap((directory) => target.toolchain === "msvc" ? [`/I${directory}`] : ["-I", directory]);
    const definitionArguments = definitions.map((definition) => `${target.toolchain === "msvc" ? "/D" : "-D"}${definitionArgument(definition)}`);
    if (target.toolchain === "msvc") {
        return {
            label: `${unit.language} compilation`, executable,
            arguments: ["/nologo", unit.language === "C" ? "/TC" : "/TP", "/c", unit.relative_path, `/Fo${object}`, ...includeArguments, ...definitionArguments]
        };
    }
    const positionIndependent = unit.language === "Cuda" ? ["-Xcompiler", "-fPIC"] : ["-fPIC"];
    return {
        label: `${unit.language} compilation`, executable,
        arguments: [...sourceLanguageArguments(unit.language), ...positionIndependent, "-c", unit.relative_path, "-o", object, ...includeArguments, ...definitionArguments]
    };
}

function linkCommand(languages, objects, target, environment) {
    if (target.toolchain === "msvc") {
        return {
            label: "native library link", executable: environment.LINK ?? "link.exe",
            arguments: ["/NOLOGO", target.linkerFlag, `/OUT:${target.relativePath}`, ...objects]
        };
    }
    const language = languages.has("Cuda") ? "Cuda" : languages.has("Cxx") ? "Cxx" : languages.has("Fortran") ? "Fortran" : "C";
    const runtimeLibraries = language === "Cxx" && languages.has("Fortran") ? ["-lgfortran"] : [];
    return {
        label: "native library link", executable: compilerFor(language, target.toolchain, environment),
        arguments: [target.linkerFlag, ...objects, ...runtimeLibraries, "-o", target.relativePath]
    };
}

function compilerFor(language, toolchain, environment) {
    if (language === "C") return environment.CC ?? (toolchain === "msvc" ? "cl.exe" : "cc");
    if (language === "Cxx") return environment.CXX ?? (toolchain === "msvc" ? "cl.exe" : "c++");
    if (language === "Fortran") return environment.FC ?? environment.F77 ?? "gfortran";
    if (language === "Cuda") return environment.NVCC ?? "nvcc";
    throw new Error(`unsupported native source language: ${language}`);
}

function sourceLanguageArguments(language) {
    if (language === "C") return ["-x", "c"];
    if (language === "Cxx") return ["-x", "c++"];
    if (language === "Fortran") return ["-x", "f95"];
    if (language === "Cuda") return ["-x", "cu"];
    throw new Error(`unsupported native source language: ${language}`);
}

function definitionArgument(definition) {
    return `${definition.name}${definition.value === null ? "" : `=${definition.value}`}`;
}

function remaining(deadline) {
    return Math.max(1, deadline - Date.now());
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

function ready(config) {
    return { available: true, error: "", config, environment: {} };
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
