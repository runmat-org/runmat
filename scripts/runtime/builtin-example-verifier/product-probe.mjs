// @ts-check

import { spawnSync } from "node:child_process";
import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { resolveArtifactPath, validateArtifactManifest } from "./artifacts.mjs";
import { digestObject, sha256 } from "./identity.mjs";
import { buildDesktopHostRequest } from "./desktop-host-protocol.mjs";
import { invokeDesktopHost } from "./desktop-host.mjs";
import { DESKTOP_HOST_PROTOCOL_PROBE, NATIVE_EMBEDDED_AOT_PROBE } from "./product-contracts.mjs";
import { digest, enumValue, exactKeys, gitRevision, integer } from "./schema.mjs";

export const PRODUCT_PROBE_SCHEMA = "runmat.builtin-examples.product-probe.v2";
export { DESKTOP_HOST_PROTOCOL_PROBE, NATIVE_EMBEDDED_AOT_PROBE } from "./product-contracts.mjs";

const NATIVE_VECTOR_SOURCE = "disp(2 + 3);\n";
const NATIVE_VECTOR_DIGEST = sha256(NATIVE_VECTOR_SOURCE);
const DESKTOP_VECTOR_SOURCE = "input('Probe: ', 's');\ndisp(2 + 3);\n";
const DESKTOP_VECTOR_DIGEST = sha256(DESKTOP_VECTOR_SOURCE);
const NATIVE_SUCCESS_STDOUT_DIGESTS = Object.freeze([sha256("5"), sha256("5\n"), sha256("5\r\n")]);

export function buildProductProbe(fields) {
    const probe = {
        schema: PRODUCT_PROBE_SCHEMA,
        sourceRevision: fields.sourceRevision,
        product: fields.product,
        artifactProfile: fields.artifactProfile,
        artifactManifestDigest: fields.artifactManifestDigest,
        probeKind: fields.probeKind,
        vectorDigest: fields.vectorDigest,
        result: structuredClone(fields.result),
        stdoutDigest: fields.stdoutDigest,
        productProbeDigest: ""
    };
    probe.productProbeDigest = digestObject(probe, ["productProbeDigest"]);
    return probe;
}

export function runNativeEmbeddedAotProbe(fields) {
    const { artifactManifest, artifactManifestPath } = fields;
    validateArtifactManifest(artifactManifest, {
        sourceRevision: fields.sourceRevision,
        verifyFiles: true,
        manifestPath: artifactManifestPath
    });
    if (artifactManifest.product !== "native-cli" || artifactManifest.artifactProfile !== "embedded-aot") {
        throw new Error(`${NATIVE_EMBEDDED_AOT_PROBE} requires native-cli/embedded-aot`);
    }
    const runmat = resolveArtifactPath(artifactManifest, artifactManifestPath, "runmat-binary");
    const directory = mkdtempSync(join(tmpdir(), "runmat-product-probe-"));
    let compileExitCode = null;
    let executeExitCode = null;
    let stdout = Buffer.alloc(0);
    try {
        const source = join(directory, "embedded-aot-probe.m");
        const output = join(directory, process.platform === "win32" ? "embedded-aot-probe.exe" : "embedded-aot-probe");
        writeFileSync(source, NATIVE_VECTOR_SOURCE, "utf8");
        const environment = { ...process.env, NO_COLOR: "1" };
        const compilation = spawnSync(runmat, ["compile", source, "-o", output], {
            cwd: directory,
            env: environment,
            encoding: null,
            maxBuffer: 16 * 1024 * 1024,
            timeout: fields.timeoutMs ?? 120_000
        });
        compileExitCode = processExitCode(compilation.status);
        if (compileExitCode === 0) {
            if (process.platform !== "win32") chmodSync(output, 0o700);
            const execution = spawnSync(output, [], {
                cwd: directory,
                env: environment,
                encoding: null,
                maxBuffer: 16 * 1024 * 1024,
                timeout: fields.timeoutMs ?? 120_000
            });
            executeExitCode = processExitCode(execution.status);
            stdout = Buffer.isBuffer(execution.stdout) ? execution.stdout : Buffer.alloc(0);
        }
        // Re-read the complete staged product after execution. This makes a probe
        // inadmissible if the entrypoint or any adjacent dependency changed while
        // it was compiling or running.
        validateArtifactManifest(artifactManifest, { verifyFiles: true, manifestPath: artifactManifestPath });
        const outputMatches = normalizeProbeOutput(stdout) === "5";
        const failureKind = compileExitCode !== 0
            ? "compile-process"
            : executeExitCode !== 0
                ? "execute-process"
                : outputMatches ? "none" : "stdout-mismatch";
        const status = failureKind === "none" ? "passed" : "failed";
        return buildProductProbe({
            sourceRevision: artifactManifest.sourceRevision,
            product: artifactManifest.product,
            artifactProfile: artifactManifest.artifactProfile,
            artifactManifestDigest: artifactManifest.artifactManifestDigest,
            probeKind: NATIVE_EMBEDDED_AOT_PROBE,
            vectorDigest: NATIVE_VECTOR_DIGEST,
            result: {
                kind: "compile-and-execute",
                status,
                failureKind: failureKind === "compile-process" ? "prepare-process" : failureKind,
                prepareExitCode: compileExitCode,
                executeExitCode
            },
            stdoutDigest: sha256(stdout)
        });
    } finally {
        rmSync(directory, { recursive: true, force: true });
    }
}

export async function runDesktopHostProtocolProbe(fields) {
    const { artifactManifest, artifactManifestPath } = fields;
    validateArtifactManifest(artifactManifest, {
        sourceRevision: fields.sourceRevision,
        producerRevision: fields.producerRevision,
        verifyFiles: true,
        manifestPath: artifactManifestPath
    });
    if (artifactManifest.product !== "desktop-native" || artifactManifest.artifactProfile !== "desktop-host") {
        throw new Error(`${DESKTOP_HOST_PROTOCOL_PROBE} requires desktop-native/desktop-host`);
    }
    const binary = resolveArtifactPath(artifactManifest, artifactManifestPath, "runmat-desktop-binary");
    const directory = mkdtempSync(join(tmpdir(), "runmat-desktop-product-probe-"));
    try {
        const workspaceRoot = join(directory, "workspace");
        const requestPath = join(directory, "request.json");
        const resultPath = join(directory, "result.json");
        const fixture = {
            DesktopHostOnly: {
                id: { local_name: "product-probe" },
                entries: [],
                interactions: [{
                    LineInput: {
                        prompt: "Probe: ",
                        echo: true,
                        outcome: { Line: "ready" }
                    }
                }]
            }
        };
        const request = buildDesktopHostRequest({
            exampleKey: "product-probe",
            program: DESKTOP_VECTOR_SOURCE,
            compatibility: "RunMat",
            workspaceRoot,
            fixture
        });
        mkdirSync(workspaceRoot);
        writeFileSync(requestPath, `${JSON.stringify(request, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
        const completed = await invokeDesktopHost(binary, request, requestPath, resultPath, workspaceRoot, fields.timeoutMs ?? 120_000);
        validateArtifactManifest(artifactManifest, {
            sourceRevision: fields.sourceRevision,
            producerRevision: fields.producerRevision,
            verifyFiles: true,
            manifestPath: artifactManifestPath
        });
        const stdout = Buffer.from(completed.result?.stdoutText ?? "", "utf8");
        const outputMatches = completed.result?.interactions.matched === true && normalizeProbeOutput(stdout) === "5";
        const failureKind = completed.error || completed.status !== 0
            ? "execute-process"
            : completed.protocolError || completed.result === null
                ? "protocol"
                : completed.result.errorText !== "" || completed.result.errorIdentifier !== ""
                    ? "execution-result"
                    : outputMatches ? "none" : "stdout-mismatch";
        return buildProductProbe({
            sourceRevision: artifactManifest.sourceRevision,
            product: artifactManifest.product,
            artifactProfile: artifactManifest.artifactProfile,
            artifactManifestDigest: artifactManifest.artifactManifestDigest,
            probeKind: DESKTOP_HOST_PROTOCOL_PROBE,
            vectorDigest: DESKTOP_VECTOR_DIGEST,
            result: {
                kind: "single-process",
                status: failureKind === "none" ? "passed" : "failed",
                failureKind,
                prepareExitCode: null,
                executeExitCode: processExitCode(completed.status)
            },
            stdoutDigest: sha256(stdout)
        });
    } finally {
        rmSync(directory, { recursive: true, force: true });
    }
}

export function validateProductProbe(probe, options = {}) {
    if (!probe || probe.schema !== PRODUCT_PROBE_SCHEMA) throw new Error("Unsupported builtin example product probe schema");
    exactKeys(probe, ["schema", "sourceRevision", "product", "artifactProfile", "artifactManifestDigest", "probeKind", "vectorDigest", "result", "stdoutDigest", "productProbeDigest"], "product probe");
    gitRevision(probe.sourceRevision, "product probe source revision");
    enumValue(probe.product, ["native-cli", "browser-wasm", "desktop-native"], "product probe product");
    digest(probe.artifactManifestDigest, "product probe artifact manifest digest");
    enumValue(probe.probeKind, [NATIVE_EMBEDDED_AOT_PROBE, DESKTOP_HOST_PROTOCOL_PROBE], "product probe kind");
    digest(probe.vectorDigest, "product probe vector digest");
    digest(probe.stdoutDigest, "product probe stdout digest");
    digest(probe.productProbeDigest, "product probe digest");
    if (digestObject(probe, ["productProbeDigest"]) !== probe.productProbeDigest) throw new Error("Product probe digest mismatch");
    exactKeys(probe.result, ["kind", "status", "failureKind", "prepareExitCode", "executeExitCode"], "product probe result");
    enumValue(probe.result.kind, ["compile-and-execute", "single-process"], "product probe execution kind");
    enumValue(probe.result.status, ["passed", "failed"], "product probe result status");
    for (const [field, label] of [["prepareExitCode", "prepare exit code"], ["executeExitCode", "execute exit code"]]) {
        const value = probe.result[field];
        if (value !== null) integer(value, `product probe ${label}`);
    }
    if (probe.probeKind === NATIVE_EMBEDDED_AOT_PROBE) {
        if (probe.product !== "native-cli" || probe.artifactProfile !== "embedded-aot") throw new Error("Native embedded-AOT probe has the wrong product profile");
        if (probe.vectorDigest !== NATIVE_VECTOR_DIGEST) throw new Error("Native embedded-AOT probe vector digest mismatch");
        if ((probe.result.status === "passed") !== (probe.result.failureKind === "none")) throw new Error("Product probe status and failure kind disagree");
        if (probe.result.kind !== "compile-and-execute") throw new Error("Native embedded-AOT probe has the wrong execution kind");
        enumValue(probe.result.failureKind, ["none", "prepare-process", "execute-process", "stdout-mismatch"], "native product probe failure kind");
        validateNativeProcessEvidence(probe.result);
        if (probe.result.status === "passed" && (probe.result.prepareExitCode !== 0 || probe.result.executeExitCode !== 0)) throw new Error("Passed native embedded-AOT probe has unsuccessful process evidence");
        if (probe.result.status === "passed" && !NATIVE_SUCCESS_STDOUT_DIGESTS.includes(probe.stdoutDigest)) throw new Error("Passed native embedded-AOT probe has unexpected stdout evidence");
    }
    if (probe.probeKind === DESKTOP_HOST_PROTOCOL_PROBE) {
        if (probe.product !== "desktop-native" || probe.artifactProfile !== "desktop-host") throw new Error("Desktop host protocol probe has the wrong product profile");
        if (probe.result.kind !== "single-process" || probe.result.prepareExitCode !== null) throw new Error("Desktop host protocol probe has the wrong execution kind");
        if (probe.vectorDigest !== DESKTOP_VECTOR_DIGEST) throw new Error("Desktop host protocol probe vector digest mismatch");
        enumValue(probe.result.failureKind, ["none", "execute-process", "protocol", "execution-result", "stdout-mismatch"], "Desktop product probe failure kind");
        validateDesktopProcessEvidence(probe.result);
        if ((probe.result.status === "passed") !== (probe.result.failureKind === "none")) throw new Error("Product probe status and failure kind disagree");
        if (probe.result.status === "passed" && probe.result.executeExitCode !== 0) throw new Error("Passed Desktop host protocol probe has unsuccessful process evidence");
        if (probe.result.status === "passed" && !NATIVE_SUCCESS_STDOUT_DIGESTS.includes(probe.stdoutDigest)) throw new Error("Passed Desktop host protocol probe has unexpected stdout evidence");
    }
    if (options.sourceRevision && probe.sourceRevision !== options.sourceRevision) throw new Error("Product probe source revision does not match the execution plan");
    if (options.artifactManifest) {
        const artifact = options.artifactManifest;
        validateArtifactManifest(artifact, {
            sourceRevision: probe.sourceRevision,
            verifyFiles: Boolean(options.verifyFiles),
            manifestPath: options.artifactManifestPath
        });
        if (probe.product !== artifact.product || probe.artifactProfile !== artifact.artifactProfile) throw new Error("Product probe product profile does not match its artifact manifest");
        if (probe.artifactManifestDigest !== artifact.artifactManifestDigest) throw new Error("Product probe artifact manifest digest mismatch");
    }
    return probe;
}

function validateNativeProcessEvidence(result) {
    if (result.failureKind === "prepare-process") {
        if (result.prepareExitCode === 0 || result.executeExitCode !== null) throw new Error("Native prepare-process evidence is inconsistent");
        return;
    }
    if (result.prepareExitCode !== 0) throw new Error("Native product probe did not record a successful preparation");
    if (result.failureKind === "execute-process") {
        if (result.executeExitCode === 0) throw new Error("Native execute-process evidence is inconsistent");
        return;
    }
    if (result.executeExitCode !== 0) throw new Error("Native product probe did not record a successful execution");
}

function validateDesktopProcessEvidence(result) {
    if (result.failureKind === "execute-process") {
        if (result.executeExitCode === 0) throw new Error("Desktop execute-process evidence is inconsistent");
        return;
    }
    if (result.executeExitCode !== 0) throw new Error("Desktop product probe did not record a successful execution");
}

function processExitCode(status) {
    return Number.isSafeInteger(status) && status >= 0 ? status : null;
}

function normalizeProbeOutput(output) {
    return output.toString("utf8").replace(/\r\n/g, "\n").trim();
}
