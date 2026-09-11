// @ts-check

import { spawnSync } from "node:child_process";
import { chmodSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { resolveArtifactPath, validateArtifactManifest } from "./artifacts.mjs";
import { digestObject, sha256 } from "./identity.mjs";
import { NATIVE_EMBEDDED_AOT_PROBE } from "./product-contracts.mjs";
import { digest, enumValue, exactKeys, gitRevision, integer } from "./schema.mjs";

export const PRODUCT_PROBE_SCHEMA = "runmat.builtin-examples.product-probe.v1";
export { NATIVE_EMBEDDED_AOT_PROBE } from "./product-contracts.mjs";

const NATIVE_VECTOR_SOURCE = "disp(2 + 3);\n";
const NATIVE_VECTOR_DIGEST = sha256(NATIVE_VECTOR_SOURCE);
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
            result: { status, failureKind, compileExitCode, executeExitCode },
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
    enumValue(probe.product, ["native-cli", "browser-wasm"], "product probe product");
    digest(probe.artifactManifestDigest, "product probe artifact manifest digest");
    enumValue(probe.probeKind, [NATIVE_EMBEDDED_AOT_PROBE], "product probe kind");
    digest(probe.vectorDigest, "product probe vector digest");
    digest(probe.stdoutDigest, "product probe stdout digest");
    digest(probe.productProbeDigest, "product probe digest");
    if (digestObject(probe, ["productProbeDigest"]) !== probe.productProbeDigest) throw new Error("Product probe digest mismatch");
    exactKeys(probe.result, ["status", "failureKind", "compileExitCode", "executeExitCode"], "product probe result");
    enumValue(probe.result.status, ["passed", "failed"], "product probe result status");
    enumValue(probe.result.failureKind, ["none", "compile-process", "execute-process", "stdout-mismatch"], "product probe failure kind");
    for (const [field, label] of [["compileExitCode", "compile exit code"], ["executeExitCode", "execute exit code"]]) {
        const value = probe.result[field];
        if (value !== null) integer(value, `product probe ${label}`);
    }
    if (probe.probeKind === NATIVE_EMBEDDED_AOT_PROBE) {
        if (probe.product !== "native-cli" || probe.artifactProfile !== "embedded-aot") throw new Error("Native embedded-AOT probe has the wrong product profile");
        if (probe.vectorDigest !== NATIVE_VECTOR_DIGEST) throw new Error("Native embedded-AOT probe vector digest mismatch");
        if ((probe.result.status === "passed") !== (probe.result.failureKind === "none")) throw new Error("Product probe status and failure kind disagree");
        if (probe.result.status === "passed" && (probe.result.compileExitCode !== 0 || probe.result.executeExitCode !== 0)) throw new Error("Passed native embedded-AOT probe has unsuccessful process evidence");
        if (probe.result.status === "passed" && !NATIVE_SUCCESS_STDOUT_DIGESTS.includes(probe.stdoutDigest)) throw new Error("Passed native embedded-AOT probe has unexpected stdout evidence");
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

function processExitCode(status) {
    return Number.isSafeInteger(status) && status >= 0 ? status : null;
}

function normalizeProbeOutput(output) {
    return output.toString("utf8").replace(/\r\n/g, "\n").trim();
}
