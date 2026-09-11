// @ts-check

import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { materializeFilesystemFixture } from "./adapters/filesystem.mjs";
import { runBoundedProcess } from "./adapters/process.mjs";
import {
    buildDesktopHostRequest,
    DESKTOP_HOST_MODE_ARGUMENT,
    DESKTOP_HOST_REQUEST_ENV,
    DESKTOP_HOST_RESULT_ENV,
    validateDesktopHostResult
} from "./desktop-host-protocol.mjs";

export async function runDesktopHostCases(cases, timeoutMs) {
    const binary = resolveDesktopBinary();
    const root = mkdtempSync(join(tmpdir(), "runmat-desktop-documentation-examples-"));
    try {
        const results = [];
        for (const testCase of cases) results.push(await runDesktopHostCase(binary, root, testCase, timeoutMs));
        return results;
    } finally {
        rmSync(root, { recursive: true, force: true });
    }
}

async function runDesktopHostCase(binary, root, testCase, timeoutMs) {
    const desktopFixture = testCase.fixture?.DesktopHostOnly;
    if (!desktopFixture) return failure(testCase.id, "Desktop host lane requires a DesktopHostOnly fixture", "");
    const caseDirectory = join(root, String(testCase.id));
    const workspaceRoot = join(caseDirectory, "workspace");
    const requestPath = join(caseDirectory, "request.json");
    const resultPath = join(caseDirectory, "result.json");
    mkdirSync(caseDirectory);
    try {
        materializeFilesystemFixture(workspaceRoot, { id: desktopFixture.id, entries: desktopFixture.entries });
        const request = buildDesktopHostRequest({
            exampleKey: testCase.exampleKey,
            program: testCase.input,
            compatibility: testCase.compatibility,
            workspaceRoot,
            fixture: testCase.fixture
        });
        writeFileSync(requestPath, `${JSON.stringify(request, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
        const completed = await invokeDesktopHost(binary, request, requestPath, resultPath, workspaceRoot, timeoutMs);
        if (completed.error) return failure(testCase.id, `Desktop host process: ${completed.error.message}`, "");
        if (completed.status !== 0) {
            const detail = completed.stderr.trim() || completed.stdout.trim() || `process exited with status ${completed.status}`;
            return failure(testCase.id, `Desktop host process: ${detail}`, "");
        }
        if (completed.protocolError) return failure(testCase.id, `Desktop host protocol: ${completed.protocolError.message}`, "");
        const result = completed.result;
        if (!result.interactions.matched) {
            return failure(testCase.id, "Desktop host interactions did not match the declared fixture", "RunMat:Verifier:InteractionMismatch", result.stdoutText);
        }
        return {
            id: testCase.id,
            stdoutText: result.stdoutText,
            valueText: result.valueText,
            errorText: result.errorText,
            errorIdentifier: result.errorIdentifier,
            figureVerified: result.interactions.figurePresentationsMatched
        };
    } catch (error) {
        return failure(testCase.id, `Desktop host adapter: ${error instanceof Error ? error.message : String(error)}`, "");
    }
}

export async function invokeDesktopHost(binary, request, requestPath, resultPath, workspaceRoot, timeoutMs) {
    const completed = await runBoundedProcess(binary, [DESKTOP_HOST_MODE_ARGUMENT], {
        cwd: workspaceRoot,
        env: {
            ...process.env,
            NO_COLOR: "1",
            [DESKTOP_HOST_REQUEST_ENV]: requestPath,
            [DESKTOP_HOST_RESULT_ENV]: resultPath
        },
        timeout: timeoutMs,
        maxBuffer: 16 * 1024 * 1024
    });
    if (completed.error || completed.status !== 0) return { ...completed, protocolError: null, result: null };
    try {
        if (!existsSync(resultPath)) throw new Error("Desktop host process produced no result file");
        if (statSync(resultPath).size > 16 * 1024 * 1024) throw new Error("Desktop host result exceeded its byte limit");
        const result = validateDesktopHostResult(JSON.parse(readFileSync(resultPath, "utf8")), request);
        return { ...completed, protocolError: null, result };
    } catch (error) {
        return { ...completed, protocolError: error instanceof Error ? error : new Error(String(error)), result: null };
    }
}

function resolveDesktopBinary() {
    const override = process.env.RUNMAT_EXAMPLE_DESKTOP_BINARY;
    if (!override) throw new Error("RUNMAT_EXAMPLE_DESKTOP_BINARY is required for the desktop-host lane");
    const binary = resolve(override);
    if (!existsSync(binary)) throw new Error(`RUNMAT_EXAMPLE_DESKTOP_BINARY does not exist: ${binary}`);
    return binary;
}

function failure(id, errorText, errorIdentifier, stdoutText = "") {
    return { id, stdoutText, valueText: "", errorText, errorIdentifier };
}
