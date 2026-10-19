// @ts-check

import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { existsSync, readFileSync, statSync } from "node:fs";
import { extname, resolve } from "node:path";

export async function runHeadlessChrome(options) {
    if (!existsSync(options.chromeWrapper) || !statSync(options.chromeWrapper).isFile()) {
        throw new Error(`Chrome wrapper not found: ${options.chromeWrapper}`);
    }

    let resolveResults;
    let rejectResults;
    /** @type {Promise<RunnerResult[]>} */
    const resultsPromise = new Promise((resolve, reject) => {
        resolveResults = resolve;
        rejectResults = reject;
    });

    let lastSnapshotLogTime = 0;
    const server = createServer((req, res) => {
        if (!req.url) {
            res.statusCode = 400;
            res.end("Missing URL");
            return;
        }
        const urlPath = req.url.split("?")[0];
        if (urlPath === "/__runner__/runner.html") {
            console.log("[runner] request runner.html");
            res.statusCode = 200;
            res.setHeader("content-type", "text/html; charset=utf-8");
            res.end(options.runnerHtml);
            return;
        }
        if (urlPath === "/__runner__/cases.json") {
            console.log("[runner] request cases.json");
            res.statusCode = 200;
            res.setHeader("content-type", "application/json");
            res.end(options.casesJson);
            return;
        }
        if (urlPath === "/__runner__/wasm.js") {
            serveExactFile(options.wasmModule, res);
            return;
        }
        if (urlPath === "/__runner__/runmat_wasm_web_bg.wasm") {
            serveExactFile(options.wasmBinary, res);
            return;
        }
        if (urlPath === "/__runner__/log" && req.method === "POST") {
            let body = "";
            req.setEncoding("utf8");
            req.on("data", (chunk) => {
                body += chunk;
            });
            req.on("end", () => {
                try {
                    const payload = JSON.parse(body);
                    if (payload && payload.type === "worker-snapshot") {
                        const isTimeout = payload.reason === "timeout";
                        const now = Date.now();

                        // Log every 2s as requested, or immediately on timeout
                        if (isTimeout || now - lastSnapshotLogTime >= 2000) {
                            lastSnapshotLogTime = now;
                            const workerList = (payload.workers || [])
                                .map((w) => `[W${w.id}: case ${w.caseId}, ${Math.round(w.elapsedMs / 1000)}s]`)
                                .join(" ");

                            const label = isTimeout ? "WATCHDOG KILL" : "SUPERVISOR REPORT";
                            console.log(
                                `[runner] ${label} - progress: ${payload.completed}/${payload.total} active: ${payload.active} ${
                                    workerList ? "\n         stalled: " + workerList : ""
                                }`
                            );
                        }
                    } else if (payload && payload.type === "runner-start") {
                        console.log(`[runner] started: ${payload.cases} cases, concurrency=${payload.concurrency}`);
                    } else if (payload && payload.type === "worker-start") {
                        // Log every start during ramp-up
                        console.log(`[runner] W${payload.workerId} start: case ${payload.caseId} (${payload.description})`);
                    } else if (payload && payload.type === "worker-finish") {
                        if (payload.caseId % 50 === 0) {
                            console.log(`[runner] progress: ${payload.caseId}/${options.totalCases}...`);
                        }
                    } else if (payload && payload.type === "worker-start") {
                        // Suppress start logs to reduce noise unless critical
                    } else if (payload && payload.type === "runner-fetch-cases-error") {
                        console.error(`[runner] error fetching cases: ${payload.message}`);
                    } else if (payload && payload.type === "runner-error") {
                        console.error(`[runner] error: ${payload.message}`);
                    } else if (payload && payload.type === "runner-unhandled-rejection") {
                        console.error(`[runner] unhandled rejection: ${payload.message}`);
                    }
                    res.statusCode = 200;
                    res.end("ok");
                } catch (err) {
                    res.statusCode = 400;
                    res.end("invalid log payload");
                }
            });
            return;
        }
        if (urlPath === "/__runner__/results" && req.method === "POST") {
            console.log("[runner] receiving results...");
            let body = "";
            req.setEncoding("utf8");
            req.on("data", (chunk) => {
                body += chunk;
            });
            req.on("end", () => {
                try {
                    const payload = JSON.parse(body);
                    if (!payload || !Array.isArray(payload.results)) {
                        throw new Error("Malformed results payload");
                    }
                    console.log(`[runner] received ${payload.results.length} results.`);
                    resolveResults(payload.results);
                    res.statusCode = 200;
                    res.end("ok");
                } catch (err) {
                    console.error(`[runner] failed to parse results: ${err.message}`);
                    res.statusCode = 400;
                    res.end("invalid results");
                    rejectResults(err);
                }
            });
            return;
        }

        serveStaticFile(options.repoRoot, urlPath, res);
    });

    const port = await new Promise((resolvePort, rejectPort) => {
        server.listen(0, "127.0.0.1", () => {
            const address = server.address();
            if (!address || typeof address === "string") {
                rejectPort(new Error("Unable to bind server"));
                return;
            }
            resolvePort(address.port);
        });
    });

    const url = `http://127.0.0.1:${port}/__runner__/runner.html`;
    const chrome = spawn(options.chromeWrapper, [url], {
        cwd: options.repoRoot,
        stdio: process.env.RUNMAT_HEADLESS_DEBUG === "1"
            ? ["ignore", "inherit", "inherit"]
            : "ignore"
    });
    chrome.once("error", (error) => {
        rejectResults(new Error(`Unable to start headless Chrome: ${error.message}`));
    });
    chrome.once("exit", (code, signal) => {
        const outcome = signal ? `signal ${signal}` : `status ${code ?? "unknown"}`;
        rejectResults(new Error(`Headless Chrome exited before returning results (${outcome})`));
    });

    const timeoutMs = options.overallTimeoutMs ?? 600000;
    const timeout = setTimeout(() => {
        rejectResults(new Error(`Timed out after ${timeoutMs}ms waiting for results`));
    }, timeoutMs);

    let results;
    try {
        results = await resultsPromise;
    } finally {
        clearTimeout(timeout);
        server.closeAllConnections();
        await Promise.all([
            terminateChild(chrome, 5000),
            new Promise((resolveClose, rejectClose) => {
                server.close((error) => {
                    if (error) {
                        rejectClose(error);
                    } else {
                        resolveClose();
                    }
                });
            })
        ]);
    }

    return results;
}

/**
 * @param {import("child_process").ChildProcess} child
 * @param {number} graceMs
 */
async function terminateChild(child, graceMs) {
    if (child.exitCode !== null || child.signalCode !== null) {
        return;
    }

    const exited = new Promise((resolveExit) => child.once("exit", resolveExit));
    child.kill("SIGTERM");
    const forced = await Promise.race([
        exited.then(() => false),
        new Promise((resolveTimeout) => {
            const timer = setTimeout(() => resolveTimeout(true), graceMs);
            timer.unref();
        })
    ]);
    if (forced && child.exitCode === null && child.signalCode === null) {
        child.kill("SIGKILL");
        await exited;
    }
}

/**
 * @param {string} root
 * @param {string} urlPath
 * @param {import("http").ServerResponse} res
 */
function serveStaticFile(root, urlPath, res) {
    const safePath = resolve(root, `.${urlPath}`);
    if (!safePath.startsWith(root)) {
        res.statusCode = 403;
        res.end("Forbidden");
        return;
    }
    if (!existsSync(safePath) || !statSync(safePath).isFile()) {
        res.statusCode = 404;
        res.end("Not found");
        return;
    }
    const extension = extname(safePath);
    const contentType = contentTypeForExtension(extension);
    if (contentType) {
        res.setHeader("content-type", contentType);
    }
    const data = readFileSync(safePath);
    res.statusCode = 200;
    res.end(data);
}

function serveExactFile(path, res) {
    if (!existsSync(path) || !statSync(path).isFile()) {
        res.statusCode = 404;
        res.end("Not found");
        return;
    }
    const contentType = contentTypeForExtension(extname(path));
    if (contentType) res.setHeader("content-type", contentType);
    res.statusCode = 200;
    res.end(readFileSync(path));
}

/**
 * @param {string} extension
 */
function contentTypeForExtension(extension) {
    switch (extension) {
        case ".html":
            return "text/html; charset=utf-8";
        case ".js":
            return "text/javascript";
        case ".json":
            return "application/json";
        case ".wasm":
            return "application/wasm";
        case ".css":
            return "text/css";
        case ".map":
            return "application/json";
        default:
            return "application/octet-stream";
    }
}
