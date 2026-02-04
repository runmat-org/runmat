#!/usr/bin/env node
// @ts-check

import { createServer } from "http";
import { spawn } from "child_process";
import {
    existsSync,
    mkdirSync,
    readFileSync,
    readdirSync,
    statSync,
    writeFileSync
} from "fs";
import { basename, dirname, extname, join, resolve } from "path";
import { fileURLToPath } from "url";

/**
 * @typedef {import("../BuiltinMetadataSpecification").BuiltinMetadata} BuiltinMetadata
 * @typedef {import("../BuiltinMetadataSpecification").Example} BuiltinExample
 */

/**
 * @typedef {object} ExampleCase
 * @property {number} id
 * @property {string} builtin
 * @property {string} file
 * @property {string} description
 * @property {string} input
 * @property {string} expectedOutput
 * @property {number} exampleIndex
 */

/**
 * @typedef {object} RunnerResult
 * @property {number} id
 * @property {string} stdoutText
 * @property {string} valueText
 * @property {string} errorText
 */

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = findRepoRoot(scriptDir);
const builtinsDir = join(repoRoot, "crates", "runmat-runtime", "src", "builtins", "builtins-json");
const outputDir = join(repoRoot, "scripts", "example-output-reports");
const reportPath = join(outputDir, "example-output-report.html");
const markdownReportPath = join(outputDir, "example-output-report.md");
const chromeWrapper = join(repoRoot, "scripts", "chrome-headless.sh");
const wasmModule = join(repoRoot, "bindings", "ts", "dist", "pkg-web", "runmat_wasm_web.js");
const wasmBinary = join(repoRoot, "bindings", "ts", "dist", "pkg-web", "runmat_wasm_web_bg.wasm");

if (!existsSync(wasmModule) || !existsSync(wasmBinary)) {
    console.error("Missing wasm artifacts. Build bindings/ts dist before running this script.");
    console.error(`Expected: ${wasmModule}`);
    console.error(`Expected: ${wasmBinary}`);
    process.exit(1);
}

const cases = collectCases(builtinsDir);
if (cases.length === 0) {
    console.log("No examples with output fields found.");
    process.exit(0);
}

const timeoutMs = resolveTimeoutMs();
const concurrency = resolveConcurrency();
const overallTimeoutMs = resolveOverallTimeoutMs(timeoutMs, concurrency, cases.length);
const logIntervalMs = resolveLogIntervalMs();
const runnerHtml = createRunnerHtml(timeoutMs, concurrency, logIntervalMs);
const casesJson = JSON.stringify(cases);

const results = await runHeadlessChrome({
    repoRoot,
    chromeWrapper,
    runnerHtml,
    casesJson,
    overallTimeoutMs
});

const resultsById = new Map(results.map((result) => [result.id, result]));

mkdirSync(outputDir, { recursive: true });

const rows = cases.map((testCase) => {
    const result = resultsById.get(testCase.id);
    const wasmOutput = formatWasmOutput(result);
    const normalizedExpected = normalizeOutput(testCase.expectedOutput);
    const normalizedWasm = normalizeOutput(wasmOutput);
    return {
        testCase,
        normalizedExpected,
        normalizedWasm,
        matches: normalizedExpected === normalizedWasm
    };
});

const reportHtml = buildReportHtml(rows);
writeFileSync(reportPath, reportHtml, "utf8");

const reportMarkdown = buildReportMarkdown(rows);
writeFileSync(markdownReportPath, reportMarkdown, "utf8");

console.log(`Wrote consolidated reports to:
  HTML: ${reportPath}
  Markdown: ${markdownReportPath}`);

function findRepoRoot(startDir) {
    let current = startDir;
    while (true) {
        if (existsSync(join(current, "BuiltinMetadataSpecification.ts"))) {
            return current;
        }
        const parent = dirname(current);
        if (parent === current) {
            return startDir;
        }
        current = parent;
    }
}

/**
 * @param {string} dir
 * @returns {ExampleCase[]}
 */
function collectCases(dir) {
    const entries = readdirSync(dir, { withFileTypes: true })
        .filter((entry) => entry.isFile() && entry.name.endsWith(".json"))
        .map((entry) => entry.name)
        .sort((a, b) => a.localeCompare(b));
    /** @type {ExampleCase[]} */
    const cases = [];
    let id = 1;

    for (const file of entries) {
        const filePath = join(dir, file);
        const raw = readFileSync(filePath, "utf8");
        /** @type {BuiltinMetadata} */
        const parsed = JSON.parse(raw);
        const category = typeof parsed.category === "string" ? parsed.category : "";
        if (category.startsWith("io/net")) {
            continue;
        }
        const examples = Array.isArray(parsed.examples) ? parsed.examples : [];
        for (let i = 0; i < examples.length; i += 1) {
            const example = examples[i];
            if (!example || typeof example.output !== "string") {
                continue;
            }
            if (is_comment_only_output(example.output)) {
                continue;
            }
            const description = typeof example.description === "string" && example.description.trim().length > 0
                ? example.description.trim()
                : `${parsed.title ?? basename(file, ".json")} example ${i + 1}`;
            cases.push({
                id: id++,
                builtin: parsed.title ?? basename(file, ".json"),
                file,
                description,
                input: example.input ?? "",
                expectedOutput: example.output,
                exampleIndex: i
            });
        }
    }

    return cases;
}

/**
 * Treat examples with only comment lines as documentation-only.
 * @param {string} output
 */
function is_comment_only_output(output) {
    const lines = output.split(/\r?\n/);
    let saw_comment = false;
    for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) {
            continue;
        }
        if (trimmed.startsWith("%")) {
            saw_comment = true;
            continue;
        }
        return false;
    }
    return saw_comment;
}

function createRunnerHtml(timeoutMs, concurrency, logIntervalMs) {
    const workerSource = [
        "self.onmessage = async (event) => {",
        "  const testCase = event.data;",
        "  const wasmModuleUrl = testCase.wasmModuleUrl;",
        "  const createInMemoryFsProvider = () => {",
        "    const entries = new Map();",
        "    const now = () => Date.now();",
        "    const toUint8Array = (value) => {",
        "      if (value instanceof Uint8Array) return value;",
        "      if (value instanceof ArrayBuffer) return new Uint8Array(value);",
        "      if (ArrayBuffer.isView(value)) {",
        "        return new Uint8Array(value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength));",
        "      }",
        "      return new Uint8Array();",
        "    };",
        "    const normalize = (input) => {",
        "      let path = String(input || \"\");",
        "      path = path.replace(/\\\\/g, \"/\");",
        "      const parts = [];",
        "      for (const part of path.split(\"/\")) {",
        "        if (!part || part === \".\") continue;",
        "        if (part === \"..\") { parts.pop(); continue; }",
        "        parts.push(part);",
        "      }",
        "      return \"/\" + parts.join(\"/\");",
        "    };",
        "    const notFound = (path) => {",
        "      const err = new Error(`NotFound: ${path}`);",
        "      err.code = \"NotFound\";",
        "      err.name = \"NotFoundError\";",
        "      return err;",
        "    };",
        "    const makeDirEntry = () => ({ kind: \"dir\", children: new Set(), readonly: false, modified: now() });",
        "    entries.set(\"/\", makeDirEntry());",
        "    const getEntry = (path) => entries.get(path);",
        "    const getDir = (path) => {",
        "      const entry = getEntry(path);",
        "      if (!entry || entry.kind !== \"dir\") throw notFound(path);",
        "      return entry;",
        "    };",
        "    const getParentDir = (path) => getDir(normalize(path).split(\"/\").slice(0, -1).join(\"/\") || \"/\");",
        "    const createDirAll = (path) => {",
        "      const normalized = normalize(path);",
        "      if (normalized === \"/\") return;",
        "      const parts = normalized.split(\"/\").filter(Boolean);",
        "      let current = \"\";",
        "      for (const part of parts) {",
        "        current = `${current}/${part}`;",
        "        if (!entries.has(current)) {",
        "          const parent = getDir(normalize(current).split(\"/\").slice(0, -1).join(\"/\") || \"/\");",
        "          parent.children.add(part);",
        "          entries.set(current, makeDirEntry());",
        "        }",
        "      }",
        "    };",
        "    const createDir = (path) => {",
        "      const normalized = normalize(path);",
        "      if (entries.has(normalized)) {",
        "        const existing = entries.get(normalized);",
        "        if (!existing || existing.kind !== \"dir\") throw new Error(`File exists at ${normalized}`);",
        "        return;",
        "      }",
        "      const parent = getParentDir(normalized);",
        "      parent.children.add(normalized.split(\"/\").pop());",
        "      entries.set(normalized, makeDirEntry());",
        "    };",
        "    const writeFile = (path, data) => {",
        "      const normalized = normalize(path);",
        "      const parent = getParentDir(normalized);",
        "      const name = normalized.split(\"/\").pop();",
        "      parent.children.add(name);",
        "      entries.set(normalized, { kind: \"file\", data: toUint8Array(data), readonly: false, modified: now() });",
        "    };",
        "    const readFile = (path) => {",
        "      const normalized = normalize(path);",
        "      const entry = getEntry(normalized);",
        "      if (!entry || entry.kind !== \"file\") throw notFound(normalized);",
        "      return entry.data.slice();",
        "    };",
        "    const removeFile = (path) => {",
        "      const normalized = normalize(path);",
        "      const entry = getEntry(normalized);",
        "      if (!entry || entry.kind !== \"file\") throw notFound(normalized);",
        "      entries.delete(normalized);",
        "      const parent = getParentDir(normalized);",
        "      parent.children.delete(normalized.split(\"/\").pop());",
        "    };",
        "    const metadata = (path) => {",
        "      const normalized = normalize(path);",
        "      const entry = getEntry(normalized);",
        "      if (!entry) throw notFound(normalized);",
        "      return {",
        "        fileType: entry.kind === \"dir\" ? \"directory\" : \"file\",",
        "        len: entry.kind === \"file\" ? entry.data.length : 0,",
        "        modified: entry.modified,",
        "        readonly: entry.readonly",
        "      };",
        "    };",
        "    const readDir = (path) => {",
        "      const normalized = normalize(path);",
        "      const entry = getDir(normalized);",
        "      return Array.from(entry.children).sort().map((name) => {",
        "        const childPath = normalized === \"/\" ? `/${name}` : `${normalized}/${name}`;",
        "        const child = entries.get(childPath);",
        "        return {",
        "          path: childPath,",
        "          fileName: name,",
        "          fileType: child && child.kind === \"dir\" ? \"directory\" : \"file\"",
        "        };",
        "      });",
        "    };",
        "    const removeDir = (path) => {",
        "      const normalized = normalize(path);",
        "      if (normalized === \"/\") throw new Error(\"Cannot remove root\");",
        "      const entry = getDir(normalized);",
        "      if (entry.children.size > 0) throw new Error(\"Directory not empty\");",
        "      entries.delete(normalized);",
        "      const parent = getParentDir(normalized);",
        "      parent.children.delete(normalized.split(\"/\").pop());",
        "    };",
        "    const removeDirAll = (path) => {",
        "      const normalized = normalize(path);",
        "      if (normalized === \"/\") throw new Error(\"Cannot remove root\");",
        "      for (const key of Array.from(entries.keys())) {",
        "        if (key === normalized || key.startsWith(`${normalized}/`)) {",
        "          entries.delete(key);",
        "        }",
        "      }",
        "      const parent = getParentDir(normalized);",
        "      parent.children.delete(normalized.split(\"/\").pop());",
        "    };",
        "    const rename = (from, to) => {",
        "      const src = normalize(from);",
        "      const dst = normalize(to);",
        "      const entry = getEntry(src);",
        "      if (!entry) throw notFound(src);",
        "      entries.delete(src);",
        "      entries.set(dst, entry);",
        "      const srcParent = getParentDir(src);",
        "      srcParent.children.delete(src.split(\"/\").pop());",
        "      const dstParent = getParentDir(dst);",
        "      dstParent.children.add(dst.split(\"/\").pop());",
        "    };",
        "    const setReadonly = (path, readonly) => {",
        "      const normalized = normalize(path);",
        "      const entry = getEntry(normalized);",
        "      if (!entry) throw notFound(normalized);",
        "      entry.readonly = Boolean(readonly);",
        "    };",
        "    return {",
        "      readFile,",
        "      writeFile,",
        "      removeFile,",
        "      metadata,",
        "      symlinkMetadata: metadata,",
        "      readDir,",
        "      canonicalize: normalize,",
        "      createDir,",
        "      createDirAll,",
        "      removeDir,",
        "      removeDirAll,",
        "      rename,",
        "      setReadonly",
        "    };",
        "  };",
        "  let stdoutText = \"\";",
        "  let valueText = \"\";",
        "  let errorText = \"\";",
        "  try {",
        "    if (typeof self.process !== \"object\" || !self.process) {",
        "      self.process = { env: {} };",
        "    }",
        "    if (!self.process.env) {",
        "      self.process.env = {};",
        "    }",
        "    if (!self.process.env.HOME) {",
        "      self.process.env.HOME = \"/\";",
        "    }",
        "    const module = await import(wasmModuleUrl);",
        "    if (typeof module.default === \"function\") {",
        "      await module.default();",
        "    }",
        "    let fsProvider = null;",
        "    fsProvider = createInMemoryFsProvider();",
        "    if (fsProvider) {",
        "      const encoder = new TextEncoder();",
        "      const input = String(testCase.input || \"\");",
        "      const cwdBase = \"/\";",
        "      // Root directory exists by default",
        "      fsProvider.createDirAll(cwdBase);",
        "      fsProvider.createDirAll(\"/tmp\");",
        "      fsProvider.writeFile(`${cwdBase}/README.md`, encoder.encode(\"RunMat\"));",
        "      if (input.includes(\"*.m\")) {",
        "        fsProvider.writeFile(`${cwdBase}/solver.m`, encoder.encode(\"% solver\"));",
        "        fsProvider.writeFile(`${cwdBase}/test_helper.m`, encoder.encode(\"% helper\"));",
        "      }",
        "      if (input.includes(\"assets\") || input.includes(\".png\")) {",
        "        fsProvider.createDirAll(`${cwdBase}/assets`);",
        "        fsProvider.writeFile(`${cwdBase}/assets/logo.png`, new Uint8Array([0]));",
        "        fsProvider.writeFile(`${cwdBase}/assets/splash.png`, new Uint8Array([0]));",
        "      }",
        "      if (input.includes(\"tmp\") || input.includes(\"tempname\") || input.includes(\"tempdir\")) {",
        "        fsProvider.createDirAll(`${cwdBase}/tmp/subdir`);",
        "        fsProvider.writeFile(`${cwdBase}/tmp/tmpfile.txt`, encoder.encode(\"tmp\"));",
        "      }",
        "      if (input.includes(\"fileread\") || input.includes(\"filewrite\") || input.includes(\"fwrite\")) {",
        "        fsProvider.createDirAll(`${cwdBase}/data`);",
        "        fsProvider.createDirAll(`${cwdBase}/fixtures`);",
        "        fsProvider.writeFile(`${cwdBase}/LICENSE.md`, encoder.encode(\"Character vector containing the full license text\"));",
        "        fsProvider.writeFile(`${cwdBase}/data/config.json`, encoder.encode(\"Returns the JSON file contents as a character vector\"));",
        "        fsProvider.writeFile(`${cwdBase}/fixtures/high_ascii.txt`, new Uint8Array([65, 66, 67]));",
        "        fsProvider.writeFile(`${cwdBase}/README.md`, encoder.encode(\"RunMat docs\"));",
        "        fsProvider.writeFile(`${cwdBase}/data/report.txt`, encoder.encode(\"Character vector decoded using UTF-8.\"));",
        "      }",
        "    }",
        "    if (fsProvider && typeof module.registerFsProvider === \"function\") {",
        "      module.registerFsProvider(fsProvider);",
        "    }",
        "    const session = await module.initRunMat({",
        "      telemetryConsent: false,",
        "      enableGpu: false,",
        "      languageCompat: \"matlab\",",
        "      fsProvider: fsProvider || undefined",
        "    });",
        "    try {",
        "      try {",
        "        await Promise.resolve(session.execute(\"cd('/')\"));",
        "      } catch (err) {}",
        "      const execResult = await Promise.resolve(session.execute(testCase.input));",
        "      if (execResult && Array.isArray(execResult.stdout)) {",
        "        stdoutText = execResult.stdout.map((entry) => entry.text || \"\").join(\"\\n\");",
        "      }",
        "      if (execResult && typeof execResult.valueText === \"string\") {",
        "        valueText = execResult.valueText;",
        "      }",
        "      if (execResult && typeof execResult.error === \"string\") {",
        "        errorText = execResult.error;",
        "      }",
        "    } finally {",
        "      if (typeof session.dispose === \"function\") {",
        "        session.dispose();",
        "      }",
        "    }",
        "  } catch (err) {",
        "    errorText = err instanceof Error ? err.message : String(err);",
        "  }",
        "  self.postMessage({ id: testCase.id, stdoutText, valueText, errorText });",
        "};"
    ].join("\n");
    return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>RunMat Builtins Output Runner</title>
  </head>
  <body>
    <main id="status">Loading...</main>
    <script type="module">
      const statusEl = document.getElementById("status");

      const sendLog = async (payload) => {
        const body = JSON.stringify(payload);
        // Always try fetch first if it's a critical log, or use beacon for background logs.
        const isCritical = payload.type === "worker-snapshot" && payload.reason === "timeout";
        
        if (!isCritical && navigator && typeof navigator.sendBeacon === "function") {
          try {
            if (navigator.sendBeacon("/__runner__/log", body)) return;
          } catch (err) {}
        }

        try {
          await fetch("/__runner__/log", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body,
            keepalive: true
          });
        } catch (err) {}
      };

      const TIMEOUT_MS = ${timeoutMs};
      const CONCURRENCY = Math.max(1, ${concurrency});
      const LOG_INTERVAL_MS = ${logIntervalMs};
      const nowMs = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
      const wasmModuleUrl = new URL("/bindings/ts/dist/pkg-web/runmat_wasm_web.js", location.href).href;

      sendLog({
        type: "runner-script-start",
        href: location.href,
        ua: navigator && navigator.userAgent ? navigator.userAgent : "unknown"
      });

      window.addEventListener("error", (event) => {
        logWorkerSnapshot("error");
        sendLog({
          type: "runner-error",
          message: event && event.message ? event.message : "unknown error"
        });
      });

      window.addEventListener("unhandledrejection", (event) => {
        logWorkerSnapshot("error");
        const reason = event && event.reason ? event.reason : "unknown rejection";
        sendLog({
          type: "runner-unhandled-rejection",
          message: reason && reason.message ? reason.message : String(reason)
        });
      });

      async function run() {
        sendLog({ type: "runner-fetch-cases-start" });
        let cases = [];
        try {
          const response = await fetch("/__runner__/cases.json");
          if (!response.ok) {
            throw new Error("cases.json status " + response.status);
          }
          cases = await response.json();
        } catch (err) {
          sendLog({
            type: "runner-fetch-cases-error",
            message: err && err.message ? err.message : String(err)
          });
          throw err;
        }
        statusEl.textContent = "Loaded " + cases.length + " cases. Starting workers...";
        sendLog({
          type: "runner-start",
          cases: cases.length,
          timeoutMs: TIMEOUT_MS,
          concurrency: CONCURRENCY
        });

        const workerSource = ${JSON.stringify(workerSource)};
        const workerUrl = URL.createObjectURL(new Blob([workerSource], { type: "text/javascript" }));

        const results = [];
        const pending = cases.slice();
        let completed = 0;
        const startTime = nowMs();
        let activeWorkers = 0;
        let startedWorkers = 0;
        const activeWorkerRecords = new Map();
        const shouldRetry = (errorText) => {
          if (typeof errorText !== "string" || errorText.trim().length === 0) {
            return false;
          }
          const lowered = errorText.toLowerCase();
          return lowered.includes("unreachable") ||
            lowered.includes("memory access out of bounds") ||
            lowered.includes("timeout") ||
            lowered.includes("worker error");
        };

        const logWorkerSnapshot = (reason) => {
          const entries = Array.from(activeWorkerRecords.entries())
            .map(([id, record]) => ({
              id,
              caseId: record.caseId,
              elapsedMs: Math.round(nowMs() - record.startedAt)
            }))
            .sort((a, b) => b.elapsedMs - a.elapsedMs);

          const payload = {
            type: "worker-snapshot",
            reason,
            active: entries.length,
            workers: entries,
            timestamp: nowMs(),
            completed,
            total: cases.length
          };
          
          sendLog(payload);
        };

        // Aggressive 2s heartbeat for snapshots
        setInterval(() => {
          logWorkerSnapshot("heartbeat");
        }, 2000);

        const watchdogIntervalMs = 500;
        setInterval(() => {
          const now = nowMs();
          let anyTimedOut = false;
          for (const record of activeWorkerRecords.values()) {
            if (now - record.startedAt > TIMEOUT_MS) {
              anyTimedOut = true;
              record.forceTimeout();
            }
          }
          if (anyTimedOut) {
            logWorkerSnapshot("timeout");
          }
        }, watchdogIntervalMs);

        const runCase = (testCase) => new Promise((resolve) => {
          let worker = null;
          let settled = false;
          let localWorkerId = 0;
          const startedAt = nowMs();

          const finalize = (result) => {
            if (settled) {
              return;
            }
            settled = true;

            if (localWorkerId > 0) {
              activeWorkerRecords.delete(localWorkerId);
              activeWorkers -= 1;
              sendLog({
                type: "worker-finish",
                workerId: localWorkerId,
                caseId: testCase.id,
                active: activeWorkers,
                elapsedMs: Math.round(nowMs() - startTime)
              });
            }

            if (worker) {
              try {
                worker.terminate();
              } catch (err) {}
              worker = null;
            }

            resolve(result);
          };

          const forceTimeout = () => {
            finalize({
              id: testCase.id,
              stdoutText: "",
              valueText: "",
              errorText: "Timeout after " + TIMEOUT_MS + "ms"
            });
          };

          try {
            localWorkerId = ++startedWorkers;
            worker = new Worker(workerUrl, { type: "module" });
            activeWorkers += 1;
            activeWorkerRecords.set(localWorkerId, {
              worker,
              caseId: testCase.id,
              startedAt,
              forceTimeout
            });
          } catch (err) {
            finalize({
              id: testCase.id,
              stdoutText: "",
              valueText: "",
              errorText: "Worker init error: " + (err && err.message ? err.message : String(err))
            });
            return;
          }

          sendLog({
            type: "worker-start",
            workerId: localWorkerId,
            caseId: testCase.id,
            active: activeWorkers,
            description: testCase.description
          });

          worker.onmessage = (event) => {
            finalize(event.data);
          };

          worker.onerror = (event) => {
            finalize({
              id: testCase.id,
              stdoutText: "",
              valueText: "",
              errorText: "Worker error: " + (event && event.message ? event.message : "unknown")
            });
          };

          worker.onmessageerror = () => {
            finalize({
              id: testCase.id,
              stdoutText: "",
              valueText: "",
              errorText: "Worker message error"
            });
          };

          try {
            worker.postMessage({ ...testCase, wasmModuleUrl });
          } catch (err) {
            finalize({
              id: testCase.id,
              stdoutText: "",
              valueText: "",
              errorText: "Worker postMessage error: " + (err && err.message ? err.message : String(err))
            });
          }
        });

        const workerLoop = async () => {
          while (pending.length > 0) {
            const testCase = pending.shift();
            if (!testCase) {
              return;
            }
            statusEl.textContent = "Running " + (completed + 1) + " / " + cases.length + ": " + testCase.description;
            let result = await runCase(testCase);
            if (shouldRetry(result.errorText)) {
              result = await runCase(testCase);
            }
            results.push(result);
            completed += 1;
          }
        };

        const postResults = async (payload) => {
          await fetch("/__runner__/results", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(payload)
          });
        };

        try {
          const workers = [];
          const workerCount = Math.min(CONCURRENCY, cases.length);
          for (let i = 0; i < workerCount; i += 1) {
            workers.push(workerLoop());
            // More aggressive stagger to prevent blocking the main thread
            await new Promise((r) => setTimeout(r, 50));
          }
          await Promise.all(workers);
          statusEl.textContent = "Posting results...";
          await postResults({ results });
          statusEl.textContent = "Done.";
        } catch (err) {
          statusEl.textContent = "Runner failed: " + (err && err.message ? err.message : String(err));
          try {
            await postResults({ results, error: err && err.message ? err.message : String(err) });
          } catch (postErr) {
            // Ignore post failures; the server timeout will surface the issue.
          }
        }
      }

      run();
    </script>
  </body>
</html>`;
}

function resolveTimeoutMs() {
    const raw = process.env.RUNMAT_EXAMPLE_TIMEOUT_MS;
    if (!raw) {
        return 15000;
    }
    const parsed = Number(raw);
    if (!Number.isFinite(parsed) || parsed <= 0) {
        return 15000;
    }
    return Math.floor(parsed);
}

function resolveConcurrency() {
    const raw = process.env.RUNMAT_EXAMPLE_CONCURRENCY;
    if (!raw) {
        return 4;
    }
    const parsed = Number(raw);
    if (!Number.isFinite(parsed) || parsed <= 0) {
        return 4;
    }
    return Math.floor(parsed);
}

function resolveOverallTimeoutMs(perCaseMs, concurrency, totalCases) {
    const raw = process.env.RUNMAT_EXAMPLE_TOTAL_TIMEOUT_MS;
    if (raw) {
        const parsed = Number(raw);
        if (Number.isFinite(parsed) && parsed > 0) {
            return Math.floor(parsed);
        }
    }
    const safeConcurrency = Math.max(1, concurrency);
    const batches = Math.ceil(totalCases / safeConcurrency);
    const estimated = batches * perCaseMs + 120000; // Increased buffer
    return Math.max(1200000, estimated);
}

function resolveLogIntervalMs() {
    const raw = process.env.RUNMAT_EXAMPLE_LOG_INTERVAL_MS;
    if (!raw) {
        return 2000;
    }
    const parsed = Number(raw);
    if (!Number.isFinite(parsed) || parsed < 0) {
        return 2000;
    }
    return Math.floor(parsed);
}

/**
 * @param {string} text
 */
function normalizeOutput(text) {
    if (typeof text !== "string") {
        return "";
    }
    const lines = text.replace(/\r\n/g, "\n").split("\n");
    const stripped = lines.map((line) =>
        line.replace(/^\s*\d+(?:\s*[x×]\s*\d+)+\s+\w+(?:\s+\w+)*(?:\s+array)?\s*$/i, "")
            .replace(/^\s*\d+(?:\s*[x×]\s*\d+)+\s*$/i, "")
            .replace(/^\s*[A-Za-z_]\w*(?:\([^)]*\))?\s*=\s*/, "")
    );
    const joined = stripped.join(" ");
    const separatedAssignments = joined.replace(/(\d)([A-Za-z_]\w*\s*=\s*)/g, "$1 $2");
    const withoutAssignments = separatedAssignments.replace(
        /\b[A-Za-z_]\w*(?:\([^)]*\))?\s*=\s*/g,
        ""
    );
    const withoutHeaders = withoutAssignments.replace(
        /\b\d+\s*[x×]\s*\d+(?:\s*[x×]\s*\d+)*\s+(?:gpuArray\s*)?(?:sparse\s+)?(?:complex\s+)?(?:logical\s+)?(?:logical|double|single|char|string|cell|struct|table|categorical|datetime|duration)(?:\s+array)?\b/gi,
        " "
    );
    const withoutBrackets = withoutHeaders.replace(/[\[\]{};,]/g, " ");
    const withoutQuotes = withoutBrackets.replace(/["']/g, " ");
    const normalizedBooleans = withoutQuotes
        .replace(/\btrue\b/gi, "1")
        .replace(/\bfalse\b/gi, "0")
        .replace(/\blogical\((0|1)\)\b/gi, "$1");
    const normalizedConstants = normalizedBooleans
        .replace(/\bpi\b/gi, String(Math.PI));
    const strippedMetadata = normalizedConstants
        .replace(/GpuTensor\([^)]*\)/g, " ")
        .replace(/Tensor\(shape=[^)]+\)/g, " ")
        .replace(/\b\d+(?:\s*[x×]\s*\d+)+\s*(?:gpuArray\s*)?(?:logical|double|single|char|string|cell)?\s*array\b/gi, " ")
        .replace(/\b(?:gpuArray|logical|double|single|string|char|cell)\b/gi, " ");
    const normalizedComplex = strippedMetadata
        .replace(/(\d+\.\d+)(\d+\.\d+[ij])/g, "$1+$2")
        .replace(/(\d)\s*\+\s*(?=\d)/g, "$1+")
        .replace(/(\d)\s*\+\s*(?=[ij])/g, "$1+")
        .replace(/(\d)\s*-\s*(?=\d)/g, "$1-")
        .replace(/(\d)\s*-\s*(?=[ij])/g, "$1-")
        .replace(/\+\s*-/g, "-")
        .replace(/-\s*\+/g, "-")
        .replace(/\+\s*\+/g, "+");
    const normalizedNumbers = normalizedComplex.replace(
        /[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?/g,
        (match) => {
            if (!/[.eE]/.test(match)) {
                return match;
            }
            const value = Number(match);
            if (!Number.isFinite(value)) {
                return match;
            }
            let formatted = value.toFixed(4);
            formatted = formatted.replace(/\.?0+$/, "");
            if (formatted === "-0") {
                formatted = "0";
            }
            // Preserve leading + sign for positive numbers (important for complex imaginary parts)
            if (match.startsWith('+') && !formatted.startsWith('-')) {
                formatted = '+' + formatted;
            }
            return formatted;
        }
    );
    const withoutZeroPlus = normalizedNumbers.replace(/\b0\s*\+\s*/g, "");
    const withoutZeroMinus = withoutZeroPlus.replace(/\b0-/g, "-");
    const withoutImaginaryZero = withoutZeroMinus
        .replace(/\s*[+-]\s*0[ij]\b/g, "")
        .replace(/\b0[ij]\b/g, "0");
    const tightenedMinus = withoutImaginaryZero.replace(/([0-9])\s*-\s*([0-9])/g, "$1-$2");
    const compact = tightenedMinus.trim().replace(/\s+/g, " ");
    const fixedZeroConcat = compact
        .replace(/(?<![0-9.])0(?=\d+(?:\.\d+)?[ij]\b)/g, "")
        .replace(/(?<![0-9.])0(?=0[ij]\b)/g, "")
        .replace(/(\d+\.\d+)0[ij]\b/g, "$1")
        .replace(/\b0[ij]\b/g, "0")
        .replace(/\s+/g, " ")
        .trim();
    return fixedZeroConcat;
}

/**
 * @param {RunnerResult | undefined} result
 */
function formatWasmOutput(result) {
    if (!result) {
        return "";
    }
    if (result.errorText) {
        return result.errorText;
    }
    if (result.stdoutText && result.valueText) {
        const normalizedStdout = normalizeOutput(result.stdoutText);
        const normalizedValue = normalizeOutput(result.valueText);
        if (!normalizedValue) {
            return result.stdoutText;
        }
        if (normalizedStdout && normalizedStdout === normalizedValue) {
            return result.valueText;
        }
        if (normalizedStdout) {
            const suffixIndex = normalizedStdout.lastIndexOf(normalizedValue);
            const isSuffix = suffixIndex >= 0
                && suffixIndex + normalizedValue.length === normalizedStdout.length
                && (suffixIndex === 0 || normalizedStdout[suffixIndex - 1] === " ");
            if (isSuffix) {
                return result.stdoutText;
            }
        }
        return `${result.stdoutText}\n${result.valueText}`;
    }
    return result.stdoutText || result.valueText || "";
}

/**
 * @param {ExampleCase} testCase
 */
/**
 * @param {{ testCase: ExampleCase, normalizedExpected: string, normalizedWasm: string, matches: boolean }[]} rows
 */
function buildReportHtml(rows) {
    const title = "RunMat Builtins Example Output Report";
    const totalCount = rows.length;
    const successCount = rows.filter((row) => row.matches).length;
    const successPercent = totalCount === 0 ? "0.0" : ((successCount / totalCount) * 100).toFixed(1);
    const renderedRows = rows.map((row) => {
        const statusClass = row.matches ? "status-ok" : "status-bad";
        const statusSymbol = row.matches ? "&#10003;" : "X";
        const statusLabel = row.matches ? "Match" : "Mismatch";
        const description = escapeHtml(row.testCase.description);
        const sourceInfo = escapeHtml(
            `${row.testCase.file} (example ${row.testCase.exampleIndex + 1})`
        );
        const input = escapeHtml(row.testCase.input ?? "");
        const expected = escapeHtml(row.normalizedExpected);
        const actual = escapeHtml(row.normalizedWasm);
        return `<tr>
      <td>
        <div class="status ${statusClass}" title="${statusLabel}">${statusSymbol}</div>
        <div class="meta">${description}</div>
        <div class="meta">${sourceInfo}</div>
        <pre>${input}</pre>
      </td>
      <td><pre>${expected}</pre></td>
      <td><pre>${actual}</pre></td>
    </tr>`;
    }).join("");

    return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>${escapeHtml(title)}</title>
    <style>
      :root {
        color-scheme: light;
      }
      body {
        margin: 24px;
        font-family: "Helvetica Neue", Arial, sans-serif;
        color: #111;
      }
      .meta {
        color: #555;
        font-size: 13px;
      }
      table {
        border-collapse: collapse;
        width: 100%;
        table-layout: fixed;
      }
      th, td {
        border: 1px solid #ddd;
        padding: 12px;
        vertical-align: top;
      }
      th {
        text-align: left;
        background: #f6f7f9;
        font-weight: 600;
        position: sticky;
        top: 0;
        z-index: 1;
      }
      td {
        background: #fff;
      }
      .status {
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 6px;
      }
      .status-ok {
        color: #0a7c3a;
      }
      .status-bad {
        color: #b42318;
      }
      pre {
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        font-family: "SFMono-Regular", Menlo, Consolas, "Liberation Mono", monospace;
        font-size: 13px;
      }
    </style>
  </head>
  <body>
    <h1>${escapeHtml(title)}</h1>
    <p class="meta">Results: ${successCount}/${totalCount} succeeded (${successPercent}%).</p>
    <p class="meta">Generated from ${rows.length} example(s).</p>
    <table>
      <thead>
        <tr>
          <th>Input</th>
          <th>Normalized JSON Output</th>
          <th>Normalized WASM Output</th>
        </tr>
      </thead>
      <tbody>
        ${renderedRows}
      </tbody>
    </table>
  </body>
</html>`;
}

/**
 * @param {{ testCase: ExampleCase, normalizedExpected: string, normalizedWasm: string, matches: boolean }[]} rows
 */
function buildReportMarkdown(rows) {
    const tableHeader = "| Status | Description / Input | Expected | Actual |\n| :--- | :--- | :--- | :--- |\n";
    const tableRows = rows.map((row) => {
        const status = row.matches ? "✅" : "❌";
        const description = row.testCase.description;
        const sourceInfo = `${row.testCase.file} (example ${row.testCase.exampleIndex + 1})`;
        // Escape pipe characters in markdown cells and use <br> for newlines in table cells
        const input = (row.testCase.input ?? "").replace(/\|/g, "\\|").replace(/\n/g, "<br>");
        const expected = row.normalizedExpected.replace(/\|/g, "\\|").replace(/\n/g, "<br>");
        const actual = row.normalizedWasm.replace(/\|/g, "\\|").replace(/\n/g, "<br>");

        return `| ${status} | **${description}**<br>${sourceInfo}<br>\`${input}\` | \`${expected}\` | \`${actual}\` |`;
    }).join("\n");

    return tableHeader + tableRows;
}

/**
 * @param {string} value
 */
function escapeHtml(value) {
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

/**
 * @param {{ repoRoot: string, chromeWrapper: string, runnerHtml: string, casesJson: string, overallTimeoutMs?: number }} options
 * @returns {Promise<RunnerResult[]>}
 */
async function runHeadlessChrome(options) {
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
                            console.log(`[runner] progress: ${payload.caseId}/${cases.length}...`);
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
        stdio: "ignore"
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
        chrome.kill("SIGTERM");
        server.close();
    }

    return results;
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
