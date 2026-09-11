export function createRunnerHtml(timeoutMs, concurrency, logIntervalMs, enableGpu = true) {
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
        "  let errorIdentifier = \"\";",
        "  let figurePngBase64 = \"\";",
        "  let figureImageError = \"\";",
        "  let plotSurfaceId = null;",
        "  const normalizeErrorText = (errorValue) => {",
        "    if (typeof errorValue === \"string\") return errorValue;",
        "    if (!errorValue || typeof errorValue !== \"object\") return String(errorValue || \"\");",
        "    if (typeof errorValue.message === \"string\" && errorValue.message.length > 0) {",
        "      return errorValue.message;",
        "    }",
        "    try {",
        "      return JSON.stringify(errorValue);",
        "    } catch (err) {",
        "      return String(errorValue);",
        "    }",
        "  };",
        "  const patchWebGpuRequestDevice = () => {",
        "    if (typeof navigator !== \"object\" || !navigator || !navigator.gpu) return;",
        "    const ctor = typeof GPUAdapter === \"function\" ? GPUAdapter : null;",
        "    const proto = ctor && ctor.prototype ? ctor.prototype : null;",
        "    if (!proto) return;",
        "    const current = proto.requestDevice;",
        "    if (typeof current !== \"function\" || current.__runmatPatched) return;",
        "    const wrapped = function(descriptor) {",
        "      if (!descriptor || typeof descriptor !== \"object\") {",
        "        return current.call(this, descriptor);",
        "      }",
        "      const safeDescriptor = { ...descriptor };",
        "      if (safeDescriptor.requiredLimits && typeof safeDescriptor.requiredLimits === \"object\") {",
        "        const limits = { ...safeDescriptor.requiredLimits };",
        "        delete limits.maxInterStageShaderComponents;",
        "        safeDescriptor.requiredLimits = limits;",
        "      }",
        "      return current.call(this, safeDescriptor);",
        "    };",
        "    wrapped.__runmatPatched = true;",
        "    proto.requestDevice = wrapped;",
        "  };",
        "  try {",
        "    patchWebGpuRequestDevice();",
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
        "    const fsProvider = createInMemoryFsProvider();",
        "    fsProvider.createDirAll(\"/tmp\");",
        "    const fixture = testCase.fixture || \"None\";",
        "    if (fixture !== \"None\") {",
        "      if (!fixture || typeof fixture !== \"object\" || !fixture.Filesystem) {",
        "        throw new Error(`The ${testCase.harness} browser adapter cannot materialize its declared fixture`);",
        "      }",
        "      const encoder = new TextEncoder();",
        "      for (const entry of fixture.Filesystem.entries) {",
        "        if (entry.Directory) {",
        "          fsProvider.createDir(`/${entry.Directory.relative_path}`);",
        "        } else if (entry.File) {",
        "          const content = entry.File.content;",
        "          const bytes = Object.prototype.hasOwnProperty.call(content, \"Utf8\")",
        "            ? encoder.encode(content.Utf8)",
        "            : new Uint8Array(content.Bytes);",
        "          fsProvider.writeFile(`/${entry.File.relative_path}`, bytes);",
        "        } else {",
        "          throw new Error(\"Browser filesystem fixture contains an unsupported entry\");",
        "        }",
        "      }",
        "    }",
        "    if (fsProvider && typeof module.registerFsProvider === \"function\") {",
        "      module.registerFsProvider(fsProvider);",
        "    }",
        "    const session = await module.initRunMat({",
        "      telemetryConsent: false,",
        `      enableGpu: ${enableGpu ? "true" : "false"},`,
        "      languageCompat: \"runmat\",",
        "      fsProvider: fsProvider || undefined",
        "    });",
        "    try {",
        "      if (typeof OffscreenCanvas !== \"undefined\" && typeof module.createPlotSurface === \"function\") {",
        "        const bootstrapCanvas = new OffscreenCanvas(16, 16);",
        "        plotSurfaceId = await Promise.resolve(module.createPlotSurface(bootstrapCanvas));",
        "        if (typeof module.resizePlotSurface === \"function\" && typeof plotSurfaceId === \"number\") {",
        "          await Promise.resolve(module.resizePlotSurface(plotSurfaceId, 16, 16, 1));",
        "        }",
        "      }",
        "    } catch (_surfaceErr) {",
        "      plotSurfaceId = null;",
        "    }",
        "    try {",
        "      try {",
        "        await Promise.resolve(session.executeRequest({ source: { kind: \"text\", name: \"<example-setup>\", text: \"cd('/')\" }, compatibility: testCase.compatibility.toLowerCase() }));",
        "      } catch (err) {}",
        "      const execResult = await Promise.resolve(session.executeRequest({ source: { kind: \"text\", name: `<builtin-example:${testCase.builtin}>`, text: testCase.input }, compatibility: testCase.compatibility.toLowerCase() }));",
        "      if (execResult && Array.isArray(execResult.stdout)) {",
        "        stdoutText = execResult.stdout.map((entry) => entry.text || \"\").join(\"\\n\");",
        "      }",
        "      if (execResult && typeof execResult.valueText === \"string\") {",
        "        valueText = execResult.valueText;",
        "      }",
        "      if (execResult && execResult.error != null) {",
        "        errorText = normalizeErrorText(execResult.error);",
        "        if (typeof execResult.error.identifier === \"string\") errorIdentifier = execResult.error.identifier;",
        "      }",
        "      if (testCase.isPlotExample === true) {",
        "        try {",
        "          if (typeof module.renderFigureImage === \"function\") {",
        "            const handle = typeof module.currentFigureHandle === \"function\" ? await Promise.resolve(module.currentFigureHandle()) : undefined;",
        "            const numericHandle = typeof handle === \"number\" && Number.isFinite(handle) ? handle : undefined;",
        "            if (typeof plotSurfaceId === \"number\" && Number.isFinite(plotSurfaceId) && typeof numericHandle === \"number\" && typeof module.presentFigureOnSurface === \"function\") {",
        "              try { await Promise.resolve(module.presentFigureOnSurface(plotSurfaceId, numericHandle)); } catch (_presentErr) {}",
        "            }",
        "            const pngBytes = await Promise.resolve(module.renderFigureImage(numericHandle, 1280, 720));",
        "            const bytes = pngBytes instanceof Uint8Array ? pngBytes : new Uint8Array(pngBytes || []);",
        "            if (bytes.length > 0) {",
        "              let binary = \"\";",
        "              const chunkSize = 0x8000;",
        "              for (let offset = 0; offset < bytes.length; offset += chunkSize) {",
        "                const chunk = bytes.subarray(offset, Math.min(offset + chunkSize, bytes.length));",
        "                binary += String.fromCharCode(...chunk);",
        "              }",
        "              figurePngBase64 = btoa(binary);",
        "            } else {",
        "              figureImageError = \"renderFigureImage returned no bytes\";",
        "            }",
        "          }",
        "        } catch (err) {",
        "          let extra = \"\";",
        "          try { extra = JSON.stringify(err); } catch (_e) {}",
        "          figureImageError = normalizeErrorText(err) + (extra ? ` :: ${extra}` : \"\");",
        "        }",
        "      }",
        "    } finally {",
        "      try {",
        "        if (typeof plotSurfaceId === \"number\" && Number.isFinite(plotSurfaceId) && typeof module.destroyPlotSurface === \"function\") {",
        "          module.destroyPlotSurface(plotSurfaceId);",
        "        }",
        "      } catch (_destroyErr) {}",
        "      if (typeof session.dispose === \"function\") {",
        "        session.dispose();",
        "      }",
        "    }",
        "  } catch (err) {",
        "    errorText = err instanceof Error ? err.message : String(err);",
        "  }",
        "  self.postMessage({ id: testCase.id, stdoutText, valueText, errorText, errorIdentifier, figurePngBase64, figureImageError });",
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
      const wasmModuleUrl = new URL("/__runner__/wasm.js", location.href).href;

      sendLog({
        type: "runner-script-start",
        href: location.href,
        ua: navigator && navigator.userAgent ? navigator.userAgent : "unknown"
      });

      let logWorkerSnapshot = () => {};

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

        logWorkerSnapshot = (reason) => {
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
            const location = event && event.filename
              ? " at " + event.filename + ":" + (event.lineno || 0) + ":" + (event.colno || 0)
              : "";
            const stack = event && event.error && typeof event.error.stack === "string"
              ? "\\n" + event.error.stack
              : "";
            finalize({
              id: testCase.id,
              stdoutText: "",
              valueText: "",
              errorText: "Worker error: " + (event && event.message ? event.message : "unknown") + location + stack
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
