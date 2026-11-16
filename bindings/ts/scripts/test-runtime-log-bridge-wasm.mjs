// The browser runtime delivers its log lines and spans to subscribeRuntimeLog and
// subscribeTraceEvents. Runs the built web package in Node, in the order a host
// uses: initRunMat, then subscribe, then run a .m script.
//   RUNMAT_WASM_PKG=<dir with runmat_wasm_web.js> overrides ../dist/pkg-web.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";

const pkgDir = process.env.RUNMAT_WASM_PKG
  ? pathToFileURL(`${process.env.RUNMAT_WASM_PKG.replace(/\/$/, "")}/`)
  : new URL("../dist/pkg-web/", import.meta.url);
const native = await import(new URL("runmat_wasm_web.js", pkgDir).href);
native.initSync({ module: await readFile(new URL("runmat_wasm_web_bg.wasm", pkgDir)) });

function memoryFs() {
  const files = new Map();
  const dirs = new Set(["/"]);
  const notFound = (path) => Object.assign(new Error(`NotFound: ${path}`), { name: "NotFoundError", code: "NotFound" });
  const metadata = (path) => {
    if (dirs.has(path)) return { fileType: "directory", len: 0, modified: Date.now(), readonly: false };
    const data = files.get(path);
    if (!data) throw notFound(path);
    return { fileType: "file", len: data.length, modified: Date.now(), readonly: false };
  };
  const createDirAll = (path) => {
    let current = "";
    for (const part of path.split("/").filter(Boolean)) dirs.add((current = `${current}/${part}`));
  };
  return {
    readFile: async (path) => { const data = files.get(path); if (!data) throw notFound(path); return data.slice(); },
    writeFile: async (path, data) => { createDirAll(path.slice(0, path.lastIndexOf("/"))); files.set(path, new Uint8Array(data)); },
    removeFile: (path) => { if (!files.delete(path)) throw notFound(path); },
    metadata,
    symlinkMetadata: metadata,
    readDir: (path) => {
      if (!dirs.has(path)) throw notFound(path);
      const prefix = path === "/" ? "/" : `${path}/`;
      const names = [...dirs, ...files.keys()].filter((p) => p !== path && p.startsWith(prefix) && !p.slice(prefix.length).includes("/"));
      return names.map((p) => ({ path: p, fileName: p.slice(prefix.length), fileType: dirs.has(p) ? "directory" : "file" }));
    },
    canonicalize: (path) => path,
    createDir: (path) => { dirs.add(path); },
    createDirAll,
    removeDir: (path) => { dirs.delete(path); },
    removeDirAll: (path) => { dirs.delete(path); },
    rename: () => { throw new Error("rename is not used by this test"); },
    setReadonly: () => {},
  };
}

const sessionOptions = (extra = {}) => ({
  applicationSettings: "isolatedMemory",
  enableGpu: false,
  enableJit: false,
  telemetryConsent: false,
  fsProvider: memoryFs(),
  ...extra,
});

const logs = [];
const traces = [];
const run = async (session) => {
  logs.length = 0;
  traces.length = 0;
  const result = await session.executeRequest({
    source: { kind: "text", name: "probe.m", text: "x = 1 + 2;\ny = x * 3;\ndisp(y)\n" },
  });
  assert.equal(result?.error ?? null, null, `the script ran: ${JSON.stringify(result?.error)}`);
  return {
    spans: traces.filter((event) => event.ph === "B").map((event) => event.name),
    ends: traces.filter((event) => event.ph === "E").length,
    started: logs.filter((record) => record.message === "Execution started").length,
  };
};

// 1. A host that passes no logLevel gets the runtime's default level (warn). The
//    trace feed still carries the run's spans.
const quiet = await native.initRunMat(sessionOptions());
native.subscribeRuntimeLog((record) => logs.push(record));
native.subscribeTraceEvents((events) => traces.push(...events));
let seen = await run(quiet);
console.log("default level:", JSON.stringify({ logs: logs.length, traces: traces.length }));
assert.ok(seen.spans.includes("runmat.execute"), `runmat.execute span reached the trace feed: ${seen.spans.length} spans`);
assert.ok(seen.spans.includes("runtime.compile"), "the core compile span reached the trace feed");
assert.equal(seen.ends, seen.spans.length, "every span that began also ended");
assert.equal(seen.started, 0, "info lines stay out of the log feed at the warn level");

// 2. setLogFilter raises the log feed's level.
native.setLogFilter("info");
seen = await run(quiet);
console.log("setLogFilter(info):", JSON.stringify({ logs: logs.length, traces: traces.length }));
assert.equal(seen.started, 1, "the info line reached the log feed");
assert.ok(logs.every((record) => typeof record.target === "string" && record.ts), "log records carry target and time");

// 3. The quietest log level leaves the trace feed whole.
native.setLogFilter("error");
seen = await run(quiet);
console.log("setLogFilter(error):", JSON.stringify({ logs: logs.length, traces: traces.length }));
assert.equal(seen.started, 0);
assert.ok(seen.spans.includes("runmat.execute"), "spans survive the error log level");

// 4. A later session's explicit logLevel replaces the live level.
quiet.dispose?.();
const verbose = await native.initRunMat(sessionOptions({ logLevel: "debug" }));
seen = await run(verbose);
console.log("second session logLevel=debug:", JSON.stringify({ logs: logs.length, traces: traces.length }));
assert.equal(seen.started, 1, "the second session's logLevel took effect");

console.log("runtime log bridge: ok");
