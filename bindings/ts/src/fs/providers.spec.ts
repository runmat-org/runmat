import { describe, expect, it, afterEach, beforeEach } from "vitest";
import { tmpdir } from "node:os";
import { mkdtempSync, rmSync } from "node:fs";
import { Worker } from "node:worker_threads";
import path from "node:path";

import {
  createDefaultFsProvider,
  createInMemoryFsProvider,
  createIndexedDbFsHandle,
  createRemoteFsProvider
} from "../index.js";

const encoder = new TextEncoder();
const decoder = new TextDecoder();

describe("in-memory filesystem provider", () => {
  it("publishes complete objects once and returns independent winner bytes", async () => {
    const provider = createInMemoryFsProvider();
    const original = encoder.encode("first");
    expect(await provider.atomicCreateNew!("/object", original)).toEqual({ kind: "published", result: { durability: "atomic_visibility" } });
    original.fill(0);
    const loser = await provider.atomicCreateNew!("/object", encoder.encode("second"));
    expect(loser).toEqual({ kind: "existing", result: Array.from(encoder.encode("first")) });
    if (loser.kind === "existing") loser.result.fill(0);
    expect(toText(provider.readFile("/object"))).toBe("first");
    provider.createDir!("/directory");
    expect(() => provider.atomicCreateNew!("/directory", original)).toThrow();
  });

  it("supports basic file workflow", () => {
    const provider = createInMemoryFsProvider({
      initialFiles: {
        "/hello.txt": encoder.encode("world")
      }
    });

    const bytes = provider.readFile("/hello.txt");
    expect(toText(bytes)).toBe("world");

    provider.writeFile("/numbers.bin", new Uint8Array([1, 2, 3]));
    const meta = provider.metadata("/numbers.bin");
    expect(meta.fileType).toBe("file");
    expect(meta.len).toBe(3);

    provider.createDir?.("/tmp");
    provider.writeFile("/tmp/file.txt", encoder.encode("temp"));
    const entries = provider.readDir("/tmp");
    expect(entries.map((e) => e.fileName)).toEqual(["file.txt"]);

    provider.removeFile("/tmp/file.txt");
    provider.removeDir?.("/tmp");
    for (const operation of [
      () => provider.metadata("/tmp/file.txt"),
      () => provider.readFile("/tmp/file.txt"),
      () => provider.readDir("/tmp"),
      () => provider.removeFile("/tmp/file.txt"),
      () => provider.rename?.("/tmp/file.txt", "/replaced"),
      () => provider.setReadonly?.("/tmp/file.txt", true)
    ]) {
      expect(operation).toThrowError(expect.objectContaining({ name: "NotFoundError", status: 404 }));
    }
    // Existing directories are not missing files; do not turn a type mismatch
    // into a source-probe miss at the JS/WASM boundary.
    expect(() => provider.readFile("/")).toThrow("not a file");
  });

  it("seeds nested initial files without changing ordinary write parent checks", () => {
    const provider = createInMemoryFsProvider({
      initialFiles: {
        "/occupied/keep": "keep",
        "/source/child": "value"
      }
    });
    expect(toText(provider.readFile("/occupied/keep"))).toBe("keep");
    expect(toText(provider.readFile("/source/child"))).toBe("value");
    expect(provider.readDir("/").map((entry) => entry.fileName)).toEqual(["occupied", "source"]);
    expect(() => provider.writeFile("/missing/child", encoder.encode("no parent"))).toThrow();
    expect(() => createInMemoryFsProvider({
      initialFiles: { "/file": "file", "/file/child": "invalid" }
    })).toThrow();
  });
});

describe("indexeddb filesystem provider", () => {
  it("persists data across handles", async () => {
    const dbName = `runmat-test-${Date.now()}`;
    const handle1 = await createIndexedDbFsHandle({ dbName, flushDebounceMs: 0 });
    const provider1 = handle1.provider;

    await provider1.createDirAll?.("/data/cache");
    await provider1.writeFile("/data/cache/foo.txt", encoder.encode("cached"));
    expect(toText(await provider1.readFile("/data/cache/foo.txt"))).toBe("cached");

    await handle1.flush();
    handle1.close();

    const handle2 = await createIndexedDbFsHandle({ dbName, flushDebounceMs: 0 });
    const provider2 = handle2.provider;

    const payload = await provider2.readFile("/data/cache/foo.txt");
    expect(toText(payload)).toBe("cached");

    const listing = await provider2.readDir("/data/cache");
    expect(listing).toHaveLength(1);
    expect(listing[0]?.fileName).toBe("foo.txt");

    await handle2.flush();
    handle2.close();

    await deleteDatabase(dbName);
  });

  it("shares a live volume across same-page handles", async () => {
    const dbName = `runmat-test-live-${Date.now()}`;
    const handle1 = await createIndexedDbFsHandle({ dbName, flushDebounceMs: 0 });
    const handle2 = await createIndexedDbFsHandle({ dbName, flushDebounceMs: 0 });

    try {
      await handle2.provider.writeFile("/test.jpg", new Uint8Array([1, 2, 3]));
      expect(await handle1.provider.readFile("/test.jpg")).toEqual(new Uint8Array([1, 2, 3]));

      handle1.close();
      await handle2.provider.writeFile("/after-close.txt", encoder.encode("still-live"));
      await handle2.flush();
      handle2.close();

      const handle3 = await createIndexedDbFsHandle({ dbName, flushDebounceMs: 0 });
      try {
        expect(toText(await handle3.provider.readFile("/after-close.txt"))).toBe("still-live");
      } finally {
        handle3.close();
      }
    } finally {
      handle1.close();
      handle2.close();
      await deleteDatabase(dbName);
    }
  });

  it("permits independent handles with transaction-local clocks", async () => {
    const dbName = `runmat-test-options-${Date.now()}`;
    const a = await createIndexedDbFsHandle({ dbName, now: () => 1 });
    const b = await createIndexedDbFsHandle({ dbName, now: () => 2 });
    try {
      await a.provider.writeFile("/a", encoder.encode("a"));
      await b.provider.writeFile("/b", encoder.encode("b"));
      expect((await b.provider.metadata("/a")).modified).toBe(1);
      expect((await a.provider.metadata("/b")).modified).toBe(2);
    } finally { a.close(); b.close(); await deleteDatabase(dbName); }
  });
});

describe("default filesystem provider", () => {
  it("shares one live provider across default calls", async () => {
    const provider1 = await createDefaultFsProvider();
    const provider2 = await createDefaultFsProvider();

    expect(provider2).toBe(provider1);
    await provider1.writeFile("/default-shared.txt", encoder.encode("shared"));
    expect(toText(await provider2.readFile("/default-shared.txt"))).toBe("shared");
  });

  it("uses persistent storage when IndexedDB is available", async () => {
    const provider1 = await createDefaultFsProvider();
    await provider1.createDirAll?.("/auto");
    await provider1.writeFile("/auto/state.txt", encoder.encode("auto"));

    await delay(100);

    const provider2 = await createDefaultFsProvider();
    const payload = await provider2.readFile("/auto/state.txt");
    expect(toText(payload)).toBe("auto");
  });
});

describe("remote filesystem provider", () => {
  let server: RemoteTestServer;

  beforeEach(async () => {
    server = new RemoteTestServer();
    await server.start();
  });

  afterEach(async () => {
    if (server) {
      await server.close();
    }
  });

  it("streams multi-chunk payloads", async () => {
    const provider = createRemoteFsProvider({
      baseUrl: server.baseUrl,
      chunkBytes: 512,
      timeoutMs: 5_000
    });

    await provider.createDirAll?.("/reports");
    const payload = new Uint8Array(5_000).map((_, idx) => (idx * 31) & 0xff);
    await provider.writeFile("/reports/data.bin", payload);
    const metadata = await provider.metadata("/reports/data.bin");
    expect(metadata.len).toBe(payload.length);

    const readBack = new Uint8Array(await provider.readFile("/reports/data.bin"));
    expect(readBack).toEqual(payload);

    await provider.removeFile("/reports/data.bin");
    await expect(provider.metadata("/reports/data.bin")).rejects.toThrow();
  });

  it("respects auth tokens when provided", async () => {
    await server.close();
    server = new RemoteTestServer("secret");
    await server.start();

    const authed = createRemoteFsProvider({
      baseUrl: server.baseUrl,
      authToken: "secret"
    });
    await authed.writeFile("/secured/info.txt", encoder.encode("ok"));
    expect(toText(await authed.readFile("/secured/info.txt"))).toBe("ok");

    const unauth = createRemoteFsProvider({ baseUrl: server.baseUrl });
    await expect(unauth.readDir("/")).rejects.toThrow(/unauthorized/i);
    await expect(unauth.readMany?.(["/secured/info.txt"]) ?? Promise.resolve([])).rejects.toThrow(/unauthorized/i);
  });

  it("propagates readonly flags", async () => {
    const provider = createRemoteFsProvider({ baseUrl: server.baseUrl });
    await provider.writeFile("/reports/lock.txt", encoder.encode("locked"));
    await provider.setReadonly?.("/reports/lock.txt", true);
    const meta = await provider.metadata("/reports/lock.txt");
    expect(meta.readonly).toBe(true);
    await expect(provider.writeFile("/reports/lock.txt", encoder.encode("fail"))).rejects.toThrow();
  });

  it("keeps readMany stable when method is unbound", async () => {
    const provider = createRemoteFsProvider({ baseUrl: server.baseUrl });
    await provider.writeFile("/reports/chunk.bin", encoder.encode("payload"));

    const readMany = provider.readMany;
    expect(typeof readMany).toBe("function");
    if (!readMany) {
      throw new Error("readMany is required for remote provider");
    }

    const [bytes] = await readMany(["/reports/chunk.bin"]);
    expect(bytes).toBeInstanceOf(Uint8Array);
    expect(toText(bytes as Uint8Array)).toBe("payload");
  });
});

function toText(data: Uint8Array | ArrayBuffer): string {
  const view = data instanceof Uint8Array ? data : new Uint8Array(data);
  return decoder.decode(view);
}

class RemoteTestServer {
  baseUrl = "";
  private readonly root: string;
  private worker?: Worker;

  constructor(private readonly authToken?: string) {
    this.root = mkdtempSync(path.join(tmpdir(), "runmat-remote-"));
  }

  async start(): Promise<void> {
    if (this.worker) {
      return;
    }
    this.worker = new Worker(remoteServerWorkerSource, { eval: true });
    await new Promise<void>((resolve, reject) => {
      const worker = this.worker!;
      const handleMessage = (msg: any) => {
        if (msg?.type === "started") {
          this.baseUrl = msg.baseUrl;
          cleanup();
          resolve();
        } else if (msg?.type === "error") {
          cleanup();
          reject(new Error(msg.error));
        }
      };
      const handleError = (err: Error) => {
        cleanup();
        reject(err);
      };
      const cleanup = () => {
        worker.off("message", handleMessage);
        worker.off("error", handleError);
      };
      worker.on("message", handleMessage);
      worker.on("error", handleError);
      worker.postMessage({ type: "start", root: this.root, authToken: this.authToken });
    });
  }

  async close(): Promise<void> {
    if (!this.worker) {
      rmSync(this.root, { recursive: true, force: true });
      return;
    }
    await new Promise<void>((resolve) => {
      const worker = this.worker!;
      const handleMessage = (msg: any) => {
        if (msg?.type === "stopped") {
          cleanup();
          resolve();
        }
      };
      const cleanup = () => worker.off("message", handleMessage);
      worker.on("message", handleMessage);
      worker.postMessage({ type: "shutdown" });
    });
    await this.worker.terminate();
    this.worker = undefined;
    rmSync(this.root, { recursive: true, force: true });
  }
}

const remoteServerWorkerSource = `
const { parentPort } = require("node:worker_threads");
const { createServer } = require("node:http");
const { readFileSync, writeFileSync, readdirSync, statSync, mkdirSync, rmSync, renameSync, chmodSync, existsSync } = require("node:fs");
const path = require("node:path");

let server = null;
let rootDir = "";
let expectedAuth = null;

parentPort.on("message", async (msg) => {
  if (msg.type === "start") {
    rootDir = msg.root;
    expectedAuth = msg.authToken || null;
    server = createServer(async (req, res) => {
      if (!req.url) {
        res.statusCode = 400;
        res.end("missing url");
        return;
      }
      if (expectedAuth && req.headers["authorization"] !== \`Bearer \${expectedAuth}\`) {
        res.statusCode = 401;
        res.end("unauthorized");
        return;
      }
      const url = new URL(req.url, "http://127.0.0.1");
      const pathParam = normalizeRemotePath(url.searchParams.get("path") || "/");
      try {
        if (req.method === "GET" && url.pathname === "/fs/metadata") {
          const stats = statSync(resolve(pathParam));
          const payload = {
            fileType: stats.isDirectory() ? "dir" : stats.isFile() ? "file" : "other",
            len: stats.size,
            modified: Math.floor(stats.mtimeMs / 1000),
            readonly: (stats.mode & 0o200) === 0
          };
          sendJson(res, 200, payload);
          return;
        }

        if (req.method === "GET" && url.pathname === "/fs/dir") {
          const entries = readdirSync(resolve(pathParam));
          const payload = entries.map((name) => {
            const full = resolve(path.join(pathParam, name));
            const stats = statSync(full);
            return {
              path: normalizeRemotePath(path.join(pathParam, name)),
              fileName: name,
              fileType: stats.isDirectory() ? "dir" : stats.isFile() ? "file" : "other"
            };
          });
          sendJson(res, 200, payload);
          return;
        }

        if (req.method === "GET" && url.pathname === "/fs/canonicalize") {
          const resolved = path.resolve(resolve(pathParam));
          const relative = path.relative(rootDir, resolved);
          sendJson(res, 200, { path: \`/\${relative}\` });
          return;
        }

        if (req.method === "GET" && url.pathname === "/fs/read") {
          const offset = Number(url.searchParams.get("offset") || "0");
          const lengthParam = url.searchParams.get("length");
          const data = readFileSync(resolve(pathParam));
          const length = lengthParam ? Number(lengthParam) : data.length - offset;
          const chunk = data.subarray(offset, offset + length);
          res.writeHead(200, { "Content-Type": "application/octet-stream" });
          res.end(chunk);
          return;
        }

        if (req.method === "PUT" && url.pathname === "/fs/write") {
          const body = await readBody(req);
          const offset = Number(url.searchParams.get("offset") || "0");
          const truncate = url.searchParams.get("truncate") === "true";
          const full = resolve(pathParam);
          let data = truncate || !existsSync(full) ? Buffer.alloc(0) : Buffer.from(readFileSync(full));
          const required = offset + body.length;
          if (required > data.length) {
            const extended = Buffer.alloc(required);
            data.copy(extended);
            data = extended;
          }
          body.copy(data, offset);
          mkdirSync(path.dirname(full), { recursive: true });
          writeFileSync(full, data);
          res.statusCode = 204;
          res.end();
          return;
        }

        if (req.method === "POST" && url.pathname === "/fs/mkdir") {
          const payload = JSON.parse((await readBody(req)).toString() || "{}");
          const full = resolve(payload.path || "/");
          if (payload.recursive) {
            mkdirSync(full, { recursive: true });
          } else {
            mkdirSync(full);
          }
          res.statusCode = 204;
          res.end();
          return;
        }

        if (req.method === "DELETE" && url.pathname === "/fs/file") {
          rmSync(resolve(pathParam), { force: true });
          res.statusCode = 204;
          res.end();
          return;
        }

        if (req.method === "DELETE" && url.pathname === "/fs/dir") {
          const recursive = url.searchParams.get("recursive") === "true";
          rmSync(resolve(pathParam), { recursive, force: true });
          res.statusCode = 204;
          res.end();
          return;
        }

        if (req.method === "POST" && url.pathname === "/fs/rename") {
          const payload = JSON.parse((await readBody(req)).toString() || "{}");
          const from = resolve(payload.from || "/");
          const to = resolve(payload.to || "/");
          mkdirSync(path.dirname(to), { recursive: true });
          renameSync(from, to);
          res.statusCode = 204;
          res.end();
          return;
        }

        if (req.method === "POST" && url.pathname === "/fs/set-readonly") {
          const payload = JSON.parse((await readBody(req)).toString() || "{}");
          const target = resolve(payload.path || "/");
          const readonly = Boolean(payload.readonly);
          const mode = readonly ? 0o444 : 0o666;
          try {
            chmodSync(target, mode);
          } catch {
            // ignore permission errors
          }
          res.statusCode = 204;
          res.end();
          return;
        }

        res.statusCode = 404;
        res.end();
      } catch (error) {
        res.statusCode = 500;
        res.end(error && error.message ? error.message : String(error));
      }
    });
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      const baseUrl = \`http://127.0.0.1:\${address.port}\`;
      parentPort.postMessage({ type: "started", baseUrl });
    });
  } else if (msg.type === "shutdown") {
    if (!server) {
      parentPort.postMessage({ type: "stopped" });
      return;
    }
    server.close(() => {
      parentPort.postMessage({ type: "stopped" });
    });
  }
});

function resolve(remotePath) {
  const normalized = normalizeRemotePath(remotePath).replace(/^\\//, "");
  return path.join(rootDir, normalized);
}

function normalizeRemotePath(input) {
  if (!input || input === "/") {
    return "/";
  }
  const segments = [];
  for (const part of input.split("/")) {
    if (!part || part === ".") continue;
    if (part === "..") {
      segments.pop();
    } else {
      segments.push(part);
    }
  }
  return "/" + segments.join("/");
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

function sendJson(res, status, payload) {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(payload));
}
`;

function deleteDatabase(name: string): Promise<void> {
  if (typeof indexedDB === "undefined") {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const request = indexedDB.deleteDatabase(name);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error ?? new Error("Failed to delete IndexedDB database"));
  });
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
