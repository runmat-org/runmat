import { Buffer } from "node:buffer";
import {
  MessageChannel,
  receiveMessageOnPort,
  Worker
} from "node:worker_threads";

type WorkerResponse =
  | {
      ok: true;
      status: number;
      statusText: string;
      body: string;
    }
  | {
      ok: false;
      error: string;
    };

export function installNodeSyncXhr(): void {
  if (typeof globalThis.XMLHttpRequest !== "undefined") {
    return;
  }
  (globalThis as any).XMLHttpRequest = NodeSyncXmlHttpRequest;
}

class NodeSyncXmlHttpRequest {
  responseType: XMLHttpRequestResponseType = "";
  response: any = null;
  responseText = "";
  status = 0;
  statusText = "";
  timeout = 0;

  private method = "GET";
  private url = "";
  private headers = new Map<string, string>();

  open(method: string, url: string, async = true): void {
    if (async) {
      throw new Error("NodeSyncXmlHttpRequest only supports synchronous requests");
    }
    this.method = method;
    this.url = url;
  }

  setRequestHeader(name: string, value: string): void {
    this.headers.set(name, value);
  }

  abort(): void {
    // Synchronous shim – abort is a no-op.
  }

  send(body?: ArrayBufferView | ArrayBuffer | string | null): void {
    if (!this.url) {
      throw new Error("XMLHttpRequest: open() must be called before send()");
    }
    const worker = ensureWorker();
    const { port1, port2 } = new MessageChannel();
    const flag = new Int32Array(new SharedArrayBuffer(4));
    const payload = {
      method: this.method,
      url: this.url,
      headers: Array.from(this.headers.entries()),
      responseType: this.responseType,
      timeout: this.timeout,
      body: encodeBody(body)
    };

    worker.postMessage(
      {
        port: port2,
        flag: flag.buffer,
        request: payload
      },
      [port2]
    );

    const waitMs = this.timeout && this.timeout > 0 ? this.timeout : undefined;
    const waitResult = Atomics.wait(flag, 0, 0, waitMs);
    if (waitResult === "timed-out") {
      port1.close();
      throw new Error("timeout");
    }

    const message = receiveMessageOnPort(port1);
    port1.close();
    if (!message?.message) {
      throw new Error("sync xhr worker produced no response");
    }
    const result = message.message as WorkerResponse;
    if (!result.ok) {
      throw new Error(result.error || "remote fs http error");
    }

    this.status = result.status;
    this.statusText = result.statusText;
    const buffer = Buffer.from(result.body, "base64");
    if (this.responseType === "arraybuffer") {
      const slice = buffer.buffer.slice(
        buffer.byteOffset,
        buffer.byteOffset + buffer.byteLength
      );
      this.response = slice;
      this.responseText = "";
    } else if (this.responseType === "json") {
      const text = buffer.toString("utf8");
      this.responseText = text;
      this.response = text.length ? JSON.parse(text) : null;
    } else {
      this.responseText = buffer.toString("utf8");
      this.response = this.responseText;
    }
  }
}

function encodeBody(body?: ArrayBufferView | ArrayBuffer | string | null): string {
  if (!body) {
    return "";
  }
  if (typeof body === "string") {
    return Buffer.from(body, "utf8").toString("base64");
  }
  if (body instanceof ArrayBuffer) {
    return Buffer.from(body).toString("base64");
  }
  return Buffer.from(body.buffer, body.byteOffset, body.byteLength).toString("base64");
}

let workerInstance: Worker | null = null;

function ensureWorker(): Worker {
  if (workerInstance) {
    return workerInstance;
  }
  workerInstance = new Worker(syncWorkerSource, { eval: true });
  workerInstance.on("error", (err) => {
    console.error("[runmat:test] sync xhr worker error:", err);
  });
  workerInstance.on("exit", (code) => {
    if (code !== 0) {
      console.error("[runmat:test] sync xhr worker exited with code", code);
    }
  });
  workerInstance.unref();
  return workerInstance;
}

const syncWorkerSource = `
const { parentPort } = require("node:worker_threads");
const { Buffer } = require("node:buffer");

const fetchImpl = globalThis.fetch || require("node-fetch");

parentPort.on("message", async (message) => {
  const flag = new Int32Array(message.flag);
  const port = message.port;
  const req = message.request;
  const headers = {};
  for (const [key, value] of req.headers) {
    headers[key] = value;
  }
  const controller = req.timeout ? new AbortController() : null;
  const timer = req.timeout
    ? setTimeout(() => controller.abort(), req.timeout)
    : null;
  try {
    const response = await fetchImpl(req.url, {
      method: req.method,
      headers,
      body: req.body ? Buffer.from(req.body, "base64") : undefined,
      signal: controller ? controller.signal : undefined
    });
    const arrayBuffer = await response.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    port.postMessage({
      ok: true,
      status: response.status,
      statusText: response.statusText,
      body: buffer.toString("base64")
    });
  } catch (error) {
    port.postMessage({
      ok: false,
      error: error && error.message ? error.message : String(error)
    });
  } finally {
    if (timer) {
      clearTimeout(timer);
    }
    Atomics.store(flag, 0, 1);
    Atomics.notify(flag, 0);
    port.close();
  }
});
`;
