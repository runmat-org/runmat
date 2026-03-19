import { normalizePath } from "./path.js";
import type {
  RunMatFilesystemProvider,
  RunMatFilesystemDirEntry,
  RunMatFilesystemMetadata,
  RunMatFsFileType
} from "./provider-types.js";

export interface RemoteProviderOptions {
  baseUrl: string;
  authToken?: string;
  headers?: Record<string, string>;
  chunkBytes?: number;
  timeoutMs?: number;
}

interface MetadataPayload {
  fileType: RunMatFsFileType;
  len: number;
  modifiedAt?: string;
  readonly?: boolean;
  hash?: string;
  etag?: string;
}

interface DownloadUrlPayload {
  downloadUrl: string;
  expiresAt: string;
}

interface DirEntryPayload {
  path: string;
  fileName: string;
  fileType?: RunMatFsFileType;
}

const DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024;

export function createRemoteFsProvider(options: RemoteProviderOptions): RunMatFilesystemProvider {
  if (!options.baseUrl) {
    throw new Error("Remote provider requires options.baseUrl");
  }
  const chunkBytes = Math.max(64 * 1024, options.chunkBytes ?? DEFAULT_CHUNK_BYTES);
  const timeout = options.timeoutMs ?? 120_000;
  const client = new RemoteHttpClient(options.baseUrl, timeout, options.authToken, options.headers);

  return {
    async readFile(path) {
      const normalized = normalizePath(path);
      if (normalized === "/") {
        throw new Error("remote fs: cannot read '/'");
      }
      return client.readFileAll(normalized);
    },

    async readMany(paths) {
      return Promise.all(
        paths.map(async (path) => {
          try {
            return await this.readFile(path);
          } catch {
            return null;
          }
        })
      );
    },

    async writeFile(path, data) {
      const normalized = normalizePath(path);
      if (normalized === "/") {
        throw new Error("remote fs: cannot overwrite '/'");
      }
      const bytes = toUint8Array(data);
      let offset = 0;
      let chunkIndex = 0;
      while (offset < bytes.length) {
        const end = Math.min(offset + chunkBytes, bytes.length);
        const chunk = bytes.subarray(offset, end);
        const finalChunk = end === bytes.length;
        await client.writeChunk(normalized, offset, chunk, chunkIndex === 0, finalChunk);
        offset = end;
        chunkIndex += 1;
      }
      if (bytes.length === 0) {
        await client.writeChunk(normalized, 0, new Uint8Array(), true, true);
      }
    },

    async removeFile(path) {
      await client.delete("/fs/file", { path: normalizePath(path) });
    },

    async metadata(path) {
      return client.fetchMetadata(normalizePath(path));
    },

    async symlinkMetadata(path) {
      return client.fetchMetadata(normalizePath(path));
    },

    async readDir(path) {
      const payload = await client.getJson<DirEntryPayload[]>("/fs/dir", { path: normalizePath(path) });
      return payload.map(mapDirEntry);
    },

    async canonicalize(path) {
      const result = await client.getJson<{ path: string }>("/fs/canonicalize", {
        path: normalizePath(path)
      });
      return result.path;
    },

    async createDir(path) {
      await client.postJson("/fs/mkdir", { path: normalizePath(path), recursive: false });
    },

    async createDirAll(path) {
      await client.postJson("/fs/mkdir", { path: normalizePath(path), recursive: true });
    },

    async removeDir(path) {
      await client.delete("/fs/dir", { path: normalizePath(path), recursive: "false" });
    },

    async removeDirAll(path) {
      await client.delete("/fs/dir", { path: normalizePath(path), recursive: "true" });
    },

    async rename(from, to) {
      await client.postJson("/fs/rename", {
        from: normalizePath(from),
        to: normalizePath(to)
      });
    },

    async setReadonly(path, readonly) {
      await client.postJson("/fs/set-readonly", {
        path: normalizePath(path),
        readonly: Boolean(readonly)
      });
    }
  };
}

class RemoteHttpClient {
  constructor(
    private readonly baseUrl: string,
    private readonly timeoutMs: number,
    private readonly authToken?: string,
    private readonly extraHeaders: Record<string, string> = {}
  ) {}

  async fetchMetadata(path: string): Promise<RunMatFilesystemMetadata> {
    const payload = await this.getJson<MetadataPayload>("/fs/metadata", { path });
    const modifiedAt = payload.modifiedAt ? Date.parse(payload.modifiedAt) : undefined;
    const modified =
      modifiedAt !== undefined && !Number.isNaN(modifiedAt)
        ? Math.floor(modifiedAt / 1000)
        : undefined;
    return {
      fileType: payload.fileType,
      len: payload.len,
      modified,
      readonly: payload.readonly
    };
  }

  async readChunk(path: string, offset: number, length: number): Promise<Uint8Array> {
    const response = await this.send("GET", "/fs/read", {
      path,
      offset: String(offset),
      length: String(length)
    });
    const contentType = response.headers.get("Content-Type") ?? "";
    if (contentType.includes("application/json")) {
      const payload = (await response.json()) as DownloadUrlPayload;
      return this.downloadRange(payload.downloadUrl, offset, length);
    }
    return toUint8Array(await response.arrayBuffer());
  }

  async readFileAll(path: string): Promise<Uint8Array> {
    const response = await this.send("GET", "/fs/read", {
      path,
      offset: "0"
    });
    const contentType = response.headers.get("Content-Type") ?? "";
    if (contentType.includes("application/json")) {
      const payload = (await response.json()) as DownloadUrlPayload;
      return this.downloadFile(payload.downloadUrl);
    }
    return toUint8Array(await response.arrayBuffer());
  }

  async writeChunk(
    path: string,
    offset: number,
    data: Uint8Array,
    truncate: boolean,
    finalChunk: boolean
  ): Promise<void> {
    const query: Record<string, string> = {
      path,
      offset: String(offset)
    };
    if (truncate) {
      query.truncate = "true";
    }
    if (finalChunk) {
      query.final = "true";
    }
    await this.send("PUT", "/fs/write", query, data);
  }

  async delete(route: string, query: Record<string, string>): Promise<void> {
    await this.send("DELETE", route, query);
  }

  async postJson(route: string, body: Record<string, unknown>): Promise<void> {
    await this.send("POST", route, {}, JSON.stringify(body), "application/json");
  }

  async getJson<T>(route: string, query: Record<string, string>): Promise<T> {
    const response = await this.send("GET", route, query);
    return (await response.json()) as T;
  }

  private async downloadRange(url: string, offset: number, length: number): Promise<Uint8Array> {
    if (length === 0) {
      return new Uint8Array();
    }
    const end = offset + length - 1;
    const response = await this.send(
      "GET",
      url,
      {},
      undefined,
      undefined,
      { Range: `bytes=${offset}-${end}` },
      false,
      false
    );
    return toUint8Array(await response.arrayBuffer());
  }

  private async downloadFile(url: string): Promise<Uint8Array> {
    const response = await this.send("GET", url, {}, undefined, undefined, {}, false, false);
    return toUint8Array(await response.arrayBuffer());
  }

  private async send(
    method: string,
    route: string,
    query: Record<string, string>,
    body?: ArrayBufferView | ArrayBuffer | string,
    contentType?: string,
    extraHeaders: Record<string, string> = {},
    includeAuth = true,
    includeClientHeaders = true
  ): Promise<Response> {
    const url = route.startsWith("http") ? route : buildUrl(this.baseUrl, route, query);
    const headers: Record<string, string> = {
      ...(includeClientHeaders ? { "X-RunMat-Client": "remote-fs", ...this.extraHeaders } : {}),
      ...extraHeaders
    };
    if (this.authToken && includeAuth) {
      headers.Authorization = `Bearer ${this.authToken}`;
    }
    if (contentType) {
      headers["Content-Type"] = contentType;
    }

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);
    let response: Response;
    try {
      response = await fetch(url, {
        method,
        headers,
        body: body === undefined ? undefined : toRequestBody(body),
        signal: controller.signal
      });
    } catch (error) {
      const reason = error instanceof Error ? error.message : String(error);
      throw new Error(`remote fs http error: ${reason}`);
    } finally {
      clearTimeout(timeout);
    }

    if (response.status < 200 || response.status >= 300) {
      let detail = response.statusText;
      try {
        detail = await response.text();
      } catch {
        // ignore body parse issues and keep status text
      }
      throw new Error(
        `remote fs request failed (${response.status}): ${detail}`
      );
    }
    return response;
  }
}

function buildUrl(base: string, route: string, query: Record<string, string>): string {
  const url = new URL(route.replace(/^\//, ""), base.endsWith("/") ? base : `${base}/`);
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value !== undefined && value !== null) {
      params.append(key, value);
    }
  }
  url.search = params.toString();
  return url.toString();
}

function mapDirEntry(entry: DirEntryPayload): RunMatFilesystemDirEntry {
  return {
    path: entry.path,
    fileName: entry.fileName,
    fileType: entry.fileType
  };
}

function toUint8Array(data: Uint8Array | ArrayBuffer | ArrayBufferView): Uint8Array {
  if (data instanceof Uint8Array) {
    return data;
  }
  if (data instanceof ArrayBuffer) {
    return new Uint8Array(data);
  }
  return new Uint8Array(
    data.buffer,
    data.byteOffset,
    data.byteLength
  );
}

function toRequestBody(
  data: Uint8Array | ArrayBuffer | ArrayBufferView | string
): BodyInit {
  if (typeof data === "string") {
    return data;
  }
  if (data instanceof ArrayBuffer) {
    return data;
  }
  return cloneViewToArrayBuffer(data);
}

function cloneViewToArrayBuffer(data: Uint8Array | ArrayBufferView): ArrayBuffer {
  const view = data instanceof Uint8Array
    ? data
    : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const copy = new Uint8Array(view.byteLength);
  copy.set(view);
  return copy.buffer;
}
