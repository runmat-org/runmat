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
    readFile(path) {
      const normalized = normalizePath(path);
      if (normalized === "/") {
        throw new Error("remote fs: cannot read '/'");
      }
      const meta = client.fetchMetadata(normalized);
      const length = meta.len ?? 0;
      if (length === 0) {
        return new Uint8Array();
      }
      const buffer = new Uint8Array(length);
      let offset = 0;
      while (offset < length) {
        const size = Math.min(chunkBytes, length - offset);
        const chunk = client.readChunk(normalized, offset, size);
        buffer.set(chunk, offset);
        offset += chunk.length;
      }
      return buffer;
    },

    writeFile(path, data) {
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
        client.writeChunk(normalized, offset, chunk, chunkIndex === 0, finalChunk);
        offset = end;
        chunkIndex += 1;
      }
      if (bytes.length === 0) {
        client.writeChunk(normalized, 0, new Uint8Array(), true, true);
      }
    },

    removeFile(path) {
      client.delete("/fs/file", { path: normalizePath(path) });
    },

    metadata(path) {
      return client.fetchMetadata(normalizePath(path));
    },

    symlinkMetadata(path) {
      return client.fetchMetadata(normalizePath(path));
    },

    readDir(path) {
      const payload = client.getJson<DirEntryPayload[]>("/fs/dir", { path: normalizePath(path) });
      return payload.map(mapDirEntry);
    },

    canonicalize(path) {
      const result = client.getJson<{ path: string }>("/fs/canonicalize", {
        path: normalizePath(path)
      });
      return result.path;
    },

    createDir(path) {
      client.postJson("/fs/mkdir", { path: normalizePath(path), recursive: false });
    },

    createDirAll(path) {
      client.postJson("/fs/mkdir", { path: normalizePath(path), recursive: true });
    },

    removeDir(path) {
      client.delete("/fs/dir", { path: normalizePath(path), recursive: "false" });
    },

    removeDirAll(path) {
      client.delete("/fs/dir", { path: normalizePath(path), recursive: "true" });
    },

    rename(from, to) {
      client.postJson("/fs/rename", {
        from: normalizePath(from),
        to: normalizePath(to)
      });
    },

    setReadonly(path, readonly) {
      client.postJson("/fs/set-readonly", {
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

  fetchMetadata(path: string): RunMatFilesystemMetadata {
    const payload = this.getJson<MetadataPayload>("/fs/metadata", { path });
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

  readChunk(path: string, offset: number, length: number): Uint8Array {
    const xhr = this.sendRaw("GET", "/fs/read", {
      path,
      offset: String(offset),
      length: String(length)
    }, "arraybuffer");
    const contentType = xhr.getResponseHeader("Content-Type") ?? "";
    if (contentType.includes("application/json")) {
      const payload = JSON.parse(arrayBufferToString(xhr.response as ArrayBuffer)) as DownloadUrlPayload;
      return this.downloadRange(payload.downloadUrl, offset, length);
    }
    return new Uint8Array((xhr.response as ArrayBuffer) ?? []);
  }

  writeChunk(
    path: string,
    offset: number,
    data: Uint8Array,
    truncate: boolean,
    finalChunk: boolean
  ): void {
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
    this.send("PUT", "/fs/write", query, "arraybuffer", data);
  }

  delete(route: string, query: Record<string, string>): void {
    this.send("DELETE", route, query, "json");
  }

  postJson(route: string, body: Record<string, unknown>): void {
    this.send("POST", route, {}, "json", JSON.stringify(body), "application/json");
  }

  getJson<T>(route: string, query: Record<string, string>): T {
    return this.send("GET", route, query, "json") as T;
  }

  private downloadRange(url: string, offset: number, length: number): Uint8Array {
    if (length === 0) {
      return new Uint8Array();
    }
    const end = offset + length - 1;
    const xhr = this.sendRaw(
      "GET",
      url,
      {},
      "arraybuffer",
      undefined,
      undefined,
      { Range: `bytes=${offset}-${end}` },
      false,
      false
    );
    return new Uint8Array((xhr.response as ArrayBuffer) ?? []);
  }

  private send(
    method: string,
    route: string,
    query: Record<string, string>,
    responseType: XMLHttpRequestResponseType,
    body?: ArrayBufferView | ArrayBuffer | string,
    contentType?: string
  ): any {
    return this.sendRaw(method, route, query, responseType, body, contentType).response;
  }

  private sendRaw(
    method: string,
    route: string,
    query: Record<string, string>,
    responseType: XMLHttpRequestResponseType,
    body?: ArrayBufferView | ArrayBuffer | string,
    contentType?: string,
    extraHeaders: Record<string, string> = {},
    includeAuth = true,
    includeClientHeaders = true
  ): XMLHttpRequest {
    const xhr = new XMLHttpRequest();
    const url = route.startsWith("http") ? route : buildUrl(this.baseUrl, route, query);
    xhr.open(method, url, false);
    xhr.timeout = this.timeoutMs;
    xhr.responseType = responseType;
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
    for (const [key, value] of Object.entries(headers)) {
      xhr.setRequestHeader(key, value);
    }

    const sendBody: Document | XMLHttpRequestBodyInit | null =
      body === undefined ? null : (body as XMLHttpRequestBodyInit);

    try {
      xhr.send(sendBody);
    } catch (error) {
      throw new Error(`remote fs http error: ${String(error)}`);
    }

    if (xhr.status < 200 || xhr.status >= 300) {
      throw new Error(
        `remote fs request failed (${xhr.status}): ${xhr.responseText || xhr.statusText}`
      );
    }
    return xhr;
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

function arrayBufferToString(buffer: ArrayBuffer): string {
  const decoder = new TextDecoder("utf-8");
  return decoder.decode(new Uint8Array(buffer));
}
