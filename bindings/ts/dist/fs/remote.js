import { normalizePath } from "./path.js";
const DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024;
export function createRemoteFsProvider(options) {
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
                client.writeChunk(normalized, offset, chunk, chunkIndex === 0);
                offset = end;
                chunkIndex += 1;
            }
            if (bytes.length === 0) {
                client.writeChunk(normalized, 0, new Uint8Array(), true);
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
            const payload = client.getJson("/fs/dir", { path: normalizePath(path) });
            return payload.map(mapDirEntry);
        },
        canonicalize(path) {
            const result = client.getJson("/fs/canonicalize", {
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
    baseUrl;
    timeoutMs;
    authToken;
    extraHeaders;
    constructor(baseUrl, timeoutMs, authToken, extraHeaders = {}) {
        this.baseUrl = baseUrl;
        this.timeoutMs = timeoutMs;
        this.authToken = authToken;
        this.extraHeaders = extraHeaders;
    }
    fetchMetadata(path) {
        const payload = this.getJson("/fs/metadata", { path });
        return {
            fileType: payload.fileType,
            len: payload.len,
            modified: payload.modified,
            readonly: payload.readonly
        };
    }
    readChunk(path, offset, length) {
        const response = this.send("GET", "/fs/read", {
            path,
            offset: String(offset),
            length: String(length)
        }, "arraybuffer");
        return new Uint8Array(response ?? []);
    }
    writeChunk(path, offset, data, truncate) {
        const query = {
            path,
            offset: String(offset)
        };
        if (truncate) {
            query.truncate = "true";
        }
        this.send("PUT", "/fs/write", query, "arraybuffer", data);
    }
    delete(route, query) {
        this.send("DELETE", route, query, "json");
    }
    postJson(route, body) {
        this.send("POST", route, {}, "json", JSON.stringify(body), "application/json");
    }
    getJson(route, query) {
        return this.send("GET", route, query, "json");
    }
    send(method, route, query, responseType, body, contentType) {
        const xhr = new XMLHttpRequest();
        xhr.open(method, buildUrl(this.baseUrl, route, query), false);
        xhr.timeout = this.timeoutMs;
        xhr.responseType = responseType;
        const headers = {
            "X-RunMat-Client": "remote-fs",
            ...this.extraHeaders
        };
        if (this.authToken) {
            headers.Authorization = `Bearer ${this.authToken}`;
        }
        if (contentType) {
            headers["Content-Type"] = contentType;
        }
        for (const [key, value] of Object.entries(headers)) {
            xhr.setRequestHeader(key, value);
        }
        const sendBody = body === undefined ? null : body;
        try {
            xhr.send(sendBody);
        }
        catch (error) {
            throw new Error(`remote fs http error: ${String(error)}`);
        }
        if (xhr.status < 200 || xhr.status >= 300) {
            throw new Error(`remote fs request failed (${xhr.status}): ${xhr.responseText || xhr.statusText}`);
        }
        return xhr.response;
    }
}
function buildUrl(base, route, query) {
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
function mapDirEntry(entry) {
    return {
        path: entry.path,
        fileName: entry.fileName,
        fileType: entry.fileType
    };
}
function toUint8Array(data) {
    if (data instanceof Uint8Array) {
        return data;
    }
    if (data instanceof ArrayBuffer) {
        return new Uint8Array(data);
    }
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
}
//# sourceMappingURL=remote.js.map