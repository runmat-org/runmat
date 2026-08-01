import type { RunMatPackageCacheProvider } from "./provider-types.js";

export interface BrowserTreeEntry {
  path: string;
  kind: "file" | "directory" | "symlink";
  digest?: string;
  byte_len: number | bigint;
  executable: boolean;
  link_target?: string;
}

export interface BrowserTreeManifest {
  digest: string;
  entries: BrowserTreeEntry[];
}

export interface BrowserMountEntry {
  path: string;
  kind: BrowserTreeEntry["kind"];
  byteLength: number;
  executable: boolean;
  linkTarget?: string;
}

/**
 * Read-only projection over a Rust-validated tree manifest and cache provider.
 * This class never mutates cache state and treats evicted payloads as recoverable misses.
 */
export class ImmutableBrowserPackageMount {
  private readonly entries: Map<string, BrowserTreeEntry>;

  constructor(
    public readonly manifest: BrowserTreeManifest,
    private readonly cache: RunMatPackageCacheProvider
  ) {
    this.entries = new Map(manifest.entries.map((entry) => [entry.path, entry]));
  }

  stat(path: string): BrowserMountEntry | null {
    const entry = this.entries.get(normalizeLookup(path));
    if (!entry) {
      return null;
    }
    return {
      path: entry.path,
      kind: entry.kind,
      byteLength: Number(entry.byte_len),
      executable: entry.executable,
      linkTarget: entry.link_target
    };
  }

  readDir(path = ""): BrowserMountEntry[] {
    const normalized = normalizeLookup(path);
    const prefix = normalized ? `${normalized}/` : "";
    const children = new Map<string, BrowserMountEntry>();
    for (const entry of this.entries.values()) {
      if (!entry.path.startsWith(prefix)) {
        continue;
      }
      const remainder = entry.path.slice(prefix.length);
      if (!remainder || remainder.includes("/")) {
        continue;
      }
      children.set(remainder, {
        path: entry.path,
        kind: entry.kind,
        byteLength: Number(entry.byte_len),
        executable: entry.executable,
        linkTarget: entry.link_target
      });
    }
    return Array.from(children.values()).sort((left, right) =>
      left.path.localeCompare(right.path)
    );
  }

  async readFile(path: string): Promise<Uint8Array> {
    const entry = this.resolveFile(normalizeLookup(path), new Set());
    if (!entry.digest) {
      throw new Error(`Package mount entry '${entry.path}' has no content digest`);
    }
    const bytes = await this.cache.readObjectBytes(entry.digest);
    if (!bytes) {
      const error = new Error(`Package cache payload ${entry.digest} was evicted`) as Error & {
        code?: string;
        digest?: string;
      };
      error.code = "PackageCacheMiss";
      error.digest = entry.digest;
      throw error;
    }
    if (BigInt(bytes.byteLength) !== BigInt(entry.byte_len)) {
      throw new Error(`Package cache payload ${entry.digest} has the wrong byte length`);
    }
    await verifyDigest(entry.digest, bytes);
    return bytes;
  }

  private resolveFile(path: string, visited: Set<string>): BrowserTreeEntry {
    if (visited.has(path)) {
      throw new Error(`Package mount contains a symlink cycle at '${path}'`);
    }
    visited.add(path);
    const entry = this.entries.get(path);
    if (!entry) {
      throw new Error(`Package mount entry '${path}' does not exist`);
    }
    if (entry.kind === "symlink") {
      if (!entry.link_target) {
        throw new Error(`Package mount symlink '${path}' has no target`);
      }
      return this.resolveFile(entry.link_target, visited);
    }
    if (entry.kind !== "file") {
      throw new Error(`Package mount entry '${path}' is not a file`);
    }
    return entry;
  }
}

async function verifyDigest(digest: string, bytes: Uint8Array): Promise<void> {
  const [algorithm, expected] = digest.split(":", 2);
  if (algorithm !== "sha256" || !expected || !/^[0-9a-f]{64}$/.test(expected)) {
    throw new Error(`Package cache payload digest '${digest}' is invalid`);
  }
  if (!globalThis.crypto?.subtle) {
    throw new Error("Web Crypto is required to verify package cache payloads");
  }
  const actualBytes = await globalThis.crypto.subtle.digest("SHA-256", bytes.slice().buffer);
  const actual = Array.from(new Uint8Array(actualBytes), (byte) =>
    byte.toString(16).padStart(2, "0")
  ).join("");
  if (actual !== expected) {
    throw new Error(`Package cache payload ${digest} failed digest verification`);
  }
}

function normalizeLookup(path: string): string {
  return path.replaceAll("\\", "/").replace(/^\/+|\/+$/g, "");
}
