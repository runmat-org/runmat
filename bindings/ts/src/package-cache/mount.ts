import type { RunMatPackageCacheProvider } from "./provider-types.js";
import type {
  RunMatFilesystemDirEntry,
  RunMatFilesystemMetadata,
  RunMatFilesystemProvider
} from "../fs/provider-types.js";

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

  statFollowing(path: string): BrowserMountEntry | null {
    const normalized = normalizeLookup(path);
    if (!this.entries.has(normalized)) {
      return null;
    }
    const entry = this.resolveEntry(normalized, new Set());
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
    const entry = this.resolveEntry(path, visited);
    if (entry.kind !== "file") {
      throw new Error(`Package mount entry '${path}' is not a file`);
    }
    return entry;
  }

  private resolveEntry(path: string, visited: Set<string>): BrowserTreeEntry {
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
      return this.resolveEntry(entry.link_target, visited);
    }
    return entry;
  }
}

/**
 * Filesystem composition adapter that projects immutable package trees beneath a
 * reserved virtual namespace and delegates every workspace path to the host provider.
 */
export class BrowserPackageMountFilesystem implements RunMatFilesystemProvider {
  private readonly mounts = new Map<
    string,
    { mount: ImmutableBrowserPackageMount; dispose?: () => void }
  >();

  constructor(private readonly workspace: RunMatFilesystemProvider) {}

  clear(): void {
    for (const value of this.mounts.values()) {
      value.dispose?.();
    }
    this.mounts.clear();
  }

  register(snapshot: PackageSnapshotMountInput, cache: RunMatPackageCacheProvider): string {
    const digest = snapshot.source.tree_digest;
    const key = digest.replace(":", "_");
    const root = `/__runmat/packages/${key}`;
    this.mounts.set(root, {
      mount: new ImmutableBrowserPackageMount(snapshot.tree, cache)
    });
    return root;
  }

  registerVolatile(snapshot: PackageSnapshotMountInput): string {
    const cache = new VolatilePackageObjectStore(snapshot.blobs ?? []);
    const digest = snapshot.source.tree_digest;
    const key = digest.replace(":", "_");
    const root = `/__runmat/packages/${key}`;
    this.mounts.set(root, {
      mount: new ImmutableBrowserPackageMount(snapshot.tree, cache),
      dispose: () => cache.clear()
    });
    return root;
  }

  async readFile(path: string): Promise<Uint8Array | ArrayBuffer> {
    const resolved = this.resolve(path);
    return resolved
      ? resolved.mount.readFile(resolved.relative)
      : this.workspace.readFile(path);
  }

  async readMany(paths: string[]): Promise<Array<Uint8Array | ArrayBuffer | null>> {
    return Promise.all(
      paths.map(async (path) => {
        try {
          return await this.readFile(path);
        } catch {
          return null;
        }
      })
    );
  }

  writeFile(
    path: string,
    data: Uint8Array | ArrayBuffer | ArrayBufferView
  ): void | Promise<void> {
    this.assertMutable(path);
    return this.workspace.writeFile(path, data);
  }

  removeFile(path: string): void | Promise<void> {
    this.assertMutable(path);
    return this.workspace.removeFile(path);
  }

  async metadata(path: string): Promise<RunMatFilesystemMetadata> {
    const normalized = normalizeAbsolute(path);
    const direct = this.mounts.get(normalized)?.mount;
    if (direct) {
      return { fileType: "directory", len: 0, readonly: true };
    }
    const resolved = this.resolve(normalized);
    if (!resolved) {
      return this.workspace.metadata(path);
    }
    const entry = resolved.mount.statFollowing(resolved.relative);
    if (!entry) {
      throw missing(path);
    }
    return {
      fileType: entry.kind === "directory" ? "directory" : entry.kind,
      len: entry.byteLength,
      readonly: true
    };
  }

  async symlinkMetadata(path: string): Promise<RunMatFilesystemMetadata> {
    const normalized = normalizeAbsolute(path);
    const direct = this.mounts.get(normalized)?.mount;
    if (direct) {
      return { fileType: "directory", len: 0, readonly: true };
    }
    const resolved = this.resolve(normalized);
    if (!resolved) {
      return this.workspace.symlinkMetadata?.(path) ?? this.workspace.metadata(path);
    }
    const entry = resolved.mount.stat(resolved.relative);
    if (!entry) {
      throw missing(path);
    }
    return {
      fileType: entry.kind === "directory" ? "directory" : entry.kind,
      len: entry.byteLength,
      readonly: true
    };
  }

  async readDir(path: string): Promise<RunMatFilesystemDirEntry[]> {
    const normalized = normalizeAbsolute(path);
    const direct = this.mounts.get(normalized)?.mount;
    const resolved = direct
      ? { root: normalized, mount: direct, relative: "" }
      : this.resolve(normalized);
    if (!resolved) {
      return this.workspace.readDir(path);
    }
    return resolved.mount.readDir(resolved.relative).map((entry) => ({
      path: `${resolved.root}/${entry.path}`,
      fileName: entry.path.split("/").at(-1) ?? entry.path,
      fileType: entry.kind === "directory" ? "directory" : entry.kind
    }));
  }

  canonicalize(path: string): string | Promise<string> {
    const normalized = normalizeAbsolute(path);
    if (this.mounts.has(normalized) || this.resolve(normalized)) {
      return normalized;
    }
    return this.workspace.canonicalize?.(path) ?? normalized;
  }

  createDir(path: string): void | Promise<void> {
    this.assertMutable(path);
    return required(this.workspace.createDir, "createDir").call(this.workspace, path);
  }

  createDirAll(path: string): void | Promise<void> {
    this.assertMutable(path);
    return required(this.workspace.createDirAll, "createDirAll").call(this.workspace, path);
  }

  removeDir(path: string): void | Promise<void> {
    this.assertMutable(path);
    return required(this.workspace.removeDir, "removeDir").call(this.workspace, path);
  }

  removeDirAll(path: string): void | Promise<void> {
    this.assertMutable(path);
    return required(this.workspace.removeDirAll, "removeDirAll").call(this.workspace, path);
  }

  rename(from: string, to: string): void | Promise<void> {
    this.assertMutable(from);
    this.assertMutable(to);
    return required(this.workspace.rename, "rename").call(this.workspace, from, to);
  }

  setReadonly(path: string, readonly: boolean): void | Promise<void> {
    this.assertMutable(path);
    return required(this.workspace.setReadonly, "setReadonly").call(
      this.workspace,
      path,
      readonly
    );
  }

  private resolve(path: string):
    | { root: string; mount: ImmutableBrowserPackageMount; relative: string }
    | undefined {
    const normalized = normalizeAbsolute(path);
    for (const [root, value] of this.mounts) {
      if (normalized.startsWith(`${root}/`)) {
        return {
          root,
          mount: value.mount,
          relative: normalized.slice(root.length + 1)
        };
      }
    }
    return undefined;
  }

  private assertMutable(path: string): void {
    const normalized = normalizeAbsolute(path);
    if (this.mounts.has(normalized) || this.resolve(normalized)) {
      throw new Error(`Package mount '${normalized}' is read-only`);
    }
  }
}

export interface PackageSnapshotMountInput {
  source: {
    tree_digest: string;
  };
  tree: BrowserTreeManifest;
  blobs?: Array<{ digest: string; bytes: number[] | Uint8Array }>;
}

export type GitSnapshotMountInput = PackageSnapshotMountInput;

class VolatilePackageObjectStore implements RunMatPackageCacheProvider {
  private readonly bytes = new Map<string, Uint8Array>();

  constructor(blobs: Array<{ digest: string; bytes: number[] | Uint8Array }>) {
    for (const blob of blobs) {
      this.bytes.set(blob.digest, Uint8Array.from(blob.bytes));
      blob.bytes.fill(0);
    }
  }

  clear(): void {
    for (const bytes of this.bytes.values()) {
      bytes.fill(0);
    }
    this.bytes.clear();
  }

  async readObjectBytes(digest: string): Promise<Uint8Array | null> {
    return this.bytes.get(digest)?.slice() ?? null;
  }

  async snapshot(): Promise<null> {
    return null;
  }

  async initialize(): Promise<never> {
    throw new Error("volatile private package storage is read-only");
  }

  async commit(): Promise<never> {
    throw new Error("volatile private package storage is read-only");
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

function normalizeAbsolute(path: string): string {
  const normalized = `/${normalizeLookup(path)}`;
  return normalized === "/" ? "/" : normalized.replace(/\/+$/g, "");
}

function required<T extends (...args: never[]) => unknown>(
  value: T | undefined,
  name: string
): T {
  if (!value) {
    throw new Error(`Workspace filesystem provider does not implement ${name}`);
  }
  return value;
}

function missing(path: string): Error {
  const error = new Error(`Package mount path '${path}' does not exist`) as Error & {
    code?: string;
  };
  error.code = "ENOENT";
  return error;
}
