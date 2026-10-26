import { MemoryWorkspaceAuthority } from "./workspace-memory.js";
import type { BrowserWorkspaceEntry } from "./browser-workspace.js";
import type { WorkspaceCaptureRequest, WorkspacePublicationRequest } from "../generated/fea-workspace-contracts.js";
import type { AtomicCreateResult } from "../generated/collaboration-contracts.js";
import { basename, dirname, isSubPath, normalizePath } from "./path.js";
import { notifyCommittedMutation } from "./mutation-notification.js";
import { notFound } from "./errors.js";
import type {
  RunMatFilesystemDirEntry,
  RunMatFilesystemMetadata,
  RunMatFilesystemProvider
} from "./provider-types.js";

type Entry = FileEntry | DirEntry;

enum RenamePolicy {
  PreserveDestination,
  ReplaceCompatible,
}

interface FileEntry {
  kind: "file";
  data: Uint8Array;
  readonly: boolean;
  modified: number;
}

interface DirEntry {
  kind: "dir";
  children: Set<string>;
  readonly: boolean;
  modified: number;
}

export interface InMemoryFsProviderOptions {
  /** Bootstrap files; missing parent directories are created only while seeding. */
  initialFiles?: Record<string, Uint8Array | ArrayBuffer | ArrayBufferView | string>;
  now?: () => number;
}

export interface VolumeSnapshotEntry {
  path: string;
  kind: "file" | "dir";
  data?: Uint8Array;
  readonly: boolean;
  modified: number;
  children?: string[];
}

export class MemoryVolume {
  private entries = new Map<string, Entry>();
  readonly workspace = new MemoryWorkspaceAuthority(
    () => new Map([...this.entries].map(([path, entry]): [string, BrowserWorkspaceEntry] =>
      [path, entry.kind === "file" ? { kind: "file", bytes: entry.data, readonly: entry.readonly }
        : { kind: "dir", readonly: entry.readonly }])),
    entries => this.replaceWorkspaceInventory(entries),
  );
  private readonly now: () => number;
  private readonly encoder = new TextEncoder();

  constructor(private readonly opts: InMemoryFsProviderOptions = {}) {
    this.now = opts.now ?? (() => Date.now());
    this.entries.set("/", this.makeDirEntry());
    if (opts.initialFiles) {
      for (const [path, value] of Object.entries(opts.initialFiles)) {
        // Seed paths describe a ready-to-use tree; ordinary writeFile still
        // requires its parent directory to exist.
        this.createDirAll(dirname(normalizeAbsolute(path)));
        this.writeFile(path, this.coerceBuffer(value));
      }
    }
  }

  /** Observers run synchronously after commit; their errors use a separate reporting sink. */
  createProvider(
    onMutate?: () => void,
    onObserverError?: (error: unknown) => void
  ): RunMatFilesystemProvider {
    return new MemoryProvider(this, onMutate, onObserverError);
  }

  serialize(): VolumeSnapshotEntry[] {
    const items: VolumeSnapshotEntry[] = [];
    for (const [path, entry] of this.entries.entries()) {
      if (entry.kind === "file") {
        items.push({
          path,
          kind: "file",
          data: entry.data.slice(),
          readonly: entry.readonly,
          modified: entry.modified
        });
      } else {
        items.push({
          path,
          kind: "dir",
          readonly: entry.readonly,
          modified: entry.modified,
          children: Array.from(entry.children)
        });
      }
    }
    return items;
  }

  load(entries: VolumeSnapshotEntry[]): void {
    this.entries.clear();
    this.entries.set("/", this.makeDirEntry());
    // Ensure deterministic load order: parents before children.
    entries
      .slice()
      .sort((a, b) => depth(a.path) - depth(b.path))
      .forEach((snapshot) => {
        const path = snapshot.path;
        const entryKind = snapshot.kind;
        if (normalizeAbsolute(path) === "/") {
          const root = this.entries.get("/") as DirEntry;
          root.readonly = snapshot.readonly;
          root.modified = snapshot.modified;
          root.children = new Set<string>(snapshot.children ?? []);
          return;
        }
        if (entryKind === "dir") {
          this.createDir(path, false, snapshot.readonly, snapshot.modified);
        } else {
          if (!snapshot.data) {
            throw new Error(`Snapshot missing data for file ${path}`);
          }
          this.writeFile(path, snapshot.data.slice(), snapshot.readonly, snapshot.modified);
        }
      });
  }

  readFile(path: string): Uint8Array {
    const file = this.getFile(path);
    return file.data.slice();
  }

  readMany(paths: string[]): Array<Uint8Array | null> {
    return paths.map((path) => {
      try {
        return this.readFile(path);
      } catch {
        return null;
      }
    });
  }

  atomicCreateNew(path: string, data: Uint8Array): AtomicCreateResult {
    const normalized = normalizeAbsolute(path);
    const existing = this.entries.get(normalized);
    if (existing) {
      if (existing.kind !== "file") throw new Error(`Path ${normalized} is a directory`);
      return { kind: "existing", result: Array.from(existing.data) };
    }
    // No await: observation and insertion belong to one volume operation.
    this.writeFile(normalized, data);
    return { kind: "published", result: { durability: "atomic_visibility" } };
  }

  writeFile(path: string, data: Uint8Array, readonly = false, modified?: number): void {
    const normalized = normalizeAbsolute(path);
    if (readonly) {
      // Allow bootstrap of readonly entries during load.
    }
    const parent = this.getDir(dirname(normalized));
    const existing = this.entries.get(normalized);
    const timestamp = modified ?? this.now();
    if (existing) {
      if (existing.kind !== "file") {
        throw new Error(`Path ${normalized} is a directory`);
      }
      if (existing.readonly && !readonly) {
        throw new Error(`Path ${normalized} is readonly`);
      }
      existing.data = data.slice();
      existing.modified = timestamp;
      existing.readonly = readonly;
    } else {
      parent.children.add(basename(normalized));
      this.entries.set(normalized, {
        kind: "file",
        data: data.slice(),
        readonly,
        modified: timestamp
      });
    }
  }

  removeFile(path: string): void {
    const normalized = normalizeAbsolute(path);
    const entry = this.entries.get(normalized);
    if (!entry) {
      throw notFound(normalized, "File");
    }
    if (entry.kind !== "file") {
      throw new Error(`Path ${normalized} is not a file`);
    }
    if (entry.readonly) {
      throw new Error(`File is readonly: ${normalized}`);
    }
    this.entries.delete(normalized);
    const parent = this.getDir(dirname(normalized));
    parent.children.delete(basename(normalized));
  }

  metadata(path: string): RunMatFilesystemMetadata {
    const normalized = normalizeAbsolute(path);
    const entry = this.entries.get(normalized);
    if (!entry) {
      throw notFound(normalized);
    }
    if (entry.kind === "file") {
      return {
        fileType: "file",
        len: entry.data.length,
        modified: entry.modified,
        readonly: entry.readonly
      };
    }
    return {
      fileType: "dir",
      len: 0,
      modified: entry.modified,
      readonly: entry.readonly
    };
  }

  readDir(path: string): RunMatFilesystemDirEntry[] {
    const dir = this.getDir(path);
    const normalized = normalizeAbsolute(path);
    return Array.from(dir.children)
      .sort()
      .map((name) => {
        const childPath = normalized === "/" ? `/${name}` : `${normalized}/${name}`;
        const entry = this.entries.get(childPath);
        const fileType = entry?.kind === "dir" ? "dir" : entry?.kind === "file" ? "file" : "unknown";
        return {
          path: childPath,
          fileName: name,
          fileType
        };
      });
  }

  canonicalize(path: string): string {
    return normalizeAbsolute(path);
  }

  createDir(path: string, failOnExists = true, readonly = false, modified?: number): boolean {
    const normalized = normalizeAbsolute(path);
    if (this.entries.has(normalized)) {
      if (failOnExists) {
        throw new Error(`Path already exists: ${normalized}`);
      }
      const existing = this.entries.get(normalized);
      if (existing?.kind !== "dir") {
        throw new Error(`File exists at ${normalized}`);
      }
      return false;
    }
    const parent = this.getDir(dirname(normalized));
    parent.children.add(basename(normalized));
    this.entries.set(normalized, {
      kind: "dir",
      children: new Set<string>(),
      readonly,
      modified: modified ?? this.now()
    });
    return true;
  }

  createDirAll(path: string): boolean {
    const normalized = normalizeAbsolute(path);
    if (normalized === "/") {
      return false;
    }
    const parts = normalized.split("/").filter(Boolean);
    let current = "";
    let changed = false;
    for (const part of parts) {
      current = `${current}/${part}`;
      if (!this.entries.has(current)) {
        this.createDir(current, false);
        changed = true;
      }
    }
    return changed;
  }

  removeDir(path: string): void {
    const normalized = normalizeAbsolute(path);
    if (normalized === "/") {
      throw new Error("Cannot remove root directory");
    }
    const dir = this.getDir(normalized);
    if (dir.readonly) {
      throw new Error(`Directory is readonly: ${normalized}`);
    }
    if (dir.children.size > 0) {
      throw new Error(`Directory not empty: ${normalized}`);
    }
    this.entries.delete(normalized);
    const parent = this.getDir(dirname(normalized));
    parent.children.delete(basename(normalized));
  }

  removeDirAll(path: string): void {
    const normalized = normalizeAbsolute(path);
    if (normalized === "/") {
      throw new Error("Cannot remove root directory");
    }
    const targets = Array.from(this.entries.keys())
      .filter((entryPath) => entryPath === normalized || entryPath.startsWith(`${normalized}/`))
      .sort((a, b) => depth(b) - depth(a));
    for (const entryPath of targets) {
      const entry = this.entries.get(entryPath);
      if (!entry) {
        continue;
      }
      if (entry.kind === "dir" && entry.readonly) {
        throw new Error(`Directory is readonly: ${entryPath}`);
      }
      if (entry.kind === "file" && entry.readonly) {
        throw new Error(`File is readonly: ${entryPath}`);
      }
      this.entries.delete(entryPath);
      if (entryPath !== normalized) {
        const parentPath = dirname(entryPath);
        const parent = this.entries.get(parentPath);
        if (parent?.kind === "dir") {
          parent.children.delete(basename(entryPath));
        }
      }
    }
    const parent = this.getDir(dirname(normalized));
    parent.children.delete(basename(normalized));
  }

  rename(from: string, to: string): void {
    this.renameWithPolicy(from, to, RenamePolicy.PreserveDestination);
  }

  renameReplace(from: string, to: string): void {
    this.renameWithPolicy(from, to, RenamePolicy.ReplaceCompatible);
  }

  private renameWithPolicy(from: string, to: string, policy: RenamePolicy): void {
    const src = normalizeAbsolute(from);
    const dst = normalizeAbsolute(to);
    const entry = this.entries.get(src);
    if (!entry) {
      throw notFound(src);
    }
    if (src === dst) {
      return;
    }
    if (src === "/" || dst === "/") {
      throw new Error("Cannot rename the filesystem root");
    }
    if (entry.readonly) {
      throw new Error(`Path is readonly: ${src}`);
    }
    const destination = this.entries.get(dst);
    if (destination) {
      if (policy === RenamePolicy.PreserveDestination) {
        throw new Error(`Destination exists: ${dst}`);
      }
      if (destination.readonly || destination.kind !== entry.kind) {
        throw new Error("Destination is readonly or has an incompatible type");
      }
      if (destination.kind === "dir" && destination.children.size !== 0) {
        throw new Error("Destination directory is not empty");
      }
    }
    if (entry.kind === "dir" && isSubPath(dst, src)) {
      throw new Error("Cannot move directory into its own subtree");
    }
    const parentFrom = this.getDir(dirname(src));
    const parentTo = this.getDir(dirname(dst));

    const updates: Array<{ oldPath: string; newPath: string }> = [];
    for (const key of this.entries.keys()) {
      if (key === src || key.startsWith(`${src}/`)) {
        const suffix = key.slice(src.length);
        updates.push({ oldPath: key, newPath: `${dst}${suffix}` });
      }
    }
    updates.sort((a, b) => depth(a.oldPath) - depth(b.oldPath));
    parentFrom.children.delete(basename(src));
    parentTo.children.add(basename(dst));
    for (const { oldPath, newPath } of updates) {
      const current = this.entries.get(oldPath);
      if (!current) {
        continue;
      }
      this.entries.delete(oldPath);
      this.entries.set(newPath, current);
    }
  }

  setReadonly(path: string, readonly: boolean): void {
    const normalized = normalizeAbsolute(path);
    const entry = this.entries.get(normalized);
    if (!entry) {
      throw notFound(normalized);
    }
    entry.readonly = readonly;
    entry.modified = this.now();
  }

  replaceWorkspaceInventory(inventory: Map<string, BrowserWorkspaceEntry>): void {
    const next = new Map<string, Entry>();
    const timestamp = this.now();
    for (const [path, entry] of inventory) {
      const previous = this.entries.get(path);
      if (entry.kind === "dir") {
        next.set(path, { kind: "dir", readonly: !!entry.readonly,
          modified: previous?.modified ?? timestamp, children: new Set() });
      } else {
        const unchanged = previous?.kind === "file" && previous.data.length === entry.bytes.length
          && previous.data.every((byte, index) => byte === entry.bytes[index]);
        next.set(path, { kind: "file", data: entry.bytes.slice(), readonly: !!entry.readonly,
          modified: unchanged ? previous.modified : timestamp });
      }
    }
    for (const path of next.keys()) if (path !== "/") {
      const parent = next.get(dirname(path));
      if (parent?.kind !== "dir") throw new Error(`Workspace parent missing: ${path}`);
      parent.children.add(basename(path));
    }
    this.entries = next;
  }

  private coerceBuffer(value: Uint8Array | ArrayBuffer | ArrayBufferView | string): Uint8Array {
    if (typeof value === "string") {
      return this.encoder.encode(value);
    }
    if (value instanceof Uint8Array) {
      return value.slice();
    }
    if (value instanceof ArrayBuffer) {
      return new Uint8Array(value.slice(0));
    }
    if (ArrayBuffer.isView(value)) {
      return new Uint8Array(value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength));
    }
    throw new Error("Unsupported buffer type");
  }

  private getFile(path: string): FileEntry {
    const normalized = normalizeAbsolute(path);
    const entry = this.entries.get(normalized);
    if (!entry) {
      throw notFound(normalized, "File");
    }
    if (entry.kind !== "file") {
      throw new Error(`Path is not a file: ${normalized}`);
    }
    return entry;
  }

  private getDir(path: string): DirEntry {
    const normalized = normalizeAbsolute(path);
    const entry = this.entries.get(normalized);
    if (!entry) {
      throw notFound(normalized, "Directory");
    }
    if (entry.kind !== "dir") {
      throw new Error(`Path is not a directory: ${normalized}`);
    }
    return entry;
  }

  private makeDirEntry(): DirEntry {
    return {
      kind: "dir",
      children: new Set<string>(),
      readonly: false,
      modified: this.now()
    };
  }

}

class MemoryProvider implements RunMatFilesystemProvider {
  constructor(
    private readonly volume: MemoryVolume,
    private readonly onMutate?: () => void,
    private readonly onObserverError?: (error: unknown) => void
  ) {}

  private notifyMutation(): void {
    notifyCommittedMutation(this.onMutate, this.onObserverError);
  }

  captureWorkspace(request: WorkspaceCaptureRequest) {
    return this.volume.workspace.captureWorkspace(request);
  }
  lookupWorkspacePublication(request: WorkspacePublicationRequest) {
    return this.volume.workspace.lookupWorkspacePublication(request);
  }
  async publishWorkspace(request: WorkspacePublicationRequest) {
    const result = await this.volume.workspace.publishWorkspace(request);
    if (result.type === "published") this.notifyMutation();
    return result;
  }

  readFile(path: string): Uint8Array {
    return this.volume.readFile(path);
  }

  readMany(paths: string[]): Array<Uint8Array | null> {
    return this.volume.readMany(paths);
  }

  atomicCreateNew(path: string, data: Uint8Array): AtomicCreateResult {
    const result = this.volume.atomicCreateNew(path, data);
    if (result.kind === "published") this.notifyMutation();
    return result;
  }

  writeFile(path: string, data: Uint8Array | ArrayBuffer | ArrayBufferView): void {
    this.volume.writeFile(path, coerceFilesystemBytes(data));
    this.notifyMutation();
  }

  removeFile(path: string): void {
    this.volume.removeFile(path);
    this.notifyMutation();
  }

  metadata(path: string): RunMatFilesystemMetadata {
    return this.volume.metadata(path);
  }

  symlinkMetadata(path: string): RunMatFilesystemMetadata {
    return this.volume.metadata(path);
  }

  readDir(path: string): RunMatFilesystemDirEntry[] {
    return this.volume.readDir(path);
  }

  canonicalize(path: string): string {
    return this.volume.canonicalize(path);
  }

  createDir(path: string): void {
    const changed = this.volume.createDir(path);
    if (changed) {
      this.notifyMutation();
    }
  }

  createDirAll(path: string): void {
    const changed = this.volume.createDirAll(path);
    if (changed) {
      this.notifyMutation();
    }
  }

  removeDir(path: string): void {
    this.volume.removeDir(path);
    this.notifyMutation();
  }

  removeDirAll(path: string): void {
    this.volume.removeDirAll(path);
    this.notifyMutation();
  }

  rename(from: string, to: string): void {
    this.volume.rename(from, to);
    this.notifyMutation();
  }

  renameReplace(from: string, to: string): void {
    this.volume.renameReplace(from, to);
    this.notifyMutation();
  }

  setReadonly(path: string, readonly: boolean): void {
    this.volume.setReadonly(path, readonly);
    this.notifyMutation();
  }
}

export function createInMemoryFsProvider(options?: InMemoryFsProviderOptions): RunMatFilesystemProvider {
  const volume = new MemoryVolume(options);
  return volume.createProvider();
}

export function coerceFilesystemBytes(value: Uint8Array | ArrayBuffer | ArrayBufferView): Uint8Array {
  if (value instanceof Uint8Array) {
    return value;
  }
  if (value instanceof ArrayBuffer) {
    return new Uint8Array(value);
  }
  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength));
  }
  throw new Error("Unsupported buffer type");
}

function normalizeAbsolute(path: string): string {
  const normalized = normalizePath(path);
  if (!normalized.startsWith("/")) {
    return normalized === "." ? "/" : `/${normalized}`;
  }
  return normalized === "" ? "/" : normalized;
}

function depth(path: string): number {
  if (path === "/" || path === ".") {
    return 0;
  }
  return normalizeAbsolute(path)
    .split("/")
    .filter(Boolean).length;
}
