import { coerceFilesystemBytes, type InMemoryFsProviderOptions, type MemoryVolume } from "./memory.js";
import type { RunMatFilesystemProvider } from "./provider-types.js";
import { IndexedDbVolumeStore, openIndexedDbVolume } from "./indexeddb-store.js";
import { captureBrowserPublication, captureBrowserWorkspace, equalBrowserInventory, mergeBrowserPublication,
  prepareBrowserPublication, publicationDigest, publicationKey, reconcileReceipt, type BrowserWorkspaceInventory } from "./browser-workspace.js";
import { MAX_WORKSPACE_PUBLICATION_BYTES, MAX_WORKSPACE_PUBLICATION_EDITS } from "../generated/fea-workspace-contracts.js";

export interface IndexedDbProviderOptions extends InMemoryFsProviderOptions {
  dbName?: string;
  storeName?: string;
  version?: number;
  /** Retained for callers; authoritative operations commit before resolving. */
  flushDebounceMs?: number;
}
export interface IndexedDbFsHandle {
  provider: RunMatFilesystemProvider;
  flush(): Promise<void>;
  close(): void;
}

function inventory(volume: MemoryVolume): BrowserWorkspaceInventory {
  return new Map(volume.serialize().map(entry => [entry.path, entry.kind === "file"
    ? { kind: "file", bytes: entry.data!, readonly: entry.readonly }
    : { kind: "dir", readonly: entry.readonly }]));
}

export async function createIndexedDbFsHandle(options: IndexedDbProviderOptions = {}): Promise<IndexedDbFsHandle> {
  if (typeof indexedDB === "undefined") throw new Error("indexedDB API is unavailable in this environment");
  const storeName = options.storeName ?? "entries";
  const db = await openIndexedDbVolume(indexedDB, options.dbName ?? "runmat-fs", storeName, options.version ?? 2);
  const storage = new IndexedDbVolumeStore(db, storeName, `${storeName}_workspace_receipts`, options.now);
  const read = <T>(operation: (volume: MemoryVolume) => T) => storage.transaction(false, operation);
  const write = <T>(operation: (volume: MemoryVolume) => T) => storage.transaction(true, operation);
  const provider: RunMatFilesystemProvider = {
    readFile: path => storage.read([path], false, volume => volume.readFile(path)),
    readMany: paths => storage.read(paths, false, volume => volume.readMany(paths)),
    writeFile: (path, data) => {
      const owned = coerceFilesystemBytes(data).slice();
      return write(volume => volume.writeFile(path, owned));
    },
    atomicCreateNew: (path, data) => {
      const owned = data.slice();
      return write(volume => {
        const result = volume.atomicCreateNew(path, owned);
        return result.kind === "published" ? { kind: "published", result: { durability: "durable" } } : result;
      });
    },
    removeFile: path => write(volume => volume.removeFile(path)),
    metadata: path => storage.read([path], false, volume => volume.metadata(path)),
    symlinkMetadata: path => storage.read([path], false, volume => volume.metadata(path)),
    readDir: path => storage.read([path], true, volume => volume.readDir(path)),
    canonicalize: path => storage.read([path], false, volume => volume.canonicalize(path)),
    createDir: path => write(volume => { volume.createDir(path); }),
    createDirAll: path => write(volume => { volume.createDirAll(path); }),
    removeDir: path => write(volume => volume.removeDir(path)),
    removeDirAll: path => write(volume => volume.removeDirAll(path)),
    rename: (from, to) => write(volume => volume.rename(from, to)),
    setReadonly: (path, readonly) => write(volume => volume.setReadonly(path, readonly)),
    async captureWorkspace(request) {
      const original = await read(inventory);
      return captureBrowserWorkspace(original, request);
    },
    async lookupWorkspacePublication(request) {
      request = captureBrowserPublication(request);
      const digest = await publicationDigest(request);
      return storage.transaction(false, (_, receipts) => reconcileReceipt(receipts.get(publicationKey(request)), digest));
    },
    async publishWorkspace(request) {
      request = captureBrowserPublication(request);
      const digest = await publicationDigest(request);
      const key = publicationKey(request);
      const original = await storage.transaction(false, (volume, receipts) => ({ entries: inventory(volume), prior: reconcileReceipt(receipts.get(key), digest) }));
      if (original.prior) return original.prior;
      const prepared = await prepareBrowserPublication(original.entries, request);
      if (prepared.type === "conflict") return prepared.outcome;
      const committed = await storage.transaction(true, (volume, receipts) => {
        const prior = reconcileReceipt(receipts.get(key), digest);
        if (prior) return prior;
        const live = inventory(volume);
        if (!equalBrowserInventory(original.entries, live, request)) return null;
        if (receipts.size >= MAX_WORKSPACE_PUBLICATION_EDITS) throw new Error("Publication receipt budget exhausted");
        volume.replaceWorkspaceInventory(mergeBrowserPublication(prepared.entries, live, request));
        receipts.set(key, prepared.receipt);
        return structuredClone(prepared.receipt.outcome);
      });
      if (committed) return committed;
      const current = await provider.captureWorkspace!({ ...request, max_files: MAX_WORKSPACE_PUBLICATION_EDITS, max_bytes: MAX_WORKSPACE_PUBLICATION_BYTES });
      return { type: "conflict", current: current.revision };
    },
  };
  return { provider, flush: () => storage.flush(), close: () => storage.close() };
}
export async function createIndexedDbFsProvider(options?: IndexedDbProviderOptions): Promise<RunMatFilesystemProvider> {
  return (await createIndexedDbFsHandle(options)).provider;
}
