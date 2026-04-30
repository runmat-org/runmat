import { MemoryVolume, type InMemoryFsProviderOptions, type VolumeSnapshotEntry } from "./memory.js";
import type { RunMatFilesystemProvider } from "./provider-types.js";

export interface IndexedDbProviderOptions extends InMemoryFsProviderOptions {
  dbName?: string;
  storeName?: string;
  version?: number;
  flushDebounceMs?: number;
}

export interface IndexedDbFsHandle {
  provider: RunMatFilesystemProvider;
  flush(): Promise<void>;
  close(): void;
}

const DEFAULT_DB_NAME = "runmat-fs";
const DEFAULT_STORE_NAME = "entries";
const SHARED_BACKINGS_SYMBOL = Symbol.for("runmat.fs.indexedDbBackings");

interface SharedIndexedDbBacking {
  db: IDBDatabase;
  storeName: string;
  volume: MemoryVolume;
  flushDebounceMs: number;
  refCount: number;
  pendingFlush: Promise<void> | null;
  flushTimer: ReturnType<typeof setTimeout> | null;
  closed: boolean;
  dirty: boolean;
}

function sharedBackings(): Map<string, Promise<SharedIndexedDbBacking>> {
  const globalState = globalThis as typeof globalThis & {
    [SHARED_BACKINGS_SYMBOL]?: Map<string, Promise<SharedIndexedDbBacking>>;
  };
  if (!globalState[SHARED_BACKINGS_SYMBOL]) {
    globalState[SHARED_BACKINGS_SYMBOL] = new Map();
  }
  return globalState[SHARED_BACKINGS_SYMBOL];
}

export async function createIndexedDbFsHandle(
  options: IndexedDbProviderOptions = {}
): Promise<IndexedDbFsHandle> {
  const idb = getIndexedDb();
  const dbName = options.dbName ?? DEFAULT_DB_NAME;
  const storeName = options.storeName ?? DEFAULT_STORE_NAME;
  const version = options.version ?? 1;
  const key = sharedBackingKey(dbName, storeName, version);
  const backings = sharedBackings();
  let backingPromise = backings.get(key);
  if (!backingPromise) {
    backingPromise = createSharedBacking(idb, dbName, storeName, version, options).catch((error) => {
      backings.delete(key);
      throw error;
    });
    backings.set(key, backingPromise);
  }
  const backing = await backingPromise;
  backing.refCount += 1;
  return new IndexedDbHandle(key, backing);
}

export async function createIndexedDbFsProvider(
  options?: IndexedDbProviderOptions
): Promise<RunMatFilesystemProvider> {
  const handle = await createIndexedDbFsHandle(options);
  return handle.provider;
}

class IndexedDbHandle implements IndexedDbFsHandle {
  public readonly provider: RunMatFilesystemProvider;
  private closed = false;

  constructor(
    private readonly key: string,
    private readonly backing: SharedIndexedDbBacking
  ) {
    this.provider = backing.volume.createProvider(() => this.onMutate());
  }

  async flush(): Promise<void> {
    if (this.backing.closed) {
      return;
    }
    if (this.backing.flushTimer) {
      clearTimeout(this.backing.flushTimer);
      this.backing.flushTimer = null;
    }
    this.queueFlush();
    if (this.backing.pendingFlush) {
      await this.backing.pendingFlush;
    }
  }

  close(): void {
    if (this.closed) {
      return;
    }
    this.closed = true;
    if (this.backing.closed) {
      return;
    }
    this.backing.refCount = Math.max(0, this.backing.refCount - 1);
    if (this.backing.refCount > 0) {
      return;
    }
    if (this.backing.flushTimer) {
      clearTimeout(this.backing.flushTimer);
      this.backing.flushTimer = null;
    }
    this.backing.closed = true;
    this.backing.db.close();
    sharedBackings().delete(this.key);
  }

  private scheduleFlush(): void {
    if (this.backing.closed) {
      return;
    }
    if (this.backing.flushDebounceMs === 0) {
      this.queueFlush();
      return;
    }
    if (this.backing.flushTimer) {
      clearTimeout(this.backing.flushTimer);
    }
    this.backing.flushTimer = setTimeout(() => {
      this.backing.flushTimer = null;
      this.queueFlush();
    }, this.backing.flushDebounceMs);
  }

  private queueFlush(): void {
    if (this.backing.closed || this.backing.pendingFlush) {
      return;
    }
    this.backing.pendingFlush = this.persistAll().finally(() => {
      this.backing.pendingFlush = null;
      if (this.backing.dirty) {
        this.queueFlush();
      }
    });
  }

  private onMutate(): void {
    this.backing.dirty = true;
    this.scheduleFlush();
  }

  private async persistAll(): Promise<void> {
    while (!this.backing.closed) {
      this.backing.dirty = false;
      const snapshot = this.backing.volume.serialize();
      await writeAllEntries(this.backing.db, this.backing.storeName, snapshot);
      if (!this.backing.dirty) {
        break;
      }
    }
  }
}

async function createSharedBacking(
  idb: IDBFactory,
  dbName: string,
  storeName: string,
  version: number,
  options: IndexedDbProviderOptions
): Promise<SharedIndexedDbBacking> {
  const db = await openDatabase(idb, dbName, storeName, version);
  const snapshot = await readAllEntries(db, storeName);
  const volume = new MemoryVolume({ now: options.now });
  volume.load(snapshot);
  return {
    db,
    storeName,
    volume,
    flushDebounceMs: options.flushDebounceMs ?? 25,
    refCount: 0,
    pendingFlush: null,
    flushTimer: null,
    closed: false,
    dirty: false
  };
}

function sharedBackingKey(dbName: string, storeName: string, version: number): string {
  return `${dbName}\0${storeName}\0${version}`;
}

function getIndexedDb(): IDBFactory {
  if (typeof indexedDB === "undefined") {
    throw new Error("indexedDB API is unavailable in this environment");
  }
  return indexedDB;
}

function openDatabase(
  idb: IDBFactory,
  dbName: string,
  storeName: string,
  version: number
): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = idb.open(dbName, version);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(storeName)) {
        db.createObjectStore(storeName, { keyPath: "path" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Failed to open IndexedDB database"));
    request.onblocked = () => reject(new Error("IndexedDB upgrade was blocked by another connection"));
  });
}

function readAllEntries(db: IDBDatabase, storeName: string): Promise<VolumeSnapshotEntry[]> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, "readonly");
    const store = tx.objectStore(storeName);
    const getAll = (store as any).getAll?.bind(store);
    if (typeof getAll === "function") {
      const request = getAll();
      request.onsuccess = () => {
        const raw = (request.result ?? []) as PersistedEntry[];
        resolve(raw.map(deserializeEntry));
      };
      request.onerror = () => reject(request.error ?? new Error("Failed to read IndexedDB entries"));
      return;
    }
    const entries: PersistedEntry[] = [];
    const cursor = store.openCursor();
    cursor.onsuccess = () => {
      const result = cursor.result;
      if (!result) {
        resolve(entries.map(deserializeEntry));
        return;
      }
      entries.push(result.value as PersistedEntry);
      result.continue();
    };
    cursor.onerror = () => reject(cursor.error ?? new Error("Failed to iterate IndexedDB cursor"));
  });
}

function writeAllEntries(
  db: IDBDatabase,
  storeName: string,
  entries: VolumeSnapshotEntry[]
): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, "readwrite");
    const store = tx.objectStore(storeName);
    const clearReq = store.clear();

    clearReq.onerror = () =>
      reject(clearReq.error ?? new Error("Failed to clear IndexedDB store before write"));
    clearReq.onsuccess = () => {
      for (const entry of entries) {
        const record: PersistedEntry = {
          path: entry.path,
          kind: entry.kind,
          readonly: entry.readonly,
          modified: entry.modified,
          children: entry.children ?? []
        };
        if (entry.kind === "file" && entry.data) {
          record.data = entry.data.slice().buffer;
        }
        store.put(record);
      }
    };

    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error("Failed to persist IndexedDB entries"));
    tx.onabort = () => reject(tx.error ?? new Error("IndexedDB write aborted"));
  });
}

interface PersistedEntry {
  path: string;
  kind: "file" | "dir";
  data?: ArrayBuffer;
  readonly: boolean;
  modified: number;
  children: string[];
}

function deserializeEntry(entry: PersistedEntry): VolumeSnapshotEntry {
  return {
    path: entry.path,
    kind: entry.kind,
    readonly: entry.readonly,
    modified: entry.modified,
    children: entry.children,
    data: entry.data ? new Uint8Array(entry.data.slice(0)) : undefined
  };
}
