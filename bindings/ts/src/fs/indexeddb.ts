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

export async function createIndexedDbFsHandle(
  options: IndexedDbProviderOptions = {}
): Promise<IndexedDbFsHandle> {
  const idb = getIndexedDb();
  const dbName = options.dbName ?? DEFAULT_DB_NAME;
  const storeName = options.storeName ?? DEFAULT_STORE_NAME;
  const version = options.version ?? 1;
  const db = await openDatabase(idb, dbName, storeName, version);
  const snapshot = await readAllEntries(db, storeName);
  const volume = new MemoryVolume({ now: options.now });
  volume.load(snapshot);
  return new IndexedDbHandle(db, storeName, volume, options.flushDebounceMs ?? 25);
}

export async function createIndexedDbFsProvider(
  options?: IndexedDbProviderOptions
): Promise<RunMatFilesystemProvider> {
  const handle = await createIndexedDbFsHandle(options);
  return handle.provider;
}

class IndexedDbHandle implements IndexedDbFsHandle {
  public readonly provider: RunMatFilesystemProvider;
  private pendingFlush: Promise<void> | null = null;
  private flushTimer: ReturnType<typeof setTimeout> | null = null;
  private closed = false;
  private dirty = false;

  constructor(
    private readonly db: IDBDatabase,
    private readonly storeName: string,
    private readonly volume: MemoryVolume,
    private readonly flushDebounceMs: number
  ) {
    this.provider = volume.createProvider(() => this.onMutate());
  }

  async flush(): Promise<void> {
    if (this.closed) {
      return;
    }
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }
    this.queueFlush();
    if (this.pendingFlush) {
      await this.pendingFlush;
    }
  }

  close(): void {
    if (this.closed) {
      return;
    }
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }
    this.closed = true;
    this.db.close();
  }

  private scheduleFlush(): void {
    if (this.closed) {
      return;
    }
    if (this.flushDebounceMs === 0) {
      this.queueFlush();
      return;
    }
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
    }
    this.flushTimer = setTimeout(() => {
      this.flushTimer = null;
      this.queueFlush();
    }, this.flushDebounceMs);
  }

  private queueFlush(): void {
    if (this.closed || this.pendingFlush) {
      return;
    }
    this.pendingFlush = this.persistAll().finally(() => {
      this.pendingFlush = null;
      if (this.dirty) {
        this.queueFlush();
      }
    });
  }

  private onMutate(): void {
    this.dirty = true;
    this.scheduleFlush();
  }

  private async persistAll(): Promise<void> {
    while (!this.closed) {
      this.dirty = false;
      const snapshot = this.volume.serialize();
      await writeAllEntries(this.db, this.storeName, snapshot);
      if (!this.dirty) {
        break;
      }
    }
  }
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
    if ("getAll" in store) {
      const request = store.getAll();
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
