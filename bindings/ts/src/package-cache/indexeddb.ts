import type {
  PackageCacheCommitOutcome,
  PackageCacheObjectWrite,
  PackageCacheSnapshot,
  PackageCacheTransaction,
  RunMatPackageCacheProvider
} from "./provider-types.js";

export interface IndexedDbPackageCacheOptions {
  dbName?: string;
  version?: number;
  requestPersistence?: boolean;
  faultInjector?: PackageCacheFaultInjector;
}

export interface PackageCacheFaultInjector {
  beforeApply?(transaction: IDBTransaction): void;
}

export interface IndexedDbPackageCacheHandle {
  provider: RunMatPackageCacheProvider;
  persistenceGranted: boolean | null;
  close(): void;
}

const DEFAULT_DB_NAME = "runmat-package-cache";
const DEFAULT_VERSION = 1;
const META_STORE = "meta";
const PAYLOAD_STORE = "payloads";
const STATE_KEY = "state";

interface StateRecord extends PackageCacheSnapshot {
  key: typeof STATE_KEY;
}

interface PayloadRecord {
  digest: string;
  bytes: ArrayBuffer;
}

export async function createIndexedDbPackageCache(
  options: IndexedDbPackageCacheOptions = {}
): Promise<IndexedDbPackageCacheHandle> {
  if (typeof indexedDB === "undefined") {
    throw new Error("indexedDB API is unavailable in this environment");
  }
  const db = await openDatabase(
    indexedDB,
    options.dbName ?? DEFAULT_DB_NAME,
    options.version ?? DEFAULT_VERSION
  );
  const persistenceGranted = options.requestPersistence
    ? await requestPersistentStorage()
    : null;
  return {
    provider: new IndexedDbPackageCacheProvider(db, options.faultInjector),
    persistenceGranted,
    close: () => db.close()
  };
}

class IndexedDbPackageCacheProvider implements RunMatPackageCacheProvider {
  constructor(
    private readonly db: IDBDatabase,
    private readonly faultInjector?: PackageCacheFaultInjector
  ) {}

  snapshot(): Promise<PackageCacheSnapshot | null> {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(META_STORE, "readonly");
      const request = transaction.objectStore(META_STORE).get(STATE_KEY);
      request.onsuccess = () => {
        const record = request.result as StateRecord | undefined;
        resolve(record ? cloneSnapshot(record) : null);
      };
      rejectOnRequest(request, reject, "Failed to read package cache state");
      rejectOnTransaction(transaction, reject, "Package cache snapshot transaction failed");
    });
  }

  initialize(initial: PackageCacheSnapshot): Promise<PackageCacheSnapshot> {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(META_STORE, "readwrite");
      const store = transaction.objectStore(META_STORE);
      const read = store.get(STATE_KEY);
      let result: PackageCacheSnapshot | null = null;
      read.onsuccess = () => {
        const existing = read.result as StateRecord | undefined;
        if (existing) {
          result = cloneSnapshot(existing);
          return;
        }
        const record: StateRecord = {
          key: STATE_KEY,
          revision: initial.revision,
          state: structuredClone(initial.state)
        };
        store.add(record);
        result = cloneSnapshot(record);
      };
      rejectOnRequest(read, reject, "Failed to initialize package cache state");
      transaction.oncomplete = () => {
        if (!result) {
          reject(new Error("Package cache initialization completed without state"));
        } else {
          resolve(result);
        }
      };
      rejectOnTransaction(transaction, reject, "Package cache initialization failed");
    });
  }

  commit(input: PackageCacheTransaction): Promise<PackageCacheCommitOutcome> {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([META_STORE, PAYLOAD_STORE], "readwrite");
      const stateStore = transaction.objectStore(META_STORE);
      const payloadStore = transaction.objectStore(PAYLOAD_STORE);
      const read = stateStore.get(STATE_KEY);
      let outcome: PackageCacheCommitOutcome | null = null;
      read.onsuccess = () => {
        const current = read.result as StateRecord | undefined;
        if (!current) {
          transaction.abort();
          reject(new Error("Package cache must be initialized before commit"));
          return;
        }
        if (!revisionEquals(current.revision, input.expected_revision)) {
          outcome = {
            outcome: "conflict",
            actual_revision: current.revision
          };
          return;
        }
        try {
          this.faultInjector?.beforeApply?.(transaction);
          for (const digest of normalizeDeletes(input.deletes)) {
            payloadStore.delete(digest);
          }
          for (const [digest, write] of normalizeWrites(input.writes)) {
            const bytes = normalizeBytes(write);
            if (bytes) {
              const copy = bytes.slice();
              payloadStore.put({
                digest,
                bytes: copy.buffer
              } satisfies PayloadRecord);
            } else {
              payloadStore.delete(digest);
            }
          }
          const revision = incrementRevision(current.revision);
          stateStore.put({
            key: STATE_KEY,
            revision,
            state: structuredClone(input.next_state)
          } satisfies StateRecord);
          outcome = { outcome: "committed", revision };
        } catch (error) {
          transaction.abort();
          reject(error);
        }
      };
      rejectOnRequest(read, reject, "Failed to read package cache commit revision");
      transaction.oncomplete = () => {
        if (!outcome) {
          reject(new Error("Package cache commit completed without an outcome"));
        } else {
          resolve(outcome);
        }
      };
      rejectOnTransaction(transaction, reject, "Package cache commit failed");
    });
  }

  readObjectBytes(digest: string): Promise<Uint8Array | null> {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(PAYLOAD_STORE, "readonly");
      const request = transaction.objectStore(PAYLOAD_STORE).get(digest);
      request.onsuccess = () => {
        const record = request.result as PayloadRecord | undefined;
        resolve(record ? new Uint8Array(record.bytes.slice(0)) : null);
      };
      rejectOnRequest(request, reject, "Failed to read package cache payload");
      rejectOnTransaction(transaction, reject, "Package cache payload transaction failed");
    });
  }
}

function openDatabase(factory: IDBFactory, name: string, version: number): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = factory.open(name, version);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(META_STORE)) {
        db.createObjectStore(META_STORE, { keyPath: "key" });
      }
      if (!db.objectStoreNames.contains(PAYLOAD_STORE)) {
        db.createObjectStore(PAYLOAD_STORE, { keyPath: "digest" });
      }
    };
    request.onsuccess = () => {
      const db = request.result;
      db.onversionchange = () => db.close();
      resolve(db);
    };
    request.onerror = () =>
      reject(request.error ?? new Error("Failed to open package cache database"));
    request.onblocked = () =>
      reject(new Error("Package cache database upgrade was blocked by another connection"));
  });
}

async function requestPersistentStorage(): Promise<boolean> {
  if (typeof navigator === "undefined") {
    return false;
  }
  try {
    return (await navigator.storage?.persist?.()) ?? false;
  } catch {
    return false;
  }
}

function normalizeWrites(
  writes: PackageCacheTransaction["writes"]
): Array<[string, PackageCacheObjectWrite]> {
  return writes instanceof Map ? Array.from(writes.entries()) : Object.entries(writes);
}

function normalizeDeletes(deletes: PackageCacheTransaction["deletes"]): string[] {
  return deletes instanceof Set ? Array.from(deletes) : deletes;
}

function normalizeBytes(write: PackageCacheObjectWrite): Uint8Array | null {
  if (write.bytes === null || write.bytes === undefined) {
    return null;
  }
  return write.bytes instanceof Uint8Array ? write.bytes : Uint8Array.from(write.bytes);
}

function cloneSnapshot(snapshot: PackageCacheSnapshot): PackageCacheSnapshot {
  return {
    revision: snapshot.revision,
    state: structuredClone(snapshot.state)
  };
}

function revisionEquals(left: number | bigint, right: number | bigint): boolean {
  return BigInt(left) === BigInt(right);
}

function incrementRevision(revision: number | bigint): number | bigint {
  if (typeof revision === "bigint") {
    return revision + 1n;
  }
  if (!Number.isSafeInteger(revision) || revision < 0 || revision === Number.MAX_SAFE_INTEGER) {
    throw new Error("Package cache revision is outside JavaScript's safe integer range");
  }
  return revision + 1;
}

function rejectOnRequest(
  request: IDBRequest,
  reject: (reason?: unknown) => void,
  message: string
): void {
  request.onerror = () => reject(request.error ?? new Error(message));
}

function rejectOnTransaction(
  transaction: IDBTransaction,
  reject: (reason?: unknown) => void,
  message: string
): void {
  transaction.onerror = () => reject(transaction.error ?? new Error(message));
  transaction.onabort = () => reject(transaction.error ?? new Error(message));
}
