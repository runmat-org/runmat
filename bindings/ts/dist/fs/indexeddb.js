import { MemoryVolume } from "./memory.js";
const DEFAULT_DB_NAME = "runmat-fs";
const DEFAULT_STORE_NAME = "entries";
export async function createIndexedDbFsHandle(options = {}) {
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
export async function createIndexedDbFsProvider(options) {
    const handle = await createIndexedDbFsHandle(options);
    return handle.provider;
}
class IndexedDbHandle {
    db;
    storeName;
    volume;
    flushDebounceMs;
    provider;
    pendingFlush = null;
    flushTimer = null;
    closed = false;
    dirty = false;
    constructor(db, storeName, volume, flushDebounceMs) {
        this.db = db;
        this.storeName = storeName;
        this.volume = volume;
        this.flushDebounceMs = flushDebounceMs;
        this.provider = volume.createProvider(() => this.onMutate());
    }
    async flush() {
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
    close() {
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
    scheduleFlush() {
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
    queueFlush() {
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
    onMutate() {
        this.dirty = true;
        this.scheduleFlush();
    }
    async persistAll() {
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
function getIndexedDb() {
    if (typeof indexedDB === "undefined") {
        throw new Error("indexedDB API is unavailable in this environment");
    }
    return indexedDB;
}
function openDatabase(idb, dbName, storeName, version) {
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
function readAllEntries(db, storeName) {
    return new Promise((resolve, reject) => {
        const tx = db.transaction(storeName, "readonly");
        const store = tx.objectStore(storeName);
        const getAll = store.getAll?.bind(store);
        if (typeof getAll === "function") {
            const request = getAll();
            request.onsuccess = () => {
                const raw = (request.result ?? []);
                resolve(raw.map(deserializeEntry));
            };
            request.onerror = () => reject(request.error ?? new Error("Failed to read IndexedDB entries"));
            return;
        }
        const entries = [];
        const cursor = store.openCursor();
        cursor.onsuccess = () => {
            const result = cursor.result;
            if (!result) {
                resolve(entries.map(deserializeEntry));
                return;
            }
            entries.push(result.value);
            result.continue();
        };
        cursor.onerror = () => reject(cursor.error ?? new Error("Failed to iterate IndexedDB cursor"));
    });
}
function writeAllEntries(db, storeName, entries) {
    return new Promise((resolve, reject) => {
        const tx = db.transaction(storeName, "readwrite");
        const store = tx.objectStore(storeName);
        const clearReq = store.clear();
        clearReq.onerror = () => reject(clearReq.error ?? new Error("Failed to clear IndexedDB store before write"));
        clearReq.onsuccess = () => {
            for (const entry of entries) {
                const record = {
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
function deserializeEntry(entry) {
    return {
        path: entry.path,
        kind: entry.kind,
        readonly: entry.readonly,
        modified: entry.modified,
        children: entry.children,
        data: entry.data ? new Uint8Array(entry.data.slice(0)) : undefined
    };
}
//# sourceMappingURL=indexeddb.js.map